#include "prefetch_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"

void
PrefetchFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node)
{
  // NodeID node_id = node->id;
  // std::size_t memory_size = node->byte_size;
  // LOG_TRITON_VERBOSE("PrefetchFlowController::RecordNode");
  // std::size_t request_id = std::hash<std::string>{}(input_id->request_id);
  PutNodeToPipeline(input_id->request_id, input_id->correlation_id, node);

  // auto high_id = input_id->correlation_id >> 32;
  auto low_id = input_id->correlation_id & 0xFFFFFFFF;

  auto now = MCIROSECONDS_SINCE_EPOCH;
  if (request_time_.find(input_id->request_id) == request_time_.end()) {
    request_time_.insert({input_id->request_id, now});
  }
  request_time_[input_id->request_id] = now;

  std::vector<std::size_t> delete_request;
  std::size_t microseconds = 60000000;
  for (auto& [request_id, time] : request_time_) {
    if (now - time > microseconds) {
      delete_request.push_back(request_id);
    }
  }

  for (auto& request_id : delete_request) {
    request_time_.erase(request_id);
    request_trace_.erase(request_id);
  }

  auto stage = pipeline_.stages[low_id];
  if (request_trace_.find(input_id->request_id) == request_trace_.end()) {
    request_trace_.insert({input_id->request_id, stage});
  }
  request_trace_[input_id->request_id] = stage;

  auto node_id = node->id;
  auto memory_size = node->byte_size;
  if (node_location_.find(node_id) == node_location_.end()) {
    if (free_cpu_memory_ > memory_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      node->default_host = CPU_DEVICE;
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
      node->default_host = DISK_DEVICE;
    }
  }
  visit_count_ += 1;
  if (visit_count_ % 500 == 0) {
    ReorderNodeLocations();
  }
}


NodeMoveVec
PrefetchFlowController::PrefetchNode(const NodePtr& node)
{
  // LOG_TRITON_VERBOSE("PrefetchFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;

  SizeFilterFunc size_filter = THIS_BIND_ARGS(
      PrefetchFlowController, GetStandbyChildByFreq, node,
      std::placeholders::_1, std::placeholders::_2);
  // UpdateInitPrefetchNodes(prefetch_nodes, size_filter);
  std::vector<bool> gpu_prefetch(GetDeviceCount(), false);
  bool cpu_prefetch = false;
  bool all_gpu_prefetch = false;

  do {
    for (std::size_t gpu = 0; gpu < gpu_prefetch.size(); ++gpu) {
      if (gpu_prefetch[gpu] == false)
        gpu_prefetch[gpu] =
            CreatePrefetchThreads(node, size_filter, CUDA_DEVICE(gpu));
    }

    // set all_gpu_prefetch to true if all gpu_prefetch is true
    all_gpu_prefetch = std::all_of(
        gpu_prefetch.begin(), gpu_prefetch.end(), [](bool v) { return v; });
    if (!cpu_prefetch)
      cpu_prefetch = CreatePrefetchThreads(node, size_filter, CPU_DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (!all_gpu_prefetch || !cpu_prefetch);

  return prefetch_nodes;
}

template <typename T>
std::vector<std::size_t>
sort_indexes(const std::vector<T>& v)
{
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(
      idx.begin(), idx.end(),
      [&v](std::size_t i1, std::size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

FilterResult
PrefetchFlowController::GetStandbyChildByFreq(
    const NodePtr& node, const std::size_t size_limit, const Device& device)
{
  NodeID current_node_id = node->corr_id;
  std::size_t low_corr_id = current_node_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = current_node_id >> 32;        // node id

  NodePtrList node_ptr_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, node_ptr_list);
  }
  // LOG_TRITON_VERBOSE(
  //     ("GetStandbyChildByFreq: low_corr_id: " + std::to_string(low_corr_id))
  //         .c_str());

  if (pipeline_.stages[low_corr_id] == nullptr) {
    return std::make_pair(total_size, node_ptr_list);
  }
  // LOG_TRITON_VERBOSE(
  //     ("GetStandbyChildByFreq: pipeline_.stages[low_corr_id]->nodes.size(): " +
  //      std::to_string(pipeline_.stages[low_corr_id]->nodes.size()))
  //         .c_str());

  high_corr_id = (high_corr_id > 0) ? (high_corr_id - 1) : (0);
  if (high_corr_id >= pipeline_.stages[low_corr_id]->nodes.size()) {
    return std::make_pair(total_size, node_ptr_list);
  }

  auto current_node_body = pipeline_.stages[low_corr_id]->nodes[high_corr_id];

  if (current_node_body == nullptr) {
    return std::make_pair(total_size, node_ptr_list);
  }

  std::deque<NodeBodyPtr> visited_sparse_nodes;
  visited_sparse_nodes.push_back(current_node_body);

  auto init_low_corr_id = low_corr_id;
  low_corr_id++;
  while (low_corr_id < pipeline_.stages.size()) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    BREAK_IF_NULL(stage);

    if (stage->is_sparse) {
      // Sparse stage, we only prefetch the first layer

      auto visited = visited_sparse_nodes;
      for (auto& node_body : visited) {
        visited_sparse_nodes.pop_front();
        auto children = node_body->children;
        auto children_cnt = node_body->children_visit_cnt;

        // LOG_TRITON_VERBOSE(("GetStandbyChildByFreq: children.size(): " +
        //                     std::to_string(children.size()))
        //                        .c_str());

        // keep only not null from children and children_cnt
        children.erase(
            std::remove_if(
                children.begin(), children.end(),
                [](const NodeBodyPtr& node_body) {
                  return node_body == nullptr;
                }),
            children.end());
        children_cnt.erase(
            std::remove_if(
                children_cnt.begin(), children_cnt.end(),
                [](const std::size_t& cnt) { return cnt == 0; }),
            children_cnt.end());

        // LOG_TRITON_VERBOSE(
        //     ("GetStandbyChildByFreq: children.size() after cleanup: " +
        //      std::to_string(children.size()))
        //         .c_str());

        // sort node child visit cnt decending
        auto argsort = sort_indexes(children_cnt);
        std::size_t layer_size = argsort.size();
        std::size_t max_layer_fetch_size = 50;

        auto low_idx = (device.is_cuda()) ? 0 : 1;
        auto high_idx = std::min(
            low_idx + max_layer_fetch_size * ((device.is_cuda()) ? 1 : 8),
            layer_size);
        for (std::size_t j = low_idx; j < high_idx; j++) {
          if (node_body->children_visit_cnt[argsort[j]] == 0) {
            break;
          }
          BREAK_IF_EXCEED_SIZE_LIMIT(total_size, size_limit, children[j]->node);
          // LOG_TRITON_VERBOSE(
          //     ("GetStandbyChildByFreq: children[argsort[j]]->node->corr_id: " +
          //      std::to_string(children[argsort[j]]->node->corr_id))
          //         .c_str());
          APPEND_NODE(
              total_size, children[argsort[j]]->node, node_ptr_list, device);
          visited_sparse_nodes.push_back(children[argsort[j]]);
        }
      }
      if (total_size >= size_limit) {
        break;
      }
    } else {
      // Dense stage
      auto node_body = stage->nodes[0];
      BREAK_IF_NULL(node_body);
      BREAK_IF_EXCEED_SIZE_LIMIT(total_size, size_limit, node_body->node);
      APPEND_NODE(total_size, node_body->node, node_ptr_list, device);
    }

    low_corr_id += 1;
    if (low_corr_id >= init_low_corr_id + 32) {
      break;
    }
  }

  // LOG_TRITON_VERBOSE(("GetStandbyChildByFreq: node_ptr_list.size(): " +
  //                     std::to_string(node_ptr_list.size()))
  //                        .c_str());

  // remove node from node_ptr_list
  node_ptr_list.erase(
      std::remove_if(
          node_ptr_list.begin(), node_ptr_list.end(),
          [node](const NodePtr& prefetch_node) {
            return prefetch_node == node;
          }),
      node_ptr_list.end());

  // LOG_TRITON_VERBOSE(
  //     ("GetStandbyChildByFreq: node_ptr_list.size() after cleanup: " +
  //      std::to_string(node_ptr_list.size()))
  //         .c_str());

  return std::make_pair(total_size, node_ptr_list);
}

void
PrefetchFlowController::ReorderNodeLocations()
{
  // Sort node body according to the visit count
  std::vector<NodeBodyPtr> node_bodies;
  for (auto& stage : pipeline_.stages) {
    CONTINUE_IF_NULL(stage);
    for (auto& node_body : stage->nodes) {
      CONTINUE_IF_NULL(node_body);
      node_bodies.push_back(node_body);
    }
  }

  std::sort(
      node_bodies.begin(), node_bodies.end(),
      [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
        return a->visit_cnt > b->visit_cnt;
      });

  // Reorder node locations highest visit count first to CPU
  auto free_cpu_mem = DEFAULT_SYSTEM_FREE_MEMORY;
  node_location_.clear();
  for (auto& node_body : node_bodies) {
    auto node_id = node_body->node->id;
    if (free_cpu_mem > node_body->node->byte_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      free_cpu_mem -= node_body->node->byte_size;
      node_body->node->default_host = CPU_DEVICE;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
      node_body->node->default_host = DISK_DEVICE;
    }
  }
}
