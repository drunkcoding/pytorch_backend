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
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
    }
  }
}


NodeMoveVec
PrefetchFlowController::PrefetchNode(const NodePtr& node)
{
  // LOG_TRITON_VERBOSE("PrefetchFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;

  // if (node->memory_type == MemoryType::kStandBy) {
  //   node->memory_type = MemoryType::kMoving;
  //   prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  // }

  // if (pipeline_.stages.empty()) {
  //   UpdateMemoryManager(prefetch_nodes);
  //   return prefetch_nodes;
  // }

  // // only prefetch from a sparse node
  // // auto high_id = node->id >> 32;
  // auto low_id = node->id & 0xFFFFFFFF;
  // auto stage = pipeline_.stages[low_id];
  // if (stage->is_sparse == false) {
  //   return prefetch_nodes;
  // }

  SizeFilterFunc size_filter = THIS_BIND_ARGS(
      PrefetchFlowController, GetStandbyChildByFreq, node,
      std::placeholders::_1, std::placeholders::_2);
  // UpdateInitPrefetchNodes(prefetch_nodes, size_filter);
  bool gpu_prefetch = false;
  bool cpu_prefetch = false;

  do {
    if (!gpu_prefetch)
      gpu_prefetch =
          CreatePrefetchThreads(node, size_filter, DEFAULT_CUDA_DEVICE);
    if (!cpu_prefetch)
      cpu_prefetch = CreatePrefetchThreads(node, size_filter, CPU_DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (!gpu_prefetch || !cpu_prefetch);

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
  std::size_t high_corr_id = current_node_id >> 32;        // request id

  NodePtrList node_ptr_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, node_ptr_list);
  }

  if (pipeline_.stages[low_corr_id] == nullptr) {
    return std::make_pair(total_size, node_ptr_list);
  }

  auto current_node_body =
      pipeline_.stages[low_corr_id]
          ->nodes[(high_corr_id > 0) ? (high_corr_id - 1) : (0)];

  std::deque<NodeBodyPtr> visited_sparse_nodes;
  visited_sparse_nodes.push_back(current_node_body);
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
          if (total_size + children[j]->node->byte_size > size_limit) {
            return std::make_pair(total_size, node_ptr_list);
          }
          if (((!children[j]->node->device.is_cuda() && device.is_cuda()) ||
               (device.is_cpu() && children[j]->node->device == DISK_DEVICE)) &&
              children[j]->node->mutex.try_lock()) {
            total_size += children[j]->node->byte_size;
            node_ptr_list.push_back(children[j]->node);
          }
          visited_sparse_nodes.push_back(children[j]);
        }
      }
      if (total_size >= size_limit) {
        break;
      }
    } else {
      // Dense stage
      auto node_body = stage->nodes[0];
      BREAK_IF_NULL(node_body);
      if (total_size + node_body->node->byte_size > size_limit) {
        return std::make_pair(total_size, node_ptr_list);
      }
      if (((!node_body->node->device.is_cuda() && device.is_cuda()) ||
           (device.is_cpu() && node_body->node->device == DISK_DEVICE)) &&
          node_body->node->mutex.try_lock()) {
        total_size += node_body->node->byte_size;
        node_ptr_list.push_back(node_body->node);
      }
    }

    low_corr_id += 1;
  }

  std::size_t size = 0;
  for (auto& node : node_ptr_list) {
    size += node->byte_size;
  }
  return std::make_pair(size, node_ptr_list);
}
