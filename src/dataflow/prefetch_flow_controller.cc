#include "prefetch_flow_controller.h"
#include "forward_def.h"
#include "utils/memory_utils.h"

void
PrefetchFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node,
    const NodeMetaPtr& node_meta)
{
  // NodeID node_id = node->id;
  // std::size_t memory_size = node->byte_size;
  LOG_TRITON_VERBOSE("PrefetchFlowController::RecordNode");
  std::size_t request_id = std::hash<std::string>{}(input_id->request_id);
  PutNodeToPipeline(request_id, input_id->correlation_id, node);

  auto high_id = input_id->correlation_id >> 32;
  auto low_id = input_id->correlation_id & 0xFFFFFFFF;

  auto now = MCIROSECONDS_SINCE_EPOCH;
  if (request_time_.find(request_id) == request_time_.end()) {
    request_time_.insert({request_id, now});
  }
  request_time_[request_id] = now;

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
  if (request_trace_.find(request_id) == request_trace_.end()) {
    request_trace_.insert({request_id, stage});
  }
  request_trace_[request_id] = stage;
}


NodeMoveVec
PrefetchFlowController::PrefetchNode(const NodePtr& node)
{
  LOG_TRITON_VERBOSE("PrefetchFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;

  if (node->memory_type == MemoryType::kStandBy) {
    node->memory_type = MemoryType::kMoving;
    prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  }

  // only prefetch from a sparse node
  // auto high_id = node->id >> 32;
  auto low_id = node->id & 0xFFFFFFFF;
  auto stage = pipeline_.stages[low_id];
  if (stage->is_sparse == false) {
    return prefetch_nodes;
  }

  auto [live_memory, live_node_list] = GetTotalLiveParamSize();
  LOG_TRITON_VERBOSE(("DeepSpeedFlowController::PrefetchNode: live_memory = " +
                      std::to_string(live_memory) + " MAX_LIVE_PARAMETERS = " +
                      std::to_string(MAX_LIVE_PARAMETERS))
                         .c_str());

  auto [gpu_memory, gpu_node_list] = GetTotalGPUParamSize();
  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: gpu_memory = " +
       std::to_string(gpu_memory) +
       " free_gpu_memory_ = " + std::to_string(free_gpu_memory_) +
       " real free gpu memory = " + std::to_string(GetFreeDeviceMemory(0)))
          .c_str());

  std::size_t prefetch_size =
      std::min(PREFETCH_BUCKET_SIZE, MAX_LIVE_PARAMETERS - live_memory);
  auto [prefetch_memory, prefetch_node_list] =
      GetStandbyChildByFreq(node, prefetch_size);
  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: prefetch_size = " +
       std::to_string(prefetch_size) +
       " prefetch_memory = " + std::to_string(prefetch_memory))
          .c_str());

  for (auto& prefetch_node : prefetch_nodes) {
    if (prefetch_node.second != DEFAULT_CUDA_DEVICE) {
      free_gpu_memory_ += prefetch_node.first->byte_size;
    }
  }

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
    const NodePtr& node, std::size_t size_limit)
{
  NodeID current_node_id = node->id;
  std::size_t low_corr_id = current_node_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = current_node_id >> 32;        // request id
  NodeBodyPtr current_node_body =
      pipeline_.stages[low_corr_id]->nodes[high_corr_id];

  NodePtrList prefetch_node_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, prefetch_node_list);
  }


  std::deque<NodeBodyPtr> visited_sparse_nodes;
  visited_sparse_nodes.push_back(current_node_body);
  while (low_corr_id < pipeline_.stages.size()) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    if (stage->is_sparse) {
      // Sparse stage, we only prefetch the first layer
      auto router = stage->root;
      total_size += router->node->byte_size;
      if (total_size >= size_limit) {
        break;
      }


      auto visited = visited_sparse_nodes;
      for (auto& node_body : visited) {
        visited_sparse_nodes.pop_front();
        auto children = node_body->children;
        auto children_cnt = node_body->children_visit_cnt;
        // sort node child visit cnt decending
        auto argsort = sort_indexes(children_cnt);
        for (int j = 0; j < 3; j++) {
          if (node_body->children_visit_cnt[argsort[j]] == 0) {
            break;
          }
          BREAK_IF(children[j]->node);
          visited_sparse_nodes.push_back(children[j]);
        }
      }
      if (total_size >= size_limit) {
        break;
      }
    } else {
      // Dense stage
      auto node_body = stage->nodes[0];
      total_size += node_body->node->byte_size;
      if (total_size >= size_limit) {
        break;
      }
      if (node_body->node->memory_type == MemoryType::kStandBy) {
        prefetch_node_list.push_back(node_body->node);
      }
    }

    low_corr_id += 1;
  }

  std::size_t size = 0;
  for (auto& node : prefetch_node_list) {
    size += node->byte_size;
  }
  return std::make_pair(size, prefetch_node_list);
}
