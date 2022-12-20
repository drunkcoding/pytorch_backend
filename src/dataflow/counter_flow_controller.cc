#include "counter_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"
#include "utils/time_utils.h"

void
CounterFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node,
    const NodeMetaPtr& node_meta)
{
  auto node_id = node->id;
  auto memory_size = node->byte_size;
  std::size_t hash_request_id =
      std::hash<std::string>{}(input_id->request_id);
  LOG_TRITON_VERBOSE("CounterFlowController::RecordNode");
  PutNodeToPipeline(hash_request_id, input_id->correlation_id, node);
  if (node_location_.find(node_id) == node_location_.end()) {
    if (free_cpu_memory_ > memory_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
    }
    // node_location_.insert({node_id, node->device});
  }
}

NodeMoveVec
CounterFlowController::PrefetchNode(const NodePtr& node)
{
  NodeID node_id = node->id;
  LOG_TRITON_VERBOSE("CounterFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;
  if (node->memory_type == MemoryType::kStandBy) {
    node->memory_type = MemoryType::kMoving;
    prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
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
      GetStandbyChildByCount(node_id, prefetch_size);
  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: prefetch_size = " +
       std::to_string(prefetch_size) +
       " prefetch_memory = " + std::to_string(prefetch_memory))
          .c_str());

  for (auto& prefetch_node : prefetch_node_list) {
    prefetch_node->memory_type = MemoryType::kMoving;
    prefetch_nodes.push_back(
        std::make_pair(prefetch_node, DEFAULT_CUDA_DEVICE));
  }

  if (prefetch_memory + gpu_memory > free_gpu_memory_) {
    // not enough memory to prefetch, remove some of the non-live nodesaccording
    // to last access time
    std::int64_t remove_size = prefetch_memory + gpu_memory - free_gpu_memory_;

    auto [removable_memory, removable_node_list] = GetRemovableNodes();
    LOG_TRITON_VERBOSE(
        ("DeepSpeedFlowController::PrefetchNode: remove_size = " +
         std::to_string(remove_size))
            .c_str());

    for (auto& remove_node : removable_node_list) {
      remove_node->memory_type = MemoryType::kMoving;
      auto device_it = node_location_.find(remove_node->id);
      prefetch_nodes.push_back(std::make_pair(remove_node, device_it->second));
      // DispatchNodeMemoryInThread(remove_node, device_it->second);
      remove_size -= remove_node->byte_size;
      if (remove_size <= 0) {
        break;
      }
    }
  }

  for (auto& prefetch_node : prefetch_nodes) {
    if (prefetch_node.second != DEFAULT_CUDA_DEVICE) {
      free_gpu_memory_ += prefetch_node.first->byte_size;
    }
  }

  return prefetch_nodes;
}


FilterResult
CounterFlowController::GetStandbyChildByCount(
    const NodeID node_id, std::size_t size_limit)
{
  NodeID current_node_id = node_id;

  std::size_t low_corr_id = current_node_id & 0xFFFFFFFF;  // stage id
  NodePtrList prefetch_node_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, prefetch_node_list);
  }

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

      auto copy_nodes = stage->nodes;

      std::sort(
          copy_nodes.begin(), copy_nodes.end(),
          [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
            return a->visit_cnt > b->visit_cnt;
          });
      for (int j = 0; j < 3; j++) {
        total_size += copy_nodes[j]->node->byte_size;
        if (total_size >= size_limit) {
          break;
        }
        if (copy_nodes[j]->node->memory_type == MemoryType::kStandBy) {
          prefetch_node_list.push_back(copy_nodes[j]->node);
        }
      }

      break;
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
