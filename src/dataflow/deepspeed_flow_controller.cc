#include "deepspeed_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"

void
DeepSpeedFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node,
    const NodeMetaPtr& node_meta)
{
  NodeID node_id = node->id;
  std::size_t memory_size = node->byte_size;
  std::size_t request_id = std::hash<std::string>{}(input_id->request_id);
  LOG_TRITON_VERBOSE("DeepSpeedFlowController::RecordNode");
  PutNodeToPipeline(request_id, input_id->correlation_id, node);
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
DeepSpeedFlowController::PrefetchNode(const NodePtr& node)
{
  NodeID node_id = node->id;
  LOG_TRITON_VERBOSE("DeepSpeedFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;
  if (node->memory_type == MemoryType::kStandBy) {
    node->memory_type = MemoryType::kMoving;
    prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  } 

  // Assign lock flag here for parameters immediately required
  // Wait for fetching at its own thread
  node->memory_type = MemoryType::kLocked;

  LOG_TRITON_VERBOSE(("DeepSpeedFlowController::PrefetchNode: node_id = " +
                      std::to_string(node_id))
                         .c_str());

  if (pipeline_.stages.empty()) {
    return prefetch_nodes;
  }

  /*
  This method does the following (in order):
      1. kick off fetch for parameters in immediately required sub module
      2. kick off fetch for next few parameters we will need later (prefetch)
      3. block on parameters in immediately required sub module
  */

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
      GetStandbyChildBySizeLimit(node, prefetch_size);
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
    // free_gpu_memory_ -= remove_size;
  }

  for (auto& prefetch_node : prefetch_nodes) {
    if (prefetch_node.second != DEFAULT_CUDA_DEVICE) {
      free_gpu_memory_ += prefetch_node.first->byte_size;
    }
  }

  for (auto& prefetch_node : prefetch_nodes) {
    if (prefetch_node.second == DEFAULT_CUDA_DEVICE) {
      free_gpu_memory_ += prefetch_node.first->byte_size;
    }
  }

  // while (node->memory_type != MemoryType::kReady) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1));
  // }
  return prefetch_nodes;
  // MAX_REUSE_DISTANCE is not implemented yet
}
