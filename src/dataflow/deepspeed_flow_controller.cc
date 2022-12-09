#include "deepspeed_flow_controller.h"
#include "utils/memory_utils.h"
#include "forward_def.h"

void
DeepSpeedFlowController::RecordNode(
    const InputIDPtr& input_id, const DAGNodePtr& node,
    const NodeMetaPtr& node_meta)
{
  NodeID node_id = node->GetNodeID();
  std::size_t memory_size = node->GetNodeByteSize();
  LOG_TRITON_VERBOSE("DeepSpeedFlowController::RecordNode");
  PutNodeTopology(input_id->correlation_id, node);
  if (node_location_.find(node_id) == node_location_.end()) {
    if (free_cpu_memory_ > memory_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
    }
    // node_location_.insert({node_id, node->GetDevice()});
  }
  node->SetLastAccessTime();
}

NodeMoveVec
DeepSpeedFlowController::PrefetchNode(const DAGNodePtr& node)
{
  NodeID node_id = node->GetNodeID();
  LOG_TRITON_VERBOSE("DeepSpeedFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;
  if (node->GetMemoryType() == MemoryType::kStandBy) {
    node->SetMemoryType(MemoryType::kMoving);
    prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  }

  // Assign lock flag here for parameters immediately required
  // Wait for fetching at its own thread
  node->SetMemoryType(MemoryType::kLocked);

  LOG_TRITON_VERBOSE(("DeepSpeedFlowController::PrefetchNode: node_id = " +
                      std::to_string(node_id))
                         .c_str());

  if (root_ == nullptr) {
    return prefetch_nodes;
  }

  /*
  This method does the following (in order):
      1. kick off fetch for parameters in immediately required sub module
      2. kick off fetch for next few parameters we will need later (prefetch)
      3. block on parameters in immediately required sub module
  */

  std::size_t total_live_params = 0;
  std::size_t used_gpu_memory = 0;

  auto live_filter = THIS_BIND_ARGS(
      DeepSpeedFlowController, ParamLiveFilter, std::placeholders::_1,
      &total_live_params);
  auto live_node_list = GetNodesByFilter(live_filter, root_->GetNodeID());

  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: total_live_params = " +
       std::to_string(total_live_params) +
       " MAX_LIVE_PARAMETERS = " + std::to_string(MAX_LIVE_PARAMETERS))
          .c_str());

  auto gpu_filter = THIS_BIND_ARGS(
      DeepSpeedFlowController, ParamGPUFilter, std::placeholders::_1,
      &used_gpu_memory);
  auto gpu_node_list = GetNodesByFilter(gpu_filter, root_->GetNodeID());

  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: used_gpu_memory = " +
       std::to_string(used_gpu_memory) +
       " free_gpu_memory_ = " + std::to_string(free_gpu_memory_) +
       " real free gpu memory = " + std::to_string(GetFreeDeviceMemory(0)))
          .c_str());

  std::size_t prefetch_size =
      std::min(PREFETCH_BUCKET_SIZE, MAX_LIVE_PARAMETERS - total_live_params);
  auto size_filter = THIS_BIND_ARGS(
      DeepSpeedFlowController, MemorySizeFilter, std::placeholders::_1,
      &prefetch_size);

  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: prefetch_size = " +
       std::to_string(prefetch_size))
          .c_str());

  auto node_ptr_list = GetNodesByFilter(size_filter, node_id);
  for (auto& prefetch_node : node_ptr_list) {
    // NodeID prefetch_node_id = prefetch_node->GetNodeID();
    if (prefetch_node->GetMemoryType() == MemoryType::kStandBy) {
      prefetch_node->SetMemoryType(MemoryType::kMoving);
      prefetch_nodes.push_back(
          std::make_pair(prefetch_node, DEFAULT_CUDA_DEVICE));
    }
  }

  std::size_t prefetch_gpu_size = 0;
  for (auto& prefetch_node : prefetch_nodes) {
    prefetch_gpu_size += prefetch_node.first->GetNodeByteSize();
  }

  LOG_TRITON_VERBOSE(
      ("DeepSpeedFlowController::PrefetchNode: prefetch_gpu_size = " +
       std::to_string(prefetch_gpu_size))
          .c_str());

  if (prefetch_gpu_size + used_gpu_memory > free_cpu_memory_) {
    // not enough memory to prefetch, remove some of the non-live nodesaccording
    // to last access time
    std::int64_t remove_size =
        prefetch_gpu_size + used_gpu_memory - free_cpu_memory_;
    auto remove_filter = THIS_BIND_ARGS(
        DeepSpeedFlowController, RemoveFilter, std::placeholders::_1);
    auto remove_node_list = GetNodesByFilter(remove_filter, root_->GetNodeID());

    std::sort(
        remove_node_list.begin(), remove_node_list.end(),
        [](const DAGNodePtr& a, const DAGNodePtr& b) {
          return a->GetLastAccessTime() < b->GetLastAccessTime();
        });

    LOG_TRITON_VERBOSE(
        ("DeepSpeedFlowController::PrefetchNode: remove_size = " +
         std::to_string(remove_size))
            .c_str());

    for (auto& remove_node : remove_node_list) {
      remove_node->SetMemoryType(MemoryType::kMoving);
      auto device_it = node_location_.find(remove_node->GetNodeID());
      prefetch_nodes.push_back(std::make_pair(remove_node, device_it->second));
      // DispatchNodeMemoryInThread(remove_node, device_it->second);
      remove_size -= remove_node->GetNodeByteSize();
      if (remove_size <= 0) {
        break;
      }
    }
    free_gpu_memory_ -= remove_size;
  }

  // for (auto& prefetch_node : prefetch_nodes) {
  //   DispatchNodeMemoryInThread(prefetch_node, DEFAULT_CUDA_DEVICE);
  // }

  // while (node->GetMemoryType() != MemoryType::kReady) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1));
  // }
  return prefetch_nodes;
  // MAX_REUSE_DISTANCE is not implemented yet
}
