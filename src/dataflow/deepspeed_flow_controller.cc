#include "deepspeed_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"

void
DeepSpeedFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node)
{
  // std::size_t request_id = std::hash<std::string>{}(input_id->request_id);
  // LOG_TRITON_VERBOSE("DeepSpeedFlowController::RecordNode");
  PutNodeToPipeline(input_id->request_id, input_id->correlation_id, node);

  NodeID node_id = node->id;
  auto memory_size = node->byte_size;
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
  // // if node is already in GPU or mving towards GPU
  // // let the model thread wait for the node to be loaded
  // if (node->device == DEFAULT_CUDA_DEVICE ||
  //     node->memory_type == MemoryType::kEmplacing) {
  //   // lock here for inference
  //   node->memory_type = MemoryType::kLocked;
  // }

  // // if node is mving out of GPU
  // // let the executiuon finished and decide later
  // if (node->device != DEFAULT_CUDA_DEVICE &&
  //     node->memory_type != MemoryType::kStandBy) {
  //   return NodeMoveVec();
  // }

  // if (gpu_memory_manager_->GetTotalMemory() <= 0) {
  //   // We are out of space here, let's wait for other models to be unloaded
  //   return NodeMoveVec();
  // }

  // if (node->memory_type == MemoryType::kStandBy) {
  //   // lock here for inference
  //   node->memory_type = MemoryType::kLocked;
  //   prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  // }

  // NodeID node_id = node->id;

  /*
  This method does the following (in order):
      1. kick off fetch for parameters in immediately required sub module
      2. kick off fetch for next few parameters we will need later (prefetch)
      3. block on parameters in immediately required sub module
  */
  NodeMoveVec prefetch_nodes;
  SizeFilterFunc size_filter = THIS_BIND_ARGS(
      DeepSpeedFlowController, GetStandbyChildBySizeLimit, node,
      std::placeholders::_1, std::placeholders::_2);
  // UpdateInitPrefetchNodes(prefetch_nodes, size_filter);

  std::vector<bool> gpu_prefetch(GetDeviceCount(), false);
  bool cpu_prefetch = true;
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
    // if (!cpu_prefetch)
    //   cpu_prefetch = CreatePrefetchThreads(node, size_filter, CPU_DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (!all_gpu_prefetch || !cpu_prefetch);

  return prefetch_nodes;
  // MAX_REUSE_DISTANCE is not implemented yet
}

FilterResult
DeepSpeedFlowController::GetStandbyChildBySizeLimit(
    const NodePtr& node, const std::int64_t size_limit, const Device& device)
{
  // std::int64_t size = node->byte_size;
  // NodePtrList node_ptr_list;

  // if (size > size_limit) {
  //   return std::make_pair(0, node_ptr_list);
  // }

  // // if (node->memory_type == MemoryType::kStandBy)
  // node_ptr_list.push_back(node);

  std::int64_t size = 0;
  NodePtrList node_ptr_list;

  // return std::make_pair(size, node_ptr_list);

  for (std::uint64_t stage_idx = (node->corr_id & 0x00000000FFFFFFFF) + 1;
       stage_idx < pipeline_.stages.size(); ++stage_idx) {
    auto stage = pipeline_.stages[stage_idx];
    BREAK_IF_NULL(stage)
    for (auto& node_body : stage->nodes) {
      CONTINUE_IF_NULL(node_body)
      if (size + node_body->node->byte_size > size_limit) {
        return std::make_pair(size, node_ptr_list);
      }

      if (!node_body->node->device.is_cuda() &&
          device == node_body->node->default_device) {
        size += node_body->node->byte_size;
        if (node_body->node->mutex.try_lock())
          node_ptr_list.push_back(node_body->node);
      }
    }
  }
  return std::make_pair(size, node_ptr_list);
}
