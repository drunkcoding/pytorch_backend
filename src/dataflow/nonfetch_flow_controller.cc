#include "nonfetch_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"

void
NonFetchFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node)
{
  PutNodeToPipeline(input_id->request_id, input_id->correlation_id, node);
}

NodeMoveVec
NonFetchFlowController::PrefetchNode(const NodePtr& node)
{
  LOG_TRITON_VERBOSE(
      ("NonFetchFlowController::PrefetchNode " + node->GetModelInstanceInfo())
          .c_str());
  NodeMoveVec prefetch_nodes;
  SizeFilterFunc size_filter = THIS_BIND_ARGS(
      NonFetchFlowController, GetStandbyChildFake, node, std::placeholders::_1,
      std::placeholders::_2);

  bool gpu_prefetch = false;
  bool cpu_prefetch = true;

  do {
    if (!gpu_prefetch)
      gpu_prefetch =
          CreatePrefetchThreads(node, size_filter, DEFAULT_CUDA_DEVICE);
    if (!cpu_prefetch)
      cpu_prefetch = CreatePrefetchThreads(node, size_filter, CPU_DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (!gpu_prefetch || !cpu_prefetch);

  return prefetch_nodes;
  // MAX_REUSE_DISTANCE is not implemented yet
}

FilterResult
NonFetchFlowController::GetStandbyChildFake(
    const NodePtr& node, const std::int64_t size_limit, const Device& device)
{
  std::int64_t size = 0;
  NodePtrList node_ptr_list;

  return std::make_pair(size, node_ptr_list);
}
