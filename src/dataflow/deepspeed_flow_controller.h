#pragma once

#include <unordered_set>

#include "flow_controller.h"
#include "utils/memory_utils.h"

class DeepSpeedFlowController
    : public FlowControllerFactory,
      public std::enable_shared_from_this<DeepSpeedFlowController> {
 public:
  STATIC_GET_INSTANCE(DeepSpeedFlowController)
  DISABLE_COPY_AND_ASSIGN(DeepSpeedFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

 private:
  DeepSpeedFlowController()
  {
    free_cpu_memory_ = SYS_MEM_CTL->GetFreeMemory();
    free_gpu_memory_ = DEFAULT_CUDA_MEM_CTL->GetFreeMemory();
  }
  virtual ~DeepSpeedFlowController() = default;

  FilterResult GetStandbyChildBySizeLimit(
      const NodePtr& node, const std::int64_t size_limit);

 private:
  std::unordered_map<NodeID, Device> node_location_;
  std::int64_t free_cpu_memory_;
  std::int64_t free_gpu_memory_;
};
