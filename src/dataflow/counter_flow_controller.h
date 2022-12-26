#pragma once

#include "flow_controller.h"

class CounterFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(CounterFlowController)
  DISABLE_COPY_AND_ASSIGN(CounterFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

 private:
  CounterFlowController() {
    free_cpu_memory_ = SYS_MEM_CTL->GetFreeMemory();
    free_gpu_memory_ = DEFAULT_CUDA_MEM_CTL->GetFreeMemory();
  }
  virtual ~CounterFlowController() = default;

  FilterResult GetStandbyChildByCount(const NodePtr& node, const std::size_t size_limit);

 private:
  std::unordered_map<NodeID, Device> node_location_;
  std::int64_t free_cpu_memory_;
  std::int64_t free_gpu_memory_;
};
