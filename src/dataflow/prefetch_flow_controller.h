#pragma once

#include "flow_controller.h"


class PrefetchFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(PrefetchFlowController)
  DISABLE_COPY_AND_ASSIGN(PrefetchFlowController)

  void RecordNode(const InputIDPtr& input_id, const NodePtr& node) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

  //   ModelProbabilityVec GetTreeProbability(const NodePtr& node);

 private:
  PrefetchFlowController()
  {
    free_cpu_memory_ = SYS_MEM_CTL->GetFreeMemory();
    free_gpu_memory_ = DEFAULT_CUDA_MEM_CTL->GetFreeMemory();
  }
  virtual ~PrefetchFlowController() = default;

  FilterResult GetStandbyChildByFreq(
      const NodePtr& node, const std::size_t size_limit);
  //   void RecursivelyUpdateProbability(
  //       const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map);

  std::unordered_map<std::size_t, std::size_t>
      request_time_;  //<request_id, time>
  std::unordered_map<std::size_t, StagePtr>
      request_trace_;  //<request_id, node_id>
  //   std::unordered_map<std::size_t, NodeFlowPtr> flow_graph_;
  std::int64_t free_cpu_memory_;
  std::int64_t free_gpu_memory_;
  std::unordered_map<NodeID, Device> node_location_;
};
