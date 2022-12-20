#pragma once

#include "flow_controller.h"


class PrefetchFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(PrefetchFlowController)
  DISABLE_COPY_AND_ASSIGN(PrefetchFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node,
      const NodeMetaPtr& node_meta) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

  FilterResult GetStandbyChildByFreq(
      const NodePtr& node, std::size_t size_limit);
  //   ModelProbabilityVec GetTreeProbability(const NodePtr& node);

 private:
  PrefetchFlowController() = default;
  virtual ~PrefetchFlowController() = default;

  //   void RecursivelyUpdateProbability(
  //       const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map);

  std::unordered_map<std::size_t, std::size_t>
      request_time_;  //<request_id, time>
  std::unordered_map<std::size_t, StagePtr>
      request_trace_;  //<request_id, node_id>
  //   std::unordered_map<std::size_t, NodeFlowPtr> flow_graph_;
  std::size_t free_cpu_memory_{0};
  std::size_t free_gpu_memory_{0};
};

