#pragma once

#include "flow_controller.h"


class PrefetchFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(PrefetchFlowController)
  DISABLE_COPY_AND_ASSIGN(PrefetchFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const DAGNodePtr& node,
      const NodeMetaPtr& node_meta) override;
  NodeMoveVec PrefetchNode(const DAGNodePtr& node) override;

  //   ModelProbabilityVec GetChildernProbability(const DAGNodePtr& node);
//   ModelProbabilityVec GetTreeProbability(const DAGNodePtr& node);

 private:
  PrefetchFlowController() : request_trace_(100) {}
  virtual ~PrefetchFlowController() = default;

//   void RecursivelyUpdateProbability(
//       const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map);

  LRUCache<std::size_t, NodeMetaPtrList> request_trace_;
//   std::unordered_map<std::size_t, NodeFlowPtr> flow_graph_;
};
