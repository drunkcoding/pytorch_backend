#pragma once

#include "flow_controller.h"

class CounterFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(CounterFlowController)
  DISABLE_COPY_AND_ASSIGN(CounterFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const DAGNodePtr& node,
      const NodeMetaPtr& node_meta) override;
  NodeMoveVec PrefetchNode(const DAGNodePtr& node) override;

 private:
  CounterFlowController() = default;
  virtual ~CounterFlowController() = default;

  std::size_t total_visit_count_{0};
  std::unordered_map<NodeID, std::size_t> visit_count_;
  std::unordered_map<NodeID, std::size_t> visit_time_;
};
