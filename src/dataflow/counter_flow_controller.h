#pragma once

#include "flow_controller.h"

class CounterFlowController : public FlowControllerFactory {
 public:
  STATIC_GET_INSTANCE(CounterFlowController)
  DISABLE_COPY_AND_ASSIGN(CounterFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node,
      const NodeMetaPtr& node_meta) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

 private:
  CounterFlowController() = default;
  virtual ~CounterFlowController() = default;

  FilterResult GetStandbyChildByCount(const NodeID node_id, std::size_t size_limit);

 private:
  std::unordered_map<NodeID, Device> node_location_;
  std::size_t free_cpu_memory_{0};
  std::size_t free_gpu_memory_{0};
};
