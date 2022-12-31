#pragma once

#include "flow_controller.h"
#include "utils/memory_utils.h"

class NonFetchFlowController
    : public FlowControllerFactory,
      public std::enable_shared_from_this<NonFetchFlowController> {
 public:
  STATIC_GET_INSTANCE(NonFetchFlowController)
  DISABLE_COPY_AND_ASSIGN(NonFetchFlowController)

  void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node) override;
  NodeMoveVec PrefetchNode(const NodePtr& node) override;

 private:
  NonFetchFlowController() = default;
  virtual ~NonFetchFlowController() = default;

  FilterResult GetStandbyChildFake(
      const NodePtr& node, const std::int64_t size_limit, const Device& device);
};
