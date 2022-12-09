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
      const InputIDPtr& input_id, const DAGNodePtr& node,
      const NodeMetaPtr& node_meta) override;
  NodeMoveVec PrefetchNode(const DAGNodePtr& node) override;

 private:
  DeepSpeedFlowController()
      : free_cpu_memory_(GetFreeSystemMemory()),
        free_gpu_memory_(GetFreeDeviceMemory(0))
  {
  }
  virtual ~DeepSpeedFlowController() = default;

 private:
  std::unordered_map<NodeID, Device> node_location_;
  std::size_t free_cpu_memory_{0};
  std::size_t free_gpu_memory_{0};
};
