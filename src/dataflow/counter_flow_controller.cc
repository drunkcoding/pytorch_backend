#include "counter_flow_controller.h"

#include "forward_def.h"
#include "utils/time_utils.h"

void
CounterFlowController::RecordNode(
    const InputIDPtr& input_id, const DAGNodePtr& node,
    const NodeMetaPtr& node_meta)
{
  LOG_TRITON_VERBOSE("CounterFlowController::RecordNode");
  PutNodeTopology(input_id->correlation_id, node);

  auto node_id = node->GetNodeID();
  if (visit_count_.find(node_id) == visit_count_.end()) {
    visit_count_[node_id] = 0;
  }
  visit_count_[node_id] += 1;

  total_visit_count_ += 1;

  if (visit_time_.find(node_id) == visit_time_.end()) {
    visit_time_[node_id] = 0;
  }
  visit_time_[node_id] = MCIROSECONDS_SINCE_EPOCH;

  auto now = MCIROSECONDS_SINCE_EPOCH;
  std::vector<NodeID> delete_nodes;
  for (auto& visit : visit_time_) {
    if (now - visit.second > 60000000) {
      delete_nodes.push_back(visit.first);
      total_visit_count_ -= 1;
      visit_count_[visit.first] -= 1;
    }
  }

  for (auto& node_id : delete_nodes) {
    visit_time_.erase(node_id);
  }
}

NodeMoveVec
CounterFlowController::PrefetchNode(const DAGNodePtr& node)
{
  NodeID node_id = node->GetNodeID();
  LOG_TRITON_VERBOSE("CounterFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;
  if (node->GetMemoryType() == MemoryType::kStandBy) {
    node->SetMemoryType(MemoryType::kMoving);
    prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  }

  std::size_t total_live_params = 0;
  auto live_filter = THIS_BIND_ARGS(
      CounterFlowController, ParamLiveFilter, std::placeholders::_1,
      &total_live_params);
  auto node_ptr_list = GetNodesByFilter(live_filter, root_->GetNodeID());

  std::size_t prefetch_size =
      std::min(PREFETCH_BUCKET_SIZE, MAX_LIVE_PARAMETERS - total_live_params);

  auto size_filter = THIS_BIND_ARGS(
      CounterFlowController, MemorySizeFilter, std::placeholders::_1,
      &prefetch_size);

  node_ptr_list = GetNodesByFilter(size_filter, node_id);
  for (auto& prefetch_node : node_ptr_list) {
    // NodeID prefetch_node_id = prefetch_node->GetNodeID();
    if (prefetch_node->GetMemoryType() == MemoryType::kStandBy) {
      prefetch_node->SetMemoryType(MemoryType::kMoving);
      prefetch_nodes.push_back(
          std::make_pair(prefetch_node, DEFAULT_CUDA_DEVICE));
    }
  }
  return prefetch_nodes;
}