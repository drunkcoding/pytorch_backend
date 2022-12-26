#include "counter_flow_controller.h"

#include "forward_def.h"
#include "utils/memory_utils.h"
#include "utils/time_utils.h"

void
CounterFlowController::RecordNode(
    const InputIDPtr& input_id, const NodePtr& node)
{
  auto node_id = node->id;
  auto memory_size = node->byte_size;
  // std::size_t hash_request_id =
  // std::hash<std::string>{}(input_id->request_id);
  // LOG_TRITON_VERBOSE("CounterFlowController::RecordNode");
  PutNodeToPipeline(input_id->request_id, input_id->correlation_id, node);
  if (node_location_.find(node_id) == node_location_.end()) {
    if (free_cpu_memory_ > memory_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
    }
    // node_location_.insert({node_id, node->device});
  }
}

NodeMoveVec
CounterFlowController::PrefetchNode(const NodePtr& node)
{
  // NodeID node_id = node->id;
  // LOG_TRITON_VERBOSE("CounterFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;
  // if (node->memory_type == MemoryType::kStandBy) {
  //   node->memory_type = MemoryType::kMoving;
  //   prefetch_nodes.push_back(std::make_pair(node, DEFAULT_CUDA_DEVICE));
  // }

  // if (pipeline_.stages.empty()) {
  //   UpdateMemoryManager(prefetch_nodes);
  //   return prefetch_nodes;
  // }

  SizeFilterFunc size_filter = THIS_BIND_ARGS(
      CounterFlowController, GetStandbyChildByCount, node,
      std::placeholders::_1);
  // UpdateInitPrefetchNodes(prefetch_nodes, size_filter);
  while (!CreatePrefetchThreads(node, size_filter)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  return prefetch_nodes;
}


FilterResult
CounterFlowController::GetStandbyChildByCount(
    const NodePtr& node, const std::size_t size_limit)
{
  NodeID current_node_id = node->corr_id;

  std::size_t low_corr_id = current_node_id & 0xFFFFFFFF;  // stage id
  NodePtrList node_ptr_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, node_ptr_list);
  }

  while (low_corr_id < pipeline_.stages.size()) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    if (stage == nullptr) {
      break;
    }

    if (stage->is_sparse) {
      // Sparse stage, we only prefetch the first layer
      auto copy_nodes = stage->nodes;

      std::sort(
          copy_nodes.begin(), copy_nodes.end(),
          [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
            if (a == nullptr || b == nullptr) {
              return false;
            }
            return a->visit_cnt > b->visit_cnt;
          });
      for (int j = 0; j < 3; j++) {
        auto node_body = copy_nodes[j];
        if (node_body == nullptr) {
          break;
        }
        if (total_size + node_body->node->byte_size > size_limit) {
          return std::make_pair(total_size, node_ptr_list);
        }
        if (!node_body->node->device.is_cuda() &&
            node_body->node->mutex.try_lock()) {
          total_size += node_body->node->byte_size;
          node_ptr_list.push_back(node_body->node);
        }
      }
      break;
    } else {
      // Dense stage
      auto node_body = stage->nodes[0];
      if (node_body == nullptr) {
        break;
      }
      if (total_size + node_body->node->byte_size > size_limit) {
        return std::make_pair(total_size, node_ptr_list);
      }
      if (!node_body->node->device.is_cuda() &&
          node_body->node->mutex.try_lock()) {
        total_size += node_body->node->byte_size;
        node_ptr_list.push_back(node_body->node);
      }
    }

    low_corr_id += 1;
  }

  // std::size_t size = 0;
  // for (auto& node : node_ptr_list) {
  //   size += node->byte_size;
  // }
  return std::make_pair(total_size, node_ptr_list);
}
