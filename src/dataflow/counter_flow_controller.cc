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
      std::placeholders::_1, std::placeholders::_2);
  // UpdateInitPrefetchNodes(prefetch_nodes, size_filter);
  std::vector<bool> gpu_prefetch(GetDeviceCount(), false);
  bool cpu_prefetch = true;
  bool all_gpu_prefetch = false;

  do {
    for (std::size_t gpu = 0; gpu < gpu_prefetch.size(); ++gpu) {
      if (gpu_prefetch[gpu] == false)
        gpu_prefetch[gpu] =
            CreatePrefetchThreads(node, size_filter, CUDA_DEVICE(gpu));
    }
    // set all_gpu_prefetch to true if all gpu_prefetch is true
    all_gpu_prefetch = std::all_of(
        gpu_prefetch.begin(), gpu_prefetch.end(), [](bool v) { return v; });
    // if (!cpu_prefetch)
    //   cpu_prefetch = CreatePrefetchThreads(node, size_filter, CPU_DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (!all_gpu_prefetch || !cpu_prefetch);

  return prefetch_nodes;
}


FilterResult
CounterFlowController::GetStandbyChildByCount(
    const NodePtr& node, const std::size_t size_limit, const Device& device)
{
  NodeID current_node_id = node->corr_id;

  std::size_t low_corr_id = current_node_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = current_node_id >> 32;        // node id
  NodePtrList node_ptr_list;
  std::size_t total_size = 0;

  if (low_corr_id >= pipeline_.stages.size()) {
    return std::make_pair(total_size, node_ptr_list);
  }

  if (pipeline_.stages[low_corr_id] == nullptr) {
    return std::make_pair(total_size, node_ptr_list);
  }

  auto current_node_body =
      pipeline_.stages[low_corr_id]
          ->nodes[(high_corr_id > 0) ? (high_corr_id - 1) : (0)];

  while (low_corr_id < pipeline_.stages.size()) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    BREAK_IF_NULL(stage);

    if (stage->is_sparse) {
      // Sparse stage, we only prefetch the first layer
      auto copy_nodes = stage->nodes;

      // remove null node body
      copy_nodes.erase(
          std::remove_if(
              copy_nodes.begin(), copy_nodes.end(),
              [](const NodeBodyPtr& a) { return a == nullptr; }),
          copy_nodes.end());

      std::sort(
          copy_nodes.begin(), copy_nodes.end(),
          [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
            return a->visit_cnt > b->visit_cnt;
          });

      std::size_t layer_size = copy_nodes.size();
      std::size_t max_layer_fetch_size = 50;
      for (std::size_t j = 0; j < std::min(layer_size, max_layer_fetch_size);
           j++) {
        auto node_body = copy_nodes[j];
        BREAK_IF_NULL(node_body);
        if (total_size + node_body->node->byte_size > size_limit) {
          return std::make_pair(total_size, node_ptr_list);
        }
        if (!node_body->node->device.is_cuda() &&
            device == node_body->node->default_device) {
          total_size += node_body->node->byte_size;
          if (node_body->node->mutex.try_lock())
            node_ptr_list.push_back(node_body->node);
        }
      }
      break;
    } else {
      // Dense stage
      auto node_body = stage->nodes[0];
      BREAK_IF_NULL(node_body);
      if (total_size + node_body->node->byte_size > size_limit) {
        return std::make_pair(total_size, node_ptr_list);
      }
      if (!node_body->node->device.is_cuda() &&
          device == node_body->node->default_device) {
        total_size += node_body->node->byte_size;
        if (node_body->node->mutex.try_lock())
          node_ptr_list.push_back(node_body->node);
      }
    }

    low_corr_id += 1;
  }
  return std::make_pair(total_size, node_ptr_list);
}
