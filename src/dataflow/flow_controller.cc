#include "flow_controller.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <queue>

#include "forward_def.h"
#include "triton/common/logging.h"
#include "utils/functor.h"
#include "utils/memory_utils.h"

// #include "memory_controller.h"

// void
// NodeFlow::AddPrevNode(const NodeFlowPtr& prev_node)
// {
//   // prev_node->AddNextNode(SELF(NodeFlow));
//   prev_nodes_.emplace(std::make_pair(prev_node->id, prev_node));
// }


// void
// NodeFlow::AddNextNode(const NodeFlowPtr& next_node)
// {
//   // auto next_node = std::make_shared<NodeFlow>(node);
//   next_nodes_.emplace(std::make_pair(next_, next_node));
//   // next_node->AddPrevNode(SELF(NodeFlow));
//   //   if (!result.second) {
//   //     result.first->second->visit_cnt++;
//   //   }
//   // return result.first->second;
// }

// void
// NodeFlow::DereferenceNode(const NodeMetaPtr& node_meta)
// {
//   *node_meta_ -= *node_meta;
// }


void
NodeTopology::AddPrevNode(const NodeTopologyPtr& prev_node)
{
  prev_nodes_.emplace(std::make_pair(prev_node->GetNodeID(), prev_node));
}


void
NodeTopology::AddNextNode(const NodeTopologyPtr& next_node)
{
  next_nodes_.emplace(std::make_pair(next_node->GetNodeID(), next_node));
}

// void
// NodeFlow::RemoveNextNode(const NodeFlowPtr& next_node)
// {
//   auto model_id = next_node->id;
//   auto it = next_nodes_.find(model_id);
//   if (it != next_nodes_.end()) {
//     return nullptr;
//   }
//   if (it->second->visit_cnt > 0) {
//     it->second->visit_cnt--;
//   } else {
//     next_nodes_.erase(it);
//   }
//   return it->second;
// }


// void
// FlowControllerFactory::PutNodeTopology(
//     const std::uint64_t& correlation_id, const NodePtr& node)
// {
//   std::uint64_t high_corr_id =
//       correlation_id >> 32;  // For childs in the same level
//   std::uint64_t low_corr_id =
//       correlation_id & 0xFFFFFFFF;  // For model inference pipeline
//   if (visited_.find(correlation_id) == visited_.end()) {
//     visited_.insert(correlation_id);
//     // visited_.insert(low_corr_id);
//     auto cur_node_topology =
//         std::make_shared<NodeTopology>(node, correlation_id);
//     topology_.insert({cur_node_topology->GetNodeID(), cur_node_topology});
//     if (root_ == nullptr) {
//       root_ = cur_node_topology;
//     } else {
//       for (auto& node_topology : topology_) {
//         // auto node_id = node_topology.first;
//         auto node_topology_ptr = node_topology.second;
//         auto prev_corr_id =
//             (high_corr_id == 0) ? (low_corr_id - 1) : low_corr_id;
//         if (node_topology_ptr->GetCorrelationID() == prev_corr_id) {
//           node_topology_ptr->AddNextNode(cur_node_topology);
//           cur_node_topology->AddPrevNode(node_topology_ptr);
//         }
//       }
//     }
//   }
// }

void
FlowControllerFactory::PutNodeToPipeline(
    const std::uint64_t& request_id, const std::uint64_t& correlation_id,
    const NodePtr& node)
{
  std::uint64_t high_corr_id =
      correlation_id >> 32;  // For childs in the same level
  std::uint64_t low_corr_id =
      correlation_id & 0xFFFFFFFF;  // For model inference pipeline

  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  // LOG_VERBOSE(5) << "PutNodeToPipeline: request_id " << request_id
  //                << " correlation_id " << std::hex << correlation_id
  //                << " high_corr_id " << high_corr_id << " low_corr_id "
  //                << low_corr_id << " is_last_node " << is_last_node;
  LOG_TRITON_VERBOSE(
      (std::string("PutNodeToPipeline: request_id ") +
       std::to_string(request_id) + std::string(" correlation_id ") +
       std::to_string(correlation_id) + std::string(" high_corr_id ") +
       std::to_string(high_corr_id) + std::string(" low_corr_id ") +
       std::to_string(low_corr_id) + std::string(" is_last_node ") +
       std::to_string(is_last_node))
          .c_str());

  node->corr_id = correlation_id;
  auto node_idx = (high_corr_id > 0) ? (high_corr_id - 1) : 0;

  if (visited_.find(correlation_id) == visited_.end()) {
    if (free_cpu_memory_ > node->byte_size) {
      free_cpu_memory_ -= node->byte_size;
      node_location_.insert({node->id, CPU_DEVICE});
    } else
      node_location_.insert({node->id, DISK_DEVICE});

    visited_.insert(correlation_id);
    auto node_body = std::make_shared<NodeBody>(node);

    // LOG_TRITON_VERBOSE(
    //     (std::string("PutNodeToPipeline: ") + pipeline_.str()).c_str());

    // assert(pipeline_.stages.size() >= low_corr_id); may skip some stages
    // since embed can run in parallel

    if (pipeline_.stages.size() < low_corr_id + 1)
      pipeline_.stages.resize(low_corr_id + 1, nullptr);

    if (pipeline_.stages[low_corr_id] == nullptr)
      pipeline_.stages[low_corr_id] = std::make_shared<Stage>();

    auto stage = pipeline_.stages[low_corr_id];

    if (stage->nodes.size() < node_idx + 1)
      stage->nodes.resize(node_idx + 1, nullptr);

    stage->nodes[node_idx] = node_body;

    if (high_corr_id > 0)
      stage->is_sparse = true;

    // LOG_TRITON_VERBOSE(
    //     (std::string("PutNodeToPipeline: Add new stage") + stage->str())
    //         .c_str());
  }

  auto stage = pipeline_.stages[low_corr_id];
  auto node_body = stage->nodes[(stage->is_sparse) ? (high_corr_id - 1) : 0];
  auto now = MCIROSECONDS_SINCE_EPOCH;

  node_body->visit_time.push_back(now);
  node_body->visit_cnt += 1;
  node_body->activate_request.insert(request_id);
  node_body->node->last_access_time = now;


  // LOG_TRITON_VERBOSE(
  //     ("PutNodeToPipeline: node_body " + node_body->str()).c_str());

  // First update the visit count
  // 1. visit to sparse nodes only count once to total visit count
  // 2. find the last sparse layer and update the visit count of parent node


  if (stage->activate_request.find(request_id) ==
      stage->activate_request.end()) {
    // this is the first node in the stage
    // update the visit count of the stage
    stage->visit_cnt += 1;
  }

  stage->activate_request.insert(request_id);

  if (correlation_id == 0) {
    // this is the first node in the pipeline
    // update the visit count of the pipeline
    pipeline_.visit_cnt += 1;
  }

  // if (stage->is_sparse && high_corr_id == 0) {
  //   // this is the route node in the sparse layer
  //   // update the visit count of the stage
  //   stage->visit_cnt += 1;
  // }

  // if (!stage->is_sparse) {
  //   // this is the only node in the stage
  //   // update the visit count of the stage
  //   stage->visit_cnt += 1;
  // }

  // LOG_TRITON_VERBOSE(
  //     (std::string("PutNodeToPipeline: ") + pipeline_.str()).c_str());

  if (stage->is_sparse && high_corr_id > 0) {
    // this is the child node in the sparse layer
    // update the visit count of the parent node
    // LOG_TRITON_VERBOSE("PutNodeToPipeline: update sparse stage");
    StagePtr last_sparse_stage;
    for (auto i = low_corr_id - 1; i >= 0; i--) {
      if (stage->is_sparse) {
        last_sparse_stage = pipeline_.stages[i];
        break;
      }
    }

    for (auto& last_stage_node : last_sparse_stage->nodes) {
      if (last_stage_node->activate_request.find(request_id) !=
          last_stage_node->activate_request.end()) {
        if (last_stage_node->children_visit_cnt.size() < high_corr_id) {
          last_stage_node->children_visit_cnt.resize(high_corr_id);
        }
        if (last_stage_node->children.size() < high_corr_id) {
          last_stage_node->children.resize(high_corr_id, nullptr);
        }
        if (last_stage_node->children_visit_time.size() < high_corr_id) {
          last_stage_node->children_visit_time.resize(high_corr_id);
        }
        last_stage_node->children_visit_cnt[high_corr_id - 1] += 1;
        last_stage_node->children[high_corr_id - 1] = node_body;
        last_stage_node->children_visit_time[high_corr_id - 1].push_back(now);
      }
    }
  }

  if (is_last_node) {
    // this is the last node in the pipeline
    // remove request_id from all nodes
    // LOG_TRITON_VERBOSE("PutNodeToPipeline: is_last_node");
    for (auto& stage : pipeline_.stages) {
      if (stage == nullptr) {
        continue;
      }
      for (auto& node : stage->nodes) {
        if (node == nullptr) {
          continue;
        }
        node->activate_request.erase(request_id);
      }
      stage->activate_request.erase(request_id);
    }
    last_active_stage_.erase(request_id);
  }

  // for all nodes and stage and children in the pipeline, reduce count of nodes
  // that are not visited for 1 hour
  std::size_t microseconds = 60000000ULL * 60;
  for (auto& stage : pipeline_.stages) {
    if (stage == nullptr) {
      continue;
    }
    for (auto& node : stage->nodes) {
      if (node == nullptr) {
        continue;
      }
      auto visit = node->visit_time.begin();
      while (visit != node->visit_time.end() && *visit > 0 &&
             now - *visit > microseconds) {
        node->visit_time.pop_front();
        node->visit_cnt -= 1;
        visit = node->visit_time.begin();
      }

      int k = 0;
      for (auto& children : node->children_visit_time) {
        auto visit = children.begin();
        while (visit != children.end() && *visit > 0 &&
               now - *visit > microseconds) {
          children.pop_front();
          node->children_visit_cnt[k] -= 1;
          visit = children.begin();
        }
        k++;
      }
    }


    auto visit = stage->visit_time.begin();
    while (visit != stage->visit_time.end() && *visit > 0 &&
           now - *visit > microseconds) {
      stage->visit_time.pop_front();
      stage->visit_cnt -= 1;
      visit = stage->visit_time.begin();
    }
  }

  auto it = last_active_stage_.find(request_id);
  if (it == last_active_stage_.end()) {
    last_active_stage_.insert({request_id, low_corr_id});
  } else {
    it->second = low_corr_id;
  }

  std::size_t byte_size = 0;
  for (auto node_body : pipeline_.stages[low_corr_id]->nodes) {
    CONTINUE_IF_NULL(node_body);
    byte_size += node_body->node->byte_size;
  }
  pipeline_.stages[low_corr_id]->byte_size = byte_size;
}

// NodeTopologyPtr
// FlowControllerFactory::GetNodeTopology(const NodeID& node_id)
// {
//   if (topology_.find(node_id) == topology_.end()) {
//     return nullptr;
//   } else {
//     return topology_[node_id];
//   }
// }

// NodePtrList
// FlowControllerFactory::GetNodesByFilter(
//     const NodeFilterFunc& filter_func, const NodeID& node_id)
// {
//   NodePtrList nodes;
//   if (topology_.find(node_id) == topology_.end()) {
//     LOG_TRITON_VERBOSE(
//         ("Node " + std::to_string(node_id) + " not found").c_str());
//     return nodes;
//   } else {
//     auto node_topology = topology_[node_id];
//     std::queue<NodeTopologyPtr> node_queue;
//     node_queue.push(node_topology);
//     while (!node_queue.empty()) {
//       auto cur_node_topology = node_queue.front();
//       node_queue.pop();
//       if (filter_func(cur_node_topology->GetNode())) {
//         nodes.push_back(cur_node_topology->GetNode());
//       }
//       for (auto& next_node : cur_node_topology->GetNextNodes()) {
//         node_queue.push(next_node.second);
//       }
//     }
//     return nodes;
//   }
// }

FlowControllerFactory::FlowControllerFactory() {}

std::int64_t
FlowControllerFactory::GetPrefetableSize()
{
  auto [live_memory, live_node_list] = GetTotalLiveParamSize();
  std::int64_t prefetch_size =
      std::min(PREFETCH_BUCKET_SIZE, MAX_LIVE_PARAMETERS - live_memory);
  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::GetPrefetableSize: live_memory = " +
       std::to_string(live_memory) +
       " prefetch_size = " + std::to_string(prefetch_size))
          .c_str());

  return prefetch_size;
}

void
FlowControllerFactory::DispatchRemoveAndFetch(
    std::int64_t remove_size, NodePtrList& remove_nodes,
    NodePtrList& fetch_nodes, bool immediate, const Device& device)
{
  std::int64_t fetch_size =
      ((device.is_cuda()) ? CUDA_MEM_CTL(device.index())->GetFreeMemory()
                          : SYS_MEM_CTL->GetFreeMemory()) +
      remove_size;
  CounterPtr remove_cnt = std::make_shared<std::atomic<int>>(0);
  while (!remove_nodes.empty()) {
    auto node = remove_nodes.front();
    if (remove_size < 0) {
      break;
    }
    remove_size -= node->byte_size;
    remove_cnt->fetch_add(1);
    LOG_TRITON_VERBOSE(
        ("FlowControllerFactory::CreatePrefetchThreadsGPU: remove node = " +
         node->GetModelInstanceInfo() +
         " remain memory to remove = " + std::to_string(remove_size))
            .c_str());


#ifdef ENABLE_PREFETCH_FLOW_CONTROLLER
    auto it = node_location_.find(node->id);
    std::thread prefetch_thread(
        FetchThreadFunc, node, it->second, false, remove_cnt);
#else
    std::thread prefetch_thread(
        FetchThreadFunc, node, DISK_DEVICE, false, remove_cnt);
#endif  // ENABLE_PREFETCH_FLOW_CONTROLLER


    prefetch_thread.detach();
    // FetchThreadFunc(prefetch_node, DISK_DEVICE, false);
    remove_nodes.erase(remove_nodes.begin());
  }


  while (!fetch_nodes.empty()) {
    auto node = fetch_nodes.front();
    if (fetch_size - node->byte_size < 0) {
      break;
    }
    fetch_size -= node->byte_size;
    LOG_TRITON_VERBOSE(
        ("FlowControllerFactory::CreatePrefetchThreadsGPU: fetch node = " +
         node->GetModelInstanceInfo())
            .c_str());
    std::thread prefetch_thread(
        FetchThreadFunc, node, device, immediate, remove_cnt);
    prefetch_thread.detach();
    // FetchThreadFunc(prefetch_node, DISK_DEVICE, false);
    fetch_nodes.erase(fetch_nodes.begin());

    auto mem_ctl =
        (device.is_cuda()) ? CUDA_MEM_CTL(device.index()) : SYS_MEM_CTL;
    mem_ctl->AllocateMemory(node->id, node->byte_size);
  }
}

bool
FlowControllerFactory::CreatePrefetchThreads(
    const NodePtr& node, const SizeFilterFunc& func, const Device& device)
{
  LOG_TRITON_VERBOSE(("FlowControllerFactory::CreatePrefetchThreads: id " +
                      std::to_string(node->id))
                         .c_str());

  std::int64_t remove_size = -1;
  std::int64_t free_memory = (device.is_cuda())
                                 ? CUDA_MEM_CTL(device.index())->GetFreeMemory()
                                 : SYS_MEM_CTL->GetFreeMemory();

  auto [removable_memory, removable_node_list] = GetLRUNodes(device);
  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::CreatePrefetchThreads: device " + device.str() +
       " removable_memory = " + std::to_string(removable_memory) +
       " nodes = " + std::to_string(removable_node_list.size()))
          .c_str());
  auto [prefetch_memory, prefetch_node_list] =
      func(PREFETCH_BUCKET_SIZE * ((device.is_cuda()) ? 1 : 50), device);
  LOG_TRITON_VERBOSE(("FlowControllerFactory::CreatePrefetchThreads: device " +
                      device.str() +
                      " prefetch_memory = " + std::to_string(prefetch_memory) +
                      " nodes = " + std::to_string(prefetch_node_list.size()) +
                      " free memory = " + std::to_string(free_memory))
                         .c_str());

  if (!node->device.is_cuda() && device.is_cuda()) {
    prefetch_memory += node->byte_size;
    if (removable_memory + free_memory < node->byte_size) {
      LOG_TRITON_ERROR(
          ("FlowControllerFactory::CreatePrefetchThreads: remove_size = " +
           std::to_string(remove_size) +
           " > removable_memory = " + std::to_string(removable_memory) +
           ", node size = " + std::to_string(node->byte_size))
              .c_str());
      RELEASE_LOCKS(removable_node_list);
      RELEASE_LOCKS(prefetch_node_list);
      return false;
    }

    // First remove what is necessary for the immedient prefetch
    if (node->byte_size > free_memory) {
      remove_size = node->byte_size - free_memory;
    }

    if (remove_size > removable_memory) {
      LOG_TRITON_ERROR(
          ("FlowControllerFactory::CreatePrefetchThreads: remove_size = " +
           std::to_string(remove_size) +
           " > removable_memory = " + std::to_string(removable_memory) +
           " free memory = " + std::to_string(free_memory))
              .c_str());
      RELEASE_LOCKS(removable_node_list);
      RELEASE_LOCKS(prefetch_node_list);
      return false;
    }

    NodePtrList immediant_nodes;
    immediant_nodes.push_back(node);
    DispatchRemoveAndFetch(
        remove_size, removable_node_list, immediant_nodes, true, device);
    remove_size = -1;
  }


  if (prefetch_memory > free_memory) {
    remove_size = prefetch_memory - free_memory;
  }

  // if (remove_size > removable_memory) {
  //   LOG_TRITON_VERBOSE(
  //       ("FlowControllerFactory::CreatePrefetchThreadsGPU: remove_size = " +
  //        std::to_string(remove_size) +
  //        " > removable_memory = " + std::to_string(removable_memory))
  //           .c_str());
  //   RELEASE_LOCKS(removable_node_list);
  //   RELEASE_LOCKS(prefetch_node_list);
  //   return false;
  // }

  // DispatchRemoveAndFetch(
  //     std::min(removable_memory, remove_size * 128), removable_node_list,
  //     prefetch_node_list, false, device);
  if (device.is_cpu())
    removable_memory = std::min(removable_memory, remove_size);
  else
    removable_memory = std::min(removable_memory, remove_size * 256);
  DispatchRemoveAndFetch(
      removable_memory, removable_node_list, prefetch_node_list, false, device);

  RELEASE_LOCKS(removable_node_list);
  RELEASE_LOCKS(prefetch_node_list);

  return true;
}

void
FlowControllerFactory::UpdateInitPrefetchNodes(
    NodeMoveVec& prefetch_nodes, const SizeFilterFunc& func)
{
  auto [live_memory, live_node_list] = GetTotalLiveParamSize();
  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::UpdateInitPrefetchNodes: live_memory = " +
       std::to_string(live_memory) +
       " live_nodes = " + std::to_string(live_node_list.size()))
          .c_str());

  auto [removable_memory, removable_node_list] = GetRemovableNodes();
  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::UpdateInitPrefetchNodes: removable_memory = " +
       std::to_string(removable_memory) +
       " removable_nodes = " + std::to_string(removable_node_list.size()))
          .c_str());

  std::vector<std::int64_t> predetch_size_candidates = {
      PREFETCH_BUCKET_SIZE, MAX_LIVE_PARAMETERS - live_memory,
      DEFAULT_CUDA_MEM_CTL->GetFreeMemory() + removable_memory};
  std::int64_t prefetch_size = *std::min_element(
      predetch_size_candidates.begin(), predetch_size_candidates.end());
  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::UpdateInitPrefetchNodes: prefetch_size = " +
       std::to_string(prefetch_size) + " free memory = " +
       std::to_string(DEFAULT_CUDA_MEM_CTL->GetFreeMemory() + removable_memory))
          .c_str());

  if (prefetch_size <= 0) {
    return;
  }

  auto [prefetch_memory, prefetch_node_list] =
      func(prefetch_size, DEFAULT_CUDA_DEVICE);

  for (auto& prefetch_node : prefetch_node_list) {
    prefetch_node->memory_type = MemoryType::kEmplacing;
    prefetch_nodes.push_back(
        std::make_pair(prefetch_node, DEFAULT_CUDA_DEVICE));
  }

  LOG_TRITON_VERBOSE(("FlowControllerFactory::UpdateInitPrefetchNodes: "
                      "prefetch_nodes size = " +
                      std::to_string(prefetch_nodes.size()) +
                      " total memory = " +
                      std::to_string(DEFAULT_CUDA_MEM_CTL->GetFreeMemory()) +
                      " prefetch_size = " + std::to_string(prefetch_size) +
                      " prefetch_memory = " + std::to_string(prefetch_memory))
                         .c_str());

  // std::int64_t total_prefetch_memory = prefetch_memory;
  // for (auto& prefetch_node : prefetch_nodes) {
  //   total_prefetch_memory += prefetch_node.first->byte_size;
  // }

  bool memory_exceeded =
      prefetch_memory > DEFAULT_CUDA_MEM_CTL->GetFreeMemory();
  std::int64_t remove_size =
      prefetch_memory - DEFAULT_CUDA_MEM_CTL->GetFreeMemory();

  if (memory_exceeded) {
    std::int64_t size_removed = 0;
    for (auto& remove_node : removable_node_list) {
      remove_node->memory_type = MemoryType::kEvicting;
      prefetch_nodes.push_back(std::make_pair(remove_node, DISK_DEVICE));
      LOG_TRITON_VERBOSE(
          ("FlowControllerFactory::PrefetchNode: remove_node = " +
           remove_node->GetModelInstanceInfo())
              .c_str());
      size_removed += remove_node->byte_size;
      if (size_removed > remove_size) {
        break;
      }
    }
  }

  // allocate memory for prefetch nodes to GPUs
  // remove nodes from GPU memory updates after actual execution
  for (auto& prefetch_node : prefetch_nodes) {
    if (prefetch_node.second == DEFAULT_CUDA_DEVICE) {
      UpdateMemoryManager(
          prefetch_node.first->device, prefetch_node.second,
          prefetch_node.first->byte_size);
      LOG_TRITON_VERBOSE(
          ("FlowControllerFactory::PrefetchNode: prefetch_node = " +
           prefetch_node.first->GetModelInstanceInfo())
              .c_str());
    }
  }
}

void
FlowControllerFactory::UpdateMemoryManager(
    const Device& from, const Device& to, const std::size_t& size)
{
  if (from == to) {
    return;
  }

  if (from == CPU_DEVICE && to == DEFAULT_CUDA_DEVICE) {
    SYS_MEM_CTL->FreeMemory(size);
    DEFAULT_CUDA_MEM_CTL->AllocateMemory(size);
  }

  if (from == DEFAULT_CUDA_DEVICE && to == CPU_DEVICE) {
    DEFAULT_CUDA_MEM_CTL->FreeMemory(size);
    SYS_MEM_CTL->AllocateMemory(size);
  }

  if (from == DISK_DEVICE && to == DEFAULT_CUDA_DEVICE) {
    DEFAULT_CUDA_MEM_CTL->AllocateMemory(size);
  }

  if (from == DISK_DEVICE && to == CPU_DEVICE) {
    SYS_MEM_CTL->AllocateMemory(size);
  }

  if (from == DEFAULT_CUDA_DEVICE && to == DISK_DEVICE) {
    DEFAULT_CUDA_MEM_CTL->FreeMemory(size);
  }

  if (from == CPU_DEVICE && to == DISK_DEVICE) {
    SYS_MEM_CTL->FreeMemory(size);
  }

  LOG_TRITON_VERBOSE(
      ("FlowControllerFactory::UpdateMemoryManager: from = " + from.str() +
       " to = " + to.str() + " size = " + std::to_string(size) +
       " SYS_MEM_CTL = " + std::to_string(SYS_MEM_CTL->GetFreeMemory()) +
       " DEFAULT_CUDA_MEM_CTL = " +
       std::to_string(DEFAULT_CUDA_MEM_CTL->GetFreeMemory()))
          .c_str());
}

void
FlowControllerFactory::UpdateMemoryManager(const NodeMoveVec& prefetch_nodes)
{
  for (auto& prefetch_node : prefetch_nodes) {
    LOG_TRITON_VERBOSE(
        ("FlowControllerFactory::UpdateMemoryManager: node_id = " +
         std::to_string(prefetch_node.first->corr_id) +
         " device = " + prefetch_node.first->device.str() +
         " target device = " + prefetch_node.second.str())
            .c_str());
    // move from cpu to gpu
    if (prefetch_node.second == DEFAULT_CUDA_DEVICE &&
        prefetch_node.first->device == CPU_DEVICE) {
      SYS_MEM_CTL->FreeMemory(prefetch_node.first->byte_size);
      DEFAULT_CUDA_MEM_CTL->AllocateMemory(prefetch_node.first->byte_size);
    }

    // move from gpu to cpu
    if (prefetch_node.second == CPU_DEVICE &&
        prefetch_node.first->device == DEFAULT_CUDA_DEVICE) {
      DEFAULT_CUDA_MEM_CTL->FreeMemory(prefetch_node.first->byte_size);
      SYS_MEM_CTL->AllocateMemory(prefetch_node.first->byte_size);
    }

    // move from gpu to disk
    if (prefetch_node.second == DISK_DEVICE &&
        prefetch_node.first->device == DEFAULT_CUDA_DEVICE) {
      DEFAULT_CUDA_MEM_CTL->FreeMemory(prefetch_node.first->byte_size);
    }

    // move from disk to gpu
    if (prefetch_node.second == DEFAULT_CUDA_DEVICE &&
        prefetch_node.first->device == DISK_DEVICE) {
      DEFAULT_CUDA_MEM_CTL->AllocateMemory(prefetch_node.first->byte_size);
    }

    // move from cpu to disk
    if (prefetch_node.second == DISK_DEVICE &&
        prefetch_node.first->device == CPU_DEVICE) {
      SYS_MEM_CTL->FreeMemory(prefetch_node.first->byte_size);
    }

    // move from disk to cpu
    if (prefetch_node.second == CPU_DEVICE &&
        prefetch_node.first->device == DISK_DEVICE) {
      SYS_MEM_CTL->AllocateMemory(prefetch_node.first->byte_size);
    }
  }
}

void
FlowControllerFactory::DispatchNodeMemoryInThread(
    const NodePtr& node, const Device& device)
{
  std::thread t(&FlowControllerFactory::DispatchNodeMemory, this, node, device);
  t.detach();
}

void
FlowControllerFactory::DispatchNodeMemory(
    const NodePtr& node, const Device& device)
{
  node->SetDevice(device);
  if (device == CPU_DEVICE || device == DISK_DEVICE) {
    node->memory_type = MemoryType::kStandBy;
  }

  if (device == DEFAULT_CUDA_DEVICE &&
      node->memory_type != MemoryType::kLocked) {
    node->memory_type = MemoryType::kReady;
  }
}

FilterResult
FlowControllerFactory::GetTotalLiveParamSize()
{
  std::int64_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    if (stage == nullptr) {
      continue;
    }
    for (auto& node_body : stage->nodes) {
      if (node_body == nullptr) {
        continue;
      }
      auto node = node_body->node;
      if (node->memory_type == MemoryType::kLocked) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }

  LOG_TRITON_VERBOSE(
      ("Total live param size: " + std::to_string(size)).c_str());
  return std::make_pair(size, node_ptr_list);
}

// FilterResult
// FlowControllerFactory::GetTotalGPUParamSize()
// {
//   std::size_t size = 0;
//   NodeFilterFunc filter = [&size](const NodePtr& node) {
//     if (node->device == DEFAULT_CUDA_DEVICE) {
//       size += node->byte_size;
//       return true;
//     }
//     return false;
//   };
//   auto node_ptr_list = GetNodesByFilter(filter, root_->GetNodeID());
//   return std::make_pair(size, node_ptr_list);
// }

FilterResult
FlowControllerFactory::GetTotalGPUParamSize()
{
  std::int64_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    if (stage == nullptr) {
      continue;
    }
    for (auto& node_body : stage->nodes) {
      if (node_body == nullptr) {
        continue;
      }
      auto node = node_body->node;
      if (node->device == DEFAULT_CUDA_DEVICE) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }

  LOG_TRITON_VERBOSE(("Total GPU param size: " + std::to_string(size)).c_str());
  return std::make_pair(size, node_ptr_list);
}

// FilterResult
// FlowControllerFactory::GetRemovableNodes()
// {
//   std::size_t size = 0;
//   NodeFilterFunc filter = [&size](const NodePtr& node) {
//     if (node->memory_type == MemoryType::kReady) {
//       size += node->byte_size;
//       return true;
//     }
//     return false;
//   };
//   auto node_ptr_list = GetNodesByFilter(filter, root_->GetNodeID());
//   std::sort(
//       node_ptr_list.begin(), node_ptr_list.end(),
//       [](const NodePtr& a, const NodePtr& b) {
//         return a->last_access_time < b->last_access_time;
//       });
//   return std::make_pair(size, node_ptr_list);
// }


FilterResult
FlowControllerFactory::GetLRUNodes(const Device& device)
{
  NodePtrList node_ptr_list;
  std::int64_t size = 0;
  for (auto& stage : pipeline_.stages) {
    BREAK_IF_NULL(stage)
    for (auto& node_body : stage->nodes) {
      CONTINUE_IF_NULL(node_body)
      auto node = node_body->node;
      if (node->mutex.try_lock()) {
        bool remove = true;
        for (auto& active_stage : last_active_stage_) {
          remove &=
              (node->corr_id & 0x00000000FFFFFFFF) >
                  (active_stage.second + 32) ||
              (node->corr_id & 0x00000000FFFFFFFF) < (active_stage.second);
        }
        if (node->device == device && remove) {
          size += node->byte_size;
          node_ptr_list.push_back(node);
        } else
          node->mutex.unlock();
      }
    }
  }
  std::sort(
      node_ptr_list.begin(), node_ptr_list.end(),
      [](const NodePtr& a, const NodePtr& b) {
        return a->last_access_time < b->last_access_time;
      });

  return std::make_pair(size, node_ptr_list);
}

FilterResult
FlowControllerFactory::GetRemovableNodes()
{
  std::int64_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    if (stage == nullptr) {
      continue;
    }
    for (auto& node_body : stage->nodes) {
      if (node_body == nullptr) {
        continue;
      }
      auto node = node_body->node;
      if (node->memory_type == MemoryType::kReady) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }

  LOG_TRITON_VERBOSE(
      ("Total removable param size: " + std::to_string(size) +
       " Number of nodes: " + std::to_string(node_ptr_list.size()))
          .c_str());

  if (node_ptr_list.size() == 0) {
    LOG_TRITON_ERROR("No removable nodes found");
  } else {
    std::sort(
        node_ptr_list.begin(), node_ptr_list.end(),
        [](const NodePtr& a, const NodePtr& b) {
          return a->last_access_time < b->last_access_time;
        });
    // LOG_TRITON_VERBOSE(
    //     ("First node: " +
    //     node_ptr_list[0]->GetModelInstanceInfo()).c_str());
  }

  return std::make_pair(size, node_ptr_list);
}

// FilterResult
// FlowControllerFactory::GetStandbyChildBySizeLimit(
//     const NodeID node_id, std::size_t size_limit)
// {
//   std::size_t size = 0;
//   NodeFilterFunc filter = [&size_limit, &size](const NodePtr& node) {
//     if (node->byte_size > size_limit) {
//       size_limit -= node->byte_size;
//     }
//     if (node->memory_type == MemoryType::kStandBy) {
//       size += node->byte_size;
//     }
//     return node->memory_type == MemoryType::kStandBy;
//   };
//   auto node_ptr_list = GetNodesByFilter(filter, node_id);
//   return std::make_pair(size, node_ptr_list);
// }
