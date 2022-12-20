#include "flow_controller.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <queue>

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
//   next_nodes_.emplace(std::make_pair(next_node->id, next_node));
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

  node->id = correlation_id;

  if (visited_.find(correlation_id) == visited_.end()) {
    visited_.insert(correlation_id);

    auto node_body = std::make_shared<NodeBody>(node);

    assert(pipeline_.stages.size() >= low_corr_id);

    // this is a new stage  at tail
    if (pipeline_.stages.size() == low_corr_id) {
      auto stage = std::make_shared<Stage>();
      stage->nodes.push_back(node_body);
      pipeline_.stages.push_back(stage);
    } else {
      // this is a new node in the middle, has to be a sparse branching
      auto stage = pipeline_.stages[low_corr_id];

      if (!stage->is_sparse) {
        assert(stage->nodes.size() == 1);
        stage->is_sparse = true;
        stage->root = stage->nodes[0];
        stage->nodes.clear();
        if (high_corr_id > stage->nodes.size()) {
          stage->nodes.resize(high_corr_id);
        }
        stage->nodes[high_corr_id - 1];
      } else {
        if (high_corr_id > stage->nodes.size()) {
          stage->nodes.resize(high_corr_id);
        }
        stage->nodes[high_corr_id - 1];
      }
    }
  }
  // auto node_id = node->id;
  auto stage = pipeline_.stages[low_corr_id];
  NodeBodyPtr node_body;

  auto now = MCIROSECONDS_SINCE_EPOCH;

  if (stage->is_sparse) {
    if (high_corr_id == 0) {
      node_body = stage->root;
      stage->visit_time.push_back(now);
    } else
      node_body = stage->nodes[high_corr_id - 1];
  } else {
    node_body = stage->nodes[0];
    stage->visit_time.push_back(now);
  }

  node_body->visit_time.push_back(now);
  node_body->visit_cnt += 1;
  node_body->activate_request.insert(request_id);
  node_body->node->last_access_time = now;

  // First update the visit count
  // 1. visit to sparse nodes only count once to total visit count
  // 2. find the last sparse layer and update the visit count of parent node

  if (correlation_id == 0) {
    // this is the first node in the pipeline
    // update the visit count of the pipeline
    pipeline_.visit_cnt += 1;
  }

  if (stage->is_sparse && high_corr_id == 0) {
    // this is the route node in the sparse layer
    // update the visit count of the stage
    stage->visit_cnt += 1;
  }

  if (!stage->is_sparse) {
    // this is the only node in the stage
    // update the visit count of the stage
    stage->visit_cnt += 1;
  }

  if (stage->is_sparse && high_corr_id > 0) {
    // this is the child node in the sparse layer
    // update the visit count of the parent node
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
          last_stage_node->children.resize(high_corr_id);
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
    for (auto& stage : pipeline_.stages) {
      for (auto& node : stage->nodes) {
        node->activate_request.erase(request_id);
      }
    }
  }

  // for all nodes and stage and children in the pipeline, reduce count of nodes
  // that are not visited for 1 minute
  std::size_t microseconds = 60000000;
  for (auto& stage : pipeline_.stages) {
    for (auto& node : stage->nodes) {
      auto visit = node->visit_time.begin();
      while (visit != node->visit_time.end() && now - *visit > microseconds) {
        node->visit_time.pop_front();
        node->visit_cnt -= 1;
        visit = node->visit_time.begin();
      }

      int k = 0;
      for (auto& children : node->children_visit_time) {
        auto visit = children.begin();
        while (visit != children.end() && now - *visit > microseconds) {
          children.pop_front();
          node->children_visit_cnt[k] -= 1;
          visit = children.begin();
        }
        k++;
      }
    }


    auto visit = stage->visit_time.begin();
    while (visit != stage->visit_time.end() && now - *visit > microseconds) {
      stage->visit_time.pop_front();
      stage->visit_cnt -= 1;
      visit = stage->visit_time.begin();
    }
  }

  // if (visit_cnt_.find(node_id) == visit_cnt_.end()) {
  //   visit_cnt_[node_id] = 0;
  // }
  // visit_cnt_[node_id] += 1;
  // if (!stage->is_sparse) {
  //   total_visit_cnt_ += 1;
  // }


  // if (visit_time_.find(node_id) == visit_time_.end()) {
  //   visit_time_[node_id] = 0;
  // }
  // visit_time_[node_id] = MCIROSECONDS_SINCE_EPOCH;

  // auto now = MCIROSECONDS_SINCE_EPOCH;
  // std::vector<NodeID> delete_nodes;
  // for (auto& visit : visit_time_) {
  //   if (now - visit.second > 60000000) {
  //     delete_nodes.push_back(visit.first);
  //     total_visit_cnt_ -= 1;
  //     visit_cnt_[visit.first] -= 1;
  //   }
  // }

  // for (auto& node_id : delete_nodes) {
  //   visit_time_.erase(node_id);
  // }
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


// FilterResult
// FlowControllerFactory::GetTotalLiveParamSize()
// {
//   std::size_t size = 0;
//   NodeFilterFunc filter = [&size](const NodePtr& node) {
//     if (node->memory_type == MemoryType::kLocked) {
//       size += node->byte_size;
//       return true;
//     }
//     return false;
//   };
//   auto node_ptr_list = GetNodesByFilter(filter, root_->GetNodeID());
//   return std::make_pair(size, node_ptr_list);
// }

FilterResult
FlowControllerFactory::GetTotalLiveParamSize()
{
  std::size_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      auto node = node_body->node;
      if (node->memory_type == MemoryType::kLocked) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }
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
  std::size_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      auto node = node_body->node;
      if (node->device == DEFAULT_CUDA_DEVICE) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }
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
FlowControllerFactory::GetRemovableNodes()
{
  std::size_t size = 0;
  NodePtrList node_ptr_list;

  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      auto node = node_body->node;
      if (node->memory_type == MemoryType::kReady) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
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

FilterResult
FlowControllerFactory::GetStandbyChildBySizeLimit(
    const NodePtr& node, std::size_t size_limit)
{
  std::size_t size = 0;
  NodePtrList node_ptr_list;

  // std::uint64_t stage_idx = node->id & 0x00000000FFFFFFFF;
  // std::uint64_t node_idx = (node->id >> 32) - 1;

  for (std::uint64_t stage_idx = node->id & 0x00000000FFFFFFFF;
       stage_idx < pipeline_.stages.size(); ++stage_idx) {
    auto stage = pipeline_.stages[stage_idx];
    for (auto& node_body : stage->nodes) {
      auto node = node_body->node;
      if (node->byte_size > size_limit) {
        size_limit -= node->byte_size;
      }
      if (node->memory_type == MemoryType::kStandBy) {
        size += node->byte_size;
        node_ptr_list.push_back(node);
      }
    }
  }
  return std::make_pair(size, node_ptr_list);
}
