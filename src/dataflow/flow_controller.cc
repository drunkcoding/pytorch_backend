#include "flow_controller.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <queue>

// #include "memory_controller.h"

void
NodeFlow::AddPrevNode(const NodeFlowPtr& prev_node)
{
  // prev_node->AddNextNode(SELF(NodeFlow));
  prev_nodes_.emplace(std::make_pair(prev_node->GetNodeID(), prev_node));
}


void
NodeFlow::AddNextNode(const NodeFlowPtr& next_node)
{
  // auto next_node = std::make_shared<NodeFlow>(node);
  next_nodes_.emplace(std::make_pair(next_node->GetNodeID(), next_node));
  // next_node->AddPrevNode(SELF(NodeFlow));
  //   if (!result.second) {
  //     result.first->second->visit_cnt++;
  //   }
  // return result.first->second;
}


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
//   auto model_id = next_node->GetNodeID();
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

void
NodeFlow::DereferenceNode(const NodeMetaPtr& node_meta)
{
  *node_meta_ -= *node_meta;
}

void
FlowControllerFactory::PutNodeTopology(
    const std::uint64_t& correlation_id, const DAGNodePtr& node)
{
  std::uint64_t high_corr_id =
      correlation_id >> 32;  // For childs in the same level
  std::uint64_t low_corr_id =
      correlation_id & 0xFFFFFFFF;  // For model inference pipeline
  if (visited_.find(correlation_id) == visited_.end()) {
    visited_.insert(correlation_id);
    visited_.insert(low_corr_id);
    auto cur_node_topology =
        std::make_shared<NodeTopology>(node, correlation_id);
    topology_.insert({cur_node_topology->GetNodeID(), cur_node_topology});
    if (root_ == nullptr) {
      root_ = cur_node_topology;
    } else {
      for (auto& node_topology : topology_) {
        // auto node_id = node_topology.first;
        auto node_topology_ptr = node_topology.second;
        auto prev_corr_id =
            (high_corr_id == 0) ? (low_corr_id - 1) : low_corr_id;
        if (node_topology_ptr->GetCorrelationID() == prev_corr_id) {
          node_topology_ptr->AddNextNode(cur_node_topology);
          cur_node_topology->AddPrevNode(node_topology_ptr);
        }
      }
    }
  }
}

NodeTopologyPtr
FlowControllerFactory::GetNodeTopology(const NodeID& node_id)
{
  if (topology_.find(node_id) == topology_.end()) {
    return nullptr;
  } else {
    return topology_[node_id];
  }
}

NodePtrList
FlowControllerFactory::GetNodesByFilter(
    const NodeFilterFunc& filter_func, const NodeID& node_id)
{
  NodePtrList nodes;
  if (topology_.find(node_id) == topology_.end()) {
    LOG_TRITON_VERBOSE(
        ("Node " + std::to_string(node_id) + " not found").c_str());
    return nodes;
  } else {
    auto node_topology = topology_[node_id];
    std::queue<NodeTopologyPtr> node_queue;
    node_queue.push(node_topology);
    while (!node_queue.empty()) {
      auto cur_node_topology = node_queue.front();
      node_queue.pop();
      if (filter_func(cur_node_topology->GetNode())) {
        nodes.push_back(cur_node_topology->GetNode());
      }
      for (auto& next_node : cur_node_topology->GetNextNodes()) {
        node_queue.push(next_node.second);
      }
    }
    return nodes;
  }
}

void
FlowControllerFactory::DispatchNodeMemoryInThread(
    const DAGNodePtr& node, const Device& device)
{
  std::thread t(&FlowControllerFactory::DispatchNodeMemory, this, node, device);
  t.detach();
}

void
FlowControllerFactory::DispatchNodeMemory(
    const DAGNodePtr& node, const Device& device)
{
  node->SetDevice(device);
  if (device == CPU_DEVICE || device == DISK_DEVICE) {
    node->SetMemoryType(MemoryType::kStandBy);
  }

  if (device == DEFAULT_CUDA_DEVICE &&
      node->GetMemoryType() != MemoryType::kLocked) {
    node->SetMemoryType(MemoryType::kReady);
  }
}
