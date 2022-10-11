#include "flow_controller.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void
NodeFlow::AddPrevNode(const NodeFlowPtr& prev_node)
{
  prev_node->AddNextNode(SELF(NodeFlow));
  prev_nodes_.emplace(std::make_pair(prev_node->GetNodeID(), prev_node));
}


void
NodeFlow::AddNextNode(const NodeFlowPtr& next_node)
{
  // auto next_node = std::make_shared<NodeFlow>(node);
  next_nodes_.insert({next_node->GetNodeID(), next_node});
  next_node->AddPrevNode(SELF(NodeFlow));
  //   if (!result.second) {
  //     result.first->second->visit_cnt++;
  //   }
  // return result.first->second;
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
FlowController::RecordNodeFlow(
    const std::string& request_id, const DAGNodePtr& node,
    const NodeMetaPtr& node_meta)
{
  //   std::lock_guard<std::mutex> lock(mutex_);
  auto hash_id = std::hash<std::string>{}(request_id);
  auto current_node_flow = std::make_shared<NodeFlow>(node);
  if (flow_graph_.find(node->GetNodeID()) == flow_graph_.end()) {
    flow_graph_.insert({node->GetNodeID(), current_node_flow});
  } else {
    current_node_flow = flow_graph_[node->GetNodeID()];
  }
  auto node_meta_list = request_trace_.Get(hash_id);

  if (node_meta_list != nullptr) {
    LOG_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
                 std::string(" already exists, continue flow recording"))
                    .c_str());
    auto patent_node_flow = flow_graph_[node_meta_list->back()->node_id];

    // add new node to the parent
    current_node_flow->AddPrevNode(patent_node_flow);
    patent_node_flow->AddNextNode(current_node_flow);
    node_meta_list->push_back(current_node_flow->GetNodeMeta());
  } else {
    LOG_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
                 std::string(" does not exist, start flow recording"))
                    .c_str());
    std::shared_ptr<NodeMetaPtrList> meta_list(new NodeMetaPtrList());
    auto del_list = request_trace_.Put(hash_id, meta_list);

    // keep the child node updated by decreasing the reference count
    if (del_list != nullptr) {
      LOG_VERBOSE(
          (std::string("Request ") + std::to_string(hash_id) +
           std::string(" cause a deletion of the previous flow of size ") +
           std::to_string(del_list->size()))
              .c_str());
      for (auto& node_meta : *del_list) {
        // decrease the parent's children_visited
        flow_graph_[node_meta->node_id]->DereferenceNode(node_meta);
        // for (auto& parent : flow_graph_[node_meta->node_id]GetNode) {
        //   parent.second->RemoveNextNode(flow_graph_[node_meta->node_id]);
        // }
      }
    }
  }

  *(current_node_flow->GetNodeMeta()) += *node_meta;

  // // update model graph
  // auto model_id = node->GetNodeID();
  // if (flow_graph_.find(model_id) == flow_graph_.end()) {
  //   flow_graph_[model_id] = node;
  // }
  LOG_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
               std::string(" has ") + std::to_string(request_trace_.Size()) +
               std::string(" flows"))
                  .c_str());
}


ModelProbabilityVec
FlowController::GetChildernProbability(const DAGNodePtr& node)
{
  // std::lock_guard<std::mutex> lock(mutex_);
  ModelProbabilityVec children_prob;
  auto node_flow = flow_graph_[node->GetNodeID()];
  for (auto& child : node_flow->GetNextNodes()) {
    auto child_node = child.second->GetNode();
    auto child_node_meta = child.second->GetNodeMeta();
    children_prob.push_back(std::make_pair(
        child_node,
        child_node_meta->input_size_cnt / child_node_meta->visit_cnt));
  }
  // // normalize the probability
  // double sum = 0;
  // for (auto& prob : children_prob) {
  //   sum += prob.second;
  // }
  // for (auto& prob : children_prob) {
  //   prob.second /= sum;
  // }
  sort(children_prob.begin(), children_prob.end(), sortbysec<DAGNodePtr>);
  return children_prob;
}

void
FlowController::RecursivelyUpdateProbability(
    const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map)
{
  if (node_flow->GetNextNodes().size() == 0) {
    return;
  }
  for (auto& child : node_flow->GetNextNodes()) {
    auto child_node_flow = child.second;
    auto child_node_meta = child.second->GetNodeMeta();
    auto child_node = child_node_flow->GetNode();
    // if (prob_map.find(child_id) == prob_map.end()) {
    //   prob_map[child_id] = 0;
    // }
    // prob_map[child_id] += child_node->visit_cnt;
    prob_map.push_back(std::make_pair(
        child_node,
        child_node_meta->input_size_cnt / child_node_meta->visit_cnt));
    RecursivelyUpdateProbability(child.second, prob_map);
  }
  
}

ModelProbabilityVec
FlowController::GetTreeProbability(const DAGNodePtr& node)
{
  // std::lock_guard<std::mutex> lock(mutex_);

  ModelProbabilityVec tree_prob;
  auto node_flow = flow_graph_[node->GetNodeID()];
  RecursivelyUpdateProbability(node_flow, tree_prob);
  sort(tree_prob.begin(), tree_prob.end(), sortbysec<DAGNodePtr>);
  return tree_prob;
}
