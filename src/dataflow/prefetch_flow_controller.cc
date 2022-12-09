#include "prefetch_flow_controller.h"


void
PrefetchFlowController::RecordNode(
    const InputIDPtr& input_id, const DAGNodePtr& node,
    const NodeMetaPtr& node_meta)
{
  // NodeID node_id = node->GetNodeID();
  // std::size_t memory_size = node->GetNodeByteSize();
  LOG_TRITON_VERBOSE("PrefetchFlowController::RecordNode");
  PutNodeTopology(input_id->correlation_id, node);
  
  // auto hash_id = std::hash<std::string>{}(input_id->request_id);
  // auto current_node_flow = std::make_shared<NodeFlow>(node);
  // if (flow_graph_.find(node->GetNodeID()) == flow_graph_.end()) {
  //   flow_graph_.insert({node->GetNodeID(), current_node_flow});
  // } else {
  //   current_node_flow = flow_graph_[node->GetNodeID()];
  // }

  // auto node_meta_list = request_trace_.Get(hash_id);
  // if (node_meta_list != nullptr) {
  //   LOG_TRITON_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
  //                       std::string(" already exists, continue flow recording"))
  //                          .c_str());
  //   NodeFlowPtr patent_node_flow = flow_graph_[node_meta_list->back()->node_id];
  //   LOG_TRITON_VERBOSE(
  //       (std::string(" patent node flow: ") +
  //        std::to_string(patent_node_flow->GetNodeMeta()->node_id))
  //           .c_str());
  //   // add new node to the parent
  //   current_node_flow->AddPrevNode(patent_node_flow);

  //   // LOG_TRITON_VERBOSE((std::string("Current node flow: ") +
  //   //              std::to_string(current_node_flow->GetNodeMeta()->node_id))
  //   //                 .c_str());
  //   patent_node_flow->AddNextNode(current_node_flow);
  //   // LOG_TRITON_VERBOSE((std::string("Patent node flow: ") +
  //   //              std::to_string(patent_node_flow->GetNodeMeta()->node_id))
  //   //                 .c_str());
  //   NodeMetaPtr node_meta_copy(new NodeMeta);
  //   *node_meta_copy = *(current_node_flow->GetNodeMeta());
  //   node_meta_list->push_back(node_meta_copy);
  //   LOG_TRITON_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
  //                       std::string(" add node ") +
  //                       std::to_string(current_node_flow->GetNodeID()) +
  //                       std::string(" to parent ") +
  //                       std::to_string(patent_node_flow->GetNodeID()))
  //                          .c_str());
  // } else {
  //   LOG_TRITON_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
  //                       std::string(" does not exist, start flow recording"))
  //                          .c_str());
  //   std::shared_ptr<NodeMetaPtrList> meta_list(new NodeMetaPtrList());
  //   NodeMetaPtr node_meta_copy(new NodeMeta);
  //   *node_meta_copy = *(current_node_flow->GetNodeMeta());
  //   meta_list->push_back(node_meta_copy);
  //   auto del_list = request_trace_.Put(hash_id, meta_list);

  //   // keep the child node updated by decreasing the reference count
  //   if (del_list != nullptr) {
  //     LOG_TRITON_VERBOSE(
  //         (std::string("Request ") + std::to_string(hash_id) +
  //          std::string(" cause a deletion of the previous flow of size ") +
  //          std::to_string(del_list->size()))
  //             .c_str());
  //     for (auto& node_meta : *del_list) {
  //       // decrease the parent's children_visited
  //       flow_graph_[node_meta->node_id]->DereferenceNode(node_meta);
  //       // for (auto& parent : flow_graph_[node_meta->node_id]GetNode) {
  //       //   parent.second->RemoveNextNode(flow_graph_[node_meta->node_id]);
  //       // }
  //     }
  //   }
  // }

  // *(current_node_flow->GetNodeMeta()) += *node_meta;

  // LOG_TRITON_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
  //                     std::string(" add node ") +
  //                     std::to_string(current_node_flow->GetNodeID()) +
  //                     std::string(" with meta ") +
  //                     current_node_flow->GetNodeMeta()->ToString())
  //                        .c_str());

  // // // update model graph
  // // auto model_id = node->GetNodeID();
  // // if (flow_graph_.find(model_id) == flow_graph_.end()) {
  // //   flow_graph_[model_id] = node;
  // // }
  // LOG_TRITON_VERBOSE((std::string("Request ") + std::to_string(hash_id) +
  //                     std::string(" has ") +
  //                     std::to_string(request_trace_.Size()) +
  //                     std::string(" flows"))
  //                        .c_str());
}


NodeMoveVec
PrefetchFlowController::PrefetchNode(const DAGNodePtr& node)
{
  LOG_TRITON_VERBOSE("PrefetchFlowController::PrefetchNode");
  NodeMoveVec prefetch_nodes;

  // prefetch for deterministic models


  // prefetch for non-deterministic models

  return prefetch_nodes;
}

// ModelProbabilityVec
// PrefetchFlowController::GetChildernProbability(const DAGNodePtr& node)
// {
//   // std::lock_guard<std::mutex> lock(mutex_);
//   ModelProbabilityVec children_prob;
//   if (flow_graph_.find(node->GetNodeID()) == flow_graph_.end()) {
//     return children_prob;
//   }
//   auto node_flow = flow_graph_[node->GetNodeID()];
//   for (auto& child : node_flow->GetNextNodes()) {
//     auto child_node = child.second->GetNode();
//     auto child_node_meta = child.second->GetNodeMeta();
//     children_prob.push_back(std::make_pair(
//         child_node,
//         child_node_meta->input_size_cnt / child_node_meta->visit_cnt));
//   }
//   // // normalize the probability
//   // double sum = 0;
//   // for (auto& prob : children_prob) {
//   //   sum += prob.second;
//   // }
//   // for (auto& prob : children_prob) {
//   //   prob.second /= sum;
//   // }
//   sort(children_prob.begin(), children_prob.end(), sortbysec<DAGNodePtr>);
//   return children_prob;
// }

// void
// PrefetchFlowController::RecursivelyUpdateProbability(
//     const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map)
// {
//   if (node_flow->GetNextNodes().size() == 0) {
//     return;
//   }
//   for (auto& child : node_flow->GetNextNodes()) {
//     auto child_node_flow = child.second;
//     auto child_node_meta = child.second->GetNodeMeta();
//     auto child_node = child_node_flow->GetNode();
//     // if (prob_map.find(child_id) == prob_map.end()) {
//     //   prob_map[child_id] = 0;
//     // }
//     // prob_map[child_id] += child_node->visit_cnt;
//     prob_map.push_back(std::make_pair(
//         child_node,
//         child_node_meta->input_size_cnt / child_node_meta->visit_cnt));
//   }

//   for (auto& child : node_flow->GetNextNodes()) {
//     RecursivelyUpdateProbability(child.second, prob_map);
//   }
// }

// ModelProbabilityVec
// PrefetchFlowController::GetTreeProbability(const DAGNodePtr& node)
// {
//   // std::lock_guard<std::mutex> lock(mutex_);

//   ModelProbabilityVec tree_prob;
//   if (flow_graph_.find(node->GetNodeID()) == flow_graph_.end()) {
//     return tree_prob;
//   }
//   auto node_flow = flow_graph_[node->GetNodeID()];

//   LOG_TRITON_VERBOSE((std::string("Get tree probability for node ") +
//                       std::to_string(node_flow->GetNodeID()))
//                          .c_str());

//   std::list<NodeFlowPtr> queue;
//   queue.push_back(node_flow);

//   LOG_TRITON_VERBOSE((std::string("Get tree probability for node ") +
//                       std::to_string(node_flow->GetNodeID()) +
//                       std::string(" with queue size ") +
//                       std::to_string(queue.size()))
//                          .c_str());

//   while (!queue.empty()) {
//     auto current_node_flow = queue.front();
//     queue.pop_front();
//     for (auto& child : current_node_flow->GetNextNodes()) {
//       auto child_node_flow = child.second;
//       LOG_TRITON_VERBOSE((std::string("Get tree probability for node ") +
//                           std::to_string(node_flow->GetNodeID()) +
//                           std::string(" with child node ") +
//                           std::to_string(child_node_flow->GetNodeID()))
//                              .c_str());
//       auto child_node_meta = child.second->GetNodeMeta();
//       LOG_TRITON_VERBOSE((std::string("Get tree probability for node ") +
//                           std::to_string(node_flow->GetNodeID()) +
//                           std::string(" with child node ") +
//                           std::to_string(child_node_flow->GetNodeID()) +
//                           std::string(" with child node meta ") +
//                           child_node_meta->ToString())
//                              .c_str());
//       auto child_node = child_node_flow->GetNode();
//       LOG_TRITON_VERBOSE((std::string("Get tree probability for node ") +
//                           std::to_string(node_flow->GetNodeID()) +
//                           std::string(" with child node ") +
//                           std::to_string(child_node_flow->GetNodeID()) +
//                           std::string(" with child node meta ") +
//                           child_node_meta->ToString() +
//                           std::string(" with child node ") +
//                           std::to_string(child_node->GetNodeID()))
//                              .c_str());
//       tree_prob.push_back(std::make_pair(
//           child_node,
//           child_node_meta->input_size_cnt / child_node_meta->visit_cnt));
//       LOG_TRITON_VERBOSE(
//           (std::string("Get tree probability for node ") +
//            std::to_string(node_flow->GetNodeID()) +
//            std::string(" with child node ") +
//            std::to_string(child_node_flow->GetNodeID()) +
//            std::string(" with child node meta ") + child_node_meta->ToString() +
//            std::string(" with child node ") +
//            std::to_string(child_node->GetNodeID()) +
//            std::string(" with child node probability ") +
//            std::to_string(
//                child_node_meta->input_size_cnt / child_node_meta->visit_cnt))
//               .c_str());
//       queue.push_back(child_node_flow);
//     }
//   }

//   // RecursivelyUpdateProbability(node_flow, tree_prob);
//   // sort(tree_prob.begin(), tree_prob.end(), sortbysec<DAGNodePtr>);
//   return tree_prob;
// }
