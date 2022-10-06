#include "libtorch_flow.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dag_registry.h"

namespace triton { namespace backend { namespace pytorch {

void
ModelFlowRecorder::RecordModelFlow(
    const std::string& request_id, const DAGNodePtr& node)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto node_trace_ptr = request_trace_.Get(request_id);
  if (node_trace_ptr != nullptr) {
    LOG_VERBOSE(
        (std::string("Request ") + request_id +
         std::string(" already exists, continue flow recording"))
            .c_str());
    // auto node_trace_ptr = request_trace_.Get(request_id);
    // if (model_graph_.find(node->GetNodeID()) == model_graph_.end()) {
    //   model_graph_.insert({node->GetNodeID(), node});
    // }
    // DAGNodePtr new_node = model_graph_.find(node->GetNodeID())->second;

    DAGNodePtr patent_node = node_trace_ptr->back();

    // add new node to the parent
    node->AddPrevNode(patent_node);
    patent_node->AddNextNode(node);
    node_trace_ptr->push_back(node);
  } else {
    LOG_VERBOSE(
        (std::string("Request ") + request_id +
         std::string(" does not exist, start flow recording"))
            .c_str());
    std::shared_ptr<std::list<DAGNodePtr>> model_list;
    model_list.reset(new std::list<DAGNodePtr>());
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Request ") + request_id +
         std::string(" add new node to the list"))
            .c_str());
    model_list->push_back(node);
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Request ") + request_id +
         std::string(" add new list to the map"))
            .c_str());
    auto del_list = request_trace_.Put(request_id, model_list);
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Request ") + request_id +
         std::string(" add new node to the graph"))
            .c_str());

    // keep the child node updated by decreasing the reference count
    if (del_list != nullptr) {
      LOG_VERBOSE(
          (std::string("Request ") + request_id +
           std::string(" cause a deletion of the previous flow of size ") +
           std::to_string(del_list->size()))
              .c_str());
      for (auto& node : *del_list) {
        // decrease the parent's children_visited
        for (auto& parent : node->GetPrevNodes()) {
          parent.second->node_ptr->RemoveNextNode(node->GetNodeID());
        }
      }
    }
  }

  // // update model graph
  // auto model_id = node->GetNodeID();
  // if (model_graph_.find(model_id) == model_graph_.end()) {
  //   model_graph_[model_id] = node;
  // }
  LOG_VERBOSE(
      (std::string("Request ") + request_id + std::string(" has ") +
       std::to_string(request_trace_.Size()) + std::string(" flows"))
          .c_str());
}


ModelProbabilityVec
ModelFlowRecorder::GetChildernProbability(const std::size_t& model_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  ModelProbabilityVec children_prob;
  auto node = GET_INSTANCE(DAGRegistry)->GetNode(model_id);
  for (auto& child : node->GetNextNodes()) {
    auto& child_node = child.second;
    children_prob.push_back(std::make_pair(
        child.first, child_node->visit_cnt / request_trace_.Size()));
  }
  // // normalize the probability
  // double sum = 0;
  // for (auto& prob : children_prob) {
  //   sum += prob.second;
  // }
  // for (auto& prob : children_prob) {
  //   prob.second /= sum;
  // }
  sort(children_prob.begin(), children_prob.end(), sortbysec);
  return children_prob;
}

void
ModelFlowRecorder::RecursivelyUpdateProbability(
    const DAGNodePtr& node, ModelProbabilityVec& prob_map)
{
  if (node->GetNextNodes().size() == 0) {
    return;
  }
  for (auto& child : node->GetNextNodes()) {
    auto child_node = child.second;
    auto child_id = child_node->node_ptr->GetNodeID();
    // if (prob_map.find(child_id) == prob_map.end()) {
    //   prob_map[child_id] = 0;
    // }
    // prob_map[child_id] += child_node->visit_cnt;
    prob_map.push_back(std::make_pair(child_id, child_node->visit_cnt));
    RecursivelyUpdateProbability(child_node->node_ptr, prob_map);
  }
}

ModelProbabilityVec
ModelFlowRecorder::GetTreeProbability(const std::size_t& model_id)
{
  std::lock_guard<std::mutex> lock(mutex_);

  ModelProbabilityVec tree_prob;
  auto node = GET_INSTANCE(DAGRegistry)->GetNode(model_id);
  RecursivelyUpdateProbability(node, tree_prob);

  // normalize the probability
  // double sum = 0;
  // for (auto& prob : tree_prob) {
  //   sum += prob.second;
  // }
  // for (auto& prob : tree_prob) {
  //   prob.second /= sum;
  // }
  sort(tree_prob.begin(), tree_prob.end(), sortbysec);
  return tree_prob;
}

// DAGNodePtr
// ModelFlowRecorder::GetDAGNode(const std::size_t& model_id)
// {
//   std::lock_guard<std::mutex> lock(mutex_);
//   if (model_graph_.find(model_id) != model_graph_.end()) {
//     return model_graph_.find(model_id)->second;
//   }
//   return nullptr;
// }

}}}  // namespace triton::backend::pytorch