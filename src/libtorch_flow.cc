#include "libtorch_flow.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace triton { namespace backend { namespace pytorch {

void
ModelFlowRecorder::RecordModelFlow(
    const std::string& request_id, const ModelMetaPtr& model_meta)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (request_trace_.Contains(request_id)) {
    auto node_trace_ptr = request_trace_.Get(request_id);
    if (model_graph_.find(model_meta->GetID()) == model_graph_.end()) {
      model_graph_.insert(
          {model_meta->GetID(), std::make_shared<DAGNode>(model_meta)});
    }
    DAGNodePtr new_node = model_graph_.find(model_meta->GetID())->second;
    DAGNodePtr patent_node = node_trace_ptr->back();

    // add new node to the parent
    new_node->AddPrevNode(patent_node);
    patent_node->AddNextNode(new_node);
  } else {
    std::list<DAGNodePtr> model_list;
    model_list.push_back(std::make_shared<DAGNode>(model_meta));
    std::list<DAGNodePtr> del_list =
        request_trace_.Put(request_id, model_list)->second;

    // keep the child node updated by decreasing the reference count
    for (auto& node : del_list) {
      // decrease the parent's children_visited
      for (auto& parent : node->GetPrevNodes()) {
        parent.second->node_ptr->RemoveNextNode(node->GetModelMeta()->GetID());
      }
    }
  }

  // update model graph
  auto model_id = model_meta->GetID();
  if (model_graph_.find(model_id) == model_graph_.end()) {
    model_graph_[model_id] = std::make_shared<DAGNode>(model_meta);
  }
}


std::unordered_map<std::size_t, double>
ModelFlowRecorder::GetChildernProbability(const std::size_t& model_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_map<std::size_t, double> children_prob;
  if (model_graph_.find(model_id) != model_graph_.end()) {
    auto node = model_graph_.find(model_id)->second;
    for (auto& child : node->GetNextNodes()) {
      auto& child_node = child.second;
      children_prob[child.first] =
          child_node->visit_cnt / request_trace_.Size();
    }
  }
  // // normalize the probability
  // double sum = 0;
  // for (auto& prob : children_prob) {
  //   sum += prob.second;
  // }
  // for (auto& prob : children_prob) {
  //   prob.second /= sum;
  // }
  return children_prob;
}

void
ModelFlowRecorder::RecursivelyUpdateProbability(
    const DAGNodePtr& node, std::unordered_map<std::size_t, double>& prob_map)
{
  if (node->GetNextNodes().size() == 0) {
    return;
  }
  for (auto& child : node->GetNextNodes()) {
    auto child_node = child.second;
    auto child_id = child_node->node_ptr->GetNodeID();
    if (prob_map.find(child_id) == prob_map.end()) {
      prob_map[child_id] = 0;
    }
    prob_map[child_id] += child_node->visit_cnt;
    RecursivelyUpdateProbability(child_node->node_ptr, prob_map);
  }
}

std::unordered_map<std::size_t, double>
ModelFlowRecorder::GetTreeProbability(const std::size_t& model_id)
{
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<std::size_t, double> tree_prob;

  RecursivelyUpdateProbability(model_graph_[model_id], tree_prob);

  // normalize the probability
  // double sum = 0;
  // for (auto& prob : tree_prob) {
  //   sum += prob.second;
  // }
  // for (auto& prob : tree_prob) {
  //   prob.second /= sum;
  // }
  return tree_prob;
}

DAGNodePtr
ModelFlowRecorder::GetDAGNode(const std::size_t& model_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (model_graph_.find(model_id) != model_graph_.end()) {
    return model_graph_.find(model_id)->second;
  }
  return nullptr;
}

}}}  // namespace triton::backend::pytorch