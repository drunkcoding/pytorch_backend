#include "dag_node.h"

#include "libtorch_flow.h"

namespace triton { namespace backend { namespace pytorch {


void
DAGNode::SetDevice(const torch::Device& device)
{
  torch::InferenceMode infer_guard(true);
  device_ = device;
  // In our context, lazy device stays on disk
  if (device == DISK_DEVICE) {
    delete model_;
    model_ = nullptr;
    return;
  }
  if (model_ == nullptr) {
    // InferenceMode should be used to guard all tensors operations including
    // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
    model_ = new ScriptModule(torch::jit::load(model_path_, device));
    // model_.reset(new Module(torch::jit::load(model_path_, device)));
  } else
    model_->to(device);
}

NodeMetaPtr
DAGNode::AddPrevNode(const DAGNodePtr& node)
{
  NodeMetaPtr prev_node = std::make_shared<NodeMeta>();
  prev_node->node_ptr = node;
  auto result = prev_nodes_.insert({node->GetNodeID(), prev_node});
  return result.first->second;
}


NodeMetaPtr
DAGNode::AddNextNode(const DAGNodePtr& node)
{
  NodeMetaPtr next_node = std::make_shared<NodeMeta>();
  next_node->node_ptr = node;
  auto result = next_nodes_.insert({node->GetNodeID(), next_node});
  if (!result.second) {
    result.first->second->visit_cnt++;
  }
  return result.first->second;
}


NodeMetaPtr
DAGNode::RemoveNextNode(const std::size_t& model_id)
{
  auto it = next_nodes_.find(model_id);
  if (it == next_nodes_.end()) {
    return nullptr;
  }
  if (it->second->visit_cnt > 0) {
    it->second->visit_cnt--;
  } else {
    next_nodes_.erase(it);
  }
  return it->second;
}

void
DAGNode::RecordMyself(const std::string& request_id)
{
  GET_INSTANCE(ModelFlowRecorder)
      ->RecordModelFlow(request_id, std::shared_ptr<DAGNode>(this));
}

}}}  // namespace triton::backend::pytorch