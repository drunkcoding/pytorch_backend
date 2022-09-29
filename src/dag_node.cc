#include "dag_node.h"


NodeMetaPtr
DAGNode::AddPrevNode(const DAGNodePtr& node)
{
  NodeMetaPtr prev_node = std::make_shared<NodeMeta>();
  prev_node->node_ptr = node;
  auto result = prev_nodes_.insert({node->GetModelMeta()->GetID(), prev_node});
  return result.first->second;
}


NodeMetaPtr
DAGNode::AddNextNode(const DAGNodePtr& node)
{
  NodeMetaPtr next_node = std::make_shared<NodeMeta>();
  next_node->node_ptr = node;
  auto result = next_nodes_.insert({node->GetModelMeta()->GetID(), next_node});
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
