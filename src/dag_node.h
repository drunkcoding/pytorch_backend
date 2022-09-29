#pragma once

#include <unordered_map>

#include "libtorch_common.h"
#include "model_meta.h"

class DAGNode;

class NodeMeta;

typedef std::shared_ptr<NodeMeta> NodeMetaPtr;
typedef std::unordered_map<std::size_t, NodeMetaPtr> NodeMetaPtrMap;
typedef std::shared_ptr<DAGNode> DAGNodePtr;
// typedef std::shared_ptr<ModelDevice> ModelDevicePtr;

struct NodeMeta {
  DAGNodePtr node_ptr;
  std::size_t visit_cnt;

  NodeMeta() : node_ptr(nullptr), visit_cnt(0) {}
};

using Kwargs = std::unordered_map<std::string, torch::jit::IValue>;


class DAGNode {
 public:
  DAGNode() = default;
  DAGNode(const ModelMetaPtr& model_meta) : model_meta_(model_meta) {}
  DAGNode(
      const std::string& model_path, const std::string& model_name,
      const std::uint64_t model_version)
  {
    model_meta_.reset(new ModelMeta(model_path, model_name, model_version));
  }
  ~DAGNode() = default;

  template <typename InputType, typename OuputType>
  OuputType& Forward(InputType& inputs)
  {
    return model_meta_->GetModel()->forward(inputs);
  }

  const ModelMetaPtr& GetModelMeta() const { return model_meta_; }
  const NodeMetaPtrMap& GetNextNodes() const { return next_nodes_; }
  const NodeMetaPtrMap& GetPrevNodes() const { return prev_nodes_; }
  const std::size_t GetNodeID() const noexcept { return model_meta_->GetID(); }
  const std::size_t GetNodeByteSize() const
  {
    return model_meta_->GetByteSize();
  }

  // Add a parent node, called at child when a new node is constructed
  NodeMetaPtr AddPrevNode(const DAGNodePtr& node);
  // Add a child node, called at parent when a new node is constructed
  // If the child node already exists, increase the visit count
  NodeMetaPtr AddNextNode(const DAGNodePtr& node);
  // Remove a child node, called at parent when a request id is deleted
  // Decrease the visit count before actually removing the child node
  NodeMetaPtr RemoveNextNode(const std::size_t& model_id);

  // Move the model to DeviceType and DeviceID
  void SetDevice(const DeviceType& device, const int& device_id)
  {
    model_meta_->SetDevice(device, device_id);
  }

 private:
  ModelMetaPtr model_meta_;
  NodeMetaPtrMap next_nodes_;
  NodeMetaPtrMap prev_nodes_;
  std::mutex mutex_;
};
