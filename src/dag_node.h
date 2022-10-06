#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "libtorch_common.h"
#include "libtorch_factory.h"
// #include "model_meta.h"

namespace triton { namespace backend { namespace pytorch {

class DAGNode;

class NodeMeta;

typedef std::size_t NodeID;
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

class DAGNode : std::enable_shared_from_this<DAGNode> {
 public:
  DAGNode() = default;
  // DAGNode(const ModelMetaPtr& model_meta) : model_meta_(model_meta) {}
  DAGNode(
      const std::string& model_path, const std::string& model_name,
      const std::uint64_t model_version)
      : model_id_(MakeID(model_name, model_version)), model_path_(model_path),
        model_(nullptr), device_(DISK_DEVICE),
        memory_state_(MemoryState::INACTIVE)
  {
    model_ = new ScriptModule(torch::jit::load(model_path_));
    // model_meta_.reset(new ModelMeta(model_path, model_name, model_version));
    model_instance_name_ = model_name + "_" + std::to_string(model_version);

    // std::uint64_t param_size = 0;
    // for (const auto& param : model_->parameters()) {
    //   param_size += param.numel() * param.element_size();
    // }

    // // Iterate model buffers and calculate the total size of the model
    // std::uint64_t buffer_size = 0;
    // for (const auto& buffer : model_->buffers()) {
    //   buffer_size += buffer.numel() * buffer.element_size();
    // }

    // model_byte_size_ = param_size + buffer_size;

    // At this point, we do not have knowledge of any memory capacity.
    // model_size is set to file size.
    model_byte_size_ = GetFileSize(model_path);
  }
  ~DAGNode() = default;

  // template <typename InputType, typename OuputType>
  // OuputType& Forward(InputType& inputs)
  // {
  //   return model_meta_->GetModel()->forward(inputs);
  // }

  void SetMemoryState(const MemoryState& memory_state) noexcept
  {
    // EVICTING overrides every other state making sure that the memory is
    // freed. std::atomic<MemoryState> evict_state(MemoryState::EVICTING);
    // ATOMIC_NEQ_EXCHANGE(memory_state_, evict_state, memory_state)
    memory_state_ = memory_state;
  }
  bool IsMemoryState(const MemoryState& memory_state) const noexcept
  {
    return memory_state_ == memory_state;
  }

  const std::string GetModelInstanceInfo() noexcept
  {
    return model_instance_name_ + "(" + MemoryStateToString(memory_state_) +
           ")";
  }
  // const ModelMetaPtr& GetModelMeta() const { return model_meta_; }
  ScriptModulePtr GetModel() const noexcept { return model_; }
  const NodeMetaPtrMap& GetNextNodes() const { return next_nodes_; }
  const NodeMetaPtrMap& GetPrevNodes() const { return prev_nodes_; }
  std::size_t GetNodeID() const noexcept { return model_id_; }
  std::size_t GetNodeByteSize() const { return model_byte_size_; }
  torch::Device GetDevice() const noexcept { return device_; }

  // Add a parent node, called at child when a new node is constructed
  NodeMetaPtr AddPrevNode(const DAGNodePtr& node);
  // Add a child node, called at parent when a new node is constructed
  // If the child node already exists, increase the visit count
  NodeMetaPtr AddNextNode(const DAGNodePtr& node);
  // Remove a child node, called at parent when a request id is deleted
  // Decrease the visit count before actually removing the child node
  NodeMetaPtr RemoveNextNode(const std::size_t& model_id);

  // // Move the model to DeviceType and DeviceID
  // void SetDevice(const DeviceType& device, const int& device_id)
  // {
  //   model_meta_->SetDevice(device, device_id);
  // }

  // @brief: This is the only management gateway for model memory
  // @param: device: the device to move the model to device
  void SetDevice(const torch::Device& device);

  void RecordMyself(const std::string& request_id);

  void WaitForModel(const MemoryState& memory_state)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // always true, just work as a lock protected assignment
    cv_.wait(lock, [this] { return !device_.is_cuda(); });
    // memory_state_ = memory_state;
  }

 private:
  std::size_t model_id_;
  std::string model_path_;
  std::string model_instance_name_;
  ScriptModulePtr model_;
  torch::Device device_;
  std::size_t model_byte_size_;
  // ModelMetaPtr model_meta_;
  NodeMetaPtrMap next_nodes_;
  NodeMetaPtrMap prev_nodes_;
  MemoryState memory_state_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}}}  // namespace triton::backend::pytorch
