#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "common/state.h"
#include "engine/loop_handle.h"
#include "forward_def.h"
#include "muduo/base/noncopyable.h"
#include "triton/core/tritonbackend.h"
#include "utils/data_utils.h"
#include "utils/enum_utils.h"
#include "utils/memory_utils.h"
#include "utils/time_utils.h"
#include "utils/torch_utils.h"

// ENUM_MACRO(MemoryType, kStandBy, kReady)

class BackendEngineHandle;
class BackendEngine;

class DAGNode;
typedef std::shared_ptr<DAGNode> DAGNodePtr;

// class NodeMeta;

// typedef std::size_t NodeID;
// typedef std::shared_ptr<NodeMeta> NodeMetaPtr;
// typedef std::unordered_map<std::size_t, NodeMetaPtr> NodeMetaPtrMap;

// typedef std::shared_ptr<ModelDevice> ModelDevicePtr;

class NodeEnginePlugin : public muduo::noncopyable {
 public:
  NodeEnginePlugin();
  void ProcessTritonRequest(
      muduo::net::EventLoop::Functor func, const DAGNodePtr& node);
  void RecordTritonRequest(
      const DAGNodePtr& node, TRITONBACKEND_Request* const request,
      const torch::jit::IValue& input_tensors, const uint64_t compute_time);

 protected:
  muduo::net::EventLoop* loop_;
  LoopHandle* handle_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::shared_ptr<BackendEngine> engine_;
  bool exec_finished_;
  // std::shared_ptr<FlowEngine> flow_engine_;
};

// class NodeBackendEnginePlugin : public NodeEnginePlugin {
//  public:
//   NodeBackendEnginePlugin();
//   void ProcessTritonRequest(muduo::net::EventLoop::Functor func, const
//   DAGNodePtr& node);

//   private:
//    std::shared_ptr<BackendEngine> engine_;
// };


// class NodeFlowEnginePlugin : public NodeEnginePlugin {
//  public:
//   NodeFlowEnginePlugin();
//   void RecordTritonRequest(
//       const DAGNodePtr& node, TRITONBACKEND_Request* const request,
//       const torch::jit::IValue& input_tensors, const uint64_t compute_time);

//   private:
//    std::shared_ptr<FlowEngine> engine_;
// };


// struct NodeMeta {
//   DAGNodePtr node_ptr;
//   std::size_t visit_cnt;
//   std::size_t input_size_cnt;
//   std::size_t exec_lat_ns_cnt;
//   std::size_t load_lat_us_cnt;
//   std::size_t unload_lat_us_cnt;
//   NodeMeta()
//       : node_ptr(nullptr), visit_cnt(0), input_size_cnt(0),
//       exec_lat_ns_cnt(0),
//         load_lat_us_cnt(0), unload_lat_us_cnt(0)
//   {
//   }
// };

// using Kwargs = std::unordered_map<std::string, torch::jit::IValue>;

class DAGNode : std::enable_shared_from_this<DAGNode> {
 public:
  DAGNode() = delete;
  DAGNode(
      const std::string& model_path, const std::string& model_name,
      const std::uint64_t model_version)
      : model_id_(MakeID(model_name, model_version)), model_path_(model_path),
        model_(nullptr), model_copy_(), device_(DISK_DEVICE),
        memory_type_(MemoryType::kReady), engine_plugin_()
  {
    model_ = new ScriptModule(torch::jit::load(model_path_));
    // model_meta_.reset(new ModelMeta(model_path, model_name, model_version));
    model_instance_name_ = model_name + "_" + std::to_string(model_version);

    std::uint64_t param_size = 0;
    for (const auto& param : model_->parameters()) {
      param_size += param.numel() * param.element_size();
    }

    // Iterate model buffers and calculate the total size of the model
    std::uint64_t buffer_size = 0;
    for (const auto& buffer : model_->buffers()) {
      buffer_size += buffer.numel() * buffer.element_size();
    }

    model_byte_size_ = param_size + buffer_size;

    // At this point, we do not have knowledge of any memory capacity.
    // model_size is set to file size.
    // model_byte_size_ = GetFileSize(model_path);
  }
  ~DAGNode() = default;
  void SetMemoryType(const MemoryType& memory_type)
  {
    memory_type_ = memory_type;
  }
  MemoryType GetMemoryType() const { return memory_type_; }
  // bool IsEvict() const { return memory_type_ == MemoryType::kStandBy; }

  // void SetMemoryState(const MemoryState& memory_state) noexcept
  // {
  //   // EVICTING overrides every other state making sure that the memory is
  //   // freed. std::atomic<MemoryState> evict_state(MemoryState::EVICTING);
  //   // ATOMIC_NEQ_EXCHANGE(memory_type_, evict_state, memory_state)
  //   memory_type_ = memory_state;
  // }
  // bool IsMemoryState(const MemoryState& memory_state) const noexcept
  // {
  //   return memory_type_ == memory_state;
  // }

  const std::string GetModelInstanceInfo() noexcept
  {
    return model_instance_name_ + "(" + MemoryTypeToString(memory_type_) + ")";
  }
  // const ModelMetaPtr& GetModelMeta() const { return model_meta_; }
  ScriptModulePtr GetModel() const noexcept { return model_; }
  // const NodeMetaPtrMap& GetNextNodes() const { return next_nodes_; }
  // const NodeMetaPtrMap& GetPrevNodes() const { return prev_nodes_; }
  std::size_t GetNodeID() const noexcept { return model_id_; }
  std::size_t GetNodeByteSize() const { return model_byte_size_; }
  torch::Device GetDevice() const noexcept { return device_; }

  // // Add a parent node, called at child when a new node is constructed
  // NodeMetaPtr AddPrevNode(const DAGNodePtr& node);
  // // Add a child node, called at parent when a new node is constructed
  // // If the child node already exists, increase the visit count
  // NodeMetaPtr AddNextNode(const DAGNodePtr& node);
  // // Remove a child node, called at parent when a request id is deleted
  // // Decrease the visit count before actually removing the child node
  // NodeMetaPtr RemoveNextNode(const std::size_t& model_id);

  // @brief: This is the only management gateway for model memory
  // @param: device: the device to move the model to device
  // @param: copy: whether to copy the model to device
  void SetDevice(const torch::Device& device);
  // void SetDeviceCopy(const torch::Device& device);

  // void RecordMyself(const std::string& request_id);

  NodeEnginePlugin* GetEnginePlugin() { return &engine_plugin_; }

  void SetLastAccessTime() { last_access_time_ = TIME_NOW; }
  std::size_t GetLastAccessTime() const
  {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        last_access_time_.time_since_epoch()).count();
  }

  // NodeBackendEnginePlugin* GetBackendEnginePlugin()
  // {
  //   return &backend_engine_plugin_;
  // }
  // NodeFlowEnginePlugin* GetFlowEnginePlugin() { return &flow_engine_plugin_;
  // }

 private:
  std::size_t model_id_;
  std::string model_path_;
  std::string model_instance_name_;
  ScriptModulePtr model_;
  ScriptModulePtr model_copy_;
  torch::Device device_;
  std::size_t model_byte_size_;
  // ModelMetaPtr model_meta_;
  // NodeMetaPtrMap next_nodes_;
  // NodeMetaPtrMap prev_nodes_;
  MemoryType memory_type_;
  NodeEnginePlugin engine_plugin_;
  TimePoint last_access_time_;
  // NodeFlowEnginePlugin flow_engine_plugin_;
  // std::mutex* exec_mutex_;  // mutex to lock execution, protect memory state
  // muduo::net::EventLoop* loop_;
};
