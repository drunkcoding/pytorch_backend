#pragma once

#include <condition_variable>

#include "backend_op.h"
#include "dataflow/dag_node.h"
#include "dataflow/forward_def.h"
#include "op_base.h"
#include "utils/enum_utils.h"
#include "dataflow/flow_controller.h"

// class LibtorchOpType : public EnumType {
//  public:
//   ENUM_ARGS(kExecute)

//   explicit LibtorchOpType(const int& type) : EnumType(type) {}
// };
ENUM_MACRO(LibtorchOpType, kExecute, kPrefetch)

struct LibtorchOpRequest {
  LibtorchOpType op_type;
  BackendEnginePtr engine;  // The backend engine that this request belongs to.
  explicit LibtorchOpRequest(const LibtorchOpType& type) : op_type(type) {}
};
struct LibtorchOpResponse {
  LibtorchOpType op_type;
  explicit LibtorchOpResponse(const LibtorchOpType& type) : op_type(type) {}
};
typedef std::shared_ptr<LibtorchOpRequest> LibtorchRequestPtr;
typedef std::shared_ptr<LibtorchOpResponse> LibtorchResponsePtr;

struct LibtorchExecuteRequest : LibtorchOpRequest {
  muduo::net::EventLoop::Functor process_requests_cb;
  DAGNodePtr node;
  Device target_device = DEFAULT_CUDA_DEVICE;
  mutable std::mutex* mutex;
  mutable std::condition_variable* cv;
  LibtorchExecuteRequest() : LibtorchOpRequest(LibtorchOpType::kExecute) {}
};
struct LibtorchExecuteResponse : LibtorchOpResponse {
  LibtorchExecuteResponse() : LibtorchOpResponse(LibtorchOpType::kExecute) {}
};

struct LibtorchPrefetchRequest : LibtorchOpRequest {
  DAGNodePtr node;
  Device target_device = DEFAULT_CUDA_DEVICE;
  LibtorchPrefetchRequest() : LibtorchOpRequest(LibtorchOpType::kPrefetch) {}
};
struct LibtorchPrefetchResponse : LibtorchOpResponse {
  LibtorchPrefetchResponse() : LibtorchOpResponse(LibtorchOpType::kPrefetch) {}
};

class LibtorchOpManager : public OpBase {
 public:
  explicit LibtorchOpManager(muduo::net::EventLoop* loop);
  ~LibtorchOpManager();

  virtual void Process() {}

  void ExecuteModel(const LibtorchRequestPtr& request);
  // void PrefetchModel(const LibtorchRequestPtr& request);

 private:
  void RunLoadInBackend(
      const BackendEnginePtr& engine, const DAGNodePtr& node,
      const Device& device);
  void RunUnloadInBackend(
      const LibtorchRequestPtr& request, const DAGNodePtr& node,
      const Device& device, std::atomic_uint64_t* wait_count);


  void DispatchToBackend(const LibtorchRequestPtr& request);

  void EntryWaitModelUnload(
      const BackendResponsePtr& response, const LibtorchRequestPtr& request,
      std::atomic_uint64_t* wait_count);

  void EntryWaitModelLoad(const BackendResponsePtr& response);

  void EntryWaitImmedientModel(
      const BackendResponsePtr& response, const LibtorchRequestPtr& request);

 private:
  NodeMoveVec node_prefetch_vec_;
  std::unordered_map<NodeID, BackendEnginePtr> backend_engine_map_;
};