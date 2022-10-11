#pragma once

#include "dataflow/dag_node.h"
#include "op_base.h"
#include "utils/enum_utils.h"
#include "utils/torch_utils.h"

ENUM_MACRO(BackendOpType, kExecute, kLoad, kUnload)

// class BackendOpType : public EnumType {
//  public:
//   ENUM_ARGS(kExecute, kLoad, kUnload)

//   explicit BackendOpType(const int& type) : EnumType(type) {}
// };
// ENUM_STRUCT_MACRO(BackendOpType, kExecute, kLoad, kUnload)

struct BackendOpRequest : OpRequest {
  BackendOpType op_type;
  explicit BackendOpRequest(const BackendOpType& type) : op_type(type) {}
};
struct BackendOpResponse : OpResponse {
  BackendOpType op_type;
  explicit BackendOpResponse(const BackendOpType& type) : op_type(type) {}
};


typedef std::shared_ptr<BackendOpRequest> BackendRequestPtr;
typedef std::shared_ptr<BackendOpResponse> BackendResponsePtr;


struct BackendExecuteRequest : BackendOpRequest {
  EventLoop::Functor process_requests_cb;
  std::mutex* mutex;
  std::condition_variable* cv;
  BackendExecuteRequest() : BackendOpRequest(BackendOpType::kExecute) {}
};
struct BackendExecuteResponse : BackendOpResponse {
  BackendExecuteResponse() : BackendOpResponse(BackendOpType::kExecute) {}
};

struct BackendLoadRequest : BackendOpRequest {
  DAGNodePtr node;
  Device target_device = DEFAULT_CUDA_DEVICE;
  BackendLoadRequest() : BackendOpRequest(BackendOpType::kLoad) {}
};
struct BackendLoadResponse : BackendOpResponse {
  DAGNodePtr node;
  BackendLoadResponse() : BackendOpResponse(BackendOpType::kLoad) {}
};

struct BackendUnloadResponse : BackendOpResponse {
  DAGNodePtr node;
  Device target_device = CPU_DEVICE;
  BackendUnloadResponse() : BackendOpResponse(BackendOpType::kUnload) {}
};
struct BackendUnloadRequest : BackendOpRequest {
  DAGNodePtr node;
  Device target_device = CPU_DEVICE;
  LoopHandle* handle;
  std::function<void(const BackendResponsePtr&)> cb;
  BackendUnloadRequest() : BackendOpRequest(BackendOpType::kUnload) {}
};
