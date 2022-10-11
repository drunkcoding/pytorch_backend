#pragma once

#include <condition_variable>

#include "dataflow/dag_node.h"
#include "op_base.h"
#include "utils/enum_utils.h"

// class LibtorchOpType : public EnumType {
//  public:
//   ENUM_ARGS(kExecute)

//   explicit LibtorchOpType(const int& type) : EnumType(type) {}
// };
ENUM_MACRO(LibtorchOpType, kExecute, kPrefetch)

struct LibtorchOpRequest : OpRequest {
  LibtorchOpType op_type;
  LoopHandle* handle;
  explicit LibtorchOpRequest(const LibtorchOpType& type) : op_type(type) {}
};
struct LibtorchOpResponse : OpResponse {
  LibtorchOpType op_type;
  explicit LibtorchOpResponse(const LibtorchOpType& type) : op_type(type) {}
};
typedef std::shared_ptr<LibtorchOpRequest> LibtorchRequestPtr;
typedef std::shared_ptr<LibtorchOpResponse> LibtorchResponsePtr;

struct LibtorchExecuteRequest : LibtorchOpRequest {
  EventLoop::Functor process_requests_cb;
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