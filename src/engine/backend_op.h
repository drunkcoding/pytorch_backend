#pragma once

#include "op_base.h"
#include "utils/enum_utils.h"
#include "utils/topology.h"
#include "utils/torch_utils.h"

ENUM_MACRO(BackendOpType, kExecute, kLoad, kUnload)

// class BackendOpType : public EnumType {
//  public:
//   ENUM_ARGS(kExecute, kLoad, kUnload)

//   explicit BackendOpType(const int& type) : EnumType(type) {}
// };
// ENUM_STRUCT_MACRO(BackendOpType, kExecute, kLoad, kUnload)

struct BackendOpRequest {
  BackendOpType op_type;
  explicit BackendOpRequest(const BackendOpType& type) : op_type(type) {}
};
struct BackendOpResponse {
  BackendOpType op_type;
  explicit BackendOpResponse(const BackendOpType& type) : op_type(type) {}
};


typedef std::shared_ptr<BackendOpRequest> BackendRequestPtr;
typedef std::shared_ptr<BackendOpResponse> BackendResponsePtr;
typedef std::function<void(const BackendResponsePtr&)> BackendCallback;

struct BackendExecuteRequest : BackendOpRequest {
  muduo::net::EventLoop::Functor process_requests_cb;
  NodePtr node;
  BackendCallback cb;
  std::mutex* mutex;
  std::condition_variable* cv;
  BackendExecuteRequest() : BackendOpRequest(BackendOpType::kExecute) {}
};
struct BackendExecuteResponse : BackendOpResponse {
  NodePtr node;
  BackendExecuteResponse() : BackendOpResponse(BackendOpType::kExecute) {}
};

struct BackendLoadRequest : BackendOpRequest {
  NodePtr node;
  Device from = DISK_DEVICE;
  Device to = DISK_DEVICE;
  BackendCallback cb;
  BackendLoadRequest() : BackendOpRequest(BackendOpType::kLoad) {}
};
struct BackendLoadResponse : BackendOpResponse {
  NodePtr node;
  Device from = DISK_DEVICE;
  Device to = DISK_DEVICE;
  BackendLoadResponse() : BackendOpResponse(BackendOpType::kLoad) {}
};

struct BackendUnloadResponse : BackendOpResponse {
  NodePtr node;
  Device from = DISK_DEVICE;
  Device to = DISK_DEVICE;
  BackendUnloadResponse() : BackendOpResponse(BackendOpType::kUnload) {}
};
struct BackendUnloadRequest : BackendOpRequest {
  NodePtr node;
  Device from = DISK_DEVICE;
  Device to = DISK_DEVICE;
  // LoopHandle* handle;
  BackendCallback cb;
  BackendUnloadRequest() : BackendOpRequest(BackendOpType::kUnload) {}
};

class BackendOpManager : public OpBase {
 public:
  explicit BackendOpManager(muduo::net::EventLoop* loop);
  ~BackendOpManager();

  virtual void Process() {}

  void ExecuteModel(const BackendRequestPtr& request);
  void LoadModel(const BackendRequestPtr& request);
  void UnloadModel(const BackendRequestPtr& request);
};
