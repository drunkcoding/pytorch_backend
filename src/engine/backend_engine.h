#pragma once

#include <unordered_map>

#include "backend_op.h"
#include "base/noncopyable.h"
#include "engine_base.h"
#include "event/eventloop_thread.h"
#include "event/eventloop_thread_pool.h"
#include "event/loop_handle.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"

class BackendEngineHandle
    : public EngineHandleBase,
      public std::enable_shared_from_this<BackendEngineHandle> {
 public:
  DEFAULT_LOOPHANDLE_MEMBER(BackendEngineHandle)

 public:
  virtual void DispachRequest(const RequestPtr& request);

 private:
  void RegisterService();

  // Will be called by engine for memory management
  void ExecuteModel(const BackendRequestPtr& request);
  void LoadModel(const BackendRequestPtr& request);
  void UnloadModel(const BackendRequestPtr& request);

  //  private:
  //   OpRegistryBase op_registry_;
};

class BackendEngine : public EngineBase,
                      public std::enable_shared_from_this<BackendEngine> {
 public:
  // DEFAULT_CLASS_MEMBER(BackendEngine)
  // STATIC_GET_INSTANCE(BackendEngine)
  BackendEngine();
  ~BackendEngine();

  LoopHandle* GetHandle() { return loop_->GetLoopHandle(); }
};
