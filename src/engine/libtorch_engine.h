#pragma once

#include <torch/script.h>

#include <atomic>
#include <unordered_map>

#include "backend_op.h"
#include "base/noncopyable.h"
#include "engine_base.h"
#include "event/eventloop_thread.h"
#include "event/eventloop_thread_pool.h"
#include "libtorch_op.h"
#include "loop_handle.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"

class LibtorchEngineHandle
    : public EngineHandleBase,
      public std::enable_shared_from_this<LibtorchEngineHandle> {
 public:
  DEFAULT_LOOPHANDLE_MEMBER(LibtorchEngineHandle)

 public:
  virtual void DispachRequest(const RequestPtr& request);

 private:
  void RegisterService();
  void ExecuteModel(const LibtorchRequestPtr& request);
  void PrefetchModel(const LibtorchRequestPtr& request);
  // void MemoryAllocation(const LibtorchRequestPtr& request);

  void RunLoadInBackend(
      EventLoop* const loop, const DAGNodePtr& node, const Device& device);
  void RunUnloadInBackend(
      const LibtorchRequestPtr& request, const DAGNodePtr& node, const Device& device,
      std::atomic_uint64_t* wait_count);

  void DispatchToBackend(const LibtorchRequestPtr& request);

  void EntryWaitModelUnloadInLoop(
      const BackendResponsePtr& response, const LibtorchRequestPtr& request,
      std::atomic_uint64_t* wait_count);

 private:
  void EntryWaitModelUnload(
      const BackendResponsePtr& response, const LibtorchRequestPtr& request,
      std::atomic_uint64_t* wait_count);

 private:
  OpRegistryBase op_registry_;
};

class LibtorchEngine : public EngineBase,
                       public std::enable_shared_from_this<LibtorchEngine> {
 public:
  LibtorchEngine();
  ~LibtorchEngine();
  // DEFAULT_CLASS_MEMBER(LibtorchEngine)
  STATIC_GET_INSTANCE(LibtorchEngine)
  DISABLE_COPY_AND_ASSIGN(LibtorchEngine)
};
