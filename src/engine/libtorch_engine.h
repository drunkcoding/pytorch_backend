#pragma once

#include <torch/script.h>

#include <atomic>
#include <unordered_map>

#include "backend_op.h"
#include "libtorch_op.h"
#include "loop_handle.h"
#include "muduo/base/noncopyable.h"
#include "muduo/net/EventLoopThread.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"

// class LibtorchEngineHandle : public LoopHandle {
//  public:
//   DEFAULT_LOOPHANDLE_MEMBER(LibtorchEngineHandle)

//  public:
//   void DispachRequest(const LibtorchRequestPtr& request);
//   void RegisterService();

//  private:
//   std::shared_ptr<LibtorchOpManager> op_manager_;
//   muduo::net::EventLoop* loop_;
// };

class LibtorchEngine : public std::enable_shared_from_this<LibtorchEngine> {
 public:
  LibtorchEngine();
  ~LibtorchEngine();
  // DEFAULT_CLASS_MEMBER(LibtorchEngine)
  // STATIC_GET_INSTANCE(LibtorchEngine)
  // DISABLE_COPY_AND_ASSIGN(LibtorchEngine)

  static void ThreadInit(muduo::net::EventLoop* loop) {}

  void ProcessRequest(const LibtorchRequestPtr& request);
  void ProcessRequestInLoop(const LibtorchRequestPtr& request);

 private:
  //   void ProcessRequestInLoop(const LibtorchRequestPtr& request);
  void DispachRequest(const LibtorchRequestPtr& request);

 private:
  std::shared_ptr<muduo::net::EventLoopThread> loop_thread_;
  muduo::net::EventLoop* loop_;
  std::shared_ptr<LibtorchOpManager> op_manager_;
};
