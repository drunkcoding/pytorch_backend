#pragma once

#include <unordered_map>

#include "backend_op.h"
// #include "engine_base.h"
#include "loop_handle.h"
#include "muduo/base/noncopyable.h"
#include "muduo/net/EventLoop.h"
#include "muduo/net/EventLoopThread.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"

// class BackendEngineHandle : public LoopHandle {
//  public:
//   DEFAULT_LOOPHANDLE_MEMBER(BackendEngineHandle)

//  public:
//   void DispachRequest(const BackendRequestPtr& request);
//   void RegisterService();

//   //  private:
//   //   OpRegistryBase op_registry_;
//  private:
//   std::shared_ptr<BackendOpManager> op_manager_;
//   muduo::net::EventLoop* loop_;
// };

class BackendEngine : public std::enable_shared_from_this<BackendEngine>,
                      public muduo::noncopyable {
 public:
  BackendEngine();
  ~BackendEngine();
  DISABLE_COPY_AND_ASSIGN(BackendEngine)

  void ProcessRequest(const BackendRequestPtr& request);

  static void ThreadInit(muduo::net::EventLoop* loop) {}

 private:
  void ProcessRequestInLoop(const BackendRequestPtr& request);
  void DispachRequest(const BackendRequestPtr& request);

 private:
  std::shared_ptr<muduo::net::EventLoopThread> loop_thread_;
  muduo::net::EventLoop* loop_;
  std::shared_ptr<BackendOpManager> op_manager_;
};
typedef std::shared_ptr<BackendEngine> BackendEnginePtr;

class BackendEngineRegistry : public muduo::noncopyable {
 public:
  BackendEngineRegistry();
  ~BackendEngineRegistry();
  DISABLE_COPY_AND_ASSIGN(BackendEngineRegistry)
  STATIC_GET_INSTANCE(BackendEngineRegistry)

  void RegisterBackendEngine(const std::size_t key, const BackendEnginePtr& engine);
  void UnregisterBackendEngine(const std::size_t key, const BackendEnginePtr& engine);

  BackendEnginePtr GetBackendEngine(const std::size_t key);

 private:
  std::unordered_map<std::size_t, BackendEnginePtr> engine_map_;
};
// static const std::unique_ptr<BackendEngineRegistry> kBackendReg = std::make_unique<BackendEngineRegistry>();
