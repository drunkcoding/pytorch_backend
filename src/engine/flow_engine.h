#pragma once

#include "flow_op.h"
#include "muduo/net/EventLoop.h"
#include "muduo/net/EventLoopThread.h"

// class FlowEngineHandle : public uevent::LoopHandle {
//  public:
//   DEFAULT_LOOPHANDLE_MEMBER(FlowEngineHandle)

//  public:
//   void DispachRequest(const FlowOpRequestPtr& request);
//   void RegisterService();

//  private:
//   std::shared_ptr<FlowOpManager> op_manager_;
//   muduo::net::EventLoop* loop_;
// };

class FlowEngine : public std::enable_shared_from_this<FlowEngine> {
 public:
  // STATIC_GET_INSTANCE(FlowEngine)
  // DISABLE_COPY_AND_ASSIGN(FlowEngine)
  FlowEngine();
  ~FlowEngine();

  static void ThreadInit(muduo::net::EventLoop* loop) {}

  void ProcessRequest(const FlowOpRequestPtr& request);
  void ProcessRequestInLoop(const FlowOpRequestPtr& request);

 private:
  void DispachRequest(const FlowOpRequestPtr& request);

 private:
  std::shared_ptr<muduo::net::EventLoopThread> loop_thread_;
  muduo::net::EventLoop* loop_;
  std::shared_ptr<FlowOpManager> op_manager_;
};
typedef std::shared_ptr<FlowEngine> FlowEnginePtr;
