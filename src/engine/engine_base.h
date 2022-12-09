#pragma once

#include <functional>
#include <memory>

#include "muduo/base/noncopyable.h"
#include "muduo/net/EventLoop.h"
#include "op_base.h"
#include "utils/class_utils.h"
#include "utils/log_utils.h"

class EngineHandleBase : public LoopHandle {
 public:
  explicit EngineHandleBase(muduo::net::EventLoop* loop) { loop_ = loop; }
  virtual void DispachRequest(const RequestPtr& request) = 0;

 protected:
  virtual void RegisterService() = 0;

 protected:
  OpRegistryBase op_registry_;
};

// class EngineImplBase : public std::enable_shared_from_this<EngineImplBase> {
//  public:
//   explicit EngineImplBase(
//       const CreateLoopHandleCb& cb1 = CreateLoopHandleCb(),
//       const ThreadInitCb& cb2 = ThreadInitCb(),
//       const std::string& name = std::string())
//   {
//     loop_thread_ = std::make_shared<EventLoopThread>(cb1, cb2, name);
//     loop_ = loop_thread_->StartLoop();
//   }
//   virtual ~EngineImplBase() = default;

//  public:
//   virtual void RequestInLoop(const RequestPtr& request)
//   {
//     auto task = SELF_BIND_ARGS(EngineImplBase, DispachRequest, request);
//     loop_->RunInLoop(task);
//   }

//  protected:
//   virtual void DispachRequest(const RequestPtr& request) = 0;

//  protected:
//   std::shared_ptr<EventLoopThread> loop_thread_;
//   muduo::net::EventLoop* loop_;
// };

class EngineBase : public std::enable_shared_from_this<EngineBase>,
                   public muduo::noncopyable {
 public:
  EngineBase() = default;
  virtual ~EngineBase() = default;
  // DISABLE_COPY_AND_ASSIGN(EngineBase)
  muduo::net::EventLoop* GetLoop() { return loop_; }

 public:
  virtual void RequestInLoop(const RequestPtr& request) = 0;
  // {
  //   LOG_TRITON_VERBOSE("EngineBase::RequestInLoop");
  //   auto* handle =
  //   reinterpret_cast<EngineHandleBase*>(loop_->GetLoopHandle()); auto task =
  //   std::bind(&EngineHandleBase::DispachRequest, handle, request);
  //   loop_->RunInLoop(task);
  //   LOG_TRITON_VERBOSE("EngineBase::RequestInLoop");
  // }

  // LoopHandle* GetHandle() { return loop_->GetLoopHandle(); }

 protected:
  // std::shared_ptr<EventLoopThread> loop_thread_;
  muduo::net::EventLoop* loop_;
};