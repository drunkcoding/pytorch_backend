#pragma once

#include <functional>
#include <memory>

#include "utils/class_utils.h"
#include "event/eventloop_thread.h"
#include "base/noncopyable.h"
#include "op_base.h"

class EngineHandleBase : public LoopHandle {
 public:
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
//   EventLoop* loop_;
// };

class EngineBase : public std::enable_shared_from_this<EngineBase>,
                   public noncopyable {
 public:
  EngineBase() = default;
  virtual ~EngineBase() = default;
  // DISABLE_COPY_AND_ASSIGN(EngineBase)
  EventLoop* GetLoop() { return loop_; }

 public:
  virtual void RequestInLoop(const RequestPtr& request)
  {
    auto* handle = reinterpret_cast<EngineHandleBase*>(loop_->GetLoopHandle());
    auto task = std::bind(&EngineHandleBase::DispachRequest, handle, request);
    loop_->RunInLoop(task);
  }

 protected:
  std::shared_ptr<EventLoopThread> loop_thread_;
  EventLoop* loop_;
};