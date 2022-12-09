#include "flow_engine.h"

#include "engine/libtorch_engine.h"
#include "engine/libtorch_op.h"
#include "flow_op.h"
#include "utils/class_utils.h"


// FlowEngineHandle::FlowEngineHandle(muduo::net::EventLoop* loop) : loop_(loop)
// {
//   LOG_TRITON_VERBOSE("FlowEngineHandle::FlowEngineHandle");
// }

// void
// FlowEngineHandle::ThreadInit(muduo::net::EventLoop* loop)
// {
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<FlowEngineHandle*>(loop_handle);
//   handle->RegisterService();
//   LOG_TRITON_VERBOSE("FlowEngineHandle::ThreadInit");
// }

// void
// FlowEngineHandle::RegisterService()
// {
//   op_manager_ = std::make_shared<FlowOpManager>(loop_);
// }


// void
// FlowEngineHandle::DispachRequest(const FlowOpRequestPtr& request)
// {
//   // auto* handle =
//   reinterpret_cast<FlowEngineHandle*>(loop_->GetLoopHandle());
//   // auto flow_request = std::static_pointer_cast<FlowOpRequest>(request);
//   switch (request->op_type) {
//     case FlowOpType::kRecord:
//       op_manager_->RecordNode(request);
//       break;
//     case FlowOpType::kPrefetch:
//       op_manager_->PrefetchNode(request);
//       break;
//     default:
//       break;
//   }
// }

/* ================== Engine ================== */

FlowEngine::FlowEngine()
{
  loop_thread_ = std::make_shared<muduo::net::EventLoopThread>(
      INIT_INS(FlowEngine), "FlowEngine");
  loop_ = loop_thread_->startLoop();
  // loop_ = new muduo::net::EventLoop(
  //     CREATE_INS(FlowEngineHandle), INIT_INS(FlowEngineHandle),
  //     "FlowEngine");
  // loop_->Start();
  op_manager_ = std::make_shared<FlowOpManager>(loop_);
}

FlowEngine::~FlowEngine()
{
  // loop_thread_->StopLoop();
  // delete loop_;
}

void
FlowEngine::ProcessRequest(const FlowOpRequestPtr& request)
{
  LOG_TRITON_VERBOSE("FlowEngine::ProcessRequest");
  auto task = SELF_BIND_ARGS(FlowEngine, ProcessRequestInLoop, request);
  loop_->runInLoop(task);
}

void
FlowEngine::ProcessRequestInLoop(const FlowOpRequestPtr& request)
{
  LOG_TRITON_VERBOSE("FlowEngine::ProcessRequestInLoop");
  // auto* handle = reinterpret_cast<FlowEngineHandle*>(loop_->GetLoopHandle());
  // LOG_TRITON_VERBOSE(("FlowEngine::ProcessRequestInLoop " +
  //              std::to_string((int)request->op_type) + " " +
  //              std::to_string(handle->GetWeight()) + " " +
  //              std::to_string(handle->GetRefs()))
  //                 .c_str());
  DispachRequest(request);
}

void
FlowEngine::DispachRequest(const FlowOpRequestPtr& request)
{
  // auto* handle = reinterpret_cast<FlowEngineHandle*>(loop_->GetLoopHandle());
  // auto flow_request = std::static_pointer_cast<FlowOpRequest>(request);
  switch (request->op_type) {
    case FlowOpType::kRecord:
      op_manager_->RecordNode(request);
      break;
    case FlowOpType::kPrefetch:
      op_manager_->PrefetchNode(request);
      break;
    default:
      break;
  }
}


// void
// FlowEngine::RequestInLoop(const RequestPtr& request)
// {
//   auto* handle = reinterpret_cast<FlowEngineHandle*>(loop_->GetLoopHandle());
//   auto task = std::bind(&FlowEngineHandle::DispachRequest, handle, request);
//   loop_->RunInLoop(task);
// }