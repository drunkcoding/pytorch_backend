#include "backend_engine.h"

// #include "backend_op_execute.h"
// #include "event/eventloop_libevent.h"
#include "libtorch_engine.h"

// BackendEngineHandle::BackendEngineHandle(muduo::net::EventLoop* loop)
//     : loop_(loop)
// {
//   LOG_TRITON_VERBOSE("BackendEngineHandle::BackendEngineHandle");
// }

// void
// BackendEngineHandle::ThreadInit(muduo::net::EventLoop* loop)
// {
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<BackendEngineHandle*>(loop_handle);
//   handle->RegisterService();
//   LOG_TRITON_VERBOSE("BackendEngineHandle::ThreadInit");
// }

// void
// BackendEngineHandle::RegisterService()
// {
//   op_manager_ = std::make_shared<BackendOpManager>(loop_);
//   LOG_TRITON_VERBOSE("BackendEngineHandle::RegisterService");
// }


// void
// BackendEngineHandle::DispachRequest(const BackendRequestPtr& request)
// {
//   // auto backend_request =
//   std::static_pointer_cast<BackendOpRequest>(request);
//   LOG_TRITON_VERBOSE("BackendEngineHandle::DispachRequest");
//   switch (request->op_type) {
//     case BackendOpType::kLoad:
//       op_manager_->LoadModel(request);
//       break;
//     case BackendOpType::kUnload:
//       op_manager_->UnloadModel(request);
//       break;
//     case BackendOpType::kExecute:
//       op_manager_->ExecuteModel(request);
//       break;
//     default:
//       break;
//   }
// }


// BackendEngineImpl::BackendEngineImpl()
//     : EngineImplBase(
//           CREATE_INS(BackendEngineHandle), INIT_INS(BackendEngineHandle),
//           "BackendEngineImpl")
// {
// }

/* ================== Engine ================== */

BackendEngine::BackendEngine()
{
  // loop_ = new muduo::net::EventLoop(
  //     CREATE_INS(BackendEngineHandle), INIT_INS(BackendEngineHandle),
  //     "BackendEngine");
  // loop_->Start();
  loop_thread_ = std::make_shared<muduo::net::EventLoopThread>(
      INIT_INS(BackendEngine), "BackendEngine");
  LOG_TRITON_VERBOSE("BackendEngine::BackendEngine");
  loop_ = loop_thread_->startLoop();
  LOG_TRITON_VERBOSE("BackendEngine::BackendEngine");
  op_manager_ = std::make_shared<BackendOpManager>(loop_);
}

BackendEngine::~BackendEngine() {}

// void
// BackendEngine::RequestInLoop(const RequestPtr& request)
// {
//   auto* handle =
//   reinterpret_cast<BackendEngineHandle*>(loop_->GetLoopHandle());
//   handle->DispachRequest(request);
//   // auto task = std::bind(&BackendEngineHandle::DispachRequest, handle,
//   request);
//   // loop_->RunInLoop(task);
// }

void
BackendEngine::ProcessRequest(const BackendRequestPtr& request)
{
  // auto* handle =
  // reinterpret_cast<BackendEngineHandle*>(loop_->GetLoopHandle());
  // LOG_TRITON_VERBOSE(("BackendEngine::ProcessRequest " +
  //              std::to_string((int)request->op_type) + " " +
  //              std::to_string(handle->GetWeight()) + " " +
  //              std::to_string(handle->GetRefs()))
  //                 .c_str());
  // loop_->queueInLoop(
  //     std::bind(&BackendEngineHandle::DispachRequest, handle, request));
  // // handle->DispachRequest(request);
  loop_->runInLoop(
      SELF_BIND_ARGS(BackendEngine, ProcessRequestInLoop, request));
}

void
BackendEngine::ProcessRequestInLoop(const BackendRequestPtr& request)
{
  // auto* handle =
  // reinterpret_cast<BackendEngineHandle*>(loop_->GetLoopHandle());
  // LOG_TRITON_VERBOSE(("BackendEngine::ProcessRequestInLoop " +
  //                     std::to_string((int)request->op_type) + " " +
  //                     std::to_string(handle->GetWeight()) + " " +
  //                     std::to_string(handle->GetRefs()))
  //                        .c_str());
  DispachRequest(request);
}

void
BackendEngine::DispachRequest(const BackendRequestPtr& request)
{
  // auto backend_request = std::static_pointer_cast<BackendOpRequest>(request);
  LOG_TRITON_VERBOSE("BackendEngineHandle::DispachRequest");
  switch (request->op_type) {
    case BackendOpType::kLoad:
      op_manager_->LoadModel(request);
      break;
    case BackendOpType::kUnload:
      op_manager_->UnloadModel(request);
      break;
    case BackendOpType::kExecute:
      op_manager_->ExecuteModel(request);
      break;
    default:
      break;
  }
}
