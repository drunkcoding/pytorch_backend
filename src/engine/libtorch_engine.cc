#include "libtorch_engine.h"

#include "backend_engine.h"
#include "backend_op.h"
// #include "dataflow/memory_controller.h"
// #include "event/eventloop_libevent.h"
#include "libtorch_op.h"

// LibtorchEngineHandle::LibtorchEngineHandle(muduo::net::EventLoop* loop)
//     : loop_(loop)
// {
//   LOG_TRITON_VERBOSE("LibtorchEngineHandle::LibtorchEngineHandle");
// }

// void
// LibtorchEngineHandle::ThreadInit(muduo::net::EventLoop* loop)
// {
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<LibtorchEngineHandle*>(loop_handle);
//   handle->RegisterService();
//   // LOG_TRITON_VERBOSE("LibtorchEngineHandle::ThreadInit");
// }

// void
// LibtorchEngineHandle::RegisterService()
// {
//   op_manager_ = std::make_shared<LibtorchOpManager>(loop_);
// }

// void
// LibtorchEngineHandle::DispachRequest(const LibtorchRequestPtr& request)
// {
//   LOG_TRITON_VERBOSE("LibtorchEngineHandle::DispachRequest");
//   // auto* handle =
//   //     reinterpret_cast<LibtorchEngineHandle*>(loop_->GetLoopHandle());
//   // auto libtorch_request =
//   // std::static_pointer_cast<LibtorchOpRequest>(request);
//   LOG_TRITON_VERBOSE((std::string("LibtorchEngineHandle::DispachRequest ") +
//                       LibtorchOpTypeToString(request->op_type))
//                          .c_str());

//   switch (request->op_type) {
//     case LibtorchOpType::kExecute:
//       op_manager_->ExecuteModel(request);
//       break;
//     case LibtorchOpType::kPrefetch:
//       op_manager_->PrefetchModel(request);
//       break;
//     default:
//       break;
//   }
// }

// LibtorchEngineImpl::LibtorchEngineImpl()
//     : EngineImplBase(
//           CREATE_INS(LibtorchEngineHandle), INIT_INS(LibtorchEngineHandle),
//           "LibtorchEngineImpl")
// {
// }


LibtorchEngine::LibtorchEngine() : is_init_(false)
{
  // loop_ = new muduo::net::EventLoop(
  //     CREATE_INS(LibtorchEngineHandle), INIT_INS(LibtorchEngineHandle),
  //     "LibtorchEngine");
  // loop_->Start();

  // loop_ =
  //     new EventLoopLibevent("LibtorchEngine",
  //     CREATE_INS(LibtorchEngineHandle));
  // loop_->Start();
}

void
LibtorchEngine::Init()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (is_init_) {
    return;
  }
  loop_thread_ = std::make_shared<muduo::net::EventLoopThread>(
      INIT_INS(LibtorchEngine), "LibtorchEngine");
  loop_ = loop_thread_->startLoop();
  op_manager_ = std::make_shared<LibtorchOpManager>(loop_);
  is_init_ = true;
}

LibtorchEngine::~LibtorchEngine()
{
  // delete loop_;
}


void
LibtorchEngine::ProcessRequestInLoop(const LibtorchRequestPtr& request)
{
  // auto* handle =
  //     reinterpret_cast<LibtorchEngineHandle*>(loop_->GetLoopHandle());
  // LOG_TRITON_VERBOSE(("LibtorchEngine::ProcessRequest " +
  //              std::to_string((int)request->op_type) + " " +
  //              std::to_string(handle->GetWeight()) + " " +
  //              std::to_string(handle->GetRefs()))
  //                 .c_str());
  DispachRequest(request);
}

void
LibtorchEngine::ProcessRequest(const LibtorchRequestPtr& request)
{
  loop_->runInLoop(
      SELF_BIND_ARGS(LibtorchEngine, ProcessRequestInLoop, request));
}

void
LibtorchEngine::DispachRequest(const LibtorchRequestPtr& request)
{
  // LOG_TRITON_VERBOSE("LibtorchEngineHandle::DispachRequest");
  // LOG_TRITON_VERBOSE((std::string("LibtorchEngineHandle::DispachRequest ") +
  //                     LibtorchOpTypeToString(request->op_type))
  //                        .c_str());
  bool ret = false;
  switch (request->op_type) {
    case LibtorchOpType::kExecute:
      ret = op_manager_->ExecuteModel(request);
      break;
    // case LibtorchOpType::kPrefetch:
    //   op_manager_->PrefetchModel(request);
    //   break;
    default:
      break;
  }

  if (!ret) {
    // LOG_TRITON_ERROR(
    //     "LibtorchEngine::DispachRequest execute model failed, do again");
    loop_->queueInLoop(SELF_BIND_ARGS(LibtorchEngine, ProcessRequestInLoop, request));
  }
}

// void
// LibtorchEngine::RequestInLoop(const RequestPtr& request)
// {
//   LOG_TRITON_VERBOSE("RequestInLoop");
//   auto* handle =
//       reinterpret_cast<LibtorchEngineHandle*>(loop_->GetLoopHandle());
//   LOG_TRITON_VERBOSE("RequestInLoop");
//   handle->DispachRequest(request);
//   LOG_TRITON_VERBOSE("RequestInLoop");
//   // auto task = std::bind(&LibtorchEngineHandle::DispachRequest, handle,
//   // request); loop_->RunInLoop(task);
// }
