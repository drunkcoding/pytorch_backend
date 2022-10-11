#include "backend_engine.h"

// #include "backend_op_execute.h"
#include "libtorch_engine.h"

void
BackendEngineHandle::RegisterService()
{
  //   op_registry_.RegisterOp(
  //       BackendOpTypeToString(BackendOpType::kLoad),
  //       CREATE_INS(BackendOpLoad));
  // op_registry_.RegisterOp(
  //     BackendOpTypeToString(BackendOpType::kUnload),
  //     CREATE_INS(BackendOpUnload));
  //   op_registry_.RegisterOp(
  //       BackendOpTypeToString(BackendOpType::kExecute),
  //       CREATE_INS(BackendOpExecute));
}


void
BackendEngineHandle::DispachRequest(const RequestPtr& request)
{
  auto* handle = reinterpret_cast<BackendEngineHandle*>(loop_->GetLoopHandle());
  auto backend_request = std::static_pointer_cast<BackendOpRequest>(request);
  switch (backend_request->op_type) {
    case BackendOpType::kLoad:
      loop_->RunInLoop(
          SELF_BIND_ARGS(BackendEngineHandle, LoadModel, backend_request));
      break;
    case BackendOpType::kUnload:
      loop_->RunInLoop(
          SELF_BIND_ARGS(BackendEngineHandle, UnloadModel, backend_request));
      break;
    case BackendOpType::kExecute:
      loop_->RunInLoop(
          SELF_BIND_ARGS(BackendEngineHandle, ExecuteModel, backend_request));
      break;
    default:
      break;
  }
//   auto op_creator =
//       op_registry_.GetOp(BackendOpTypeToString(backend_request->op_type));
//   auto op = op_creator(loop_);
//   op->SendRequest(backend_request);
}

void
BackendEngineHandle::ExecuteModel(const BackendRequestPtr& request)
{
  auto exec_request = std::static_pointer_cast<BackendExecuteRequest>(request);
  exec_request->process_requests_cb();
  // notify triton that the request is finished
  exec_request->mutex->unlock();
  exec_request->cv->notify_all();
  // no call back to memory management needed.
}

void
BackendEngineHandle::LoadModel(const BackendRequestPtr& request)
{
  auto load_request = std::static_pointer_cast<BackendLoadRequest>(request);
  load_request->node->SetDevice(load_request->target_device);
  // no call back to memory management needed. since memory allocation already
  // registered.
}

void
BackendEngineHandle::UnloadModel(const BackendRequestPtr& request)
{
  auto unload_request =
      std::static_pointer_cast<BackendUnloadRequest>(request);
  unload_request->node->SetDevice(unload_request->target_device);
  // callback is needed to memory management to unblock load of other models.

  auto unload_response = std::make_shared<BackendUnloadResponse>();
  unload_response->node = unload_request->node;
  unload_response->target_device = unload_request->target_device;

  
  unload_request->cb(unload_response);
  // auto* loop = unload_request->handle->GetLoop();
  // loop->RunInLoop(std::bind(unload_request->cb, unload_response));
}

// BackendEngineImpl::BackendEngineImpl()
//     : EngineImplBase(
//           CREATE_INS(BackendEngineHandle), INIT_INS(BackendEngineHandle),
//           "BackendEngineImpl")
// {
// }

/* ================== Engine ================== */

BackendEngine::BackendEngine()
{
  loop_thread_ = std::make_shared<EventLoopThread>(
      CREATE_INS(BackendEngineHandle), INIT_INS(BackendEngineHandle),
      "BackendEngine");
  loop_ = loop_thread_->StartLoop();
}

BackendEngine::~BackendEngine() {}
