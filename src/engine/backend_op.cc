#include "backend_op.h"

#include "utils/log_utils.h"

BackendOpManager::BackendOpManager(muduo::net::EventLoop* loop) : OpBase(loop)
{
}

BackendOpManager::~BackendOpManager() {}


void
BackendOpManager::ExecuteModel(const BackendRequestPtr& request)
{
  LOG_TRITON_VERBOSE("BackendOpManager::ExecuteModel");
  auto exec_request = std::static_pointer_cast<BackendExecuteRequest>(request);
  exec_request->process_requests_cb();
  // notify triton that the request is finished
  exec_request->mutex->unlock();
  exec_request->cv->notify_all();
  // send response here to switch back flags to ready
  auto exec_response = std::make_shared<BackendExecuteResponse>();
  exec_response->node = exec_request->node;

  exec_request->cb(exec_response);
}

void
BackendOpManager::LoadModel(const BackendRequestPtr& request)
{
  LOG_TRITON_VERBOSE("BackendOpManager::LoadModel");
  auto load_request = std::static_pointer_cast<BackendLoadRequest>(request);
  load_request->node->SetDevice(load_request->target_device);

  auto load_response = std::make_shared<BackendLoadResponse>();
  load_response->node = load_request->node;
  load_request->cb(load_response);
  // no call back to memory management needed. since memory allocation already
  // registered.
}

void
BackendOpManager::UnloadModel(const BackendRequestPtr& request)
{
  LOG_TRITON_VERBOSE("BackendOpManager::UnloadModel");
  auto unload_request = std::static_pointer_cast<BackendUnloadRequest>(request);
  unload_request->node->SetDevice(unload_request->target_device);
  // callback is needed to memory management to unblock load of other models.

  auto unload_response = std::make_shared<BackendUnloadResponse>();
  unload_response->node = unload_request->node;
  unload_response->target_device = unload_request->target_device;


  unload_request->cb(unload_response);
  // auto* loop = unload_request->handle->GetLoop();
  // loop->RunInLoop(std::bind(unload_request->cb, unload_response));
}
