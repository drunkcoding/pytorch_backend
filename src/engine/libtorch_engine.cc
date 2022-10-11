#include "libtorch_engine.h"

#include "backend_engine.h"
#include "backend_op.h"
#include "dataflow/memory_controller.h"
#include "libtorch_op.h"

void
LibtorchEngineHandle::RegisterService()
{
  // op_registry_.RegisterOp(
  //     LibtorchOpTypeToString(LibtorchOpType::kLoad),
  //     CREATE_INS(LibtorchOpLoad));
  // op_registry_.RegisterOp(
  //     LibtorchOpTypeToString(LibtorchOpType::kUnload),
  //     CREATE_INS(LibtorchOpUnload));
  // op_registry_.RegisterOp(
  //     LibtorchOpTypeToString(LibtorchOpType::kExecute),
  //     CREATE_INS(LibtorchOpExecute));
}


void
LibtorchEngineHandle::DispachRequest(const RequestPtr& request)
{
  // auto* handle =
  //     reinterpret_cast<LibtorchEngineHandle*>(loop_->GetLoopHandle());
  auto libtorch_request = std::static_pointer_cast<LibtorchOpRequest>(request);
  switch (libtorch_request->op_type) {
    case LibtorchOpType::kExecute:
      loop_->RunInLoop(
          SELF_BIND_ARGS(LibtorchEngineHandle, ExecuteModel, libtorch_request));
      break;
    case LibtorchOpType::kPrefetch:
      loop_->RunInLoop(SELF_BIND_ARGS(
          LibtorchEngineHandle, PrefetchModel, libtorch_request));
      break;
    default:
      break;
  }
}

void
LibtorchEngineHandle::PrefetchModel(const LibtorchRequestPtr& request)
{
  auto prefetch_request =
      std::static_pointer_cast<LibtorchPrefetchRequest>(request);
  // auto prob_vec =
  // GET_INSTANCE(LibtorchEngine)->GetTreeProbability(prefetch_request->node);
  // for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
  //   prefetch_key_set.insert(prob_vec[i].first);
  // }

  EvictionCandidates eviction_candidates;
  bool is_evictable = false;
  // Device target_device = DISK_DEVICE;
  std::tie(eviction_candidates, is_evictable, prefetch_request->target_device) =
      GET_INSTANCE(MemoryController)
          ->AllocMemory(
              prefetch_request->node, prefetch_request->target_device);

  std::atomic_uint64_t wait_count{eviction_candidates.size()};
  if (wait_count > 0) {
    for (auto& eviction_candidate : eviction_candidates) {
      RunUnloadInBackend(request, eviction_candidate, CPU_DEVICE, &wait_count);
    }
  } else {
    DispatchToBackend(prefetch_request);
  }
}

void
LibtorchEngineHandle::ExecuteModel(const LibtorchRequestPtr& request)
{
  auto exec_request = std::static_pointer_cast<LibtorchExecuteRequest>(request);
  // auto* caller_loop = exec_request->handle->GetLoop();

  EvictionCandidates eviction_candidates;
  bool is_evictable = false;
  // Device target_device = DISK_DEVICE;
  std::tie(eviction_candidates, is_evictable, exec_request->target_device) =
      GET_INSTANCE(MemoryController)
          ->AllocMemory(
              exec_request->node,
              DEFAULT_CUDA_DEVICE);  // memory allocation here
  std::atomic_uint64_t wait_count{eviction_candidates.size()};

  if (wait_count > 0) {
    for (auto& eviction_candidate : eviction_candidates) {
      RunUnloadInBackend(request, eviction_candidate, CPU_DEVICE, &wait_count);
    }
  } else {
    DispatchToBackend(exec_request);
  }
}

void
LibtorchEngineHandle::RunUnloadInBackend(
    const LibtorchRequestPtr& request, const DAGNodePtr& node,
    const Device& device, std::atomic_uint64_t* wait_count)
{
  auto backend_unload_request = std::make_shared<BackendUnloadRequest>();
  backend_unload_request->node = node;
  backend_unload_request->handle = loop_->GetLoopHandle();
  backend_unload_request->target_device = device;
  backend_unload_request->cb = SELF_BIND_ARGS(
      LibtorchEngineHandle, EntryWaitModelUnloadInLoop, std::placeholders::_1,
      request, wait_count);

  node->SetMemoryType(MemoryType::kEvict);
  request->handle->GetLoop()->RunInLoop(std::bind(
      &BackendEngineHandle::DispachRequest,
      reinterpret_cast<BackendEngineHandle*>(backend_unload_request->handle),
      backend_unload_request));
}

void
LibtorchEngineHandle::RunLoadInBackend(
    EventLoop* const loop, const DAGNodePtr& node, const Device& device)
{
  auto backend_load_request = std::make_shared<BackendLoadRequest>();
  backend_load_request->node = node;
  backend_load_request->target_device = device;
  loop->RunInLoop(std::bind(
      &BackendEngineHandle::DispachRequest,
      reinterpret_cast<BackendEngineHandle*>(loop->GetLoopHandle()),
      backend_load_request));
}

void
LibtorchEngineHandle::DispatchToBackend(const LibtorchRequestPtr& request)
{
  if (request->op_type == LibtorchOpType::kPrefetch) {
    auto prefetch_request =
        std::static_pointer_cast<LibtorchPrefetchRequest>(request);
    RunLoadInBackend(
        prefetch_request->handle->GetLoop(), prefetch_request->node,
        prefetch_request->target_device);
    return;
  }

  if (request->op_type == LibtorchOpType::kExecute) {
    auto exec_request =
        std::static_pointer_cast<LibtorchExecuteRequest>(request);
    RunLoadInBackend(
        exec_request->handle->GetLoop(), exec_request->node,
        exec_request->target_device);

    auto backend_execute_request = std::make_shared<BackendExecuteRequest>();
    backend_execute_request->op_type = BackendOpType::kExecute;
    backend_execute_request->process_requests_cb =
        exec_request->process_requests_cb;
    backend_execute_request->mutex = exec_request->mutex;
    backend_execute_request->cv = exec_request->cv;

    exec_request->handle->GetLoop()->RunInLoop(std::bind(
        &BackendEngineHandle::DispachRequest,
        reinterpret_cast<BackendEngineHandle*>(exec_request->handle),
        backend_execute_request));
  }
}

void
LibtorchEngineHandle::EntryWaitModelUnloadInLoop(
    const BackendResponsePtr& response, const LibtorchRequestPtr& request,
    std::atomic_uint64_t* wait_count)
{
  loop_->RunInLoop(SELF_BIND_ARGS(
      LibtorchEngineHandle, EntryWaitModelUnload, response, request,
      wait_count));
}

void
LibtorchEngineHandle::EntryWaitModelUnload(
    const BackendResponsePtr& response, const LibtorchRequestPtr& request,
    std::atomic_uint64_t* wait_count)
{
  (*wait_count)--;
  auto unload_response =
      std::static_pointer_cast<BackendUnloadResponse>(response);

  GET_INSTANCE(MemoryController)->FreeMemory(unload_response->node);
  unload_response->node->SetMemoryType(MemoryType::kReady);
  if (*wait_count > 0)
    return;

  // auto exec_request =
  // std::static_pointer_cast<LibtorchExecuteRequest>(request); Process the
  // response until all models are unloaded
  DispatchToBackend(request);
}

// LibtorchEngineImpl::LibtorchEngineImpl()
//     : EngineImplBase(
//           CREATE_INS(LibtorchEngineHandle), INIT_INS(LibtorchEngineHandle),
//           "LibtorchEngineImpl")
// {
// }


LibtorchEngine::LibtorchEngine()
{
  loop_thread_ = std::make_shared<EventLoopThread>(
      CREATE_INS(LibtorchEngineHandle), INIT_INS(LibtorchEngineHandle),
      "LibtorchEngine");
  loop_ = loop_thread_->StartLoop();
}

LibtorchEngine::~LibtorchEngine() {}
