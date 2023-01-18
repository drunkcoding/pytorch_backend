#include "libtorch_op.h"

#include "backend_engine.h"
// #include "engine_ctx.h"
#include "dataflow/counter_flow_controller.h"
#include "dataflow/deepspeed_flow_controller.h"
#include "dataflow/prefetch_flow_controller.h"
#include "dataflow/nonfetch_flow_controller.h"

LibtorchOpManager::LibtorchOpManager(muduo::net::EventLoop* loop) : OpBase(loop)
{
}

LibtorchOpManager::~LibtorchOpManager() {}

// void
// LibtorchOpManager::PrefetchModel(const LibtorchRequestPtr& request)
// {
//   auto prefetch_request =
//       std::static_pointer_cast<LibtorchPrefetchRequest>(request);
//   // auto prob_vec =
//   //
//   GET_INSTANCE(LibtorchEngine)->GetTreeProbability(prefetch_request->node);
//   // for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
//   //   prefetch_key_set.insert(prob_vec[i].first);
//   // }

//   // EvictionCandidates eviction_candidates;
//   // bool is_evictable = false;
//   // // Device target_device = DISK_DEVICE;
//   // std::tie(eviction_candidates, is_evictable,
//   // prefetch_request->target_device) =
//   //     GET_INSTANCE(MemoryController)
//   //         ->AllocMemory(
//   //             prefetch_request->node, prefetch_request->target_device);

//   // FLOW_CONTROLLER->PrefetchNode(prefetch_request->node);

//   // std::atomic_uint64_t wait_count{eviction_candidates.size()};
//   // if (wait_count > 0) {
//   //   for (auto& eviction_candidate : eviction_candidates) {
//   //     RunUnloadInBackend(request, eviction_candidate, CPU_DEVICE,
//   //     &wait_count);
//   //   }
//   // } else {
//   //   DispatchToBackend(prefetch_request);
//   // }
// }

bool
LibtorchOpManager::ExecuteModel(const LibtorchRequestPtr& request)
{
  auto exec_request = std::static_pointer_cast<LibtorchExecuteRequest>(request);

  // backend_engine_map_.emplace(exec_request->node->id, exec_request->engine);

  auto node_move_vec = FLOW_CONTROLLER->PrefetchNode(exec_request->node);
  FLOW_CONTROLLER->RecordNode(exec_request->input_id, exec_request->node);

  return true;

  if (exec_request->node->device != DEFAULT_CUDA_DEVICE &&
      node_move_vec.empty() &&
      exec_request->node->memory_type != MemoryType::kEmplacing) {
    // this request wait at the end of the queue
    return false;
  }
  assert(
      exec_request->node->memory_type == MemoryType::kReady ||
      exec_request->node->memory_type == MemoryType::kEmplacing);
  // exec_request->node->memory_type = MemoryType::kLocked;
  FLOW_CONTROLLER->RecordNode(exec_request->input_id, exec_request->node);

  // eviction_candidates is node where device not to CUDA
  NodeMoveVec eviction_candidates;
  // NodeMoveVec prefetch_candidates;
  for (auto& node_move : node_move_vec) {
    if (node_move.second != DEFAULT_CUDA_DEVICE) {
      eviction_candidates.push_back(node_move);
    } else {
      node_prefetch_vec_.push_back(node_move);
    }
  }
  std::atomic_uint64_t wait_count{eviction_candidates.size()};
  LOG_TRITON_VERBOSE(
      ("Wait for " + std::to_string(wait_count) + " models to be evicted")
          .c_str());

  if (wait_count > 0) {
    for (auto& eviction_candidate : eviction_candidates) {
      RunUnloadInBackend(
          request, eviction_candidate.first, eviction_candidate.second,
          &wait_count);
    }
  } else {
    DispatchToBackend(exec_request);
  }

  // auto prefetch_request = std::make_shared<FlowPrefetchRequest>();
  // prefetch_request->node = exec_request->node;
  // prefetch_request->engine = exec_request->engine;
  // LOG_TRITON_VERBOSE(
  //     "LibtorchOpManager::ExecuteModel FlowOpManager::PrefetchNode");
  // ENGINE_CTX->ProcessFlowRequest(prefetch_request);

  // for (auto& prefetch_candidate : prefetch_candidates) {
  //   RunLoadInBackend(
  //       exec_request->engine, prefetch_candidate.first,
  //       prefetch_candidate.second);
  // }
  return true;
}

void
LibtorchOpManager::RunUnloadInBackend(
    const LibtorchRequestPtr& request, const NodePtr& node,
    const Device& device, std::atomic_uint64_t* wait_count)
{
  auto backend_unload_request = std::make_shared<BackendUnloadRequest>();
  backend_unload_request->node = node;
  backend_unload_request->to = device;
  backend_unload_request->from = node->device;
  backend_unload_request->cb = SELF_BIND_ARGS(
      LibtorchOpManager, EntryWaitModelUnload, std::placeholders::_1, request,
      wait_count);

  // node->memory_type = MemoryType::kMoving;
  request->engine->ProcessRequest(backend_unload_request);
}

void
LibtorchOpManager::RunLoadInBackend(
    const BackendEnginePtr& engine, const NodePtr& node, const Device& device)
{
  auto backend_load_request = std::make_shared<BackendLoadRequest>();
  backend_load_request->node = node;
  backend_load_request->to = device;
  backend_load_request->from = node->device;
  backend_load_request->cb = SELF_BIND_ARGS(
      LibtorchOpManager, EntryWaitModelLoad, std::placeholders::_1);
  engine->ProcessRequest(backend_load_request);
}

void
LibtorchOpManager::DispatchToBackend(const LibtorchRequestPtr& request)
{
  if (request->op_type == LibtorchOpType::kPrefetch) {
    auto prefetch_request =
        std::static_pointer_cast<LibtorchPrefetchRequest>(request);
    LOG_TRITON_VERBOSE(("Prefetch model " +
                        prefetch_request->node->GetModelInstanceInfo() +
                        " to " + prefetch_request->target_device.str())
                           .c_str());
    RunLoadInBackend(
        prefetch_request->engine, prefetch_request->node,
        prefetch_request->target_device);
    return;
  }

  if (request->op_type == LibtorchOpType::kExecute) {
    auto exec_request =
        std::static_pointer_cast<LibtorchExecuteRequest>(request);
    // LOG_TRITON_VERBOSE(("Execute model " +
    //                     exec_request->node->GetModelInstanceInfo() + " on " +
    //                     DEFAULT_CUDA_DEVICE.str())
    //                        .c_str());
    if (exec_request->node->device != DEFAULT_CUDA_DEVICE)
      RunLoadInBackend(
          exec_request->engine, exec_request->node, DEFAULT_CUDA_DEVICE);

    auto backend_execute_request = std::make_shared<BackendExecuteRequest>();
    backend_execute_request->op_type = BackendOpType::kExecute;
    backend_execute_request->process_requests_cb =
        exec_request->process_requests_cb;
    backend_execute_request->node = exec_request->node;
    backend_execute_request->mutex = exec_request->mutex;
    backend_execute_request->cv = exec_request->cv;
    backend_execute_request->cb = SELF_BIND_ARGS(
        LibtorchOpManager, EntryWaitModelExecute, std::placeholders::_1);

    exec_request->engine->ProcessRequest(backend_execute_request);
  }

  for (auto& prefetch_candidate : node_prefetch_vec_) {
    auto engine_ptr = GET_INSTANCE(BackendEngineRegistry)->GetBackendEngine(prefetch_candidate.first->id);
    RunLoadInBackend(
        engine_ptr, prefetch_candidate.first, prefetch_candidate.second);
  }
  node_prefetch_vec_.clear();
}

void
LibtorchOpManager::EntryWaitModelExecute(const BackendResponsePtr& response)
{
  if (loop_->isInLoopThread()) {
    auto execute_response =
        std::static_pointer_cast<BackendExecuteResponse>(response);
    // execute_response->node->memory_type = MemoryType::kReady;
  } else {
    loop_->queueInLoop(
        SELF_BIND_ARGS(LibtorchOpManager, EntryWaitModelExecute, response));
  }
}

void
LibtorchOpManager::EntryWaitModelUnload(
    const BackendResponsePtr& response, const LibtorchRequestPtr& request,
    std::atomic_uint64_t* wait_count)
{
  if (loop_->isInLoopThread()) {
    (*wait_count)--;
    auto unload_response =
        std::static_pointer_cast<BackendUnloadResponse>(response);
    // unload_response->node->memory_type = MemoryType::kStandBy;
    FLOW_CONTROLLER->UpdateMemoryManager(
        unload_response->from, unload_response->to,
        unload_response->node->byte_size);
    if (*wait_count > 0)
      return;
  } else {
    loop_->queueInLoop(SELF_BIND_ARGS(
        LibtorchOpManager, EntryWaitModelUnload, response, request,
        wait_count));
  }

  DispatchToBackend(request);
}


void
LibtorchOpManager::EntryWaitImmedientModel(
    const BackendResponsePtr& response, const LibtorchRequestPtr& request)
{
  if (loop_->isInLoopThread()) {
    DispatchToBackend(request);
  } else {
    loop_->queueInLoop(SELF_BIND_ARGS(
        LibtorchOpManager, EntryWaitImmedientModel, response, request));
  }
}

void
LibtorchOpManager::EntryWaitModelLoad(const BackendResponsePtr& response)
{
  if (loop_->isInLoopThread()) {
    auto load_response =
        std::static_pointer_cast<BackendLoadResponse>(response);
    if (load_response->node->memory_type != MemoryType::kReady &&
        load_response->node->memory_type != MemoryType::kLocked)
      load_response->node->memory_type = MemoryType::kReady;
  } else {
    loop_->queueInLoop(
        SELF_BIND_ARGS(LibtorchOpManager, EntryWaitModelLoad, response));
  }
}
