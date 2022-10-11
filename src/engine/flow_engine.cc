#include "flow_engine.h"

#include "flow_op.h"
#include "utils/class_utils.h"
#include "engine/libtorch_engine.h"
#include "engine/libtorch_op.h"

void
FlowEngineHandle::RegisterService()
{
}


void
FlowEngineHandle::DispachRequest(const RequestPtr& request)
{
  auto* handle = reinterpret_cast<FlowEngineHandle*>(loop_->GetLoopHandle());
  auto flow_request = std::static_pointer_cast<FlowOpRequest>(request);
  switch (flow_request->op_type) {
    case FlowOpType::kRecord:
      loop_->RunInLoop(
          SELF_BIND_ARGS(FlowEngineHandle, RecordNode, flow_request));
      break;
    case FlowOpType::kPrefetch:
      loop_->RunInLoop(
          SELF_BIND_ARGS(FlowEngineHandle, PrefetchNode, flow_request));
      break;
    default:
      break;
  }
}

void
FlowEngineHandle::RecordNode(const FlowOpRequestPtr& request)
{
  auto record_request = std::static_pointer_cast<FlowRecordRequest>(request);
  GET_INSTANCE(FlowController)
      ->RecordNodeFlow(
          record_request->request_id, record_request->node,
          record_request->node_meta);

  auto prefetch_request = std::make_shared<FlowPrefetchRequest>();
  prefetch_request->node = record_request->node;
  prefetch_request->handle = record_request->handle;
  loop_->QueueInLoop(SELF_BIND_ARGS(FlowEngineHandle, PrefetchNode, request));
}
void
FlowEngineHandle::PrefetchNode(const FlowOpRequestPtr& request)
{
  auto prefetch_request =
      std::static_pointer_cast<FlowPrefetchRequest>(request);
  auto prob_vec =
      GET_INSTANCE(FlowController)->GetTreeProbability(prefetch_request->node);


  for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
    auto libtorch_request = std::make_shared<LibtorchPrefetchRequest>();
    libtorch_request->node = prob_vec[i].first;
    libtorch_request->handle = prefetch_request->handle;
    GET_INSTANCE(LibtorchEngine)->RequestInLoop(libtorch_request);
  }
}

/* ================== Engine ================== */

FlowEngine::FlowEngine()
{
  loop_thread_ = std::make_shared<EventLoopThread>(
      CREATE_INS(FlowEngineHandle), INIT_INS(FlowEngineHandle), "FlowEngine");
  loop_ = loop_thread_->StartLoop();
}

FlowEngine::~FlowEngine() {}