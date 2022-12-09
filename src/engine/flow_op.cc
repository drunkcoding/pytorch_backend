#include "flow_op.h"

#include "dataflow/counter_flow_controller.h"
#include "dataflow/deepspeed_flow_controller.h"
#include "dataflow/prefetch_flow_controller.h"
#include "engine_ctx.h"
#include "libtorch_op.h"

FlowOpManager::FlowOpManager(muduo::net::EventLoop* loop) : OpBase(loop) {}

FlowOpManager::~FlowOpManager() {}

void
FlowOpManager::RecordNode(const FlowOpRequestPtr& request)
{
  auto record_request = std::static_pointer_cast<FlowRecordRequest>(request);
  FLOW_CONTROLLER->RecordNode(
      record_request->input_id, record_request->node,
      record_request->node_meta);

  // auto prefetch_request = std::make_shared<FlowPrefetchRequest>();
  // prefetch_request->node = record_request->node;
  // prefetch_request->engine = record_request->engine;
  // ENGINE_CTX->ProcessRequest(prefetch_request);
}

void
FlowOpManager::PrefetchNode(const FlowOpRequestPtr& request)
{
  auto prefetch_request =
      std::static_pointer_cast<FlowPrefetchRequest>(request);
  LOG_TRITON_VERBOSE("FlowOpManager::PrefetchNode");

  auto node_vec = FLOW_CONTROLLER->PrefetchNode(prefetch_request->node);
  for (auto& node : node_vec) {
    auto libtorch_request = std::make_shared<LibtorchPrefetchRequest>();
    libtorch_request->node = node.first;
    libtorch_request->engine = prefetch_request->engine;
    ENGINE_CTX->ProcessLibtorchRequest(libtorch_request);
  }

  // for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
  //   LOG_TRITON_VERBOSE((std::string("prob_vec ") + std::to_string(i) +
  //                std::string(": ") + std::to_string(prob_vec[i].second))
  //                   .c_str());
  //   auto libtorch_request = std::make_shared<LibtorchPrefetchRequest>();
  //   libtorch_request->node = prob_vec[i].first;
  //   libtorch_request->engine = prefetch_request->engine;
  //   ENGINE_CTX->ProcessLibtorchRequest(libtorch_request);
  // }
}