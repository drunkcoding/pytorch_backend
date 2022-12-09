#include "dag_node.h"

#include "engine/backend_engine.h"
#include "engine/backend_op.h"
#include "engine/engine_ctx.h"
#include "engine/flow_engine.h"
#include "engine/flow_op.h"
#include "engine/libtorch_engine.h"
#include "engine/libtorch_op.h"
#include "libtorch_utils.h"
#include "utils/log_utils.h"
// #include "libtorch_flow.h"

/* ============== NodeBackendEnginePlugin ============== */

NodeEnginePlugin::NodeEnginePlugin()
{
  // loop_ = engine_->GetLoop();
  // handle_ = loop_->GetLoopHandle();
  engine_ = std::make_shared<BackendEngine>();
}

void
NodeEnginePlugin::ProcessTritonRequest(
    muduo::net::EventLoop::Functor func, const DAGNodePtr& node)
{
  // exec_finished_ = false;
  // mutex_.lock();

  auto execute_request = std::make_shared<LibtorchExecuteRequest>();
  execute_request->node = node;
  execute_request->engine = engine_;
  execute_request->process_requests_cb = func;
  execute_request->mutex = &mutex_;
  execute_request->cv = &cond_;

  LOG_TRITON_VERBOSE(("NodeBackendEnginePlugin::ProcessTritonRequest: node: " +
                      node->GetModelInstanceInfo() + ", op: " +
                      LibtorchOpTypeToString(execute_request->op_type) +
                      ", device: " + node->GetDevice().str())
                         .c_str());
  ENGINE_CTX->ProcessLibtorchRequest(execute_request);

  // no response needed just wait
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [node] { return node->GetDevice() == DEFAULT_CUDA_DEVICE; });

  LOG_TRITON_VERBOSE(("NodeBackendEnginePlugin::ProcessTritonRequest: node: " +
                      node->GetModelInstanceInfo() + ", op: " +
                      LibtorchOpTypeToString(execute_request->op_type) +
                      ", device: " + node->GetDevice().str() + " done")
                         .c_str());
}

/* ============== NodeFlowEnginePlugin ============== */

// NodeFlowEnginePlugin::NodeFlowEnginePlugin()
// {
//   engine_ = std::make_shared<FlowEngine>();
//   // loop_ = engine_->GetLoop();
//   // handle_ = loop_->GetLoopHandle();
// }

void
NodeEnginePlugin::RecordTritonRequest(
    const DAGNodePtr& node, TRITONBACKEND_Request* const request,
    const torch::jit::IValue& input_tensor, const uint64_t compute_time)
{
  // auto request_id =
  //     std::hash<std::string>{}(triton::backend::GetRequestId(request));
  auto input_size = GetIValueByteSize(input_tensor);

  // Send Request to FlowEngine
  auto flow_request = std::make_shared<FlowRecordRequest>();
  auto input_id = std::make_shared<InputID>();
  input_id->request_id = GetRequestId(request);
  input_id->correlation_id = GetCorrelationId(request);
  flow_request->input_id = input_id;
  flow_request->node = node;
  flow_request->engine = engine_;
  auto node_meta = std::make_shared<NodeMeta>();
  node_meta->node_id = node->GetNodeID();
  node_meta->visit_cnt = 1;
  node_meta->input_size_cnt = input_size;
  node_meta->exec_lat_ns_cnt = compute_time;
  flow_request->node_meta = node_meta;
  // std::size_t load_lat_us_cnt;
  // std::size_t unload_lat_us_cnt;
  ENGINE_CTX->ProcessFlowRequest(flow_request);
}


/* ============== DAGNode ============== */

void
DAGNode::SetDevice(const torch::Device& device)
{
  if (device == device_)
    return;
  torch::InferenceMode infer_guard(true);
  device_ = device;

  // In our context, lazy device stays on disk
  if (device == DISK_DEVICE) {
    delete model_;
    model_ = nullptr;
    // model_copy_ = nullptr;
    return;
  }

  if (model_ == nullptr) {
    // InferenceMode should be used to guard all tensors operations including
    // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
    model_ = new ScriptModule(torch::jit::load(model_path_, device));
    // model_.reset(new Module(torch::jit::load(model_path_, device)));
  } else {
    model_->to(device);
  }
  // model_copy_ = model_;
}

// void
// DAGNode::SetDeviceCopy(const torch::Device& device)
// {
//   if (device == device_)
//     return;
//   torch::InferenceMode infer_guard(true);
//   device_ = device;

//   if (device == DISK_DEVICE) {
//     if (model_ == model_copy_) {
//       delete model_copy_;
//       model_ = nullptr;
//       model_copy_ = nullptr;
//     }

//     if (model_copy_ != nullptr) {
//       delete model_copy_;
//       model_copy_ = nullptr;
//     }

//     if (model_ != nullptr) {
//       delete model_;
//       model_ = nullptr;
//     }

//     return;
//   }

//   if (model_ == nullptr) {
//     // InferenceMode should be used to guard all tensors operations including
//     // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
//     model_ = new ScriptModule(torch::jit::load(model_path_, device));
//     model_copy_ = model_;
//   } else {
//     if (device == CPU_DEVICE) {
//       delete model_copy_;
//       model_copy_ = nullptr;
//     }

//     if (device == DEFAULT_CUDA_DEVICE) {
//       model_copy_ = new ScriptModule(model_->clone());
//       model_copy_->to(device);
//     }
//   }
// }

// NodeMetaPtr
// DAGNode::AddPrevNode(const DAGNodePtr& node)
// {
//   NodeMetaPtr prev_node = std::make_shared<NodeMeta>();
//   prev_node->node_ptr = node;
//   auto result = prev_nodes_.insert({node->GetNodeID(), prev_node});
//   return result.first->second;
// }


// NodeMetaPtr
// DAGNode::AddNextNode(const DAGNodePtr& node)
// {
//   NodeMetaPtr next_node = std::make_shared<NodeMeta>();
//   next_node->node_ptr = node;
//   auto result = next_nodes_.insert({node->GetNodeID(), next_node});
//   if (!result.second) {
//     result.first->second->visit_cnt++;
//   }
//   return result.first->second;
// }


// NodeMetaPtr
// DAGNode::RemoveNextNode(const std::size_t& model_id)
// {
//   auto it = next_nodes_.find(model_id);
//   if (it == next_nodes_.end()) {
//     return nullptr;
//   }
//   if (it->second->visit_cnt > 0) {
//     it->second->visit_cnt--;
//   } else {
//     next_nodes_.erase(it);
//   }
//   return it->second;
// }

// void
// DAGNode::RecordMyself(const std::string& request_id)
// {
//   GET_INSTANCE(ModelFlowRecorder)
//       ->RecordModelFlow(request_id, std::shared_ptr<DAGNode>(this));
// }
