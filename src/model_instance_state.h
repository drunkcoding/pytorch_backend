#pragma once

#include <torch/script.h>

#include "engine/backend_engine.h"
#include "utils/topology.h"
// #include "eventloop_thread.h"
#include "model_state.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model_instance.h"
#include "utils/enum_utils.h"

namespace triton { namespace backend { namespace pytorch {

// The naming convention followed for inputs/outputs in the model configuration.
// Outputs don't support FORWARD_ARGUMENT.
ENUM_MACRO(
    NamingConvention, NAMED_INDEX, FORWARD_ARGUMENT, STRICT_CONFIG_ORDERING)

// enum class NamingConvention {
//   NAMED_INDEX,
//   FORWARD_ARGUMENT,
//   STRICT_CONFIG_ORDERING
// };

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState
    : public BackendModelInstance,
      public std::enable_shared_from_this<ModelInstanceState> {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);
  //   void ProcessTritonRequest();

  // Clear CUDA cache
  void ClearCache();

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<torch::jit::IValue>* output_tensors);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size,
      const std::vector<torch::jit::IValue>& output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  TRITONSERVER_Error* RecordBackendTimestamp(
      uint64_t* timestamp, void* cuda_event);

  // Get the naming convention for inputs/outputs from the model configuration
  TRITONSERVER_Error* GetNamingConvention(
      NamingConvention* naming_convention,
      const std::vector<std::string>& allowed_io);

  //   void TryAllocateNodeOnGPU();
  //   void TryReleaseNodeOnGPU();

  ModelState* model_state_;

  // The full path to the TorchScript model file.
  std::string model_path_;

  std::shared_ptr<torch::jit::script::Module> torch_model_;
  //   torch::jit::script::Module* torch_model_;
  NodePtr node_;
  //   std::mutex exec_mutex_;
  //   std::condition_variable exec_cv_;
  torch::Device device_;

  // muduo::net::EventLoop base programming
  BackendEnginePtr engine_;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;

  // If the input to the tensor is a dictionary of tensors.
  bool is_dict_input_;

  // If the model supports batching.
  bool supports_batching_;

  cudaEvent_t compute_input_start_event_;
  cudaEvent_t compute_infer_start_event_;
  cudaEvent_t compute_output_start_event_;

  uint64_t num_experts_;
  uint64_t num_layers_;
  uint64_t stage_id_;
  uint64_t node_id_;

  //   std::mutex mutex_;
  //   std::condition_variable cv_;
};

static std::atomic<int> g_stream_count{0};

}}}  // namespace triton::backend::pytorch