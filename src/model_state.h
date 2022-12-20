#pragma once

#include <torch/script.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "utils/topology.h"

namespace triton { namespace backend { namespace pytorch {

// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load a TorchScript model using 'artifact_name' as the name for the
  // TorchScript file. Return in 'model_path' the full path to the
  // TorchScript file, return in 'torch_model' the Torch Module
  // representing the model.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, const torch::Device device,
      std::string* model_path,
      std::shared_ptr<torch::jit::script::Module>* torch_model, NodePtr* node);

  bool EnabledOptimizedExecution() { return enable_optimized_execution_; }
  const std::pair<bool, bool>& EnabledTensorExprFuser() const
  {
    return enable_tensor_fuser_pair_;
  }
  const std::pair<bool, bool>& EnabledJitProfiling() const
  {
    return enable_jit_profiling_pair_;
  }
  const std::pair<bool, bool>& EnabledJitExecutor() const
  {
    return enable_jit_executor_pair_;
  }
  bool EnabledInferenceMode() { return enable_inference_mode_; }
  const std::pair<bool, bool>& EnabledNvfuserPair() const
  {
    return enable_nvfuser_pair_;
  }
  bool EnabledCacheCleaning() { return enable_cache_cleaning_; }

  bool EnabledWeightSharing() { return enable_weight_sharing_; }
  const std::vector<std::string>& ModelOutputs() { return output_names_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  // Parses and validates parameters in config
  TRITONSERVER_Error* ParseParameters();

  // Flag to indicate whether optimized execution is enabled. Defaults to true.
  bool enable_optimized_execution_;

  // Flag to indicate whether inference mode is enabled. Defaults to false.
  bool enable_inference_mode_;

  // Flag to indicate whether cache cleaning after each run is enabled.
  // Defaults to false.
  bool enable_cache_cleaning_;

  // Flag to indicate whether weight sharing is enabled. Defaults to false.
  bool enable_weight_sharing_;

  // Flag pairs to indicate if various JIT settings are set and
  // enabled respectively. Defaults to (false, true). Default behavior
  // is to do nothing if not explicitly set. Tensor fuser flag is
  // ignore if nvfuser is explicitly set.
  std::pair<bool, bool> enable_tensor_fuser_pair_;
  std::pair<bool, bool> enable_jit_profiling_pair_;
  std::pair<bool, bool> enable_jit_executor_pair_;

  // Flag pair to indicate whether nvfuser is set and enabled respectively.
  // Defaults to (false, false).
  std::pair<bool, bool> enable_nvfuser_pair_;

  // Model mapping for shared TorchScript model across all instances on the
  // same device. The key is a pair of isGPU and device index.
  std::map<
      std::pair<bool, int64_t>, std::shared_ptr<torch::jit::script::Module>>
      torch_models_;

  // List of all the outputs specified in the output section of model
  // configuration.
  std::vector<std::string> output_names_;
};


}}}  // namespace triton::backend::pytorch