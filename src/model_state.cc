#include "model_state.h"

#include "dag_registry.h"
#include "libtorch_utils.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace pytorch {

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  RETURN_IF_ERROR((*state)->ParseParameters());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), enable_optimized_execution_(true),
      enable_inference_mode_(false), enable_cache_cleaning_(false),
      enable_weight_sharing_(false), enable_tensor_fuser_pair_({false, true}),
      enable_jit_profiling_pair_({false, true}),
      enable_jit_executor_pair_({false, true}),
      enable_nvfuser_pair_({false, false})
{
  output_names_.clear();

  triton::common::TritonJson::Value ios;
  THROW_IF_BACKEND_INSTANCE_ERROR(ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    THROW_IF_BACKEND_INSTANCE_ERROR(ios.IndexAsObject(i, &io));

    // Use names from ModelConfig by reference since the model
    // config will persist longer than this inference execution.
    const char* io_name;
    size_t io_name_len;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        io.MemberAsString("name", &io_name, &io_name_len));
    output_names_.emplace_back(io_name);
  }
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, const torch::Device device,
    std::string* model_path,
    std::shared_ptr<torch::jit::script::Module>* torch_model, DAGNodePtr* node)
{
  // Find the TorchScript file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.pt").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.pt";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  // Check if the model exist in repository
  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // InferenceMode should be used to guard all tensors operations including
  // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
  torch::InferenceMode infer_guard(EnabledInferenceMode());

  // Create a new torch model as DAG node
  *node = std::make_shared<DAGNode>(*model_path, Name(), Version());
  GET_INSTANCE(DAGRegistry)->AddNode((*node)->GetNodeID(), *node);
  // // Allocate Memory on Management Memory Pool
  // auto request = std::make_shared<MemoryManageRequest>(
  //     node, CPU_DEVICE, ManageType::PREFETCH);
  // GET_INSTANCE(DynamicMemoryBatcher)->Enqueue(request);

  *torch_model = MODULE_PTR_NODELETE((*node)->GetModel());

  // RETURN_IF_ERROR(
  //     GET_INSTANCE(LibTorchPool)->RegisterModule(*model_path, Name(),
  //     Version()));

  // // If the model is not already loaded, load it.
  // auto start_time = std::chrono::steady_clock::now();
  // RETURN_IF_ERROR(GET_INSTANCE(LibTorchPool)->FetchModule(Name(), Version(),
  // device)); auto end_time = std::chrono::steady_clock::now(); LOG_MESSAGE(
  //     TRITONSERVER_LOG_VERBOSE,
  //     (std::string("TorchScript model load time: ") +
  //      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
  //                         end_time - start_time)
  //                         .count()) +
  //      " usec")
  //         .c_str());


  // *torch_model = GET_INSTANCE(LibTorchPool)->GetModule(Name(), Version());
  TRITONSERVER_Message* model_config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model_, 1 /* config_version */, &model_config_message));

  //
  // // If weight sharing is enabled, skip loading model if
  // // it is already available on the target device
  // std::pair<bool, int> device_pair;
  // if (enable_weight_sharing_) {
  //   device_pair = std::make_pair(!device.is_cpu(), device.index());
  //   auto mit = torch_models_.find(device_pair);
  //   if (mit != torch_models_.end()) {
  //     *torch_model = mit->second;
  //     LOG_MESSAGE(
  //         TRITONSERVER_LOG_INFO,
  //         (std::string("Reusing TorchScript model for instance '") + Name() +
  //          "'")
  //             .c_str());
  //     return nullptr;  // success
  //   }
  // }

  // // Serialize the torch model to string
  // std::string model_data_str;
  // RETURN_IF_ERROR(ReadTextFile(*model_path, &model_data_str));

  // try {
  //   std::istringstream model_stream(model_data_str);
  //   torch_model->reset(
  //       new torch::jit::Module(torch::jit::load(model_stream, device)));
  // }
  // catch (const std::exception& ex) {
  //   return TRITONSERVER_ErrorNew(
  //       TRITONSERVER_ERROR_INTERNAL,
  //       ("failed to load model '" + Name() + "': " + ex.what()).c_str());
  // }

  // if (enable_weight_sharing_) {
  //   if (!((torch_models_.emplace(device_pair, *torch_model)).second)) {
  //     std::string type = device.is_cpu() ? "CPU" : "GPU";
  //     LOG_MESSAGE(
  //         TRITONSERVER_LOG_WARN,
  //         (std::string("Model already found on target ") + type + " device "
  //         +
  //          "(id " + std::to_string(device.index()) + ") for '" + Name() +
  //          "'")
  //             .c_str());
  //   }
  // }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since PyTorch does not
  // store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for pytorch backend")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    // If 'DISABLE_OPTIMIZED_EXECUTION' is not present in 'parameters' then no
    // update is made to 'enable_optimized_execution_'.
    bool disable_optimized_execution = false;
    TRITONSERVER_Error* err = ParseParameter(
        params, "DISABLE_OPTIMIZED_EXECUTION", &disable_optimized_execution);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    enable_optimized_execution_ = !disable_optimized_execution;

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Optimized execution is ") +
         (enable_optimized_execution_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'ENABLE_CACHE_CLEANING' is not present in 'parameters' then
    // no update is made to 'enable_cache_cleaning_'.
    err = ParseParameter(
        params, "ENABLE_CACHE_CLEANING", &enable_cache_cleaning_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Cache Cleaning is ") +
         (enable_cache_cleaning_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'INFERENCE_MODE' is not present in 'parameters' then no update is made
    // to 'enable_inference_mode_'.
    err = ParseParameter(params, "INFERENCE_MODE", &enable_inference_mode_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Inference Mode is ") +
         (enable_inference_mode_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'ENABLE_TENSOR_FUSER' is not present in 'parameters' then no
    // update is made to 'enable_tensor_fuser'.
    bool enable_tensor_fuser = false;
    err = ParseParameter(params, "ENABLE_TENSOR_FUSER", &enable_tensor_fuser);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_tensor_fuser_pair_ = {true, enable_tensor_fuser};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Tensor fuser is ") +
           (enable_tensor_fuser ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_WEIGHT_SHARING' is not present in 'parameters' then no
    // update is made to 'enable_weight_sharing'.
    err = ParseParameter(
        params, "ENABLE_WEIGHT_SHARING", &enable_weight_sharing_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Weight sharing is ") +
           (enable_weight_sharing_ ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_JIT_PROFILING' is not present in 'parameters' then no update
    // is made to 'enable_jit_profiling'.
    bool enable_jit_profiling = false;
    err = ParseParameter(params, "ENABLE_JIT_PROFILING", &enable_jit_profiling);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_jit_profiling_pair_ = {true, enable_jit_profiling};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Jit profiling is ") +
           (enable_jit_profiling ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_JIT_EXECUTOR' is not present in 'parameters' then no update is
    // made to 'enable_jit_executor'.
    bool enable_jit_executor = false;
    err = ParseParameter(params, "ENABLE_JIT_EXECUTOR", &enable_jit_executor);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_jit_executor_pair_ = {true, enable_jit_executor};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Jit executor is ") +
           (enable_jit_executor ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // TODO Re-enable NvFuser once fixed
    // If 'ENABLE_NVFUSER' is not present in 'parameters' then no
    // update is made to 'enable_nvfuser'.
    bool enable_nvfuser = false;
    err = ParseParameter(params, "ENABLE_NVFUSER", &enable_nvfuser);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO, (std::string("NvFuser is not specified") +
                                    " for model instance '" + Name() + "'")
                                       .c_str());
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      // Override, disable NvFuser till fixed
      enable_nvfuser = false;
      enable_nvfuser_pair_ = {true, enable_nvfuser};
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, (std::string("NvFuser is ") +
                                  (enable_nvfuser ? "enabled" : "disabled") +
                                  " for model instance '" + Name() + "'")
                                     .c_str());
    }
  }

  return nullptr;
}

}}}  // namespace triton::backend::pytorch