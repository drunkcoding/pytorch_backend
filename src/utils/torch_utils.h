#pragma once

#include <torch/script.h>

#include <string>

#include "data_utils.h"

typedef torch::Device Device;

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kLazy)
#define META_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

typedef torch::jit::script::Module ScriptModule;
// typedef std::shared_ptr<Module> LibTorchModulePtr;
typedef ScriptModule* ScriptModulePtr;  // always use raw pointer, since we need
                                        // to manage the memory by ourselves


class ModuleInstance {
 public:
  ModuleInstance() = default;
  ModuleInstance(const std::string& model_path, const Device& device)
      : model_path_(model_path), device_(device)
  {
    LoadModel();
  }
  ~ModuleInstance()
  {
    if (module_) {
      delete module_;
    }
  }
  void LoadModel()
  {
    if (module_) {
      delete module_;
    }
    module_ = new ScriptModule(torch::jit::load(model_path_));
    module_->to(device_);
  }
  ScriptModulePtr GetModule() { return module_; }
  void SetDevice(const Device& device)
  {
    device_ = device;
    LoadModel();
  }

 private:
  NodeID model_id_;
  std::string model_path_;
  std::string model_instance_name_;
  Device device_;
  ScriptModulePtr module_;
  std::size_t model_byte_size_;
};