#pragma once
#if 0
#include <torch/script.h>  // One-stop header for TorchScript

#include <condition_variable>
#include <memory>
#include <mutex>

#include "libtorch_common.h"

class ModelMeta {
 public:
  ModelMeta() = default;
  ModelMeta(
      const std::string& model_path, const std::string& model_name,
      const std::uint64_t model_version)
      : model_byte_size_(0), model_id_(MakeID(model_name, model_version)),
        model_name_(model_name), model_version_(model_version),
        model_path_(model_path)
  {
    model_.reset(new Module(torch::jit::load(model_path)));

    // Iterate model parameters and calculate the total size of the model
    std::uint64_t param_size = 0;
    for (const auto& param : model_->parameters()) {
      param_size += param.numel() * param.element_size();
    }

    // Iterate model buffers and calculate the total size of the model
    std::uint64_t buffer_size = 0;
    for (const auto& buffer : model_->buffers()) {
      buffer_size += buffer.numel() * buffer.element_size();
    }

    model_byte_size_ = param_size + buffer_size;

    model_.reset();
  }

  LibTorchModulePtr& GetModel() { return model_; }
  std::uint64_t GetID() const noexcept { return model_id_; }

  void SetDevice(const torch::Device& device) noexcept { model_->to(device); }

  void SetDevice(const DeviceType& device, const int& device_id) noexcept
  {
    if (device == DeviceType::DISK) {
      model_.reset();
      return;
    }

    if (model_ == nullptr) {
      model_.reset(new Module(torch::jit::load(model_path_)));
    }

    if (device == DeviceType::CPU) {
      model_->to(torch::kCPU);
    } else if (device == DeviceType::GPU) {
      // if (model_->Device() == torch::Device(torch::kCUDA, device_id)) {
      //   return;
      // }
      // cudaStreamSynchronize(cuda_stream_); // sync with previous stream
      // cudaStreamDestroy(cuda_stream_);     // destroy previous stream
      model_->to(torch::Device(torch::kCUDA, device_id));
      // cudaSetDevice(device_id); // set device for current stream
      // cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
    }
  }

  // return model_bytes_size
  std::size_t GetByteSize() const noexcept { return model_byte_size_; }

 private:
  std::size_t model_byte_size_;
  std::size_t model_id_;
  std::string model_name_;
  std::uint64_t model_version_;
  std::string model_path_;
  LibTorchModulePtr model_;
  // std::condition_variable cv_;
  // std::mutex mutex_;
  // at::cuda::CUDAStream cuda_stream_; // Stream for GPU execution and memory
  // transfer
};

typedef std::shared_ptr<ModelMeta> ModelMetaPtr;
#endif