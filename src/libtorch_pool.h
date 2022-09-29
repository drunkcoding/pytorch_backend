#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <torch/script.h>  // One-stop header for TorchScript
#include <unistd.h>

#include <cstdint>

#include "libtorch_factory.h"
#include "lru_cache.h"
#include "model_meta.h"
#include "triton/core/tritonbackend.h"


namespace triton { namespace backend { namespace pytorch {

enum class TorchDevice {
  DEVICE_CPU = -1,
  DEVICE_GPU,
  DEVICE_DISK = -2,
  DEVICE_INVALID
};

inline TorchDevice
GetTorchDevice(const torch::Device& device) noexcept
{
  if (device.is_cpu()) {
    return TorchDevice::DEVICE_CPU;
  } else if (device.is_cuda()) {
    return TorchDevice::DEVICE_GPU;
  } else {
    return TorchDevice::DEVICE_INVALID;
  }
}

// struct ModelCheckpoint {
//   TorchDevice model_device = TorchDevice::DEVICE_INVALID;
//   std::string model_path = "";
//   std::string model_name = "";
//   std::uint64_t model_version = 0;
//   std::uint64_t mode_byte_size = 0;
//   std::shared_ptr<torch::jit::Module> checkpoint = nullptr;

//   // return model_bytes_size
//   std::uint64_t size() const noexcept { return mode_byte_size; }

//   ModelCheckpoint(const std::string& model_path, const std::string&
//   model_name)
//       : model_path(model_path), model_name(model_name)
//   {
//     model_name_hash_ = std::hash<std::string>{}(model_name);
//     checkpoint.reset(new torch::jit::Module(torch::jit::load(model_path)));
//     model_device = TorchDevice::DEVICE_CPU;

//     // Iterate model parameters and calculate the total size of the model
//     std::uint64_t param_size = 0;
//     for (const auto& param : checkpoint->parameters()) {
//       param_size += param.numel() * param.element_size();
//     }

//     // Iterate model buffers and calculate the total size of the model
//     std::uint64_t buffer_size = 0;
//     for (const auto& buffer : checkpoint->buffers()) {
//       buffer_size += buffer.numel() * buffer.element_size();
//     }

//     mode_byte_size = param_size + buffer_size;
//   }

//  private:
//   std::size_t model_name_hash_;
//   std::condition_variable cv_;
//   ModelCheckpoint() = default;
// };

// typedef std::shared_ptr<ModelCheckpoint> ModelCheckpointPtr;

typedef LRUCache<std::size_t, ModelMetaPtr> ModelCache;

class LibTorchPool : public SingletonFactory {
 public:
  FACTORY_STATIC_GET_INSTANCE(LibTorchPool)

  // Register a model to the pool, put the model into the cache
  ModelMetaPtr RegisterModule(
      const std::string& model_path, const std::string& model_name,
      const std::uint64_t model_version);
  // TRITONSERVER_Error* DeregisterModule(const std::string& model_name);
  // TRITONSERVER_Error* FetchModule(
  //     const std::string& model_name, const std::uint64_t model_version,
  //     const torch::Device device);

  // TODO: Add algorithm to release models from cache
  // current implementation only do this in a lazy way
  // TRITONSERVER_Error* ReleaseModule(
  //     const std::string& model_name, const torch::Device device);

  // LibTorchModulePtr GetModule(
  //     const std::string& model_name, const std::uint64_t model_version);

  DISABLE_COPY_AND_ASSIGN(LibTorchPool)

 private:
  // LibTorchPool() = default;
  LibTorchPool()
  {
    // CPU LRU Cache, use 80% of the total memory
    cache_.insert({-1, ModelCache(GetTotalSystemMemory() * 0.8)});

    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      cudaSetDevice(gpu_id);
      cudaMemGetInfo(&free, &total);
      // GPU LRU Cache, leave 100MB for the share memory
      cache_.insert({gpu_id, ModelCache(total - 100 * 1024 * 1024)});
    }
  }
  ~LibTorchPool();

  // std::unordered_map<std::size_t, ModelCheckpoint> modules_;
  std::mutex mutex_;


  std::unordered_map<int, ModelCache> cache_;
  std::unordered_map<std::size_t, int>
      registered_models_;  // model_id -> device_id
};

#define LIBTORCHPOOL_GUARD \
  std::lock_guard<std::mutex> lock(LibTorchPool::GetInstance()->mutex_)

}}}  // namespace triton::backend::pytorch
