#include "topology.h"

#include "controller/mem_ctrl.h"

std::atomic_int32_t kGPUDeviceCount = 0;

Node::Node(const std::string& model_path)
    : id(std::hash<std::string>{}(model_path)), corr_id(0), byte_size(0),
      last_access_time(MCIROSECONDS_SINCE_EPOCH), device(CPU_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE), model_path_(model_path)
{
  model = new ScriptModule(torch::jit::load(model_path));
  for (const auto& param : model->parameters()) {
    byte_size += param.nbytes();
  }
  for (const auto& buffer : model->buffers()) {
    byte_size += buffer.nbytes();
  }
}

const std::string
Node::GetModelInstanceInfo() noexcept
{
  // write same string using c style sprintf
  char buffer[1024];
  memset(buffer, 0, 1024);
  sprintf(
      buffer, "%s (%ldMB) ID[%lx,%lx] TS[%ld] DEVICE[%s,%d];",
      model_path_.c_str(), byte_size / MB, id, corr_id, last_access_time,
      device.str().c_str(), static_cast<int>(memory_type));

  return std::string(buffer);
}

void
Node::SetDevice(const Device& target_device) noexcept
{
  // std::lock_guard<std::mutex> lock(mutex);
  if (device == target_device)
    return;

  // InferenceMode should be used to guard all tensors operations including
  // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
  at::InferenceMode infer_guard(true);


  LOG_TRITON_VERBOSE(("SetDevice: infer_guard " +
                      std::to_string(infer_guard.is_enabled()) + " " +
                      GetModelInstanceInfo() + " " + target_device.str())
                         .c_str());

  auto move_device = (target_device.is_cuda()) ? default_device : target_device;

  if (move_device == DISK_DEVICE) {
    delete model;
    model = nullptr;
    if (host_memory_ptr != nullptr) {
      kHostMemoryPool->FreeMemory(id, host_memory_ptr, byte_size, CPU_DEVICE);
      host_memory_ptr = nullptr;
    }
    if (device_memory_ptr != nullptr) {
      kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
      device_memory_ptr = nullptr;
    }
  } else {
    if (model == nullptr) {
      model = new ScriptModule(torch::jit::load(model_path_));
      int numa_id = default_device.index() / 4;
      host_memory_ptr =
          kHostMemoryPool->AllocateMemory(id, byte_size, torch::Device(torch::kCPU, numa_id));
      SetModuleContinuousMemory(model);
      CopyModulePinnedMemory(model, host_memory_ptr);
      SetModulePinnedMemory(model, host_memory_ptr);
    }

    if (move_device.is_cuda()) {
      device_memory_ptr =
          kDeviceMemoryPool->AllocateMemory(id, byte_size, move_device);
      cudaMemcpy(
          device_memory_ptr, host_memory_ptr, byte_size,
          cudaMemcpyHostToDevice);
      // cudaStreamSynchronize(0);
      // cudaPointerAttributes attr{};
      // cudaPointerGetAttributes(&attr, device_memory_ptr);
      // assert(attr.type != cudaMemoryTypeUnregistered);
      SetModuleCudaMemory(model, device_memory_ptr, move_device);
      LOG_TRITON_VERBOSE(("SetDevice: " + GetModelInstanceInfo() + " " +
                          move_device.str() + " " + std::to_string(byte_size))
                             .c_str());
    }

    if (move_device.is_cpu() && device.is_cuda()) {
      kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
      device_memory_ptr = nullptr;
      SetModulePinnedMemory(model, host_memory_ptr);
    }
  }
  c10::cuda::CUDACachingAllocator::emptyCache();
  device = move_device;
}
