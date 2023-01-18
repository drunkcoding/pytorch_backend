#include "topology.h"

Node::Node(const std::string& model_path)
    : id(std::hash<std::string>{}(model_path)), corr_id(0), byte_size(0),
      last_access_time(MCIROSECONDS_SINCE_EPOCH), device(CPU_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE), model_path_(model_path)
{
  model = ScriptModule(torch::jit::load(model_path));
  cpu_model = model.copy();
  std::int64_t param_size = 0;
  for (const auto& param : model.parameters()) {
    param_size += param.numel() * param.element_size();
  }

  // Iterate model buffers and calculate the total size of the model
  std::int64_t buffer_size = 0;
  for (const auto& buffer : model.buffers()) {
    buffer_size += buffer.numel() * buffer.element_size();
  }
  byte_size = param_size + buffer_size;
  // mutex.unlock();
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
  if (device == target_device)
    return;

  // at::cuda::CUDAStreamGuard guard(
  //     at::cuda::getStreamFromExternal(stream, default_device.index()));

  // InferenceMode should be used to guard all tensors operations including
  // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html

  // LOG_TRITON_VERBOSE(("SetDevice: infer_guard " +
  //                     std::to_string(infer_guard.is_enabled()) + " " +
  //                     GetModelInstanceInfo() + " " + target_device.str())
  //                        .c_str());

  auto move_device = (target_device.is_cuda()) ? default_device : target_device;

  if (move_device == DISK_DEVICE) {
    // move from CPU/GPU to SSD
    RemoveModuleFromCache(&model);
    cpu_model = ScriptModule();
    model = ScriptModule();
    is_loaded = false;
  }

  if (device.is_cpu() && move_device.is_cuda()) {
    // move from CPU to GPU
    assert(is_loaded);
    model = cpu_model.clone();
    model.to(move_device);
  }

  if (device == DISK_DEVICE && move_device.is_cuda()) {
    // move from SSD to GPU
    model = ScriptModule(torch::jit::load(model_path_, move_device));

    if (default_host.is_cpu()) {
      cpu_model = model.copy();
      is_loaded = true;
    }
  }

  if (device == DISK_DEVICE && move_device.is_cpu()) {
    // move from SSD to CPU
    cpu_model = ScriptModule(torch::jit::load(model_path_, move_device));
    is_loaded = true;
  }

  if (device.is_cuda() && move_device.is_cpu()) {
    // move from GPU to CPU
    if (default_host.is_cpu() && !is_loaded) {
      model.to(move_device);
      cpu_model = model.clone();
      is_loaded = true;
    }
    model = ScriptModule();
  }


  // if (target_device == DISK_DEVICE) {
  //   delete model;
  //   model = nullptr;
  // } else {
  //   if (model == nullptr)
  //     model = new ScriptModule(torch::jit::load(model_path_, move_device));
  //   else {
  //     model.to(move_device);
  //   }
  // }

  // if (waiting) {
  //   mem_future.wait();
  // }

  // // In our context, lazy device stays on disk
  // if (target_device == DISK_DEVICE) {
  //   // RemoveModuleFromCache(&model);
  //   delete model;
  //   model = nullptr;
  // } else if (model == nullptr) {
  //   if (move_device.is_cuda())
  //     Load(move_device);
  //   else if (move_device.is_cpu()) {
  //     waiting = true;
  //     mem_future =
  //         std::async(std::launch::async, &Node::Load, this, move_device);
  //   }
  // } else if (device.is_cuda() && move_device.is_cpu()) {
  //   delete model;
  //   model = nullptr;
  //   waiting = true;
  //   mem_future =
  //       std::async(std::launch::async, &Node::Load, this, move_device);
  // } else if (device.is_cpu() && move_device.is_cuda()) {
  //   model.to(move_device);
  // } else {
  //   assert(false);
  // }

  // LOG_TRITON_VERBOSE(("SetDevice: infer_guard " +
  //                     std::to_string(infer_guard.is_enabled()) + " " +
  //                     GetModelInstanceInfo() + " " + target_device.str())
  //                        .c_str());
  c10::cuda::CUDACachingAllocator::emptyCache();
  device = move_device;
}