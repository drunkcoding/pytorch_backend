#include "mem_ctrl.h"

#include "utils/log_utils.h"
#include "utils/memory_utils.h"

MemoryStatus
MemoryController::AllocateMemory(const std::size_t key, const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  if (allocated_memory_.find(key) != allocated_memory_.end()) {
    return MemoryStatus::kAllocated;
  }
  free_memory_ -= size;
  allocated_memory_.insert(key);
  return MemoryStatus::kSuccess;
}

MemoryStatus
MemoryController::TryAllocateMemory(
    const std::size_t key, const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  if (allocated_memory_.find(key) != allocated_memory_.end()) {
    return MemoryStatus::kAllocated;
  }
  if (free_memory_ < size) {
    return MemoryStatus::kFailed;
  }
  free_memory_ -= size;
  allocated_memory_.insert(key);
  return MemoryStatus::kSuccess;
}

MemoryStatus
MemoryController::AllocateMemory(const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  free_memory_ -= size;
  return MemoryStatus::kSuccess;
}


MemoryStatus
MemoryController::FreeMemory(const std::size_t key, const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  if (allocated_memory_.find(key) == allocated_memory_.end()) {
    return MemoryStatus::kFreed;
  }
  free_memory_ += size;
  allocated_memory_.erase(key);
  return MemoryStatus::kSuccess;
}

MemoryStatus
MemoryController::FreeMemory(const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  free_memory_ += size;
  return MemoryStatus::kSuccess;
}


HostMemoryPool* kHostMemoryPool = HostMemoryPool::GetInstance();
DeviceMemoryPool* kDeviceMemoryPool = DeviceMemoryPool::GetInstance();


void
SetModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  auto tensor_options = FLOAT32_TENSOR_OPTIONS(CPU_DEVICE);
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)host_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    param_size += size;
    LOG_TRITON_VERBOSE((std::string("SetModulePinnedMemory: ") + (*it).name() +
                        " param_size: " + std::to_string(param_size) +
                        " size: " + std::to_string(size))
                           .c_str());
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)host_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    param_size += size;
    LOG_TRITON_VERBOSE((std::string("SetModulePinnedMemory: ") + (*it).name() +
                        " param_size: " + std::to_string(param_size) +
                        " size: " + std::to_string(size))
                           .c_str());
  }
}

void
SetModuleCudaMemory(
    torch::jit::script::Module* model, void* device_ptr,
    const torch::Device& device)
{
  std::int64_t param_size = 0;
  auto tensor_options = FLOAT32_TENSOR_OPTIONS(device);
  // std::cout << "tensor_options: " << tensor_options << std::endl;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)device_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    // std::cout << "param: " << (*it).device() << std::endl;
    param_size += size;
    LOG_TRITON_VERBOSE((std::string("SetModuleCudaMemory: ") + (*it).name() +
                        " param_size: " + std::to_string(param_size) +
                        " size: " + std::to_string(size))
                           .c_str());
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)device_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    // std::cout << "buffer: " << (*it).device() << std::endl;
    param_size += size;
    LOG_TRITON_VERBOSE((std::string("SetModuleCudaMemory: ") + (*it).name() +
                        " param_size: " + std::to_string(param_size) +
                        " size: " + std::to_string(size))
                           .c_str());
  }
}


void
SetModuleContinuousMemory(torch::jit::script::Module* model)
{
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    if (!(*it).is_contiguous())
      (*it).set_data((*it).contiguous());
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    if (!(*it).is_contiguous())
      (*it).set_data((*it).contiguous());
  }
}

void
CopyModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    cudaMemcpy((char*)host_ptr + param_size, ptr, size, cudaMemcpyHostToHost);
    param_size += size;
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    cudaMemcpy((char*)host_ptr + param_size, ptr, size, cudaMemcpyHostToHost);
    param_size += size;
  }
}
