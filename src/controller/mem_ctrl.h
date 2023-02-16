#pragma once

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include "muduo/base/noncopyable.h"
#include "stream_ctrl.h"
#include "utils/class_utils.h"
#include "utils/memory_utils.h"
#include "utils/state.h"
#include "utils/torch_utils.h"

#define DEFAULT_SYSTEM_FREE_MEMORY GetTotalSystemMemory() * 0.5
#define DEFAULT_CUDA_FREE_MEMORY(idx) GetTotalDeviceMemory(idx) * 0.4

typedef rmm::mr::cuda_memory_resource CudaMemoryResource;
typedef std::shared_ptr<CudaMemoryResource> CudaMemoryResourcePtr;
typedef rmm::mr::pinned_memory_resource PinnedMemoryResource;
typedef std::shared_ptr<PinnedMemoryResource> PinnedMemoryResourcePtr;
typedef rmm::mr::arena_memory_resource<CudaMemoryResource> ArenaMemoryResource;
typedef std::shared_ptr<ArenaMemoryResource> ArenaMemoryResourcePtr;

class MemoryController {
 public:
  explicit MemoryController(std::size_t free_memory)
  {
    free_memory_ = free_memory;
  }

  /**
   * Allocate memory for a node.
   * @param node The node to allocate memory for.
   * @return True if the memory is allocated successfully, false otherwise.
   */
  MemoryStatus AllocateMemory(const std::size_t key, const std::int64_t size);
  MemoryStatus TryAllocateMemory(
      const std::size_t key, const std::int64_t size);
  MemoryStatus AllocateMemory(const std::int64_t size);

  /**
   * Free memory for a node.
   * @param node The node to free memory for.
   * @return True if the memory is freed successfully, false otherwise.
   */
  MemoryStatus FreeMemory(const std::size_t key, const std::int64_t size);
  MemoryStatus FreeMemory(const std::int64_t size);

  /**
   * Get the total memory usage.
   * @return The total memory usage.
   */
  std::int64_t GetFreeMemory() const
  {
    std::shared_lock lock(mutex_);
    return free_memory_;
  }

 private:
  std::int64_t free_memory_;
  std::unordered_set<std::size_t> allocated_memory_;
  mutable std::shared_mutex mutex_;
};
typedef std::shared_ptr<MemoryController> MemoryControllerPtr;

class MemCtrl : public muduo::noncopyable {
 public:
  DISABLE_COPY_AND_ASSIGN(MemCtrl)
  STATIC_GET_INSTANCE(MemCtrl)

  MemoryControllerPtr& GetSysMemCtrl() { return sys_mem_ctrl_; }

  MemoryControllerPtr& GetCudaMemCtrl(std::size_t idx)
  {
    return cuda_mem_ctrls_[idx];
  }

  MemCtrl()
  {
    sys_mem_ctrl_ =
        std::make_shared<MemoryController>(DEFAULT_SYSTEM_FREE_MEMORY);
    for (int i = 0; i < GetDeviceCount(); ++i) {
      cuda_mem_ctrls_.emplace_back(
          std::make_shared<MemoryController>(DEFAULT_CUDA_FREE_MEMORY(i)));
    }
  }
  virtual ~MemCtrl() = default;

 private:
  MemoryControllerPtr sys_mem_ctrl_;
  rmm::mr::pinned_memory_resource pinned_mr_;
  std::vector<MemoryControllerPtr> cuda_mem_ctrls_;
};

class HostMemoryPool : public muduo::noncopyable {
 public:
  static HostMemoryPool* GetInstance() { return new HostMemoryPool(); }

  void PreAllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    assert(device.is_cpu());
    std::unique_lock lock(mutex_);
    if (allocated_id_.find(key) != allocated_id_.end()) {
      return;
    }
    allocated_id_.insert(key);
    free_memory_ -= size;
  }

  void* AllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    assert(device.is_cpu());
    std::unique_lock lock(mutex_);
    if (allocated_id_.find(key) == allocated_id_.end()) {
      return nullptr;
    }
    return pinned_mr_.allocate(size);
  }

  int FreeMemory(
      const std::size_t key, void* data, const std::int64_t size,
      const torch::Device& device)
  {
    assert(device.is_cpu());
    std::unique_lock lock(mutex_);
    if (allocated_id_.find(key) == allocated_id_.end()) {
      return -1;
    }
    allocated_id_.erase(key);
    if (data != nullptr)
      pinned_mr_.deallocate(data, size);
    free_memory_ += size;
    return 0;
  }

  std::int64_t GetFreeMemory()
  {
    std::lock_guard lock(mutex_);
    return free_memory_;
  }

 private:
  HostMemoryPool() = default;
  virtual ~HostMemoryPool() = default;

 private:
  std::unordered_set<std::uint64_t> allocated_id_;
  rmm::mr::pinned_memory_resource pinned_mr_;
  std::int64_t free_memory_ = DEFAULT_SYSTEM_FREE_MEMORY;
  std::mutex mutex_;
};

class DeviceMemoryPool : public muduo::noncopyable {
 public:
  static DeviceMemoryPool* GetInstance() { return new DeviceMemoryPool(); }

  void PreAllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    int device_id = device.index();
    std::unique_lock lock(mutex_);
    if (allocated_id_[device_id].find(key) != allocated_id_[device_id].end()) {
      return;
    }
    allocated_id_[device_id].insert(key);
    free_memory_[device_id] -= size;
  }


  void* AllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    int device_id = device.index();
    std::unique_lock lock(mutex_);
    if (allocated_id_[device_id].find(key) == allocated_id_[device_id].end()) {
      return nullptr;
    }
    return arena_mr_[device_id]->allocate(
        size, CUDA_STREAM_H2D_VIEW(device_id));
  }

  int FreeMemory(
      const std::size_t key, void* data, const std::int64_t size,
      const torch::Device& device)
  {
    int device_id = device.index();
    std::unique_lock lock(mutex_);
    if (allocated_id_[device_id].find(key) == allocated_id_[device_id].end()) {
      return -1;
    }
    allocated_id_[device_id].erase(key);
    if (data != nullptr)
      arena_mr_[device_id]->deallocate(
          data, size, CUDA_STREAM_H2D_VIEW(device_id));
    free_memory_[device_id] += size;
    return 0;
  }

  std::int64_t GetFreeMemory(const torch::Device& device)
  {
    int device_id = device.index();
    std::lock_guard lock(mutex_);
    return free_memory_[device_id];
  }

 private:
  DeviceMemoryPool()
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; ++i) {
      cudaSetDevice(i);
      cuda_mr_.emplace_back(std::make_shared<CudaMemoryResource>());
      arena_mr_.emplace_back(std::make_shared<ArenaMemoryResource>(
          cuda_mr_[i].get(), DEFAULT_CUDA_FREE_MEMORY(i), false));
      std::unordered_set<std::uint64_t> allocated_id;
      allocated_id_.emplace_back(allocated_id);
      free_memory_.emplace_back(DEFAULT_CUDA_FREE_MEMORY(i));
    }
  }
  virtual ~DeviceMemoryPool() = default;

 private:
  std::vector<std::unordered_set<std::uint64_t>> allocated_id_;
  std::vector<CudaMemoryResourcePtr> cuda_mr_;
  std::vector<ArenaMemoryResourcePtr> arena_mr_;
  std::vector<std::int64_t> free_memory_;
  std::mutex mutex_;
};

extern HostMemoryPool* kHostMemoryPool;
extern DeviceMemoryPool* kDeviceMemoryPool;

#define CUDA_MEM_CTL(idx) GET_INSTANCE(MemCtrl)->GetCudaMemCtrl(idx)
#define SYS_MEM_CTL GET_INSTANCE(MemCtrl)->GetSysMemCtrl()
#define DEFAULT_CUDA_MEM_CTL GET_INSTANCE(MemCtrl)->GetCudaMemCtrl(0)

#define FLOAT32_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat32).device(target)
#define FLOAT16_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat16).device(target)
#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

void SetModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr);
void SetModuleCudaMemory(
    torch::jit::script::Module* model, void* device_ptr,
    const torch::Device& device);
void SetModuleContinuousMemory(torch::jit::script::Module* model);
void CopyModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr);
