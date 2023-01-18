#pragma once

#include "muduo/base/noncopyable.h"
#include "utils/class_utils.h"
#include "utils/memory_utils.h"
#include "utils/state.h"

#define DEFAULT_SYSTEM_FREE_MEMORY GetTotalSystemMemory() * 0.3
#define DEFAULT_CUDA_FREE_MEMORY(idx) GetTotalDeviceMemory(idx) * 0.7

class MemoryController {
 public:
  explicit MemoryController(std::size_t free_memory) {
    free_memory_ = free_memory;
  }

  /**
   * Allocate memory for a node.
   * @param node The node to allocate memory for.
   * @return True if the memory is allocated successfully, false otherwise.
   */
  MemoryStatus AllocateMemory(const std::size_t key, const std::int64_t size);
  MemoryStatus TryAllocateMemory(const std::size_t key, const std::int64_t size);
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
    sys_mem_ctrl_ = std::make_shared<MemoryController>(DEFAULT_SYSTEM_FREE_MEMORY);
    for (int i = 0; i < GetDeviceCount(); ++i) {
      cuda_mem_ctrls_.emplace_back(
          std::make_shared<MemoryController>(DEFAULT_CUDA_FREE_MEMORY(i)));
    }
  }
  virtual ~MemCtrl() = default;

 private:
  MemoryControllerPtr sys_mem_ctrl_;
  std::vector<MemoryControllerPtr> cuda_mem_ctrls_;
};

#define CUDA_MEM_CTL(idx) GET_INSTANCE(MemCtrl)->GetCudaMemCtrl(idx)
#define SYS_MEM_CTL GET_INSTANCE(MemCtrl)->GetSysMemCtrl()
#define DEFAULT_CUDA_MEM_CTL GET_INSTANCE(MemCtrl)->GetCudaMemCtrl(0)
