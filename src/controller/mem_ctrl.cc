#include "mem_ctrl.h"

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