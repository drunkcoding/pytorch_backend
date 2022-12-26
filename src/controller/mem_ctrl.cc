#include "mem_ctrl.h"

bool
MemoryController::AllocateMemory(const std::size_t key, const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  // if (size > free_memory_) {
  //   return false;
  // }
  if (allocated_memory_.find(key) != allocated_memory_.end()) {
    return false;
  }
  free_memory_ -= size;
  allocated_memory_.insert(key);
  return true;
}

bool
MemoryController::AllocateMemory(const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  free_memory_ -= size;
  return true;
}


bool
MemoryController::FreeMemory(const std::size_t key, const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  if (allocated_memory_.find(key) == allocated_memory_.end()) {
    return false;
  }
  free_memory_ += size;
  allocated_memory_.erase(key);
  return true;
}

bool
MemoryController::FreeMemory(const std::int64_t size)
{
  std::unique_lock lock(mutex_);
  free_memory_ += size;
  return true;
}