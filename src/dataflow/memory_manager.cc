#include "memory_manager.h"

MemoryManager::MemoryManager(std::size_t free_memory)
    : total_memory_(free_memory), free_memory_(free_memory)
{
}

bool
MemoryManager::AllocateMemory(const NodePtr& node)
{
  return AllocateMemory(node->byte_size);
}

bool
MemoryManager::AllocateMemory(const std::int64_t size)
{
  if (total_memory_ <= size) {
    LOG_TRITON_ERROR("MemoryManager::AllocateMemory: Not enough memory");
  }
  total_memory_ -= size;
  return true;
}

bool
MemoryManager::FreeMemory(const NodePtr& node)
{
  return FreeMemory(node->byte_size);
}

bool
MemoryManager::FreeMemory(const std::int64_t size)
{
  total_memory_ += size;
  assert(total_memory_ <= free_memory_);
  return true;
}
