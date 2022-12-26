#pragma once

#include "utils/class_utils.h"
#include "utils/topology.h"

/**
 * A simple memory manager that keep a counter of the total memory usage.
*/

class MemoryManager {
 public:
  explicit MemoryManager(std::size_t free_memory);

  /**
   * Allocate memory for a node.
   * @param node The node to allocate memory for.
   * @return True if the memory is allocated successfully, false otherwise.
   */
  bool AllocateMemory(const NodePtr& node);
  bool AllocateMemory(const std::int64_t size);

  /**
   * Free memory for a node.
   * @param node The node to free memory for.
   * @return True if the memory is freed successfully, false otherwise.
   */
  bool FreeMemory(const NodePtr& node);
  bool FreeMemory(const std::int64_t size);

  /**
   * Get the total memory usage.
   * @return The total memory usage.
   */
  std::int64_t GetTotalMemory() const { return total_memory_; }

 private:
  std::int64_t total_memory_;
  std::int64_t free_memory_;
};

typedef std::shared_ptr<MemoryManager> MemoryManagerPtr;
