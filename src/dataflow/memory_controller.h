#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "base/noncopyable.h"
#include "dataflow/dag_node.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"
#include "utils/lru_cache.h"
#include "utils/memory_utils.h"

ENUM_MACRO(
    RetCode, RET_SUCCESS, RET_FAILED, RET_NOT_ENOUGH_MEMORY,
    RET_ALREADY_ALLOCATED, RET_NOT_ALLOCATED)

// class MemoryType : public EnumType {
//  public:
//   ENUM_ARGS(kEvict, kReady)

//   explicit MemoryType(const int& type) : EnumType(type) {}
// };
// ENUM_STRUCT_MACRO(MemoryType, kEvict, kReady)

typedef std::vector<DAGNodePtr> EvictionCandidates;

// This class is always accessed by the same thread, so no need to protect it
class MemoryAllocator : public noncopyable {
 public:
  DISABLE_COPY_AND_ASSIGN(MemoryAllocator);
  explicit MemoryAllocator(
      const Device& device, std::size_t memory_limit, std::size_t free_memory)
      : memory_limit_(memory_limit), free_memory_(free_memory), device_(device), lru_cache_(100000)
  {
  }
  ~MemoryAllocator() = default;

  RetCode AllocMemory(const DAGNodePtr& node);
  //   RetCode TryAllocMemory(const std::size_t& key, const std::size_t& size);
  RetCode FreeMemory(const DAGNodePtr& node);
  //   void PrepareFreeMemory(const std::size_t& key);
  //   void ConfirmFreeMemory(const std::size_t& key);
  //   std::size_t GetFreeMemory() const { return free_memory_; }
  std::pair<EvictionCandidates, bool> GetEvictionCadidates(
      const std::size_t& size) const;

  Device GetDevice() const { return device_; }

 private:
  std::size_t memory_limit_;
  std::size_t free_memory_;
  Device device_;
  LRUCache<std::size_t, DAGNode> lru_cache_;
};


class MemoryController : public noncopyable {
 public:
  DISABLE_COPY_AND_ASSIGN(MemoryController);
  STATIC_GET_INSTANCE(MemoryController);
  MemoryController();
  ~MemoryController() = default;

  // Will always success by evicting some other models
  std::tuple<EvictionCandidates, bool, Device> AllocMemory(
      const DAGNodePtr& node, const Device& device);
  // RetCode TryAllocMemory(const std::size_t& key, onst std::size_t& size);
  std::tuple<EvictionCandidates, bool, Device> FreeMemory(
      const DAGNodePtr& node);

  //  private:
  //   void MoveMemory(
  //       const std::size_t& key, const std::size_t& size,
  //       const torch::Device& src_device, const torch::Device& dst_device);

 private:
  std::unordered_map<torch::Device, std::shared_ptr<MemoryAllocator>>
      device_allocator_;
  std::unordered_map<std::size_t, torch::Device> device_map_;
};