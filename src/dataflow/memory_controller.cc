#include "memory_controller.h"

#include <tuple>
#include <algorithm>

#include "utils/data_utils.h"
#include "utils/log_utils.h"
#include "utils/torch_utils.h"

RetCode
MemoryAllocator::AllocMemory(const DAGNodePtr& node)
{
  auto key = node->GetNodeID();
  auto size = node->GetNodeByteSize();
  if (lru_cache_.Get(key) != nullptr) {
    return RetCode::RET_ALREADY_ALLOCATED;
  }
  assert(size <= free_memory_);

  // if (size > free_memory_) {
  //   static_assert(
  //       std::is_same<std::size_t, decltype(size)>::value,
  //       "size is not std::size_t");
  //   return RET_NOT_ENOUGH_MEMORY;
  // }
  lru_cache_.Put(key, node);
  free_memory_ -= size;
  return RetCode::RET_SUCCESS;
}

// RetCode
// MemoryAllocator::TryAllocMemory(const std::size_t& key, const std::size_t&
// size)
// {
//   if (lru_cache_.Get(key) != nullptr) {
//     return RetCode::RET_ALREADY_ALLOCATED;
//   }
//   if (size > free_memory_) {
//     return RET_NOT_ENOUGH_MEMORY;
//   }
//   return RET_SUCCESS;
// }

RetCode
MemoryAllocator::FreeMemory(const DAGNodePtr& node)
{
  auto key = node->GetNodeID();
  auto size = node->GetNodeByteSize();
  if (lru_cache_.Get(key) == nullptr) {
    return RetCode::RET_NOT_ALLOCATED;
  }
  lru_cache_.Remove(key);
  free_memory_ += size;
  node->SetMemoryType(MemoryType::kReady);
  return RetCode::RET_SUCCESS;
}

// void
// MemoryAllocator::PrepareFreeMemory(const std::size_t& key)
// {
//   auto node = lru_cache_.Get(key);
//   node->SetMemoryType(MemoryType::kEvict);
// }

// void
// MemoryAllocator::ConfirmFreeMemory(const std::size_t& key)
// {
//   auto node = lru_cache_.Get(key);
//   node->SetMemoryType(MemoryType::kReady);
//   lru_cache_.Remove(key);
//   free_memory_ += node->GetNodeByteSize();
// }

std::pair<EvictionCandidates, bool>
MemoryAllocator::GetEvictionCadidates(const std::size_t& size) const
{
  EvictionCandidates eviction_candidates;
  auto cache_in_order = lru_cache_.GetCacheInOrder();
  cache_in_order.reverse();
  auto free_memory = free_memory_;
  for (auto iter = cache_in_order.begin();
       iter != cache_in_order.end() && size > free_memory; ++iter) {
    if (iter->second->IsEvict()) {
      continue;
    }
    eviction_candidates.push_back(iter->second);
    free_memory += iter->second->GetNodeByteSize();
  }
  return std::make_pair(eviction_candidates, free_memory >= size);
}

MemoryController::MemoryController()
{
  auto cpu_memory_total = GetTotalSystemMemory();
  auto cpu_memory_free = GetFreeSystemMemory();
  LOG_INFO((std::string("Total System Memory: ") +
            std::to_string(cpu_memory_total / MB) + std::string(" MB"))
               .c_str());
  LOG_INFO((std::string("Free System Memory: ") +
            std::to_string(cpu_memory_free / MB) + std::string(" MB"))
               .c_str());

  device_allocator_.insert(
      {CPU_DEVICE, std::make_shared<MemoryAllocator>(
                       CPU_DEVICE, cpu_memory_total, cpu_memory_free)});

  device_allocator_.insert(
      {DISK_DEVICE,
       std::make_shared<MemoryAllocator>(DISK_DEVICE, UINT64_MAX, UINT64_MAX)});

  // create MemoryAllocator for all gpu devices
  for (int i = 0; i < GetDeviceCount(); i++) {
    auto gpu_memory_total = GetTotalDeviceMemory(i);
    auto gpu_memory_free = GetFreeDeviceMemory(i);
    LOG_INFO((std::string("Total GPU Memory: ") +
              std::to_string(gpu_memory_total / MB) + std::string(" MB") +
              std::string(" on device ") + std::to_string(i))
                 .c_str());
    LOG_INFO((std::string("Free GPU Memory: ") +
              std::to_string(gpu_memory_free / MB) + std::string(" MB") +
              std::string(" on device ") + std::to_string(i))
                 .c_str());
    device_allocator_.insert(
        {CUDA_DEVICE(i),
         std::make_shared<MemoryAllocator>(
             CUDA_DEVICE(i), gpu_memory_total, gpu_memory_free)});
  }
}

std::tuple<EvictionCandidates, bool, Device>
MemoryController::AllocMemory(const DAGNodePtr& node, const Device& device)
{
  // auto iter = device_allocator_.find(device);
  // if (iter == device_allocator_.end()) {
  //   return RET_INVALID_DEVICE;
  // }
  auto key = node->GetNodeID();
  auto size = node->GetNodeByteSize();

  std::shared_ptr<MemoryAllocator> allocator;
  EvictionCandidates eviction_candidates;
  bool is_evictable = false;
  // std::vector<std::pair<std::size_t, Device>> device_candidate;

  Device target_device = DISK_DEVICE;
  std::size_t min_eviction_size = UINT64_MAX;

  if (device.is_cpu()) {
    allocator = device_allocator_.at(CPU_DEVICE);
    // CPU_DEVICE is always evictable
    std::tie(eviction_candidates, is_evictable) =
        allocator->GetEvictionCadidates(size);
    // device_candidate.push_back(std::make_pair(size, CPU_DEVICE));
  }
  if (device.is_cuda()) {
    for (int device_id = 0; device_id < GetDeviceCount(); device_id++) {
      allocator = device_allocator_.at(CUDA_DEVICE(device_id));
      std::tie(eviction_candidates, is_evictable) =
          allocator->GetEvictionCadidates(size);
      if (is_evictable) {
        std::size_t memsize = 0;
        for (auto iter = eviction_candidates.begin();
             iter != eviction_candidates.end(); iter++) {
          memsize += (*iter)->GetNodeByteSize();
        }
        // device_candidate.push_back(
        //     std::make_pair(memsize, CUDA_DEVICE(device_id)));
        if (memsize < min_eviction_size) {
          min_eviction_size = memsize;
          target_device = CUDA_DEVICE(device_id);
        }
      }
    }
  }

  if (min_eviction_size == UINT64_MAX) {
    // cannot evict any model, let caller handle this
    return std::make_tuple(eviction_candidates, false, DISK_DEVICE);
  }

  // std::sort(device_candidate.begin(), device_candidate.end());
  // std::reverse(device_candidate.begin(), device_candidate.end());
  // auto target_device = device_candidate[0].second;
  allocator = device_allocator_.at(target_device);
  std::tie(eviction_candidates, is_evictable) =
      allocator->GetEvictionCadidates(size);
  // for (auto iter = eviction_candidates.begin();
  //      iter != eviction_candidates.end(); iter++) {
  //   // FreeMemory(iter->first, iter->second, allocator->GetDevice());
  //   allocator->PrepareFreeMemory(iter->GetNodeID());
  // }
  if (eviction_candidates.empty())
    allocator->AllocMemory(node);
  return std::make_tuple(eviction_candidates, true, target_device);
}

std::tuple<EvictionCandidates, bool, Device>
MemoryController::FreeMemory(const DAGNodePtr& node)
{
  // auto key = node->GetNodeID();
  // auto size = node->GetNodeByteSize();
  auto origin_device = node->GetDevice();

  auto origin_allocator = device_allocator_.at(origin_device);
  auto target_device = (origin_device.is_cpu()) ? DISK_DEVICE : CPU_DEVICE;
  auto target_allocator = device_allocator_.at(target_device);


  if (origin_device.is_cpu()) {
    origin_allocator->FreeMemory(node);
    target_allocator->AllocMemory(node);
    EvictionCandidates eviction_candidates;
    bool is_evictable = true;
    return std::make_tuple(eviction_candidates, is_evictable, target_device);
  }

  if (origin_device.is_cuda()) {
    origin_allocator->FreeMemory(node);
    return AllocMemory(node, CPU_DEVICE);
  }
}

// void
// MemoryController::MoveMemory(
//     const std::size_t& key, const std::size_t& size, const Device&
//     src_device, const Device& dst_device)
// {
//   auto src_iter = device_allocator_.find(src_device);
//   auto dst_iter = device_allocator_.find(dst_device);
//   if (src_iter == device_allocator_.end() ||
//       dst_iter == device_allocator_.end()) {
//     return;
//   }
//   src_iter->second->FreeMemory(key, size);
//   dst_iter->second->AllocMemory(key, size);
// }