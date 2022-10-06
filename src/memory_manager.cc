#include "memory_manager.h"

#include <algorithm>

#include "dag_registry.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace pytorch {

RetCode
MemoryAllocator::AllocMemory(const DAGNodePtr& node)
{
  auto memory_size = node->GetNodeByteSize();
  RETURN_IF_NOT_SUCCESS(TryAllocMemory(node));
  free_memory_ -= memory_size;
  memory_map_.insert(std::make_pair(node->GetNodeID(), node));
  LOG_VERBOSE((std::string("Allocated ") + std::to_string(memory_size / MB) +
               " MB for node " + node->GetModelInstanceInfo() +
               ". Free memory is " + std::to_string(free_memory_ / MB) + " MB.")
                  .c_str());
  return RetCode::RET_SUCCESS;
}

RetCode
MemoryAllocator::TryAllocMemory(const DAGNodePtr& node)
{
  auto memory_size = node->GetNodeByteSize();
  if (memory_size > free_memory_) {
    LOG_VERBOSE((std::string("Not enough memory to allocate ") +
                 std::to_string(memory_size / MB) + " MB for node " +
                 node->GetModelInstanceInfo() + ". Free memory is " +
                 std::to_string(free_memory_ / MB) + " MB.")
                    .c_str());
    return RetCode::RET_NOT_ENOUGH_MEMORY;
  }
  return RetCode::RET_SUCCESS;
}
// int
// MemoryAllocator::AllocMemory(const std::size_t& key)
// {
//   auto iter = memory_map_.find(key);
//   if (iter == memory_map_.end()) {
//     return -1;
//   }
//   auto memory_size = iter->second->GetNodeByteSize();
//   if (memory_size > free_memory_) {
//     return 1;
//   }
//   free_memory_ -= memory_size;
//   return 0;
// }

RetCode
MemoryAllocator::FreeMemory(const DAGNodePtr& node)
{
  auto memory_size = node->GetNodeByteSize();
  auto iter = memory_map_.find(node->GetNodeID());
  if (iter == memory_map_.end()) {
    LOG_VERBOSE((std::string("Node ") + node->GetModelInstanceInfo() +
                 " is not allocated memory." + " Free memory is " +
                 std::to_string(free_memory_ / MB) + " MB.")
                    .c_str());
    return RetCode::RET_NOT_ALLOCATED;
  }
  free_memory_ += memory_size;
  // node->SetMemoryState(MemoryState::FINISHED);
  memory_map_.erase(node->GetNodeID());
  LOG_VERBOSE((std::string("Released ") + std::to_string(memory_size / MB) +
               " MB for node " + node->GetModelInstanceInfo() +
               ". Free memory is " + std::to_string(free_memory_ / MB) + " MB.")
                  .c_str());
  return RetCode::RET_SUCCESS;
}
// int
// MemoryAllocator::FreeMemory(const std::size_t& key)
// {
//   auto iter = memory_map_.find(key);
//   if (iter == memory_map_.end()) {
//     return -1;
//   }
//   auto memory_size = iter->second->GetNodeByteSize();
//   free_memory_ += memory_size;
//   memory_map_.erase(key);
//   return 0;
// }

void
MemoryAllocator::SetDevice(const std::size_t& key, const torch::Device& device)
{
  if (memory_map_.find(key) != memory_map_.end()) {
    memory_map_[key]->SetDevice(device);
  }
}

// void
// MemoryAllocator::SetDevice(
//     const std::size_t& key, const DeviceType& device_type, const int&
//     device_id)
// {
//   if (memory_map_.find(key) != memory_map_.end()) {
//     memory_map_[key]->SetDevice(device_type, device_id);
//   }
// }

std::size_t
MemoryAllocator::GetRandomKey()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, memory_map_.size() - 1);
  auto random_index = dis(gen);
  auto it = memory_map_.begin();
  std::advance(it, random_index);
  return it->first;
}

std::vector<DAGNodePtr>
MemoryAllocator::GetNodeWithState(const MemoryState& state)
{
  std::vector<DAGNodePtr> nodes;
  for (auto& node : memory_map_) {
    if (node.second->IsMemoryState(state)) {
      nodes.push_back(node.second);
    }
  }
  return nodes;
}


DynamicMemoryBatcher::DynamicMemoryBatcher()
{
  batcher_thread_ = std::thread(&DynamicMemoryBatcher::BatcherThread, this);
  auto cpu_memory_total = GetTotalSystemMemory();
  auto cpu_memory_free = GetFreeSystemMemory();
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Total System Memory: ") +
       std::to_string(cpu_memory_total / MB) + std::string(" MB"))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Free System Memory: ") +
       std::to_string(cpu_memory_free / MB) + std::string(" MB"))
          .c_str());

  device_allocator_.insert(
      {CPU_DEVICE, std::make_shared<MemoryAllocator>(
                       CPU_DEVICE.str(), cpu_memory_total, cpu_memory_free)});

  device_allocator_.insert(
      {DISK_DEVICE, std::make_shared<MemoryAllocator>(
                        DISK_DEVICE.str(), UINT64_MAX, UINT64_MAX)});

  // create MemoryAllocator for all gpu devices
  for (int i = 0; i < GetDeviceCount(); i++) {
    auto gpu_memory_total = GetTotalDeviceMemory(i);
    auto gpu_memory_free = GetFreeDeviceMemory(i);
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Total GPU Memory: ") +
         std::to_string(gpu_memory_total / MB) + std::string(" MB") +
         std::string(" on device ") + std::to_string(i))
            .c_str());
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Free GPU Memory: ") +
         std::to_string(gpu_memory_free / MB) + std::string(" MB") +
         std::string(" on device ") + std::to_string(i))
            .c_str());
    device_allocator_.insert(
        {CUDA_DEVICE(i),
         std::make_shared<MemoryAllocator>(
             CUDA_DEVICE(i).str(), gpu_memory_total, gpu_memory_free)});
  }
}

int
DynamicMemoryBatcher::Enqueue(RequestPtr request)
{
  LOG_VERBOSE(
      (request->node->GetModelInstanceInfo() + " waiting to be enqueued.")
          .c_str());
  {
    // std::lock_guard<std::mutex> batcher_lock(batcher_mutex_);
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    request_queue_.push_front(request);
    LOG_VERBOSE((std::string("Enqueue request ") +
                 ManageTypeToString(request->manage_type) + " for node " +
                 request->node->GetModelInstanceInfo() + " to batcher queue.")
                    .c_str());
  }
  // wake up queue to process data
  // queue_cv_.notify_one();

  return 0;
}


void
DynamicMemoryBatcher::BatcherThread()
{
  std::deque<RequestPtr> req_queue;
  while (true) {
    if (request_queue_.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    {
      // wait for the queue to have data
      // std::unique_lock<std::mutex> batcher_lock(batcher_mutex_);
      // queue_cv_.wait(batcher_lock, [this] { return !request_queue_.empty();
      // });
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
      req_queue.clear();

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE, (std::string("BatcherThread: ") +
                                     std::to_string(request_queue_.size()) +
                                     std::string(" requests in queue"))
                                        .c_str());
      req_queue.swap(request_queue_);
      // // hold the lock for short period to swap the queue
      // for (auto& request : request_queue_) {
      //   if (request->node->GetDevice().is_cuda()) {
      //     // already on gpu, just run
      //     Notify(request, MemoryState::READY);
      //     continue;
      //   }
      //   // if the device is different, move the request to the new queue
      //   req_queue.push_back(request);
      // }
      // request_queue_.clear();  // clear the queue after swap
    }


    // process the new queue
    std::vector<RequestPtr> front_queue, back_queue;
    for (auto& request : req_queue) {
      auto node = request->node;
      auto device = node->GetDevice();
      auto target_device = request->device;

      LOG_VERBOSE((node->GetModelInstanceInfo() +
                   " is being processed. From device" + device.str() +
                   " to device " + target_device.str() + ". With request " +
                   ManageTypeToString(request->manage_type))
                      .c_str());

      if (request->manage_type == ManageType::RELEASE) {
        LOG_VERBOSE((std::string("BatcherThread: ") + std::string("release ") +
                     std::to_string(node->GetNodeID()) + std::string(" on ") +
                     device.str())
                        .c_str());
        if (!request->node->IsMemoryState(MemoryState::ACTIVE)) {
          device_allocator_[device]->FreeMemory(node);
          device_allocator_[target_device]->AllocMemory(node);
          node->SetDevice(target_device);
          Notify(request, MemoryState::INACTIVE);
        }
      }1

      if (request->manage_type == ManageType::ON_DEMAND) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") +
             std::string("Allocating on-demand ") +
             std::to_string(node->GetNodeID()) + std::string(" to GPU"))
                .c_str());
        device_allocator_[DEFAULT_CUDA_DEVICE]->AllocMemory(node);
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") +
             std::string("Allocating on-demand ") +
             std::to_string(node->GetNodeID()) + std::string(" free CPU Mem"))
                .c_str());
        device_allocator_[CPU_DEVICE]->FreeMemory(node);
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") +
             std::string("Allocating on-demand ") +
             std::to_string(node->GetNodeID()) + std::string(" free DISK Mem"))
                .c_str());
        device_allocator_[DISK_DEVICE]->FreeMemory(node);
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") +
             std::string("Allocating on-demand ") +
             std::to_string(node->GetNodeID()) + std::string(" SetDevice GPU"))
                .c_str());
        node->SetDevice(DEFAULT_CUDA_DEVICE);
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE, (std::string("BatcherThread: ") +
                                       std::string("Allocating on-demand ") +
                                       std::to_string(node->GetNodeID()) +
                                       std::string(" Notify wait for memory"))
                                          .c_str());
        Notify(request, MemoryState::ACTIVE);

        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") + std::string("Prefetching from ") +
             std::to_string(node->GetNodeID()))
                .c_str());

        PrefetchFromKey(node->GetNodeID());
      }

      if (request->manage_type == ManageType::PREFETCH && !device.is_cuda()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("BatcherThread: ") + std::string("Prefetching ") +
             std::to_string(node->GetNodeID()) + std::string(" to CPU"))
                .c_str());
        device_allocator_[DISK_DEVICE]->FreeMemory(node);
        device_allocator_[DEFAULT_CUDA_DEVICE]->AllocMemory(node);
        node->SetDevice(DEFAULT_CUDA_DEVICE);
        Notify(request, MemoryState::READY);
      }
    }

    //     // process the request lock free
    //     std::unordered_multimap<std::size_t, RequestPtr> batch_map;
    //     std::vector<std::size_t> key_set;
    //     for (auto& request : req_queue) {
    //       auto model_key = request->node->GetNodeID();
    //       batch_map.insert({model_key, request});
    //       key_set.push_back(model_key);
    //     }


    //     /*
    //      * Rules for Memory Allocation
    //      * 1. ON_DEMAND: allocation has the top priority, if allocation is
    //      not
    //      * successfull, this thread will be waiting (no GPU memory availble).
    //      * 2. PREFETCH: fail of allocation will be placed back into the queue
    //      * 3. RELEASE: will be overriden by ON_DEMAND and PREFETCH
    //      */


    //     std::vector<RequestPtr> front_queue, back_queue;
    //     std::unordered_set<std::size_t> running_key_set;
    //     for (auto& key : key_set) {
    //       auto range = batch_map.equal_range(key);
    //       std::uint32_t release = 0;
    //       for (auto it = range.first; it != range.second; ++it) {
    //         release = release |
    //         static_cast<std::uint32_t>(it->second->manage_type);
    //       }

    //       auto node = GET_INSTANCE(DAGRegistry)->GetNode(key);

    //       // Is a real release only if there is no other request
    //       if (release == 0) {
    //         LOG_MESSAGE(
    //             TRITONSERVER_LOG_INFO,
    //             (std::string("BatcherThread: ") +
    //             node->GetModelInstanceInfo() +
    //              std::string(" is a real release"))
    //                 .c_str());
    //         ReleaseMemory(key);
    //         // need to notify all the requests
    //         for (auto it = range.first; it != range.second; ++it) {
    //           Notify(it->second, MemoryState::INACTIVE);
    //         }
    //         continue;
    //       }

    //       // If there is a release, it is now overriden by ON_DEMAND and
    //       PREFETCH
    //       // Just skip the release for simplicity

    //       for (auto it = range.first; it != range.second; ++it) {
    //         if (it->second->manage_type == ManageType::RELEASE) {
    //           LOG_MESSAGE(
    //               TRITONSERVER_LOG_INFO, (std::string("BatcherThread: ") +
    //                                       it->second->node->GetModelInstanceInfo()
    //                                       + std::string(" skipping a
    //                                       release"))
    //                                          .c_str());
    //           Notify(it->second, MemoryState::INACTIVE);
    //           continue;
    //         }


    //         // auto device_it =
    //         device_map_.find(it->second->node->GetNodeID());
    //         // if (device_it != device_map_.end()) {
    //         //   if (device_it->second == it->second->device) {
    //         //     // if the device is the same, just notify the model to
    //         continue
    //         //     Notify(it->second);
    //         //     continue;
    //         //   }
    //         // }

    //         for (int device_id = 0; device_id < GetDeviceCount();
    //         device_id++) {
    //           auto target_device = CUDA_DEVICE(device_id);
    //           if (device == it->second->device) {
    //             // if the device is the same, just notify the model to
    //             continue Notify(it->second, MemoryState::READY); continue;
    //           }
    //         }

    //         auto target_device = it->second->device;
    //         auto& target_allocator = device_allocator_.at(target_device);

    //         LOG_MESSAGE(
    //             TRITONSERVER_LOG_INFO,
    //             (std::string("BatcherThread: ") +
    //              it->second->node->GetModelInstanceInfo() +
    //              std::string(" is requesting allocation on ") +
    //              target_device.str() + std::string(" with size ") +
    //              std::to_string(it->second->node->GetNodeByteSize() / MB) +
    //              std::string(" MB"))
    //                 .c_str());

    //         auto alloc_success =
    //         target_allocator->AllocMemory(it->second->node); if
    //         (alloc_success == RetCode::RET_SUCCESS) {
    //           LOG_MESSAGE(
    //               TRITONSERVER_LOG_VERBOSE,
    //               (std::string("BatcherThread: ") +
    //                it->second->node->GetModelInstanceInfo() +
    //                std::string(" allocation success"))
    //                   .c_str());
    //           it->second->node->SetDevice(target_device);
    //           // it->second->node->SetMemoryState(MemoryState::READY);

    //           auto device_it = device_map_.find(it->first);
    //           if (device_it != device_map_.end()) {
    //             LOG_MESSAGE(
    //                 TRITONSERVER_LOG_INFO,
    //                 (std::string("BatcherThread: ") +
    //                  it->second->node->GetModelInstanceInfo() +
    //                  std::string(" device changed from ") +
    //                  device_it->second.str() + std::string(" to ") +
    //                  target_device.str())
    //                     .c_str());
    //             auto origin_device = device_it->second;
    //             auto origin_allocator = device_allocator_.at(origin_device);
    //             origin_allocator->FreeMemory(it->second->node);
    //             device_it->second = target_device;
    //           } else {
    //             LOG_MESSAGE(
    //                 TRITONSERVER_LOG_INFO,
    //                 (std::string("BatcherThread: ") +
    //                  it->second->node->GetModelInstanceInfo() +
    //                  std::string(" device set to ") + target_device.str())
    //                     .c_str());
    //             device_map_.insert({it->first, target_device});
    //           }

    //           Notify(it->second, MemoryState::READY);
    //         }

    //         // FIXME: naive implementation, just find a GPU with enough
    //         memory
    //         // int num_devices = GetDeviceCount();
    //         // RetCode alloc_success = RetCode::RET_FAILED;
    //         // for (int i = 0; i < num_devices; i++) {
    //         //   auto device = torch::Device(torch::kCUDA, i);
    //         //   auto& allocator = device_allocator_[device];
    //         //   alloc_success = allocator->AllocMemory(it->second->node);
    //         //   if (alloc_success == RetCode::RET_SUCCESS) {
    //         //     it->second->node->SetDevice(device);
    //         //     auto device_it = device_map_.find(it->first);
    //         //     if (device_it != device_map_.end())
    //         //       device_it->second = device;
    //         //     else
    //         //       device_map_.insert({it->first, device});
    //         //     Notify(it->second);
    //         //     running_key_set.insert(it->first);
    //         //     break;
    //         //   }
    //         // }
    //         if (alloc_success != RetCode::RET_SUCCESS) {
    //           LOG_MESSAGE(
    //               TRITONSERVER_LOG_INFO,
    //               (std::string("BatcherThread: ") +
    //                it->second->node->GetModelInstanceInfo() +
    //                std::string(" allocation failed on device ") +
    //                target_device.str())
    //                   .c_str());
    //           if (it->second->manage_type == ManageType::ON_DEMAND) {
    //             // if ON_DEMAND allocation is not successfull, this request
    //             has the
    //             // most priorityat front of request_queue.
    //             front_queue.push_back(it->second);
    //           } else {
    //             // if PREFETCH allocation is not successfull, this request
    //             has the
    //             // least priority at the back of request_queue.
    //             back_queue.push_back(it->second);
    //           }
    //         }
    //       }
    //     }

    //     std::unordered_set<std::size_t> gpu_key_set;
    //     for (auto& dpair : device_map_) {
    //       if (dpair.second.is_cuda()) {
    //         gpu_key_set.insert(dpair.first);
    //         if (GET_INSTANCE(DAGRegistry)
    //                 ->GetNode(dpair.first)
    //                 ->IsMemoryState(MemoryState::RUNNING)) {
    //           running_key_set.insert(dpair.first);
    //         }
    //       }
    //     }

    //     // only fetch the immedient child for now, regardless of GPU load and
    //     // probability
    //     std::vector<ModelProbabilityVec> model_prob_map_list;
    //     std::unordered_set<std::size_t> prefetch_key_set;
    //     for (auto& key : running_key_set) {
    //       auto prob_vec =
    //       GET_INSTANCE(ModelFlowRecorder)->GetTreeProbability(key);
    //       model_prob_map_list.push_back(prob_vec);
    //       for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
    //         prefetch_key_set.insert(prob_vec[i].first);
    //       }
    //     }

    //     std::unordered_set<std::size_t> diff;
    //     std::set_difference(
    //         prefetch_key_set.begin(), prefetch_key_set.end(),
    //         gpu_key_set.begin(), gpu_key_set.end(), std::inserter(diff,
    //         diff.begin()));

    //     for (auto& key : diff) {
    //       back_queue.push_back(std::make_shared<MemoryManageRequest>(
    //           GET_INSTANCE(DAGRegistry)->GetNode(key),
    //           torch::Device(torch::kCUDA), ManageType::PREFETCH));
    //     }

    //     {
    //       std::unique_lock<std::mutex> lock(queue_mutex_);
    //       for (auto& request : front_queue) {
    //         request_queue_.push_front(request);
    //       }
    //       for (auto& request : back_queue) {
    //         request_queue_.push_back(request);
    //       }
    //     }
  }
}

void
DynamicMemoryBatcher::PrefetchFromKey(const std::size_t& key)
{
  std::vector<RequestPtr> prefetch_queue;
  auto prob_vec = GET_INSTANCE(ModelFlowRecorder)->GetTreeProbability(key);
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("BatcherThread: ") +
       GET_INSTANCE(DAGRegistry)->GetNode(key)->GetModelInstanceInfo() +
       std::string(" prefetching ") + std::to_string(prob_vec.size()) +
       std::string(" models"))
          .c_str());
  for (std::size_t i = 0; i < prob_vec.size() && i < 10; i++) {
    auto node = GET_INSTANCE(DAGRegistry)->GetNode(prob_vec[i].first);
    prefetch_queue.push_back(std::make_shared<MemoryManageRequest>(
        node, DEFAULT_CUDA_DEVICE, ManageType::PREFETCH));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("BatcherThread: ") + node->GetModelInstanceInfo() +
         std::string(" prefetch request added"))
            .c_str());
  }

  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    for (auto& request : prefetch_queue) {
      request_queue_.push_back(request);
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("BatcherThread: ") +
         GET_INSTANCE(DAGRegistry)->GetNode(key)->GetModelInstanceInfo() +
         std::string(" prefetching done. Request queue size ") +
         std::to_string(request_queue_.size()))
            .c_str());
  }
}


// void
// DynamicMemoryBatcher::ReleaseMemory(const RequestPtr& request)
// {
//   auto node = request->node;
//   if (node->GetDevice().is_cuda()) {
//     MoveMemory(request, CPU_DEVICE);
//   }

//   if (node->GetDevice().is_cpu()) {
//     MoveMemory(request, DISK_DEVICE;
//   }

//   node->SetMemoryState(MemoryState::INACTIVE);
// }

// void
// DynamicMemoryBatcher::FetchMemory(const RequestPtr& request)
// {
//   auto node = request->node;
//   if (node->GetDevice().is_cuda()) {
//     MoveMemory(request, CPU_DEVICE);
//   }

//   if (node->GetDevice().is_cpu()) {
//     MoveMemory(request, DISK_DEVICE;
//   }

//   node->SetMemoryState(MemoryState::INACTIVE);
// }

// void
// DynamicMemoryBatcher::ReleaseMemory(const std::size_t& key)
// {
//   // auto key = request->node->GetNodeID();
//   auto node = GET_INSTANCE(DAGRegistry)->GetNode(key);

//   if (node->GetDevice().is_cuda()) {
//     MoveMemory(key, node->GetDevice(), CPU_DEVICE);
//   }

//   if (node->GetDevice().is_cpu()) {
//     MoveMemory(key, node->GetDevice(), DISK_DEVICE;
//   }

//   node->SetMemoryState(MemoryState::INACTIVE);

//   // auto device_it = device_map_.find(key);
//   // if (device_it == device_map_.end()) {
//   //   return;  // no memory allocated, already released
//   // }
//   // auto device = device_it->second;
//   // if (!device.is_cuda()) {
//   //   return;  // if the device is not on GPU, no need to release memory
//   // }

//   // // First remove from gpu and skip the CPU part
//   // auto gpu_allocator = device_allocator_[device];
//   // gpu_allocator->FreeMemory(node);
//   // gpu_allocator->SetDevice(key, CPU_DEVICE);

//   // // try add to CPU MemoryAllocator
//   // auto cpu_allocator = device_allocator_[CPU_DEVICE];
//   // while (cpu_allocator->AllocMemory(node) != RetCode::RET_SUCCESS) {
//   //   // randomly remove models from CPU MemoryAllocator until this key fits
//   //   auto rnd_key = cpu_allocator->GetRandomKey();
//   //   auto rnd_node = GET_INSTANCE(DAGRegistry)->GetNode(rnd_key);
//   //   cpu_allocator->SetDevice(rnd_key, DISK_DEVICE);
//   //   cpu_allocator->FreeMemory(rnd_node);
//   //   device_map_.erase(rnd_key);
//   // }

//   // gpu_allocator->FreeMemory(node);
// }

// void
// DynamicMemoryBatcher::MoveMemory(
//     const RequestPtr& request, const torch::Device& dst_device)
// {
//   auto node = request->node;
//   auto src_allocator = device_allocator_[src_device];
//   auto dst_allocator = device_allocator_[dst_device];

//   // Remove from src_device
//   src_allocator->FreeMemory(node);

//   // Add to dst_device
//   while (dst_allocator->AllocMemory(node) != RetCode::RET_SUCCESS) {
//     // find a random key to remove from dst_device
//     auto evict_node = dst_allocator->GetNodeWithState(MemoryState::INACTIVE);
//     if (evict_node == nullptr) {
//       std::this_thread::sleep_for(std::chrono::milliseconds(10));
//       continue;
//     }
//     dst_allocator->FreeMemory(evict_node);
//     // FIXME: Need to implement a better way to evict, using multi-tier
//     memory evict_node->SetDevice(dst_device);
//   }

//   node->SetDevice(dst_device);
// }

// void
// DynamicMemoryBatcher::MoveMemory(
//     const std::size_t& key, const torch::Device& src_device,
//     const torch::Device& dst_device)
// {
//   auto node = GET_INSTANCE(DAGRegistry)->GetNode(key);
//   auto src_allocator = device_allocator_[src_device];
//   auto dst_allocator = device_allocator_[dst_device];

//   // Remove from src_device
//   src_allocator->FreeMemory(node);

//   // Add to dst_device
//   while (dst_allocator->AllocMemory(node) != RetCode::RET_SUCCESS) {
//     // find a random key to remove from dst_device
//     auto evict_node = dst_allocator->GetNodeWithState(MemoryState::INACTIVE);
//     if (evict_node == nullptr) {
//       std::this_thread::sleep_for(std::chrono::milliseconds(10));
//       continue;
//     }
//     dst_allocator->FreeMemory(evict_node);
//     // FIXME: Need to implement a better way to evict, using multi-tier
//     memory evict_node->SetDevice(dst_device);
//   }

//   node->SetDevice(dst_device);
// }

}}}  // namespace triton::backend::pytorch