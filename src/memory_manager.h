#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "dag_node.h"
#include "dag_registry.h"
#include "libtorch_factory.h"
#include "libtorch_flow.h"
#include "lru_cache.h"

// class ModelDevice {
//  public:
//   ModelDevice()
//       : device_id_(DefaultDeviceID::INVALID), device_path_(""),
//         device_type_(DeviceType::INVALID)
//   {
//   }
//   ModelDevice(
//       int device_id, const std::string& device_path,
//       const DeviceType& device_type)
//       : device_id_(device_id), device_path_(device_path),
//         device_type_(device_type)
//   {
//   }
//   ~ModelDevice() = default;
//   const std::string& GetDevicePath() const { return device_path_; }
//   int GetDeviceId() const { return device_id_; }
//   const DeviceType& GetDeviceType() const { return device_type_; }
//   bool operator==(const ModelDevice& other) const
//   {
//     return (device_id_ == other.device_id_) &&
//            (device_type_ == other.device_type_) &&
//            (device_path_ == other.device_path_);
//   }
//   bool operator!=(const ModelDevice& other) const { return !(*this == other);
//   }

//  private:
//   int device_id_;
//   std::string device_path_;  // path point to the checkpoint path
//   DeviceType device_type_;
// };


// class ModelDeviceCPU : public ModelDevice {
//  public:
//   ModelDeviceCPU()
//       : ModelDevice(DefaultDeviceID::CPU, std::string(""), DeviceType::CPU)
//   {
//   }
//   ModelDeviceCPU(int device_id, const std::string& device_path)
//       : ModelDevice(device_id, device_path, DeviceType::CPU)
//   {
//   }
// };

// class ModelDeviceGPU : public ModelDevice {
//  public:
//   ModelDeviceGPU()
//       : ModelDevice(DefaultDeviceID::GPU, std::string(""), DeviceType::GPU)
//   {
//   }
//   ModelDeviceGPU(int device_id, const std::string& device_path)
//       : ModelDevice(device_id, device_path, DeviceType::GPU)
//   {
//   }
// };

// class DiskDevice : public ModelDevice {
//  public:
//   DiskDevice()
//       : ModelDevice(DefaultDeviceID::DISK, std::string(""), DeviceType::DISK)
//   {
//   }
//   DiskDevice(int device_id, const std::string& device_path)
//       : ModelDevice(device_id, device_path, DeviceType::DISK)
//   {
//   }
// };

// inline long int
// sysinfo_mempages(unsigned long int num, unsigned int mem_unit)
// {
//   unsigned long int ps = sysconf(_SC_PAGE_SIZE);
//   while (mem_unit > 1 && ps > 1) {
//     mem_unit >>= 1;
//     ps >>= 1;
//   }
//   num *= mem_unit;
//   while (ps > 1) {
//     ps >>= 1;
//     num >>= 1;
//   }
//   return num;
// }

inline std::size_t
GetTotalSystemMemory()
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

inline std::size_t
GetFreeSystemMemory()
{
  // struct sysinfo memInfo;
  // sysinfo(&memInfo);
  // return sysinfo_mempages(memInfo.totalram, memInfo.mem_unit) +
  //        sysinfo_mempages(memInfo.bufferram, memInfo.mem_unit);
  // long long totalVirtualMem = memInfo.totalram;
  // long pages = sysconf(_SC_AVPHYS_PAGES);
  // long page_size = sysconf(_SC_PAGE_SIZE);
  // return pages * page_size;

  // This is some how hacky, but _SC_AVPHYS_PAGES does not give us cached memory
  // This assume that we are the only one using the system memory
  return GetTotalSystemMemory() * 0.7;

  // FILE* meminfo = fopen("/proc/meminfo", "r");
  // if (meminfo == NULL)
  //   return 0;
  // char line[256];
  // while (fgets(line, sizeof(line), meminfo)) {
  //   unsigned int ram;
  //   if (sscanf(line, "MemAvailable: %d kB", &ram) == 1) {
  //     fclose(meminfo);
  //     return ram * 1024;
  //   }
  // }

  // // If we got here, then we couldn't find the proper line in the meminfo
  // file:
  // // do something appropriate like return an error code, throw an exception,
  // // etc.
  // fclose(meminfo);
  // return 0;
}

inline std::size_t
GetTotalDeviceMemory(int device_id)
{
  size_t free_memory, total_memory;
  cudaSetDevice(device_id);
  cudaMemGetInfo(&free_memory, &total_memory);
  return total_memory;
}

inline std::size_t
GetFreeDeviceMemory(int device_id)
{
  size_t free_memory, total_memory;
  cudaSetDevice(device_id);
  cudaMemGetInfo(&free_memory, &total_memory);
  return free_memory;
}

inline int
GetDeviceCount()
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
}

namespace triton { namespace backend { namespace pytorch {

struct MemoryManageRequest {
  DAGNodePtr node;
  std::mutex mutex;
  std::condition_variable cv;
  torch::Device device;
  ManageType manage_type;

  MemoryManageRequest(
      const DAGNodePtr& node, const torch::Device& device,
      const ManageType& manage_type)
      : node(node), device(device), manage_type(manage_type)
  {
    mutex.lock();  // lock the mutex while constructing, the mutex is released
                   // after the request has been handled. The ModelInstanceState
                   // will wait for this lock.
  }
};
typedef std::shared_ptr<MemoryManageRequest> RequestPtr;

inline void
WaitForMemoryManageRequest(
    const RequestPtr& request, const MemoryState& memory_state)
{
  std::unique_lock<std::mutex> lock(request->mutex);
  request->cv.wait(lock, [&request, &memory_state] {
    return request->node->IsMemoryState(memory_state) &&
           (request->device.is_cuda() == request->node->GetDevice().is_cuda());
  });
}

// This class is always accessed by the same thread, so no need to protect it
class MemoryAllocator {
 public:
  DISABLE_COPY_AND_ASSIGN(MemoryAllocator);
  MemoryAllocator() = delete;
  MemoryAllocator(
      const std::string& name, std::size_t memory_limit,
      std::size_t free_memory)
      : name_(name), memory_limit_(memory_limit), free_memory_(free_memory)
  {
  }
  ~MemoryAllocator() = default;
  typedef std::pair<std::size_t, DAGNodePtr> MemoryPair;

  RetCode AllocMemory(const DAGNodePtr& node);
  RetCode TryAllocMemory(const DAGNodePtr& node);
  // RetCode AllocMemory(const std::size_t& key);
  RetCode FreeMemory(const DAGNodePtr& node);
  // RetCode FreeMemory(const std::size_t& key);

  void SetDevice(const std::size_t& key, const torch::Device& device);

  // void SetDevice(
  //     const std::size_t& key, const DeviceType& device_type,
  //     const int& device_id);

  std::size_t GetRandomKey();
  std::vector<DAGNodePtr> GetNodeWithState(const MemoryState& state);

 private:
  std::string name_;
  std::unordered_map<std::size_t, DAGNodePtr> memory_map_;
  std::size_t memory_limit_;
  std::size_t free_memory_;
};

class DynamicMemoryBatcher : public SingletonFactory {
 public:
  DISABLE_COPY_AND_ASSIGN(DynamicMemoryBatcher)
  FACTORY_STATIC_GET_INSTANCE(DynamicMemoryBatcher)


  int Enqueue(RequestPtr request);


 private:
  DynamicMemoryBatcher();
  ~DynamicMemoryBatcher() = default;

  void BatcherThread();
  void PrefetchFromKey(const std::size_t& key);

  // void MoveMemory(const std::size_t& key, const torch::Device& src_device,
  // const torch::Device& dst_device); void MoveMemory(const RequestPtr&
  // request, const torch::Device& dst_device); void ReleaseMemory(const
  // std::size_t& key); void ReleaseMemory(const RequestPtr& request); void
  // FetchMemory(const RequestPtr& request);
  void Notify(const RequestPtr& request, const MemoryState& state)
  {
    request->node->SetMemoryState(state);
    request->mutex.unlock();
    request->cv.notify_one();
  }

  std::deque<RequestPtr> request_queue_;
  std::thread batcher_thread_;
  std::condition_variable queue_cv_;
  std::mutex queue_mutex_;
  std::mutex batcher_mutex_;
  std::unordered_map<torch::Device, std::shared_ptr<MemoryAllocator>>
      device_allocator_;
  std::unordered_map<std::size_t, torch::Device> device_map_;
};

/*
This Manager is used to manage the memory usage of the model.
It will try to load the model into the GPU memory if there is enough memory.
If there is not enough memory, it will wait for memory availibility.
*/

/*
template <typename KeyType, typename ValueType>
class MemoryManager : public SingletonFactory {
 public:
  FACTORY_STATIC_GET_INSTANCE(MemoryManager)
  DISABLE_COPY_AND_ASSIGN(MemoryManager)


  MemoryManager();
  ~MemoryManager();

  inline std::size_t GetTotalSystemMemory()
  {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
  }

  inline std::size_t GetFreeSystemMemory()
  {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
  }

  inline std::size_t GetUsedSystemMemory()
  {
    return GetTotalSystemMemory() - GetFreeSystemMemory();
  }

  inline std::vector<std::size_t> GetTotalGPUMemory()
  {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    std::vector<std::size_t> gpu_memory(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice(i);
      cudaMemGetInfo(&free, &total);
      gpu_memory[i] = total;
    }
    return gpu_memory;
  }


  inline std::vector<std::size_t> GetFreeGPUMemory()
  {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    std::vector<std::size_t> gpu_memory(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice(i);
      cudaMemGetInfo(&free, &total);
      gpu_memory[i] = free;
    }
    return gpu_memory;
  }

  inline std::vector<std::size_t> GetUsedGPUMemory()
  {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    std::vector<std::size_t> gpu_memory(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice(i);
      cudaMemGetInfo(&free, &total);
      gpu_memory[i] = total - free;
    }
    return gpu_memory;
  }


  typedef std::pair<KeyType, ValueType> MemoryPair;

  // TODO: model runtime memory usage is different from the model size, we need
  // to consider this.

  int ConsumeObject(
      const KeyType& key, const ValueType& value,
      const std::size_t& memory_size, bool force = false)
  {
    int add_success = AddKeyValue(key, value);

    if (add_success == 0) {
      int consume_success = ConsumeMemory(memory_size);
      if (consume_success < 0) {
        if (!force) {
          // memory is not enough, wait for memory
          std::unique_lock<std::mutex> lock(mgr_mutex_);
          memory_cv_.wait(lock, [this, memory_size]() {
            return ConsumeMemory(memory_size) == 0;
          });
        } else {
          // TODO: force to consume memory by removing objects until free memory
          // is enough
        }
      }
    }


    return 0;
  }

  int ReleaseObject(const KeyType& key, const std::size_t& memory_size)
  {
    int success = RemoveKeyValue(key);
    if (success < 0) {
      return success;
    }
    success = ReleaseMemory(memory_size);
    memory_cv_.notify_all();

    return success;
  }

  std::size_t GetMemoryLimit() const { return memory_limit_; }
  std::size_t GetFreeMemory() const { return free_memory_; }

  ValueType GetObject(const KeyType& key)
  {
    ValueType* it = kv_map_.Get(key);
    if (it != nullptr) {
      return *it;
    }
    return ValueType();
  }

 protected:
  int ConsumeMemory(const std::size_t& memory_size)
  {
    // std::lock_guard<std::mutex> lock(memory_mutex_);
    if (memory_size > free_memory_) {
      return -1;
    }
    free_memory_ -= memory_size;
    return 0;
  }

  int AddKeyValue(const KeyType& key, const ValueType& value)
  {
    // std::lock_guard<std::mutex> lock(kv_mutex_);
    if (kv_map_.Get(key) != nullptr) {
      return -1;
    }
    // key_lru_list_.push_front(key);
    kv_map_.Put(key, value);
    return 0;
  }
  int ReleaseMemory(const std::size_t& memory_size)
  {
    // std::lock_guard<std::mutex> lock(memory_mutex_);
    free_memory_ += memory_size;
    if (free_memory_ > memory_limit_) {
      free_memory_ = memory_limit_;
    }
    return 0;
  }
  int RemoveKeyValue(const KeyType& key)
  {
    // std::lock_guard<std::mutex> lock(kv_mutex_);
    return kv_map_.Remove(key);
  }

  LRUCache<KeyType, ValueType> kv_map_;
  std::size_t memory_limit_;
  std::size_t free_memory_;
  std::mutex memory_mutex_;
  std::mutex kv_mutex_;
  std::mutex mgr_mutex_;
  std::condition_variable memory_cv_;
};

template <typename KeyType, typename ValueType>
class CPUMemoryManager : public MemoryManager<KeyType, ValueType> {
 public:
  CPUMemoryManager()
  {
    this->memory_limit_ = this->GetTotalSystemMemory();
    this->free_memory_ = this->GetFreeSystemMemory();
  }
};

template <typename KeyType, typename ValueType>
class GPUMemoryManager : public MemoryManager<KeyType, ValueType> {
 public:
  GPUMemoryManager()
  {
    this->memory_limit_ = this->GetTotalGPUMemory()[0];
    this->free_memory_ = this->GetFreeGPUMemory()[0];
  }

  GPUMemoryManager(int device_id)
  {
    this->memory_limit_ = this->GetTotalGPUMemory()[device_id];
    this->free_memory_ = this->GetFreeGPUMemory()[device_id];
  }
};


// TODO: current implementation only considers greedy allocation
// TODO: add DiskMemoryManager
class DAGMemoryManager : public SingletonFactory {
 public:
  DISABLE_COPY_AND_ASSIGN(DAGMemoryManager)
  FACTORY_STATIC_GET_INSTANCE(DAGMemoryManager)

  DAGMemoryManager()
      : cpu_manager_(new CPUMemoryManager<std::size_t, DAGNodePtr>()),
        gpu_managers_(GetDeviceCount())
  {
    for (std::size_t i = 0; i < gpu_managers_.size(); ++i) {
      gpu_managers_[i] =
          std::make_shared<GPUMemoryManager<std::size_t, DAGNodePtr>>(i);
    }
  }
  ~DAGMemoryManager();

  std::size_t GetCPUFreeMemory() { return cpu_manager_->GetFreeMemory(); }
  std::size_t GetGPUFreeMemory(int device_id)
  {
    return gpu_managers_[device_id]->GetFreeMemory();
  }

  void AllocateMemory(
      const std::size_t& key, const DAGNodePtr& value,
      const std::size_t& memory_size, const DeviceType& device_type,
      const int& device_id)
  {
    DAGNodePtr cpu_object = nullptr;
    DAGNodePtr gpu_object = nullptr;
    int old_device_id = -1;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      cpu_object = cpu_manager_->GetObject(key);
      for (std::size_t i = 0; i < gpu_managers_.size(); ++i) {
        gpu_object = gpu_managers_[i]->GetObject(key);
        if (gpu_object != nullptr) {
          old_device_id = i;
          break;
        }
      }
    }

    auto node_prob_map =
        GET_INSTANCE(ModelFlowRecorder)->GetTreeProbability(value->GetNodeID());

    int buffer_size = on_the_fly_keys_.size();
    if (buffer_size < 5) {
      int count = 0;
      for (auto& node_prob : node_prob_map) {
        if (node_prob.first == value->GetNodeID()) {
          {
            std::lock_guard<std::mutex> lock(on_the_fly_mutex_);
            on_the_fly_keys_.insert(value->GetNodeID());
          }
          continue;
        }
        DAGNodePtr node = GET_INSTANCE(DAGRegistry)->GetNode(node_prob.first);

        std::async([this, node] {
          {
            std::lock_guard<std::mutex> lock(on_the_fly_mutex_);
            on_the_fly_keys_.insert(node->GetNodeID());
          }
          GET_INSTANCE(DAGMemoryManager)
              ->AllocateMemory(
                  node->GetNodeID(), node, node->GetNodeByteSize(),
                  DeviceType::GPU, 0);
        });
        count++;
        if (count >= 10) {
          break;
        }
      };
    }


    // value->SetDevice(device_type, device_id);

    if (device_type == DeviceType::CPU) {
      if (cpu_object == nullptr) {
        cpu_manager_->ConsumeObject(key, value, memory_size);
      }
    } else if (device_type == DeviceType::GPU) {
      if (gpu_object == nullptr || old_device_id != device_id) {
        gpu_managers_[device_id]->ConsumeObject(key, value, memory_size);
      }
      if (cpu_object != nullptr) {
        cpu_manager_->ReleaseObject(key, memory_size);
      }
      if (gpu_object != nullptr && old_device_id != device_id) {
        gpu_managers_[old_device_id]->ReleaseObject(key, memory_size);
      }
    } else if (device_type == DeviceType::DISK) {
      // TODO: implement disk memory manager
    }
  }

  void ReleaseMemory(const std::size_t& key, const std::size_t& memory_size)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cpu_manager_->ReleaseObject(key, memory_size);
      for (auto& gpu_manager : gpu_managers_) {
        gpu_manager->ReleaseObject(key, memory_size);
      }
    }

    {
      std::lock_guard<std::mutex> lock(on_the_fly_mutex_);
      on_the_fly_keys_.erase(key);
    }
  }
  std::shared_ptr<MemoryManager<std::size_t, DAGNodePtr>> cpu_manager_;
  std::vector<std::shared_ptr<MemoryManager<std::size_t, DAGNodePtr>>>
      gpu_managers_;
  std::unordered_set<std::size_t> on_the_fly_keys_;
  std::mutex on_the_fly_mutex_;
  std::mutex mutex_;
};


*/
}}}  // namespace triton::backend::pytorch