#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "dag_node.h"
#include "libtorch_factory.h"
#include "lru_cache.h"

inline int
GetDeviceCount()
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
}

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


/*
This Manager is used to manage the memory usage of the model.
It will try to load the model into the GPU memory if there is enough memory.
If there is not enough memory, it will wait for memory availibility.
*/
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
    std::lock_guard<std::mutex> lock(memory_mutex_);
    if (memory_size > free_memory_) {
      return -1;
    }
    free_memory_ -= memory_size;
    return 0;
  }

  int AddKeyValue(const KeyType& key, const ValueType& value)
  {
    std::lock_guard<std::mutex> lock(kv_mutex_);
    if (kv_map_.Get(key) != nullptr) {
      return -1;
    }
    // key_lru_list_.push_front(key);
    kv_map_.Put(key, value);
    return 0;
  }
  int ReleaseMemory(const std::size_t& memory_size)
  {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    free_memory_ += memory_size;
    if (free_memory_ > memory_limit_) {
      free_memory_ = memory_limit_;
    }
    return 0;
  }
  int RemoveKeyValue(const KeyType& key)
  {
    std::lock_guard<std::mutex> lock(kv_mutex_);
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

  std:size_t GetCPUFreeMemory() { return cpu_manager_->GetFreeMemory(); }
  std:size_t GetGPUFreeMemory(int device_id)
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

    value->SetDevice(device_type, device_id);

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
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_manager_->ReleaseObject(key, memory_size);
    for (auto& gpu_manager : gpu_managers_) {
      gpu_manager->ReleaseObject(key, memory_size);
    }
  }
  std::shared_ptr<MemoryManager<std::size_t, DAGNodePtr>> cpu_manager_;
  std::vector<std::shared_ptr<MemoryManager<std::size_t, DAGNodePtr>>>
      gpu_managers_;
  std::mutex mutex_;
};
