#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/sysinfo.h>
#include <torch/script.h>
#include <unistd.h>

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

#define MODULE_PTR_NODELETE(ptr)        \
  std::shared_ptr<ScriptModule>         \
  {                                     \
    ptr, DoNothingDeleter<ScriptModule> \
    {                                   \
    }                                   \
  }

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
  return GetTotalSystemMemory() * 0.8;

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

inline std::size_t
GetFileSize(const std::string& file_path)
{
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  return file.tellg();
}

// Get Byte size of torch::jit::IValue
inline std::size_t
GetIValueByteSize(const torch::jit::IValue& value)
{
  return value.toTensor().nbytes();
}

inline std::size_t
GetSumOfIValueByteSize(const std::vector<torch::jit::IValue>& values)
{
  std::size_t size = 0;
  for (const auto& value : values) {
    size += GetIValueByteSize(value);
  }
  return size;
}

// #define DEFAULT_DEVICE_ID 0
// #define DEFAULT_CUDA_FREE_MEMORY GetFreeDeviceMemory(DEFAULT_DEVICE_ID)
#define CUDA_FREE_MEMORY(idx) GetFreeDeviceMemory(idx)
#define SYS_FREE_MEMORY GetFreeSystemMemory()