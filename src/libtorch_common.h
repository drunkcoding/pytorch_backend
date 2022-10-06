#pragma once

#include <torch/script.h>

#include <atomic>
#include <boost/preprocessor.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>

#include "triton/backend/backend_common.h"

#define TIME_NOW std::chrono::high_resolution_clock::now();

/*=============LOGGING===============*/
#define LOG_VERBOSE(MSG) LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, MSG)
#define LOG_INFO(MSG) LOG_MESSAGE(TRITONSERVER_LOG_INFO, MSG)

typedef torch::jit::script::Module ScriptModule;
// typedef std::shared_ptr<Module> LibTorchModulePtr;
typedef ScriptModule* ScriptModulePtr;  // always use raw pointer, since we need
                                        // to manage the memory by ourselves
template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

#define MODULE_PTR_NODELETE(ptr)              \
  std::shared_ptr<ScriptModule>               \
  {                                           \
    ptr, DoNothingDeleter<ScriptModule> {} \
  }

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kLazy)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define PROCESS_ONE_ELEMENT(r, unused, idx, elem) \
  BOOST_PP_COMMA_IF(idx) BOOST_PP_STRINGIZE(elem)

#ifndef ENUM_MACRO
#define ENUM_MACRO(name, ...)                                            \
  enum class name { __VA_ARGS__ };                                       \
  static const char* name##Strings[] = {BOOST_PP_SEQ_FOR_EACH_I(         \
      PROCESS_ONE_ELEMENT, % %, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))}; \
  template <typename T>                                                  \
  constexpr const std::string name##ToString(T value)                    \
  {                                                                      \
    return std::string(name##Strings[static_cast<int>(value)]);          \
  }
#endif  // ENUM_MACRO

ENUM_MACRO(
    RetCode, RET_SUCCESS, RET_FAILED, RET_NOT_ENOUGH_MEMORY,
    RET_ALREADY_ALLOCATED, RET_NOT_ALLOCATED)


#define RETURN_IF_NOT_SUCCESS(ret)     \
  do {                                 \
    if (ret != RetCode::RET_SUCCESS) { \
      return ret;                      \
    }                                  \
  } while (false)

/*
 * @brief: This class is used to manage the memory of the model.
 * READY: The module is waiting for its change of execution.
 * ACTIVE: The module is loaded into GPU memory and running inference.
 * INACTIVE: The module is loaded into GPU memory but not running inference.
 * EVICTING: The module is evicting from GPU/CPU memory, will be PENDING after
 * eviction finished.
 */
ENUM_MACRO(MemoryState, READY, ACTIVE, INACTIVE, EVICTING)
ENUM_MACRO(DeviceType, CPU, CUDA, GPU, DISK, INVALID)
ENUM_MACRO(ManageType, RELEASE = 0, PREFETCH = 1, ON_DEMAND = 2)

typedef std::atomic<MemoryState> AtomicMemoryState;

// BETTER_ENUM(
//     RetCode, std::uint8_t, EXIT_SUCCESS, EXIT_NOT_ENOUGH_MEMORY,
//     EXIT_ALREADY_ALLOCATED, EXIT_NOT_ALLOCATED)

// BETTER_ENUM(DeviceType, std::uint8_t, CPU, GPU, DISK, INVALID)

// enum class DeviceID {
//   CPU = -1,
//   GPU = 0,
//   DISK = -2,
//   INVALID = -100,
// };

// typedef std::size_t KeyType;


inline std::size_t
MakeID(const std::string& model_name, const std::size_t& model_index) noexcept
{
  return std::hash<std::string>{}(model_name + std::to_string(model_index));
}

inline std::size_t
MakeID(const std::string& model_instance_name) noexcept
{
  return std::hash<std::string>{}(model_instance_name);
}

inline std::size_t
GetFileSize(const std::string& file_path)
{
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  return file.tellg();
}
