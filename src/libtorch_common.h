#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <string>

#include <torch/script.h>

enum class DeviceType {
  CPU = 0,
  GPU = 1,
  DISK = 2,
  INVALID = -1,
};

enum class DeviceID {
  CPU = -1,
  GPU = 0,
  DISK = -2,
  INVALID = -100,
};

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