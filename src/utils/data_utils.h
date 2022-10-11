#pragma once

#include <cstdint>
#include <functional>
#include <string>

typedef std::size_t NodeID;

template <class T1, class T2, class Pred = std::greater<T2>>
struct sort_pair_second {
  bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right)
  {
    Pred p;
    return p(left.second, right.second);
  }
};

template <class T>
inline bool
sortbysec(const std::pair<T, double>& a, const std::pair<T, double>& b)
{
  return (a.second > b.second);
}

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

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)