#pragma once

#include <atomic>

#include "utils/topology.h"

typedef std::vector<std::pair<NodePtr, Device>> NodeMoveVec;
typedef std::shared_ptr<std::atomic_int> CounterPtr;

void FetchThreadFunc(
    const NodePtr node, const Device device, std::uint32_t immediate,
    CounterPtr counter);
void FetchThread(
    const NodePtr node, const Device device);
void PrefetchThreadFunc(const NodePtr& node);

// bool IsNodeReady(const NodePtr& node);

static std::mutex kPrefetchMutex;

void SetThreadScheduling(std::thread& th, int policy, int priority);
void SetThreadAffinity(std::thread& th, int cpu_id);
void SetThreadAffinity(std::thread& th);

static std::atomic_uint64_t kCPUCounter{0};

// struct LockDeleter {
//   void operator()(T* ptr) const {}
// };