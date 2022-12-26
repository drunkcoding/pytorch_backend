#pragma once

#include "utils/topology.h"
#include <atomic>

typedef std::vector<std::pair<NodePtr, Device>> NodeMoveVec;
typedef std::shared_ptr<std::atomic_int> CounterPtr;

void FetchThreadFunc(const NodePtr node, const Device device, bool immediate, CounterPtr counter);
void PrefetchThreadFunc(const NodePtr& node);

// bool IsNodeReady(const NodePtr& node);

static std::mutex kPrefetchMutex;

// struct LockDeleter {
//   void operator()(T* ptr) const {}
// };