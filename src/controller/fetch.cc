#include "fetch.h"

#include <pthread.h>

#include <iostream>

#include "mem_ctrl.h"
#include "stream_ctrl.h"
#include "utils/log_utils.h"
#include "utils/shm_utils.h"
#include "utils/time_utils.h"

void
SetThreadScheduling(std::thread& th, int policy, int priority)
{
  sched_param sch_params;
  sch_params.sched_priority = priority;
  if (pthread_setschedparam(th.native_handle(), policy, &sch_params)) {
    std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno)
              << std::endl;
    assert(false);
  }
}

void
SetThreadAffinity(std::thread& th, int cpu_id)
{
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset)) {
    std::cerr << "Failed to set Thread affinity : " << std::strerror(errno)
              << std::endl;
    assert(false);
  }
}

void
SetThreadAffinity(std::thread& th)
{
  // get number of cpus
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  kCPUCounter++;
  int cpu_id = kCPUCounter % num_cpus;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);

  if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset)) {
    std::cerr << "Failed to set Thread affinity : " << std::strerror(errno)
              << std::endl;
    assert(false);
  }
}

void
FetchThread(const NodePtr node, const Device device)
{
  // Memory operation on node must be synchronized
  LOG_TRITON_VERBOSE(
      ("FetchThread: node: " + node->GetModelInstanceInfo() +
       " target device: " + device.str() + " cpu free memory " +
       std::to_string(kHostMemoryPool->GetFreeMemory()) + " bytes" +
       " gpu free memory " +
       std::to_string(kDeviceMemoryPool->GetFreeMemory(device)) +
       " bytes")
          .c_str());
  auto start_time = MCIROSECONDS_SINCE_EPOCH;
  if (node->device == device) {
    if (device.is_cuda()) {
      node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
      node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
    }
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    if (device.is_cuda()) {
      char buffer[1024];
      memset(buffer, 0, 1024);
      sprintf(
          buffer, "FetchThread: node: %s, device: %s, time cost %ld us",
          node->GetModelInstanceInfo().c_str(), device.str().c_str(),
          end_time - start_time);
      LOG_TRITON_INFO(buffer);
    }
    return;
  }

  LOG_TRITON_VERBOSE(("FetchThread: node: " + node->GetModelInstanceInfo() +
                      " lock acuired ")
                         .c_str());
  if (device.is_cuda()) {
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
  }

  LOG_TRITON_VERBOSE(("FetchThread: node: " + node->GetModelInstanceInfo() +
                      " stream acuired ")
                         .c_str());

  torch::InferenceMode infer_guard(true);
  auto origin_device = node->device;
  auto node_start_time = MCIROSECONDS_SINCE_EPOCH;
  node->SetDevice(device);
  auto node_end_time = MCIROSECONDS_SINCE_EPOCH;
  if (device.is_cuda() && origin_device.is_cpu()) {
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "FetchThread: node: %s, device: %s, move cost %ld us",
        node->GetModelInstanceInfo().c_str(), device.str().c_str(),
        node_end_time - node_start_time);
    LOG_TRITON_INFO(buffer);
  }

  LOG_TRITON_VERBOSE(("FetchThread: node: " + node->GetModelInstanceInfo() +
                      " memset acuired ")
                         .c_str());
  auto end_time = MCIROSECONDS_SINCE_EPOCH;

  {
    char buffer[2048];
    memset(buffer, 0, 2048);
    sprintf(
        buffer, "FetchThreadFunc: node: %s, device: %s, time cost %ld us",
        node->GetModelInstanceInfo().c_str(), device.str().c_str(),
        end_time - start_time);
    LOG_TRITON_INFO(buffer);
  }
}

void
FetchThreadFunc(
    const NodePtr node, const Device device, std::uint32_t immediate,
    CounterPtr counter)
{
  // Memory operation on node must be synchronized
  LOG_TRITON_VERBOSE(
      ("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
       " target device: " + device.str() +
       " immediate: " + std::to_string(immediate) + " cpu free memory " +
       std::to_string(SYS_MEM_CTL->GetFreeMemory()) + " bytes" +
       " gpu free memory " + std::to_string(CUDA_MEM_CTL(0)->GetFreeMemory()) +
       " bytes")
          .c_str());
  auto start_time = MCIROSECONDS_SINCE_EPOCH;
  if (node->device == device) {
    if (device.is_cuda()) {
      node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
      node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
    }

    if (immediate > 0)
      node->mutex.unlock();
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    if (device.is_cuda()) {
      char buffer[1024];
      memset(buffer, 0, 1024);
      sprintf(
          buffer, "FetchThreadFunc: node: %s, device: %s, time cost %ld us",
          node->GetModelInstanceInfo().c_str(), device.str().c_str(),
          end_time - start_time);
      LOG_TRITON_INFO(buffer);
    }
    return;
  }

  // work as a barrier
  if (device.is_cuda() || (device.is_cpu() && node->device == DISK_DEVICE)) {
    int wait_cnt = 0;
    while (counter->load() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      wait_cnt++;
      if (wait_cnt % 1000 == 0) {
        LOG_TRITON_ERROR(
            ("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
             " wait for counter " + std::to_string(counter->load()))
                .c_str());
      }
    }
  }

  // if (!immediate)
  //   lock.lock();
  // if (!immediate)
  //   node->mutex.lock();

  LOG_TRITON_VERBOSE(("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
                      " lock acuired ")
                         .c_str());
  if (device.is_cuda()) {
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
  }

  auto device_id = device.index();

  LOG_TRITON_VERBOSE(("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
                      " stream acuired ")
                         .c_str());

  if (node->device.is_cuda() && !device.is_cuda()) {
    counter->fetch_add(-1);
    CUDA_MEM_CTL(device_id)->FreeMemory(node->id, node->byte_size);
  }
  if (node->device.is_cpu() && !device.is_cpu()) {
    counter->fetch_add(-1);
    SYS_MEM_CTL->FreeMemory(node->id, node->byte_size);
  }

  torch::InferenceMode infer_guard(true);
  auto origin_device = node->device;
  auto node_start_time = MCIROSECONDS_SINCE_EPOCH;
  node->SetDevice(device);
  auto node_end_time = MCIROSECONDS_SINCE_EPOCH;
  if (device.is_cuda() && origin_device.is_cpu()) {
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "FetchThreadFunc: node: %s, device: %s, move cost %ld us",
        node->GetModelInstanceInfo().c_str(), device.str().c_str(),
        node_end_time - node_start_time);
    LOG_TRITON_INFO(buffer);
  }

  LOG_TRITON_VERBOSE(("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
                      " memset acuired ")
                         .c_str());

  if (immediate > 0)
    node->mutex.unlock();

  auto end_time = MCIROSECONDS_SINCE_EPOCH;

  {
    char buffer[2048];
    memset(buffer, 0, 2048);
    sprintf(
        buffer, "FetchThreadFunc: node: %s, device: %s, time cost %ld us",
        node->GetModelInstanceInfo().c_str(), device.str().c_str(),
        end_time - start_time);
    LOG_TRITON_INFO(buffer);
  }
}

void
PrefetchThreadFunc(const NodePtr& node)
{
  std::lock_guard<std::mutex> lock(kPrefetchMutex);
}
