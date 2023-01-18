#include "fetch_pool.h"

#include <sstream>

#include "utils/log_utils.h"
#include "utils/shm_utils.h"
#include "utils/time_utils.h"

void
FetchPool::D2HThreadFunc()
{
  auto target_stream = CUDA_STREAM_CTRL(0)->GetStream(3);
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(target_stream, 0);
  at::cuda::CUDAStreamGuard guard(torch_stream);
  while (true) {
    std::uint32_t max_priority = 1000;

    std::unique_lock<std::mutex> lock(d2h_mutex_);
    // d2h_cond_.wait(lock, [this, &max_priority]() {
    //   for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    //     if (!D2HTasks(i).empty()) {
    //       max_priority = i;
    //       return true;
    //     }
    //   }
    //   return false;
    // });
    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!D2HTasks(i).empty()) {
        max_priority = i;
        break;
      }
    }
    // find the highest priority task, when the priorities_ is not empty


    // for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    //   if (!Tasks(i).empty()) {
    //     max_priority = i;
    //     break;
    //   }
    // }

    // if (max_priority == 1000 || max_priority < 3) {
    //   lock.unlock();
    //   continue;
    // }

    if (max_priority == 1000) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // get random it from the highest priority tasks
    auto [key, cb] = *D2HTasks(max_priority).begin();
    D2HTasks(max_priority).erase(key);

    // if (nodes_.find(key) == nodes_.end()) {
    //   lock.unlock();
    //   continue;
    // }

    // auto device_id = nodes_[key]->default_device.index();
    NodePtr node = nodes_[key];
    // nodes_.erase(key);

    lock.unlock();


    auto start_time = MCIROSECONDS_SINCE_EPOCH;
    cb();
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "ThreadFunc: node: %s, priority: %d, time: %ld us",
        node->GetModelInstanceInfo().c_str(), max_priority,
        end_time - start_time);
    LOG_TRITON_INFO(buffer);

    // cudaStreamSynchronize(target_stream);
  }
}


void
FetchPool::TopThreadFunc()
{
  auto target_stream = CUDA_STREAM_CTRL(0)->GetStream(0);
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(target_stream, 0);
  at::cuda::CUDAStreamGuard guard(torch_stream);
  while (true) {
    std::uint32_t max_priority = 1000;

    std::unique_lock<std::mutex> lock(mutex_);
    // cond_.wait(lock, [this, &max_priority]() {
    //   for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    //     if (!Tasks(i).empty()) {
    //       max_priority = i;
    //       return true;
    //     }
    //   }
    //   return false;
    // });

    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!Tasks(i).empty()) {
        max_priority = i;
        break;
      }
    }

    if (max_priority == 1000 || max_priority >= 3) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // get random it from the highest priority tasks
    auto [key, cb] = *Tasks(max_priority).begin();
    Tasks(max_priority).erase(key);

    // if (nodes_.find(key) == nodes_.end()) {
    //   lock.unlock();
    //   continue;
    // }

    // auto device_id = nodes_[key]->default_device.index();
    NodePtr node = nodes_[key];
    // nodes_.erase(key);
    lock.unlock();

    auto start_time = MCIROSECONDS_SINCE_EPOCH;
    cb();
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "TopThreadFunc: node: %s, priority: %d, time: %ld us",
        node->GetModelInstanceInfo().c_str(), max_priority,
        end_time - start_time);
    LOG_TRITON_INFO(buffer);

    // cudaStreamSynchronize(target_stream);
  }
}

void
FetchPool::ThreadFunc()
{
  auto target_stream = CUDA_STREAM_CTRL(0)->GetStream(1);
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(target_stream, 0);
  at::cuda::CUDAStreamGuard guard(torch_stream);

  int wait_cnt = 0;
  while (true) {
    std::uint32_t max_priority = 1000;
    std::unique_lock<std::mutex> lock(mutex_);

    // cond_.wait(lock, [this, &max_priority]() {
    //   for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    //     if (!Tasks(i).empty()) {
    //       max_priority = i;
    //       return true;
    //     }
    //   }
    //   return false;
    // });

    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!Tasks(i).empty()) {
        max_priority = i;
        break;
      }
    }

    // find the highest priority task, when the priorities_ is not empty

    if (max_priority == 1000 || max_priority < 3) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // get begin using move
    auto [key, cb] = *Tasks(max_priority).begin();
    Tasks(max_priority).erase(key);

    // if (nodes_.find(key) == nodes_.end()) {
    //   lock.unlock();
    //   continue;
    // }

    // auto device_id = nodes_[key]->default_device.index();
    NodePtr node = nodes_[key];
    // nodes_.erase(key);
    lock.unlock();

    auto start_time = MCIROSECONDS_SINCE_EPOCH;
    cb();
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "ThreadFunc: node: %s, priority: %d, time: %ld us",
        node->GetModelInstanceInfo().c_str(), max_priority,
        end_time - start_time);
    LOG_TRITON_INFO(buffer);

    // cudaStreamSynchronize(target_stream);

    wait_cnt = 0;
  }
}

void
FetchPool::AddD2HTask(
    const NodePtr& node, std::uint32_t priority, DefaultCb task)
{
  assert(priority < NUM_PRIORITY);
  auto key = node->id;
  std::stringstream ss;
  ss << std::hex << key;
  std::unique_lock<std::mutex> lock(d2h_mutex_);
  nodes_.insert({node->id, node});
  // for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
  if (D2HTasks(priority).find(key) != D2HTasks(priority).end()) {
    LOG_TRITON_ERROR((ss.str() + " already in the fetch pool, priority " +
                      std::to_string(priority))
                         .c_str());
    lock.unlock();
    return;
  }
  // }
  D2HTasks(priority).emplace(key, std::move(task));
  LOG_TRITON_INFO(
      (ss.str() + " add with priority " + std::to_string(priority)).c_str());
  lock.unlock();
  d2h_cond_.notify_one();
}

void
FetchPool::AddTask(const NodePtr& node, std::uint32_t priority, DefaultCb task)
{
  assert(priority < NUM_PRIORITY);
  auto key = node->id;
  std::stringstream ss;
  ss << std::hex << key;
  std::unique_lock<std::mutex> lock(mutex_);
  nodes_.insert({node->id, node});
  // for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
  if (Tasks(priority).find(key) != Tasks(priority).end()) {
    LOG_TRITON_ERROR((ss.str() + " already in the fetch pool, priority " +
                      std::to_string(priority))
                         .c_str());
    lock.unlock();
    cond_.notify_all();
  }
  // }
  Tasks(priority).emplace(key, std::move(task));
  LOG_TRITON_INFO(
      (ss.str() + " add with priority " + std::to_string(priority)).c_str());
  lock.unlock();
}

void
FetchPool::PrioritizeTask(const NodePtr& node, std::uint32_t priority)
{
  assert(priority < NUM_PRIORITY);
  std::unique_lock<std::mutex> lock(mutex_);

  auto key = node->id;
  std::stringstream ss;
  ss << std::hex << key;

  // find the task in the priorities_ and move it to the
  // priorities_[priority]
  for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    if (Tasks(i).find(key) != Tasks(i).end()) {
      Tasks(priority).emplace(key, Tasks(i)[key]);
      Tasks(i).erase(key);
      LOG_TRITON_INFO(
          (ss.str() + " move to priority " + std::to_string(priority)).c_str());
      if (i <= priority) {
        lock.unlock();
        return;
      } else {
        Tasks(i).erase(key);
      }
    }
  }
  LOG_TRITON_ERROR((ss.str() + " not in the fetch pool, priority " +
                    std::to_string(priority))
                       .c_str());
  lock.unlock();
  cond_.notify_all();
}

void
FetchPool::RemoveTask(std::uint64_t corr_id)
{
  // auto high_id = corr_id >> 32;
  std::uint32_t low_id = corr_id & 0xFFFFFFFF;

  std::unique_lock<std::mutex> lock(mutex_);

  std::vector<std::uint64_t> delete_keys;
  for (auto node_item : nodes_) {
    std::uint32_t node_low_id = node_item.second->corr_id & 0xFFFFFFFF;
    if (node_low_id <= low_id) {
      delete_keys.push_back(node_item.first);
    }
  }

  for (auto del_key : delete_keys) {
    for (std::uint32_t i = 3; i < NUM_PRIORITY; ++i) {
      if (Tasks(i).find(del_key) != Tasks(i).end()) {
        Tasks(i).erase(del_key);
        nodes_[del_key]->mutex.unlock();
        // nodes_.erase(del_key);
      }
    }
  }

  // adjust node priority
  // for (auto node_item : nodes_) {
  //   auto node_low_id = node_item.second->corr_id & 0xFFFFFFFF;
  //   std::uint32_t priority = std::min(NUM_PRIORITY - 1, node_low_id -
  //   low_id); for (std::uint32_t i = 3; i < NUM_PRIORITY; ++i) {
  //     if (Tasks(i).find(node_item.first) != Tasks(i).end()) {
  //       Tasks(priority).emplace(node_item.first, Tasks(i)[node_item.first]);
  //       Tasks(i).erase(node_item.first);
  //     }
  //   }
  // }
  lock.unlock();
}
