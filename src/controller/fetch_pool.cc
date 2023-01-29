#include "fetch_pool.h"

#include <sstream>

#include "dataflow/flow_controller.h"
#include "topology_pool.h"
#include "utils/log_utils.h"
#include "utils/shm_utils.h"
#include "utils/time_utils.h"

TaskPool* kTaskPool = TaskPool::GetInstance();

TaskPool::TaskPool()
{
  h2d_queue_.resize(NUM_PRIORITY);
  d2h_queue_.resize(NUM_PRIORITY);
  unified_queue_.resize(NUM_PRIORITY);

  for (int i = 0; i < 2; ++i) {
    // auto h2d_thread = std::thread(&TaskPool::H2DThreadFunc, this);
    // SetThreadAffinity(h2d_thread);
    // h2d_thread.detach();
    // h2d_threads_.push_back(std::move(h2d_thread));

    // auto d2h_thread = std::thread(&TaskPool::D2HThreadFunc, this);
    // SetThreadAffinity(d2h_thread);
    // d2h_thread.detach();
    // d2h_threads_.push_back(std::move(d2h_thread));

    auto unified_thread = std::thread(&TaskPool::UnifiedThreadFunc, this);
    SetThreadAffinity(unified_thread);
    unified_thread.detach();
    unified_threads_.push_back(std::move(unified_thread));
  }
}

void
TaskPool::StartExec(const std::uint64_t& request_id, const NodePtr& node)
{
  auto task = std::make_shared<Task>();
  task->node = node;
  task->priority = 0;
  task->src_device = node->device;
  task->dst_device = node->default_device;
  task->request_id = request_id;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (exec_queue_.find(node->id) != exec_queue_.end()) {
      std::stringstream ss;
      ss << "Node " << std::hex << node->id << " is already in exec queue";
      LOG_TRITON_ERROR(ss.str().c_str());
      return;
    }
    exec_queue_.insert({node->id, task});
  }

  if (task->src_device == task->dst_device) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    unified_queue_[task->priority].push_back(task);
  }

  // ScheduleTask(request_id, task);
}

void
TaskPool::StopExec(const std::uint64_t& request_id, const NodePtr& node)
{
  auto task = std::make_shared<Task>();
  task->node = node;
  task->priority = 0;
  task->src_device = node->default_device;
  task->dst_device = node->default_host;
  task->request_id = request_id;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (exec_queue_.find(node->id) == exec_queue_.end()) {
      std::stringstream ss;
      ss << "Node " << std::hex << node->id << " is not in exec queue";
      LOG_TRITON_ERROR(ss.str().c_str());
      return;
    }
    exec_queue_.erase(node->id);
  }

  assert(task->src_device != task->dst_device);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    unified_queue_[task->priority].push_back(task);
  }

  // ScheduleTask(request_id, task);
}


void
TaskPool::UnifiedThreadFunc()
{
  while (true) {
    std::uint32_t max_priority = 1000;
    std::unique_lock<std::mutex> lock(mutex_);
    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!unified_queue_[i].empty()) {
        max_priority = i;
        break;
      }
    }

    if (max_priority == 1000) {
      SKIP_TO_NEXT_ITERATION
    }

    LOG_TRITON_VERBOSE(DebugString(unified_queue_).c_str());

    // pop the task from the highest priority queue
    TaskPtr task = unified_queue_[max_priority].front();
    auto node = task->node;
    unified_queue_[max_priority].pop_front();

    std::uint64_t low_corr_id = node->corr_id & 0xFFFFFFFF;
    if (low_corr_id < kTopologyPool->GetLastActivateStage(task->request_id) &&
        task->priority > 0) {
      SKIP_TO_NEXT_ITERATION
    }

    lock.unlock();

    // do not execute the rest if node memory move has conflict
    node->mutex.lock();
    std::stringstream ss;
    ss << std::hex << node->id;

    // DISK->CPU
    if (task->src_device == DISK_DEVICE && task->dst_device == CPU_DEVICE) {
      if (node->device == DISK_DEVICE) {
        kHostMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, CPU_DEVICE);
      } else if (node->device.is_cuda()) {
        LOG_TRITON_ERROR((ss.str() + " is on GPU, cannot move to CPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device == CPU_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " already on CPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else {
        assert(false);
      }
    }

    // DISK->GPU
    if (task->src_device == DISK_DEVICE && task->dst_device.is_cuda()) {
      if (node->device == DISK_DEVICE) {
        kHostMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, CPU_DEVICE);
        kDeviceMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, task->dst_device);
      } else if (node->device.is_cuda()) {
        LOG_TRITON_ERROR((ss.str() + " already on GPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device == CPU_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " is on CPU, move to GPU").c_str());
        kDeviceMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, task->dst_device);
      } else {
        assert(false);
      }
    }

    // CPU->GPU
    if (task->src_device == CPU_DEVICE && task->dst_device.is_cuda()) {
      if (node->device == DISK_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " is on DISK, move to GPU").c_str());
        kHostMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, CPU_DEVICE);
        kDeviceMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, task->dst_device);
      } else if (node->device.is_cuda()) {
        LOG_TRITON_ERROR((ss.str() + " already on GPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device == CPU_DEVICE) {
        kDeviceMemoryPool->PreAllocateMemory(
            node->id, node->byte_size, task->dst_device);
      } else {
        assert(false);
      }
    }

    // CPU->DISK
    if (task->src_device == CPU_DEVICE && task->dst_device == DISK_DEVICE) {
      if (node->device == DISK_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " already on DISK").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device.is_cuda()) {
        LOG_TRITON_ERROR(
            (ss.str() + " is on GPU, cannot move to DISK").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device == CPU_DEVICE) {
        // do nothing
      } else {
        assert(false);
      }
    }

    // GPU->CPU
    if (task->src_device.is_cuda() && task->dst_device == CPU_DEVICE) {
      if (node->device == DISK_DEVICE) {
        LOG_TRITON_ERROR(
            (ss.str() + " is on DISK, cannot move to CPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device.is_cuda()) {
        // do nothing
      } else if (node->device == CPU_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " already on CPU").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else {
        assert(false);
      }
    }

    // GPU->DISK
    if (task->src_device.is_cuda() && task->dst_device == DISK_DEVICE) {
      if (node->device == DISK_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " already on DISK").c_str());
        node->mutex.unlock();
        SKIP_TO_NEXT_ITERATION
      } else if (node->device.is_cuda()) {
        // do nothing
      } else if (node->device == CPU_DEVICE) {
        LOG_TRITON_ERROR((ss.str() + " is on CPU, move to DISK").c_str());
      } else {
        assert(false);
      }
    }

    {
      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      auto success = RemoveMemoryForNode(node, task->src_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      {
        char buffer[2048];
        memset(buffer, 0, 2048);
        sprintf(
            buffer, "UnifiedThreadFunc: task: %s, evict time %ld us",
            task->DebugString().c_str(), end_time - start_time);
        LOG_TRITON_INFO(buffer);
      }

      if (!success) {
        // insert the task back to the queue
        node->mutex.unlock();
        lock.lock();
        unified_queue_[task->priority].push_back(task);
        SKIP_TO_NEXT_ITERATION
      }
    }

    {
      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      node->SetDevice(task->dst_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      {
        char buffer[2048];
        memset(buffer, 0, 2048);
        sprintf(
            buffer, "UnifiedThreadFunc: task: %s, emplace time %ld us",
            task->DebugString().c_str(), end_time - start_time);
        LOG_TRITON_INFO(buffer);
      }
    }
    node->mutex.unlock();
  }
}

void
TaskPool::D2HThreadFunc()
{
  while (true) {
    std::uint32_t max_priority = 1000;
    std::unique_lock<std::mutex> lock(mutex_);
    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!d2h_queue_[i].empty()) {
        max_priority = i;
        break;
      }
    }
    if (max_priority == 1000) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    LOG_TRITON_VERBOSE(DebugString(d2h_queue_).c_str());

    // get random it from the highest priority tasks
    auto task = *d2h_queue_[max_priority].begin();
    d2h_queue_[max_priority].pop_front();

    if (task->node != nullptr) {
      for (auto& exec_task : exec_queue_) {
        std::remove(
            task->remove_nodes.begin(), task->remove_nodes.end(),
            exec_task.second->node);
      }
    }
    lock.unlock();

    for (auto& node : task->remove_nodes) {
      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      FetchThread(node, task->dst_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      char buffer[1024];
      memset(buffer, 0, 1024);
      sprintf(
          buffer, "D2HThreadFunc: node: %s, priority: %d, time: %ld us",
          node->GetModelInstanceInfo().c_str(), max_priority,
          end_time - start_time);
      LOG_TRITON_INFO(buffer);
    }

    lock.lock();
    // if (task->node == nullptr) {
    //   // stop exec, remove the task from the exec queue
    //   std::remove_if(exec_queue_.begin(), exec_queue_.end(), [&](auto& it) {
    //     return it.second->task == task;
    //   });
    //   task->remove_nodes.clear();
    // } else {
    //   // add the task to the h2d queue
    //   task->remove_nodes.clear();
    //   h2d_queue_[task->priority].push_back(task);
    // }
  }
}

void
TaskPool::H2DThreadFunc()
{
  while (true) {
    std::uint32_t max_priority = 1000;
    std::unique_lock<std::mutex> lock(mutex_);
    for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
      if (!h2d_queue_[i].empty()) {
        max_priority = i;
        break;
      }
    }
    if (max_priority == 1000) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    LOG_TRITON_VERBOSE(DebugString(h2d_queue_).c_str());

    // get random it from the highest priority tasks
    auto task = *h2d_queue_[max_priority].begin();
    h2d_queue_[max_priority].pop_front();
    auto node = task->node;

    if (node->device.type() >= task->dst_device.type() ||
        node->device.type() <= task->src_device.type()) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // for (std::uint32_t i = task->priority; i < NUM_PRIORITY; i++) {
    //   std::remove_if(
    //       d2h_queue_[i].begin(), d2h_queue_[i].end(),
    //       [node](auto& t) { return t->node == node; });
    //   std::remove_if(
    //       h2d_queue_[i].begin(), h2d_queue_[i].end(),
    //       [node](auto& t) { return t->node == node; });
    // }

    lock.unlock();

    auto start_time = MCIROSECONDS_SINCE_EPOCH;
    FetchThread(node, task->dst_device);
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "H2DThreadFunc: node: %s, priority: %d, time: %ld us",
        node->GetModelInstanceInfo().c_str(), max_priority,
        end_time - start_time);
    LOG_TRITON_INFO(buffer);
  }
}

void
TaskPool::Prefetch(const std::uint64_t& request_id, const NodePtr& node)
{
  auto root_id = node->corr_id & 0xFFFFFFFF;
  auto gpu_candidates = kTopologyPool->GetTopKChildNodes(node, 0, 20);
  auto cpu_candidates = kTopologyPool->GetTopKChildNodes(node, 20, 50);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::uint32_t priority = 0; priority < NUM_PRIORITY; priority++) {
      for (auto& task : unified_queue_[priority]) {
        std::remove(gpu_candidates.begin(), gpu_candidates.end(), task->node);
        std::remove(cpu_candidates.begin(), cpu_candidates.end(), task->node);
      }
    }
  }

  std::int64_t cpu_size_limit = 1024 * 1024 * 1024 * 10LL;
  for (auto& candidate : cpu_candidates) {
    if (candidate->device.is_cpu() || candidate->device.is_cuda()) {
      continue;
    }
    cpu_size_limit -= candidate->byte_size;
    if (cpu_size_limit < 0) {
      break;
    }
    auto low_id = candidate->corr_id & 0xFFFFFFFF;
    auto task = std::make_shared<Task>();
    task->node = candidate;
    task->priority = std::min(low_id - root_id + 3, NUM_PRIORITY - 1);
    task->dst_device = CPU_DEVICE;
    // task->type = TaskType::kDisk2Host;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      unified_queue_[task->priority].push_back(task);
    }

    // ScheduleTask(request_id, task);
  }

  // Get number of GPUs
  int num_gpus = GetDeviceCount();
  std::vector<NodePtrList> gpu_candidates_list(num_gpus);
  for (auto& candidate : gpu_candidates) {
    if (candidate->device.is_cuda()) {
      continue;
    }
    gpu_candidates_list[candidate->default_device.index()].push_back(candidate);
  }


  for (int i = 0; i < num_gpus; i++) {
    std::int64_t gpu_size_limit = 1024 * 1024 * 1024 * 2LL;
    for (auto& candidate : gpu_candidates_list[i]) {
      gpu_size_limit -= candidate->byte_size;
      if (gpu_size_limit < 0) {
        break;
      }

      auto low_id = candidate->corr_id & 0xFFFFFFFF;
      auto task = std::make_shared<Task>();
      task->node = candidate;
      task->priority = std::min(low_id - root_id + 3, NUM_PRIORITY - 1);
      task->dst_device = candidate->default_device;
      // task->type = TaskType::kHost2Device;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        unified_queue_[task->priority].push_back(task);
      }
      // ScheduleTask(request_id, task);
    }
  }
}

bool
TaskPool::RemoveMemoryForNode(const NodePtr& node, const Device& device)
{
  if (node == nullptr || device == DISK_DEVICE) {
    return true;
  }

  // while (true) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto free_memory = (device.is_cuda())
                         ? kDeviceMemoryPool->GetFreeMemory(device)
                         : kHostMemoryPool->GetFreeMemory();
  auto removable_node_list = kTopologyPool->GetLFUNodes(device);
  auto remove_size = node->byte_size - free_memory;

  // nodes in exec queue is not removable
  for (auto& exec_task : exec_queue_) {
    std::remove(
        removable_node_list.begin(), removable_node_list.end(),
        exec_task.second->node);
  }

  lock.unlock();

  while (remove_size > 0 && !removable_node_list.empty()) {
    NodePtr remove_node = removable_node_list.front();
    removable_node_list.pop_front();
    if (remove_node->mutex.try_lock()) {
      remove_node->SetDevice(DISK_DEVICE);
      remove_node->mutex.unlock();
    }
    remove_size -= remove_node->byte_size;
  }
  if (remove_size > 0) {
    LOG_TRITON_ERROR(
        ("ScheduleTask: remove_size > 0: " + std::to_string(remove_size) +
         " free_memory: " + std::to_string(free_memory) +
         " node->byte_size: " + std::to_string(node->byte_size))
            .c_str());
  }
  return remove_size <= 0;

  // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  // }
}

void
TaskPool::ScheduleTask(const std::uint64_t& request_id, const TaskPtr& task)
{
  auto device = task->dst_device;
  auto node = task->node;

  if (node == nullptr && task->remove_nodes.empty()) {
    return;
  }

  if (node != nullptr) {
    std::uint64_t low_corr_id = node->corr_id & 0xFFFFFFFF;
    while (low_corr_id >= kTopologyPool->GetLastActivateStage(request_id) ||
           task->priority == 0) {
      std::unique_lock<std::mutex> lock(mutex_);
      auto free_memory = (device.is_cuda())
                             ? kDeviceMemoryPool->GetFreeMemory(device)
                             : kHostMemoryPool->GetFreeMemory();
      auto removable_node_list = kTopologyPool->GetLFUNodes(device);
      auto remove_size = node->byte_size - free_memory;

      // nodes in exec queue is not removable
      for (auto& exec_task : exec_queue_) {
        std::remove(
            removable_node_list.begin(), removable_node_list.end(),
            exec_task.second->node);
      }

      // nodes in d2h queue is not removable
      for (std::uint32_t priority = 0; priority < NUM_PRIORITY; priority++) {
        for (auto& d2h_task : d2h_queue_[priority]) {
          std::remove(
              removable_node_list.begin(), removable_node_list.end(),
              d2h_task->node);
        }
      }

      while (remove_size > 0 && !removable_node_list.empty()) {
        auto it = removable_node_list.begin();
        auto remove_node = *it;
        task->remove_nodes.push_back(remove_node);
        remove_size -= remove_node->byte_size;
        removable_node_list.erase(it);
      }
      if (remove_size <= 0) {
        break;
      }
      LOG_TRITON_ERROR(
          ("ScheduleTask: remove_size > 0: " + std::to_string(remove_size) +
           " free_memory: " + std::to_string(free_memory) +
           " node->byte_size: " + std::to_string(node->byte_size))
              .c_str());
      lock.unlock();
      task->remove_nodes.clear();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (low_corr_id < kTopologyPool->GetLastActivateStage(request_id) &&
        task->priority > 0) {
      LOG_TRITON_ERROR(
          ("ScheduleTask: low_corr_id < last_activate_stage: " +
           std::to_string(low_corr_id) + " < " +
           std::to_string(kTopologyPool->GetLastActivateStage(request_id)))
              .c_str());
      return;
    }

    if (node->device == DISK_DEVICE && device == CPU_DEVICE) {
      kHostMemoryPool->PreAllocateMemory(node->id, node->byte_size, CPU_DEVICE);
    }
    if (node->device == DISK_DEVICE && device.is_cuda()) {
      kHostMemoryPool->PreAllocateMemory(node->id, node->byte_size, CPU_DEVICE);
      kDeviceMemoryPool->PreAllocateMemory(node->id, node->byte_size, device);
    }
    if (node->device == CPU_DEVICE && device.is_cuda()) {
      kDeviceMemoryPool->PreAllocateMemory(node->id, node->byte_size, device);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!task->remove_nodes.empty()) {
    d2h_queue_[task->priority].push_back(task);
  } else {
    h2d_queue_[task->priority].push_back(task);
  }
}


void
FetchPool::D2HThreadFunc()
{
  // auto target_stream = CUDA_STREAM(0, 2);
  // at::cuda::CUDAStream torch_stream =
  //     at::cuda::getStreamFromExternal(target_stream, 0);
  // at::cuda::CUDAStreamGuard guard(torch_stream);
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
    nodes_.erase(key);

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
  // auto target_stream = CUDA_STREAM(0, 0);
  // at::cuda::CUDAStream torch_stream =
  //     at::cuda::getStreamFromExternal(target_stream, 0);
  // at::cuda::CUDAStreamGuard guard(torch_stream);
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
  // auto target_stream = CUDA_STREAM(0, 0);
  // at::cuda::CUDAStream torch_stream =
  //     at::cuda::getStreamFromExternal(target_stream, 0);
  // at::cuda::CUDAStreamGuard guard(torch_stream);

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
  // if (nodes_.find(node->id) != nodes_.end()) {
  //   LOG_TRITON_ERROR((ss.str() + " already in the evict pool").c_str());
  //   lock.unlock();
  //   cond_.notify_all();
  //   return;
  // }
  nodes_.insert({node->id, node});
  for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    if (D2HTasks(i).find(key) != D2HTasks(i).end()) {
      LOG_TRITON_ERROR((ss.str() + " already in the evict pool, priority " +
                        std::to_string(i))
                           .c_str());
      if (priority >= i) {
        lock.unlock();
        return;
      }
      D2HTasks(i).erase(key);
    }
  }
  D2HTasks(priority).emplace(key, std::move(task));
  LOG_TRITON_INFO(
      (ss.str() + " evict with priority " + std::to_string(priority)).c_str());
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
  for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
    if (Tasks(i).find(key) != Tasks(i).end()) {
      LOG_TRITON_ERROR((ss.str() + " already in the fetch pool, priority " +
                        std::to_string(i))
                           .c_str());
      if (priority >= i) {
        lock.unlock();
        return;
      }
      Tasks(i).erase(key);
    }
  }

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
      LOG_TRITON_INFO(
          (ss.str() + " move to priority " + std::to_string(priority)).c_str());
      if (i <= priority) {
        lock.unlock();
        return;
      } else {
        Tasks(priority).emplace(key, Tasks(i)[key]);
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
    if (node_low_id < low_id) {
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

std::string
TaskPool::DebugString(const std::vector<std::deque<TaskPtr> >& queue)
{
  std::stringstream ss;
  for (std::uint32_t i = 0; i < queue.size(); ++i) {
    ss << "priority " << i << " : ";
    for (auto task : queue[i]) {
      auto node = task->node;
      if (node == nullptr && task->remove_nodes.size() == 0) {
        continue;
      }
      if (node == nullptr) {
        node = task->remove_nodes[0];
      }
      ss << std::hex << node->id << "[" << node->device << "->"
         << task->dst_device.str() << "," << task->remove_nodes.size() << ","
         << "]"
         << " " << std::dec;
    }
    ss << std::endl;
  }
  return ss.str();
}
