#include "fetch.h"

#include "mem_ctrl.h"
#include "utils/log_utils.h"
#include "utils/time_utils.h"

void
FetchThreadFunc(
    const NodePtr node, const Device device, bool immediate, CounterPtr counter)
{
  // Memory operation on node must be synchronized
  auto start_time = MCIROSECONDS_SINCE_EPOCH;
  LOG_TRITON_VERBOSE(
      ("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
       " target device: " + device.str() +
       " immediate: " + std::to_string(immediate) + " cpu free memory " +
       std::to_string(SYS_MEM_CTL->GetFreeMemory()) + " bytes" +
       " gpu free memory " + std::to_string(CUDA_MEM_CTL(0)->GetFreeMemory()) +
       " bytes")
          .c_str());

  if (node->device == device) {
    if (device.is_cuda()) {
      node->memory_type = MemoryType::kReady;
      node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    }
      
    if (device.is_cpu() || device == DISK_DEVICE)
      node->memory_type = MemoryType::kStandBy;

    if (!immediate)
      node->mutex.unlock();
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    LOG_TRITON_VERBOSE(
        ("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
         " already on target device: " + device.str() + " time cost " +
         std::to_string(end_time - start_time) + " us" + ", immediate " +
         std::to_string(immediate))
            .c_str());
    return;
  }

  // if (device.is_cuda())
  //   node->memory_type = MemoryType::kEmplacing;
  // if (device.is_cpu() || device == DISK_DEVICE)
  //   node->memory_type = MemoryType::kEvicting;

  // // work as a barrier
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

  // LOG_TRITON_VERBOSE(("FetchThreadFunc: node: " +
  // node->GetModelInstanceInfo() +
  //                     " lock acuired ")
  //                        .c_str());
  if (device.is_cuda()) {
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
  }
  // if (device.is_cuda()) {
  //   node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
  //   // LOG_TRITON_VERBOSE(
  //   //     ("FetchThreadFunc: move node to cuda" +
  //   node->GetModelInstanceInfo())
  //   //         .c_str());
  //   int wait_cnt = 0;
  //   while (CUDA_MEM_CTL(device.index())->AllocateMemory(node->byte_size) ==
  //          false) {
  //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
  //     wait_cnt++;
  //     if (wait_cnt % 1000 == 0) {
  //       LOG_TRITON_VERBOSE(
  //           ("FetchThreadFunc: wait for cuda memory" +
  //            node->GetModelInstanceInfo() + " " +
  //            std::to_string(CUDA_MEM_CTL(device.index())->GetFreeMemory()) +
  //            " bytes" + ", immediate " + std::to_string(immediate))
  //               .c_str());
  //     }
  //   }
  // }

  // if (device.is_cpu()) {
  //   LOG_TRITON_VERBOSE(
  //       ("FetchThreadFunc: move node to cpu" + node->GetModelInstanceInfo())
  //           .c_str());
  //   while (SYS_MEM_CTL->AllocateMemory(node->byte_size) == false) {
  //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
  //   }
  // }

  if (node->device.is_cuda() && !device.is_cuda()) {
    counter->fetch_add(-1);
    CUDA_MEM_CTL(node->device.index())->FreeMemory(node->id, node->byte_size);
  }
  if (node->device.is_cpu() && !device.is_cpu()) {
    counter->fetch_add(-1);
    SYS_MEM_CTL->FreeMemory(node->id, node->byte_size);
  }

  node->SetDevice(device);

  if (device.is_cuda())
    node->memory_type = MemoryType::kReady;
  if (device.is_cpu() || device == DISK_DEVICE)
    node->memory_type = MemoryType::kStandBy;

  if (!immediate)
    node->mutex.unlock();

  auto end_time = MCIROSECONDS_SINCE_EPOCH;
  LOG_TRITON_VERBOSE(
      ("FetchThreadFunc: node: " + node->GetModelInstanceInfo() +
       " immediate " + std::to_string(immediate) + " cpu free memory " +
       std::to_string(SYS_MEM_CTL->GetFreeMemory()) + " bytes" +
       " gpu free memory " + std::to_string(CUDA_MEM_CTL(0)->GetFreeMemory()) +
       " bytes" + " time cost " + std::to_string(end_time - start_time) + " us")
          .c_str());
}

void
PrefetchThreadFunc(const NodePtr& node)
{
  std::lock_guard<std::mutex> lock(kPrefetchMutex);
}


//   // When moving node to GPU
//   if (device.is_cuda()) {
//   }

//   // When moving node to CPU
//   if (device.is_cpu()) {
//   }

//   // When moving node to SSD
//   if (device.is_lazy()) {
//     if (node->device.is_cpu()) {
//       // Node is on CPU, move it to SSD
//       node->SetDevice(device);
//       node->memory_type = MemoryType::kStandBy;
//     } else if (node->device.is_cuda()) {
//       // Node is on GPU, move it to SSD
//       while (node->memory_type != MemoryType::kReady) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//       }
//     }
//   }