#pragma once 
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "muduo/base/noncopyable.h"
#include "utils/class_utils.h"
#include "utils/memory_utils.h"
#include <mutex>

#define NUM_PRIORITY 20UL

class StreamController {
 public:
  //   StreamController() = default;
  //   ~StreamController() = default;

  DISABLE_COPY_AND_ASSIGN(StreamController)

  explicit StreamController(const int device_id, const int num_priority)
  {
    // First three used for computation, the rest for data transfer
    for (int i = 0; i < num_priority; ++i) {
      cudaStream_t stream;
      cudaStreamCreateWithPriority(
          &stream, cudaStreamNonBlocking, -num_priority + i);
      cudaStreamSynchronize(stream);
      streams_.push_back(stream);
    }
  }

  cudaStream_t GetStream(const std::size_t idx) const { return streams_[idx]; }

  ~StreamController()
  {
    for (auto& stream : streams_) {
      cudaStreamDestroy(stream);
    }
  }

 private:
  std::vector<cudaStream_t> streams_;
};
typedef std::shared_ptr<StreamController> StreamControllerPtr;

class StreamCtrl : public muduo::noncopyable {
 public:
  DISABLE_COPY_AND_ASSIGN(StreamCtrl)
  STATIC_GET_INSTANCE(StreamCtrl)

  StreamControllerPtr& GetCudaStreamCtrl(std::size_t idx)
  {
    return cuda_mem_ctrls_[idx];
  }

  StreamCtrl()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_initialized_) return;
    for (int i = 0; i < GetDeviceCount(); ++i) {
      cuda_mem_ctrls_.emplace_back(
          std::make_shared<StreamController>(i, NUM_PRIORITY));
    }
    is_initialized_ = true;
  }
  virtual ~StreamCtrl() = default;

  //   cudaStream_t GetStreamWithPriority(const std::size_t idx) const {
  //     return cuda_mem_ctrls_[idx]->GetStream(0);
  //   }

 private:
  std::vector<StreamControllerPtr> cuda_mem_ctrls_;
  bool is_initialized_ = false;
  std::mutex mutex_;
};

#define CUDA_STREAM_CTRL(idx) GET_INSTANCE(StreamCtrl)->GetCudaStreamCtrl(idx)
#define DEFAULT_CUDA_STREAM_CTL GET_INSTANCE(StreamCtrl)->GetCudaStreamCtrl(0)