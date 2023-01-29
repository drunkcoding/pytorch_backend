#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <rmm/cuda_stream_pool.hpp>

#include "muduo/base/noncopyable.h"
#include "utils/class_utils.h"
#include "utils/memory_utils.h"


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

typedef std::unique_ptr<rmm::cuda_stream_pool> CudaStreamPoolPtr;

class CudaStreamPool : public muduo::noncopyable {
 public:
  DISABLE_COPY_AND_ASSIGN(CudaStreamPool)
  CudaStreamPoolPtr& operator()(const int device_id)
  {
    return cuda_streams_[device_id];
  }

  static CudaStreamPool* GetInstance() { return new CudaStreamPool(); }

 private:
  CudaStreamPool()
  {
    int num_devices = GetDeviceCount();
    cuda_streams_.resize(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      cudaSetDevice(i);
      cuda_streams_[i] = std::make_unique<rmm::cuda_stream_pool>(3);
    }
  }
  virtual ~CudaStreamPool() = default;

 private:
  std::vector<CudaStreamPoolPtr> cuda_streams_;
};


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
    if (is_initialized_)
      return;
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

extern CudaStreamPool* kCudaStreamPool;
#define CUDA_STREAM_VIEW(device_id, stream_id) \
  (*kCudaStreamPool)(device_id)->get_stream(stream_id)
#define CUDA_STREAM_H2D_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 0)
#define CUDA_STREAM_D2H_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 1)
#define CUDA_STREAM_COMPUTE_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 2)
#define CUDA_STREAM(device_id, stream_id) CUDA_STREAM_VIEW(device_id, stream_id).value()
#define CUDA_STREAM_H2D(device_id) CUDA_STREAM(device_id, 0)
#define CUDA_STREAM_D2H(device_id) CUDA_STREAM(device_id, 1)
#define CUDA_STREAM_COMPUTE(device_id) CUDA_STREAM(device_id, 2)