#pragma once


#include <cuda_runtime_api.h>
#include <cufile.h>

#include <future>
#include <mutex>
#include <thread>
#include <vector>


class GDSLoader {
 public:
  GDSLoader() : loaded_(false), cuda_inited_(false) {}

  ~GDSLoader() = default;


  int load(const char* filename, int device_id, void** devPtr_p);

  int loadAsync(
      const char* filename, int device_id, void** devPtr_p,
      std::future<ssize_t>& future_p);

  int loadAsyncInited(
      const char* filename, int device_id, void** devPtr_p,
      std::future<ssize_t>& future_p);

  bool cudaInit();

  inline bool isLoaded() const { return loaded_; }


 private:
  bool loaded_;

  bool cuda_inited_;

  int fd_;

  ssize_t size_;

  void* devPtr_;

  cudaStream_t stream_;
};


// read 1 MB each time and copy to GPU

#define BLOCK_SIZE (1 << 23)


struct transition {
  void* buffer;

  size_t offset;

  size_t size;
};


// a thread-safe ring buffer to store the data to be copied to GPU

class RingBuffer {
 public:
  RingBuffer(size_t size) : size_(size), head_(0), tail_(0)
  {
    buffer_ = new transition[size];

    for (int i = 0; i < size; i++) {
      buffer_[i].buffer = aligned_alloc(4096, BLOCK_SIZE);

      buffer_[i].offset = 0;

      buffer_[i].size = 0;
    }
  }

  ~RingBuffer()
  {
    for (int i = 0; i < size_; i++) {
      free(buffer_[i].buffer);
    }

    delete[] buffer_;
  }


  transition* enqueue()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (isFull()) {
      return nullptr;
    }

    transition* t = &buffer_[tail_];

    tail_ = (tail_ + 1) % size_;

    return t;
  }


  transition* getHead()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (isEmpty()) {
      return nullptr;
    }

    return &buffer_[head_];
  }


  void dequeue()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (isEmpty()) {
      return;
    }

    buffer_[head_].offset = 0;

    buffer_[head_].size = 0;

    head_ = (head_ + 1) % size_;
  }


 private:
  size_t size_;

  size_t head_;

  size_t tail_;

  transition* buffer_;

  std::mutex mutex_;


  bool isFull() { return (tail_ + 1) % size_ == head_; }

  bool isEmpty() { return head_ == tail_; }
};


class PipelineLoader {
  // This class is used to load data from disk to GPU memory in a pipeline

  // fashion. It loads a block of data from disk to CPU memory and copies it to

  // GPU memory in background threads.

 public:
  // explicit PipelineLoader(int num_threads): num_threads_(num_threads) {}

  PipelineLoader(int num_threads, size_t ring_buffer_size = 0)

      : num_threads_(num_threads),

        file_size_(0),

        ring_buffer_size_(ring_buffer_size),

        ring_buffer_(ring_buffer_size)
  {
  }

  ~PipelineLoader() = default;


  int load(const char* filename, int device_id, void** devPtr_p);

  int loadAsync(const char* filename, int device_id, void** devPtr_p);

  bool waitUntilReady();

  int readCopy(const char* filename, int device_id, void** devPtr_p);

  int backgroundLoad(size_t offset, size_t size);

  int backgroundRead(size_t offset, size_t size);

  int backgroundCopy();


 private:
  int num_threads_;

  size_t file_size_;

  int device_id_;

  // future to wait for the background threads to finish

  std::vector<std::future<int>> futures_;

  std::vector<std::thread> threads_;

  int fd_;

  void* devPtr_;

  size_t ring_buffer_size_;

  RingBuffer ring_buffer_;
};