#include "gds_load.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <future>
#include <thread>

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define checkCublasErrors(call)                               \
  do {                                                        \
    cublasStatus_t err = call;                                \
    if (err != CUBLAS_STATUS_SUCCESS) {                       \
      printf("CUBLAS error at %s %d: %d\n", __FILE__, __LINE__, err); \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



int GDSLoader::load(const char *filename, int device_id, void** devPtr_p) {
  void *devPtr = nullptr;

  // create a cuda stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int fd = open(filename, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cout << "Error opening file: " << strerror(errno) << std::endl;
    return 1;
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    std::cout << "Error getting file size: " << strerror(errno) << std::endl;
    return 1;
  }
  size_t size = st.st_size;

  // use GPU Direct Storage to read the file
  CUfileError_t status;
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;

  memset(&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "Error registering file handle: " << status.err << std::endl;
    close(fd);
    return 1;
  }

  checkCudaErrors(cudaMalloc(&devPtr, size));
  checkCudaErrors(cudaMemset(devPtr, 0, size));
  // checkCudaErrors(cudaDeviceSynchronize(0));
  // std::cout << "Allocated " << size << " bytes on GPU" << std::endl;

  ssize_t ret = cuFileRead(cf_handle, devPtr, size, 0, 0);
  if (ret < 0 || ret != size) {
    std::cerr << "Error reading file: " << ret << std::endl;
    cuFileHandleDeregister(cf_handle);
    close(fd);
    checkCudaErrors(cudaFree(devPtr));
    return 1;
  }

  // checkCudaErrors(cudaDeviceSynchronize(0));
  // std::cout << "Read " << ret << " bytes from file" << std::endl;
  cuFileHandleDeregister(cf_handle);
  close(fd);
  // checkCudaErrors(cudaFree(devPtr));
  *devPtr_p = devPtr;
  return 0;
}

int GDSLoader::loadAsync(const char *filename, int device_id, void** devPtr_p, std::future<ssize_t>& future_p) {
  cudaInit();
  return loadAsyncInited(filename, device_id, devPtr_p, future_p);
}

int GDSLoader::loadAsyncInited(const char* filename, int device_id,
                               void** devPtr_p,
                               std::future<ssize_t>& future_p) {
  if (cuda_inited_ == false) {
    std::cerr << "Error: cuda not inited" << std::endl;
    return -1;
  }
  devPtr_ = nullptr;

  fd_ = open(filename, O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    std::cout << "Error opening file: " << strerror(errno) << std::endl;
    return 1;
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    std::cout << "Error getting file size: " << strerror(errno) << std::endl;
    return 1;
  }
  size_ = st.st_size;

  checkCudaErrors(cudaMalloc(&devPtr_, size_));
  checkCudaErrors(cudaMemset(devPtr_, 0, size_));

  // create a cuFileRead function and lunch it asynchronously
  auto cu_file_read = [&]() {
    // use GPU Direct Storage to read the file
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd_;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "Error registering file handle: " << status.err << std::endl;
      close(fd_);
      return (ssize_t)1;
    }
    ssize_t ret = cuFileRead(cf_handle, devPtr_, size_, 0, 0);
    if (ret < 0 || ret != size_) {
      std::cerr << "Error reading file: " << ret << std::endl;
      cuFileHandleDeregister(cf_handle);
      close(fd_);
      checkCudaErrors(cudaFree(devPtr_));
      return ret;
    }
    cuFileHandleDeregister(cf_handle);
    close(fd_);
    return ret;
  };

  // launch the cuFileRead function asynchronously
  std::future<ssize_t> future = std::async(std::launch::async, cu_file_read);
  future_p = std::move(future);
  // std::future<int> result = std::async(std::launch::async, cu_file_read);

  *devPtr_p = devPtr_;
  return 0;
}

bool GDSLoader::cudaInit() {
  // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (cuda_inited_ == false) {
    checkCudaErrors(cudaStreamCreate(&stream_));
    // create cublas handle
    checkCublasErrors(cublasCreate(&cublas_handle_));
    cuda_inited_ = true;
  }
  // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return cuda_inited_;
}

int PipelineLoader::load(const char* filename, int device_id, void** devPtr_p) {
  device_id_ = device_id;
  fd_ = open(filename, O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    std::cout << "Error opening file: " << strerror(errno) << std::endl;
    return 1;
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    std::cout << "Error getting file size: " << strerror(errno) << std::endl;
    return 1;
  }
  size_t size = st.st_size;

  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(cudaMalloc(devPtr_p, size));
  checkCudaErrors(cudaMemset(*devPtr_p, 0, size));
  devPtr_ = *devPtr_p;

  // start background threads
  // total number of blocks, note that the last block may be smaller than
  // block_size
  size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // block per thread
  size_t bpt = (num_blocks + num_threads_ - 1) / num_threads_;
  // std::cout << "num_blocks: " << num_blocks << ", bpt: " << bpt << std::endl;
  
  for (int i = 0; i < num_threads_; i++) {
    size_t offset = i * bpt * BLOCK_SIZE;
    size_t thread_size = std::min(bpt * BLOCK_SIZE, size - offset);
    threads_.emplace_back(&PipelineLoader::backgroundLoad, this, offset, thread_size);
  }

  for (auto& t : threads_) {
    t.join();
  }

  return 0;
}

int PipelineLoader::loadAsync(const char* filename, int device_id, void** devPtr_p) {
  device_id_ = device_id;
  fd_ = open(filename, O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    std::cout << "Error opening file: " << strerror(errno) << std::endl;
    return 1;
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    std::cout << "Error getting file size: " << strerror(errno) << std::endl;
    return 1;
  }
  size_t size = st.st_size;

  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(cudaMalloc(devPtr_p, size));
  checkCudaErrors(cudaMemset(*devPtr_p, 0, size));
  devPtr_ = *devPtr_p;

  // start background threads
  // total number of blocks, note that the last block may be smaller than
  // block_size
  size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // block per thread
  size_t bpt = (num_blocks + num_threads_ - 1) / num_threads_;
  // std::cout << "num_blocks: " << num_blocks << ", bpt: " << bpt << std::endl;
  
  for (int i = 0; i < num_threads_; i++) {
    size_t offset = i * bpt * BLOCK_SIZE;
    size_t thread_size = std::min(bpt * BLOCK_SIZE, size - offset);
    futures_.emplace_back(std::async(std::launch::async, &PipelineLoader::backgroundLoad, this, offset, thread_size));
    // threads_.emplace_back(&PipelineLoader::backgroundLoad, this, offset, thread_size);
  }

  return 0;
}

bool PipelineLoader::waitUntilReady() {
  for (auto& f : futures_) {
    f.wait();
  }
  return true;
}

int PipelineLoader::backgroundLoad(size_t offset, size_t size) {
  checkCudaErrors(cudaSetDevice(device_id_));
  void* buf = aligned_alloc(4096, BLOCK_SIZE);
  while (size > 0) {
    // read data from disk, note the last block may be smaller than block_size
    size_t read_size = std::min((size_t)BLOCK_SIZE, size);
    ssize_t ret = pread(fd_, buf, read_size, offset);
    if (ret < 0) {
      std::cout << "Error reading file: " << strerror(errno) << std::endl;
      return 1;
    }

    // copy data to GPU
    checkCudaErrors(cudaMemcpy((char*)devPtr_ + offset, buf, ret, cudaMemcpyHostToDevice));

    // update offset and size
    offset += ret;
    size -= ret;
    // std::cout << "offset: " << offset << ", size: " << size << ", ret: " << ret << std::endl;
  }

  free(buf);
  return 0;
}

int PipelineLoader::backgroundRead(size_t offset, size_t size) {
  while (size > 0) {
    // allocate a buffer from ring buffer
    transition* t = ring_buffer_.enqueue();
    if (t == nullptr) {
      // std::cerr << "Queue is full, waiting..." << std::endl;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    // read data from disk, note the last block may be smaller than block_size
    size_t read_size = std::min((size_t)BLOCK_SIZE, size);
    ssize_t ret = pread(fd_, t->buffer, read_size, offset);
    if (ret < 0) {
      std::cout << "Error reading file: " << strerror(errno) << std::endl;
      return 1;
    }

    t->offset = offset;
    t->size = ret;

    // update offset and size
    offset += ret;
    size -= ret;
    // std::cout << "offset: " << offset << ", size: " << size << ", ret: " << ret << std::endl;
  }
  return 0;
}

int PipelineLoader::backgroundCopy() {
  checkCudaErrors(cudaSetDevice(device_id_));
  size_t copied_size = 0;
  while (copied_size < file_size_) {
    // get a buffer from ring buffer
    transition* t = ring_buffer_.getHead();
    if (t == nullptr || t->size == 0) {
      // std::cout << "no data to copy" << std::endl;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    // copy data to GPU
    checkCudaErrors(cudaMemcpy((char*)devPtr_ + t->offset, t->buffer, t->size, cudaMemcpyHostToDevice));
    copied_size += t->size;
    ring_buffer_.dequeue();
    // std::cout << "copied_size: " << copied_size << std::endl;
  }
  return 0;
}

int PipelineLoader::readCopy(const char* filename, int device_id, void** devPtr_p) {
  if (ring_buffer_size_ == 0) {
    std::cerr << "ring buffer size must be greater than 0" << std::endl;
    return load(filename, device_id, devPtr_p);
  }
  device_id_ = device_id;
  fd_ = open(filename, O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    std::cout << "Error opening file: " << strerror(errno) << std::endl;
    return 1;
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    std::cout << "Error getting file size: " << strerror(errno) << std::endl;
    return 1;
  }
  file_size_ = st.st_size;

  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(cudaMalloc(devPtr_p, file_size_));
  checkCudaErrors(cudaMemset(*devPtr_p, 0, file_size_));
  devPtr_ = *devPtr_p;

  // start background threads
  // total number of blocks, note that the last block may be smaller than
  // block_size
  size_t num_blocks = (file_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // block per thread
  size_t bpt = (num_blocks + num_threads_ - 1) / num_threads_;
  // std::cout << "num_blocks: " << num_blocks << ", bpt: " << bpt << std::endl;
  
  for (int i = 0; i < num_threads_; i++) {
    size_t offset = i * bpt * BLOCK_SIZE;
    size_t thread_size = std::min(bpt * BLOCK_SIZE, file_size_ - offset);
    threads_.emplace_back(&PipelineLoader::backgroundRead, this, offset, thread_size);
  }

  threads_.emplace_back(&PipelineLoader::backgroundCopy, this);

  for (auto& t : threads_) {
    t.join();
  }

  return 0;
}