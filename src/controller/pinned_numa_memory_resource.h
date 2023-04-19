#pragma once

#include <numa.h>
#include <sys/mman.h>

#include <cstddef>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>
#include <utility>

namespace rmm::mr {


class bad_numa_alloc : public std::bad_alloc {
 public:
  bad_numa_alloc() : std::bad_alloc() {}
  bad_numa_alloc(const char* msg) : std::bad_alloc(), msg_(msg) {}
  const char* what() const noexcept override { return msg_; }

 private:
  const char* msg_;
};

/*
 * @brief A `host_memory_resource` that uses `cudaMallocHost` to allocate
 * pinned/page-locked host memory.
 *
 * See https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */
class pinned_numa_memory_resource final : public host_memory_resource {
 public:
  pinned_numa_memory_resource() = default;
  ~pinned_numa_memory_resource() override = default;
  pinned_numa_memory_resource(int node) : numa_node_(node) {}
  pinned_numa_memory_resource(pinned_numa_memory_resource const&) = default;
  pinned_numa_memory_resource(pinned_numa_memory_resource&&) = default;
  pinned_numa_memory_resource& operator=(pinned_numa_memory_resource const&) =
      default;
  pinned_numa_memory_resource& operator=(pinned_numa_memory_resource&&) =
      default;

 private:
  /**
   * @brief Allocates pinned memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) {
      return nullptr;
    }

    // If the requested alignment isn't supported, use default
    alignment = (rmm::detail::is_supported_alignment(alignment))
                    ? alignment
                    : rmm::detail::RMM_DEFAULT_HOST_ALIGNMENT;

    return rmm::detail::aligned_allocate(
        bytes, alignment, [this](std::size_t size) {
          void* ptr{nullptr};
          numa_set_preferred(numa_node_);
          //   std::cout << "numa_tonode_memory" << std::endl;

          auto status = cudaMallocHost(&ptr, size);
          if (cudaSuccess != status) {
            throw std::bad_alloc{};
          }
          return ptr;

          //   ptr = numa_alloc_onnode(size, numa_node_);

          //   if (ptr == nullptr) {
          //     throw bad_numa_alloc("numa_alloc_onnode failed");
          //   }

          //   //   if (numa_tonode_memory(ptr, 1024) != numa_node_) {
          //   //     throw bad_numa_alloc("numa_tonode_memory failed");
          //   //   }

          //   std::cout << "numa_tonode_memory" << std::endl;

          //   if (mlock(ptr, size) != 0)  // Lock the memory in physical memory
          //   {
          //     throw bad_numa_alloc("mlock failed");
          //   }

          //   std::cout << "mlock" << std::endl;

          //   return ptr;
        });
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * `ptr` must have been returned by a prior call to
   * `allocate(bytes,alignment)` on a `host_memory_resource` that compares equal
   * to `*this`, and the storage it points to must not yet have been
   * deallocated, otherwise behavior is undefined.
   *
   * @throws Nothing.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned
   * `ptr`.
   * @param alignment Alignment of the allocation. This must be equal to the
   * value of `alignment` that was passed to the `allocate` call that returned
   * `ptr`.
   */
  void do_deallocate(
      void* ptr, std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) override
  {
    if (nullptr == ptr) {
      return;
    }
    rmm::detail::aligned_deallocate(
        ptr, bytes, alignment, [this, bytes](void* ptr) {
          RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
          //   munlock(ptr, bytes);    // Unlock the memory
          //   numa_free(ptr, bytes);  // Free the memory
        });
  }

 private:
  int numa_node_ = -1;
};
}  // namespace rmm::mr
