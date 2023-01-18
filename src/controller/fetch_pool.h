#pragma once

#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "fetch.h"
#include "stream_ctrl.h"
#include "utils/class_utils.h"
#include "utils/functor.h"
#include "utils/topology.h"

/*
 * A Basic Thread Pool using std::thread
 */
class FetchPool : public muduo::noncopyable {
 public:
  void Start(int num_threads)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (running_) {
      lock.unlock();
      return;
    }
    priorities_.resize(NUM_PRIORITY);
    d2h_priorities_.resize(NUM_PRIORITY);

    for (int i = 0; i < num_threads; ++i) {
      auto thread = std::thread(&FetchPool::ThreadFunc, this);
      SetThreadAffinity(thread);
      // SetThreadScheduling(thread, SCHED_RR, 0);
      thread.detach();
      threads_.push_back(std::move(thread));

      auto d2h_thread = std::thread(&FetchPool::D2HThreadFunc, this);
      SetThreadAffinity(d2h_thread);
      // SetThreadScheduling(d2h_thread, SCHED_RR, -10);
      d2h_thread.detach();
      d2h_threads_.push_back(std::move(d2h_thread));

      // auto top_thread = std::thread(&FetchPool::TopThreadFunc, this);
      // SetThreadAffinity(top_thread);
      // SetThreadScheduling(top_thread, SCHED_RR, -20);
      // top_thread.detach();
      // top_threads_.push_back(std::move(top_thread));
    }
    running_ = true;
    top_thread_ = std::thread(&FetchPool::TopThreadFunc, this);

    lock.unlock();
  }
  FetchPool() {}
  ~FetchPool() {}
  STATIC_GET_INSTANCE(FetchPool)
  DISABLE_COPY_AND_ASSIGN(FetchPool)

  void AddTask(const NodePtr& node, std::uint32_t priority, DefaultCb task);
  void AddD2HTask(const NodePtr& node, std::uint32_t priority, DefaultCb task);
  void PrioritizeTask(const NodePtr& node, std::uint32_t priority);
  void RemoveTask(std::uint64_t corr_id);

 private:
  using tasks_t = std::unordered_map<std::uint64_t, DefaultCb>;
  void ThreadFunc();
  void D2HThreadFunc();
  void TopThreadFunc();

  tasks_t& Tasks(std::uint32_t priority) { return priorities_[priority]; }
  tasks_t& D2HTasks(std::uint32_t priority)
  {
    return d2h_priorities_[priority];
  }

 private:
  std::list<std::thread> threads_;
  std::list<std::thread> d2h_threads_;
  std::list<std::thread> top_threads_;
  std::thread top_thread_;
  std::vector<tasks_t> priorities_;
  std::vector<tasks_t> d2h_priorities_;
  std::unordered_map<std::uint64_t, NodePtr> nodes_;
  std::mutex mutex_;
  std::mutex d2h_mutex_;
  // std::mutex top_mutex_;
  std::condition_variable cond_;
  std::condition_variable d2h_cond_;
  // std::condition_variable top_cond_;
  bool running_ = false;
};


#define FETCH_POOL FetchPool::GetInstance()
