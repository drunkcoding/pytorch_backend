#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <deque>

#include "callbacks.h"
#include "loop_handle.h"

class LoopHandle;


int CreateEventfd();

#define THIS_THREAD_ID std::this_thread::get_id()

inline std::size_t
ConvertThreadID(const std::thread::id& id)
{
  std::stringstream ss;
  ss << id;
  std::size_t ret;
  ss >> ret;
  return ret;
}

struct Option {
  enum LoopSelectStrategy {
    kRoundRobin,
    kEmptyOne,
    kLightestOne,
  };

  Option() : loop_strategy(kRoundRobin), reuse_port(true) {}

  LoopSelectStrategy loop_strategy;
  bool reuse_port;
};


class EventLoopBase {
 public:
  typedef std::function<void()> Functor;

  EventLoopBase();
  virtual ~EventLoopBase();

  bool IsInLoopThread() const { return thread_id_ == THIS_THREAD_ID; }

  virtual void Start() = 0;
  virtual void Quit() = 0;

  virtual int QueueInLoop(Functor cb) = 0;
  virtual void RunInLoop(Functor cb) = 0;

  virtual int QueueInLoop(void* arg) = 0;
  virtual void RunInLoop(void* arg) = 0;

  virtual int Wakeup() = 0;

  void AssertInLoopThread();

 protected:
  bool started_;
  const std::thread::id thread_id_;
};

//================================EventLoop===================================
class EventLoop : public EventLoopBase {
 public:
  typedef std::function<void()> Functor;

  explicit EventLoop(const std::string& thread_name);
  virtual ~EventLoop();

  inline LoopHandle* GetLoopHandle() { return loop_handle_; }
  const std::string& thread_name() const { return thread_name_; }

  virtual int QueueInLoop(Functor cb);
  virtual void RunInLoop(Functor cb);

  virtual int QueueInLoopFront(Functor cb);
  virtual void RunInLoopFront(Functor cb);

  virtual int QueueInLoop(void* arg);
  virtual void RunInLoop(void* arg);

  virtual int Wakeup();

  virtual std::size_t GetLoopLoad() const;

  // //@brief Runs callback at 'time'.
  // // Safe to call from other threads.
  // TimerId RunAt(base::Timestamp time, TimerCb cb);
  // //@brief Runs callback after @c delay seconds.
  // // Safe to call from other threads.
  // TimerId RunAfter(double delay, TimerCb cb);
  // //@brief Runs callback every @c interval seconds.
  // // Safe to call from other threads.
  // TimerId RunEvery(double interval, TimerCb cb);
  // //@brief Cancels the timer.
  // // Safe to call from other threads.
  // void CancelTimer(const TimerId& timer_id);

  virtual void Start() = 0;
  virtual void Quit() = 0;

  LoopArgFunc loog_arg_func() const { return loog_arg_func_; }

  void set_loog_arg_func(LoopArgFunc loop_arg_func)
  {
    loog_arg_func_ = loop_arg_func;
  }

 protected:
  int WakeupReadCb();  // waked up
  virtual void DoPendingFunctors();

  std::string thread_name_;
  LoopArgFunc loog_arg_func_;
  int wakeup_fd_;
  int timer_fd_;
  // std::unique_ptr<TimerQueueEvent> timer_queue_ptr_;
  mutable std::mutex mutex_;
  std::deque<Functor> pending_functors_;  // @GuardedBy mutex_
  std::deque<void*> pending_tasks_;       // @GuardedBy mutex_
  LoopHandle* loop_handle_;
};
