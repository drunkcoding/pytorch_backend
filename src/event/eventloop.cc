#include "eventloop.h"

#include <signal.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "eventloop_thread_pool.h"

// using base::Timestamp;


__thread EventLoop* loop_in_this_thread = 0;

#pragma GCC diagnostic ignored "-Wold-style-cast"
class IgnoreSigPipe {
 public:
  IgnoreSigPipe()
  {
    ::signal(SIGPIPE, SIG_IGN);
    // LOG_TRACE << "Ignore SIGPIPE";
  }
};

IgnoreSigPipe init_obj;


int
CreateEventfd()
{
  int evtfd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (evtfd < 0) {
    fprintf(stderr, "Failed in eventfd");
    abort();
  }
  return evtfd;
}

EventLoopBase::EventLoopBase() : started_(false), thread_id_(THIS_THREAD_ID) {}

EventLoopBase::~EventLoopBase() {}

void
EventLoopBase::AssertInLoopThread()
{
  if (!IsInLoopThread()) {
    fprintf(
        stderr,
        "EventLoop::AssertInLoopThread %p was created in thread id: = %ld, "
        "current thread id: = %ld",
        this, ConvertThreadID(thread_id_), ConvertThreadID(THIS_THREAD_ID));
    assert(false);
  }
}


//=======================================EventLoop============================

EventLoop::EventLoop(const std::string& thread_name)
    : EventLoopBase(), thread_name_(thread_name), loog_arg_func_(NULL),
      wakeup_fd_(CreateEventfd()), timer_fd_(-1), loop_handle_(NULL)
{
  fprintf(
      stdout, "EventLoop created %p in thread id: %ld in thread name: %s", this,
      ConvertThreadID(thread_id_), thread_name_.c_str());
  if (loop_in_this_thread) {
    fprintf(
        stderr, "Another EventLoop %p exists in this thread id: %ld",
        loop_in_this_thread, ConvertThreadID(thread_id_));
    assert(false);
  } else {
    loop_in_this_thread = this;
  }
}

EventLoop::~EventLoop()
{
  ::close(wakeup_fd_);
}

int
EventLoop::Wakeup()
{
  uint64_t one = 1;
  ssize_t n = ::write(wakeup_fd_, &one, sizeof one);
  if (n != sizeof one) {
    fprintf(stderr, "EventLoop::Wakeup() writes %zd bytes instead of 8", n);
    return -1;
  }
  return 0;
}

int
EventLoop::WakeupReadCb()
{
  uint64_t one = 1;
  ssize_t n = ::read(wakeup_fd_, &one, sizeof one);
  if (n != sizeof one) {
    fprintf(
        stderr, "EventLoop::WakeupReadCb() reads %zd bytes instead of 8", n);
    return -1;
  }
  DoPendingFunctors();
  return 0;
}

void
EventLoop::DoPendingFunctors()
{
  std::deque<Functor> functors;
  std::deque<void*> tasks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    //使用swap减少了加锁时间，也防止嵌套调用QueueInLoop死锁
    functors.swap(pending_functors_);
    tasks.swap(pending_tasks_);
  }
  for (size_t i = 0; i < functors.size(); ++i) {
    functors[i]();
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    loog_arg_func_(tasks[i]);
  }
}

void
EventLoop::RunInLoop(Functor cb)
{
  if (IsInLoopThread()) {
    cb();
  } else {
    QueueInLoop(std::move(cb));
  }
}

int
EventLoop::QueueInLoopFront(Functor cb)
{
  bool empty = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    empty = pending_functors_.empty();
    pending_functors_.push_front(std::move(cb));
  }
  if (empty)
    return Wakeup();
  return 0;
}
void
EventLoop::RunInLoopFront(Functor cb)
{
  if (IsInLoopThread()) {
    cb();
  } else {
    QueueInLoopFront(std::move(cb));
  }
}

int
EventLoop::QueueInLoop(Functor cb)
{
  bool empty = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    empty = pending_functors_.empty();
    pending_functors_.push_back(std::move(cb));
  }
  if (empty)
    return Wakeup();
  return 0;
}

void
EventLoop::RunInLoop(void* arg)
{
  if (IsInLoopThread()) {
    loog_arg_func_(arg);
  } else {
    QueueInLoop(arg);
  }
}

int
EventLoop::QueueInLoop(void* arg)
{
  bool empty = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    empty = pending_tasks_.empty();
    pending_tasks_.push_back(arg);
  }
  // TODO 这里使用event base 的循环无法修改，每次QueueInLoop
  //都需要唤醒，多次唤醒会触发几次可读事件回调 ？？？
  //  if (fake_pending_wake_ && fake_wake_count_ < 100) {
  //    ++fake_wake_count_;
  //  } else {
  //    fake_wake_count_ = 0;
  //  }
  if (empty)
    return Wakeup();
  return 0;
}

std::size_t
EventLoop::GetLoopLoad() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_tasks_.size() + pending_functors_.size();
}

// TimerId
// EventLoop::RunAt(base::Timestamp time, TimerCb cb)
// {
//   return timer_queue_ptr_->AddTimer(std::move(cb), time, 0.0);
// }

// TimerId
// EventLoop::RunAfter(double delay, TimerCb cb)
// {
//   Timestamp time(base::addTime(Timestamp::now(), delay));
//   return RunAt(time, std::move(cb));
// }

// TimerId
// EventLoop::RunEvery(double interval, TimerCb cb)
// {
//   Timestamp time(base::addTime(Timestamp::now(), interval));
//   return timer_queue_ptr_->AddTimer(std::move(cb), time, interval);
// }
// void
// EventLoop::CancelTimer(const TimerId& timer_id)
// {
//   timer_queue_ptr_->Cancel(timer_id);
// }
