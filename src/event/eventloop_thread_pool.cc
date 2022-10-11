#include "eventloop_thread_pool.h"

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "eventloop.h"
#include "eventloop_thread.h"

EventLoopThreadPool::EventLoopThreadPool(
    EventLoop* base_loop, const std::string& name)
    : base_loop_(base_loop), name_(name), started_(false), thread_num_(0),
      next_(0)
{
}


EventLoopThreadPool::~EventLoopThreadPool() {}

void
EventLoopThreadPool::Start(
    const CreateLoopHandleCb& cb1, const ThreadInitCb& cb2)
{
  // assert(!started_);
  base_loop_->AssertInLoopThread();
  started_ = true;
  for (int i = 0; i < thread_num_; ++i) {
    char buf[name_.size() + 32];
    snprintf(buf, sizeof buf, "%s%d", name_.c_str(), i);
    EventLoopThread* t = new EventLoopThread(cb1, cb2, buf);
    threads_.push_back(std::unique_ptr<EventLoopThread>(t));
    loops_.push_back(t->StartLoop());
  }
  if (thread_num_ == 0 && cb2) {
    cb2(base_loop_);
  }
  fprintf(stdout, "thread_pool loop size: %ld", loops_.size());
}

EventLoop*
EventLoopThreadPool::GetNextLoop()
{
  base_loop_->AssertInLoopThread();
  EventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    // round-robin
    loop = loops_[next_];
    ++next_;
    if (static_cast<size_t>(next_) >= loops_.size()) {
      next_ = 0;
    }
  }
  return loop;
}

EventLoop*
EventLoopThreadPool::GetEmptyLoop()
{
  base_loop_->AssertInLoopThread();
  assert(started_);
  EventLoop* loop = base_loop_;
  for (uint32_t i = 0; i < loops_.size(); i++) {
    if (loops_[i]->GetLoopHandle()->GetRefs() == 0) {
      fprintf(stdout, "loop index: %d will be used", i);
      return loops_[i];
    }
  }
  if (!loops_.empty()) {
    // 有线程池且没有找到可用的loop返回 NULL
    return NULL;
  } else {  // 没有线程池需要检查base_loop是否为空
    if (loop->GetLoopHandle()->GetRefs() == 0) {
      return loop;
    } else {
      return NULL;
    }
  }
}
// TODO
EventLoop*
EventLoopThreadPool::GetLightestLoop()
{
  base_loop_->AssertInLoopThread();
  assert(started_);
  EventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    // round-robin
    loop = loops_[next_];
    ++next_;
    if (static_cast<size_t>(next_) >= loops_.size()) {
      next_ = 0;
    }
  }
  return loop;
}


EventLoop*
EventLoopThreadPool::GetLoopForHash(size_t hashCode)
{
  base_loop_->AssertInLoopThread();
  EventLoop* loop = base_loop_;

  if (!loops_.empty()) {
    loop = loops_[hashCode % loops_.size()];
  }
  return loop;
}

std::vector<EventLoop*>
EventLoopThreadPool::GetAllLoops()
{
  // base_loop_->AssertInLoopThread();
  // assert(started_);
  if (thread_num_ == 0) {
    return std::vector<EventLoop*>(1, base_loop_);
  } else {
    while ((int)loops_.size() < thread_num_) {
      usleep(100);
    }
    return loops_;
  }
}
