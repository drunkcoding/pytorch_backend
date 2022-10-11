#include "eventloop_thread.h"

#include "eventloop.h"
#include "eventloop_libevent.h"


EventLoopThread::EventLoopThread(
    const CreateLoopHandleCb& cb1, const ThreadInitCb& cb2,
    const std::string& name)
    : name_(name), loop_(NULL), create_loop_handle_cb_(cb1),
      thread_init_cb_(cb2)
{
}

EventLoopThread::~EventLoopThread()
{
  if (loop_ != NULL) {
    // not 100% race-free, eg. ThreadFunc could be running callback_.
    // still a tiny chance to call destructed object, if ThreadFunc exits just
    // now. but when EventLoopThread destructs, usually programming is exiting
    // anyway.
    loop_->Quit();
    thread_.join();
  }
}

EventLoop*
EventLoopThread::StartLoop()
{
  assert(!thread_.joinable());
  // thread_.start();
  thread_ = std::thread(std::bind(&EventLoopThread::ThreadFunc, this));
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (loop_ == NULL) {
      cond_.wait(lock);
    }
  }
  return loop_;
}

void
EventLoopThread::ThreadFunc()
{
  EventLoop* loop = new EventLoopLibevent(name_, create_loop_handle_cb_);
  if (thread_init_cb_) {
    thread_init_cb_(loop);
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    loop_ = loop;
    cond_.notify_one();
  }
  loop_->Start();
  loop_ = NULL;
}
