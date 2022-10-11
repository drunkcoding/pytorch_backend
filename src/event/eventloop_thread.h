#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "callbacks.h"

class EventLoop;

class EventLoopThread {
 public:
  EventLoopThread(
      const CreateLoopHandleCb& cb1 = CreateLoopHandleCb(),
      const ThreadInitCb& cb2 = ThreadInitCb(),
      const std::string& name = std::string());
  ~EventLoopThread();
  EventLoop* StartLoop();

 private:
  void ThreadFunc();
  std::string name_;
  EventLoop* loop_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cond_;
  CreateLoopHandleCb create_loop_handle_cb_;
  ThreadInitCb thread_init_cb_;
};

