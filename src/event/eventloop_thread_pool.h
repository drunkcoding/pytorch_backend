#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "callbacks.h"

class EventLoop;
class EventLoopThread;


class EventLoopThreadPool {
 public:
  EventLoopThreadPool(EventLoop* base_loop, const std::string& name);
  ~EventLoopThreadPool();

  void Start(
      const CreateLoopHandleCb& cb1, const ThreadInitCb& cb2 = ThreadInitCb());

  // valid after calling Start()
  // 轮流的的方式获取event_loop
  EventLoop* GetNextLoop();
  // 获取一个不负责任何连接的event_loop
  EventLoop* GetEmptyLoop();
  // 获取一个负载最轻的event_loop
  EventLoop* GetLightestLoop();

  /// with the same hash code, it will always return the same EventLoop
  EventLoop* GetLoopForHash(size_t hashCode);

  std::vector<EventLoop*> GetAllLoops();

  void set_thread_num(int num) { thread_num_ = num; }

  int thread_num() const { return thread_num_; }

  void set_name(const std::string& name) { name_ = name; }

  const std::string& name() const { return name_; }

  bool is_started() const { return started_; }


 private:
  EventLoop* base_loop_;
  std::string name_;
  bool started_;
  int thread_num_;
  int next_;
  std::vector<std::unique_ptr<EventLoopThread>> threads_;
  std::vector<EventLoop*> loops_;
};
