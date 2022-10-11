#pragma once

#include <assert.h>

#include <atomic>

#include "eventloop.h"

class LoopHandle {
 public:
  LoopHandle() : weight_(0), refs_(0) {}

  EventLoop* GetLoop() { return loop_; }

  void SetLoop(EventLoop* loop)
  {
    loop_ = loop;
    return;
  }
  inline uint64_t GetWeight() const { return weight_.load(); }

  inline void SetWeight(uint64_t w) { weight_.store(w); }

  inline int32_t GetRefs() const { return refs_.load(); }

  inline void IncRefs() { refs_.fetch_add(1); }

  inline void DecRefs()
  {
    refs_.fetch_sub(1);
    assert(refs_.load() >= 0);
  }

 protected:
  EventLoop* loop_;

 private:
  std::atomic<uint64_t> weight_;
  std::atomic<int32_t> refs_;
};

