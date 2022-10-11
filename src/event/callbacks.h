#pragma once 

#include <stdint.h>
#include <functional>
#include <memory>


// All client visible callbacks go here.
class EventLoop;
class LoopHandle;

typedef std::function<void()> TimerCb;

typedef std::function<void(EventLoop*)> ThreadInitCb;
typedef std::function<LoopHandle*(EventLoop*)> CreateLoopHandleCb;

typedef int (*LoopArgFunc) (void *arg);

