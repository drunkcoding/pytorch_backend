#pragma once

#include <memory>

#define CREATE_INS(ClassName) ClassName::CreateMyself
#define INIT_INS(ClassName) ClassName::ThreadInit
#define SELF(ClassName) std::static_pointer_cast<ClassName>(shared_from_this())
#define SELF_BIND(ClassName, MethodName) \
  std::bind(&ClassName::MethodName, SELF(ClassName))
#define SELF_BIND_ARGS(ClassName, MethodName, ...) \
  std::bind(&ClassName::MethodName, SELF(ClassName), __VA_ARGS__)

#define DEFAULT_CLASS_MEMBER(ClassName) \
  ClassName() = default;                \
  virtual ~ClassName() = default;       \
  virtual const char* Name() const      \
  {                                     \
    return #ClassName;                  \
  }

#define DEFAULT_OP_MEMBER(ClassName)              \
  virtual const char* Name() const                \
  {                                               \
    return #ClassName;                            \
  }                                               \
  explicit ClassName(EventLoop* loop);            \
  virtual ~ClassName() = default;                 \
  static ClassName* CreateMyself(EventLoop* loop) \
  {                                               \
    return new ClassName(loop);                   \
  }                                               \
                                                  \
 protected:                                       \
  virtual void Process();

#define DEFAULT_LOOPHANDLE_MEMBER(ClassName)       \
  ClassName(EventLoop* loop);                      \
  static void ThreadInit(EventLoop* loop);         \
  static LoopHandle* CreateMyself(EventLoop* loop) \
  {                                                \
    return new ClassName(loop);                    \
  }

#define STATIC_GET_INSTANCE(CLASSNAME) \
  static CLASSNAME* GetInstance()      \
  {                                    \
    static CLASSNAME instance;         \
    return &instance;                  \
  }

#define DISABLE_COPY_AND_ASSIGN(CLASSNAME) \
  CLASSNAME(const CLASSNAME&) = delete;    \
  CLASSNAME& operator=(const CLASSNAME&) = delete;


#define GET_INSTANCE(CLASSNAME) CLASSNAME::GetInstance()
