#pragma once

#include <boost/preprocessor.hpp>
#include <string>

#define FACTORY_STATIC_GET_INSTANCE(CLASSNAME) \
  static CLASSNAME* GetInstance()              \
  {                                            \
    static CLASSNAME instance;                 \
    return &instance;                          \
  }



#define DISABLE_COPY_AND_ASSIGN(CLASSNAME) \
  CLASSNAME(const CLASSNAME&) = delete;    \
  CLASSNAME& operator=(const CLASSNAME&) = delete;

class SingletonFactory {
 public:
  FACTORY_STATIC_GET_INSTANCE(SingletonFactory)
  DISABLE_COPY_AND_ASSIGN(SingletonFactory)
 protected:
  SingletonFactory() = default;
  ~SingletonFactory() = default;
};

#define GET_INSTANCE(CLASSNAME) CLASSNAME::GetInstance()
