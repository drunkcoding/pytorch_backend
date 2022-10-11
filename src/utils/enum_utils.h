#pragma once

#include <boost/preprocessor.hpp>
#include <string>

#define PROCESS_ONE_ELEMENT(r, unused, idx, elem) \
  BOOST_PP_COMMA_IF(idx) BOOST_PP_STRINGIZE(elem)

#define ENUM_MACRO(name, ...)                                            \
  enum class name { __VA_ARGS__ };                                       \
  static const char* name##Strings[] = {BOOST_PP_SEQ_FOR_EACH_I(         \
      PROCESS_ONE_ELEMENT, % %, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))}; \
  template <typename T>                                                  \
  constexpr const std::string name##ToString(T value)                    \
  {                                                                      \
    return std::string(name##Strings[static_cast<int>(value)]);          \
  }

#define ENUM_ARGS(...) enum : int { __VA_ARGS__ };

#define ENUM_STRUCT_MACRO(name, ...)                                     \
  static const char* name##Strings[] = {BOOST_PP_SEQ_FOR_EACH_I(         \
      PROCESS_ONE_ELEMENT, % %, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))}; \
  template <typename T>                                                  \
  constexpr const std::string name##ToString(T container)                \
  {                                                                      \
    return std::string(name##Strings[container.value]);                  \
  }


class EnumType {
 public:
  explicit EnumType(const int& type) : value(type) {}
  enum : int {
    kInvalid = -1,
  };
  int value;
  EnumType& operator=(const int& value)
  {
    this->value = value;
    return *this;
  }
  EnumType& operator=(const EnumType& type)
  {
    this->value = type.value;
    return *this;
  }
  bool operator==(const EnumType& other) const
  {
    return this->value == other.value;
  }
  std::size_t operator()(const EnumType& type) const
  {
    return std::hash<std::size_t>()(type.value);
  }
};