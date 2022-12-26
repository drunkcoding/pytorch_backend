#pragma once

#include <functional>

#include "topology.h"

typedef std::function<void()> DefaultCb;
typedef std::function<FilterResult(const std::int64_t)> SizeFilterFunc;