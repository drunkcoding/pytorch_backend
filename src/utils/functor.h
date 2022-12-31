#pragma once

#include <functional>
#include "torch_utils.h"
#include "topology.h"

typedef std::function<void()> DefaultCb;
typedef std::function<FilterResult(const std::int64_t, const Device&)> SizeFilterFunc;