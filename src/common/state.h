#pragma once

#include "utils/enum_utils.h"

ENUM_MACRO(
    MemoryType,
    kReady,   // memory is in GPU and ready to run
    kLocked,  // memory is in GPU and used by by model
    kMoving,  // memory is not in GPU, memory movement is in progress
    kStandBy  // memory is not in GPU, memory movement is finished
)