#include "stream_ctrl.h"


// Stream0 is used for H2D, Stream1 is used for Kernel, Stream2 is used for D2H
CudaStreamPool* kCudaStreamPool = CudaStreamPool::GetInstance();

