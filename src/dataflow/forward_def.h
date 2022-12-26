#pragma once

#include <memory>

class BackendEngine;
typedef std::shared_ptr<BackendEngine> BackendEnginePtr;

class LibtorchEngine;
typedef std::shared_ptr<LibtorchEngine> LibtorchEnginePtr;

class FlowEngine;
typedef std::shared_ptr<FlowEngine> FlowEnginePtr;

// struct OpRequest;
// typedef std::shared_ptr<OpRequest> RequestPtr;
// struct OpResponse;
// typedef std::shared_ptr<OpResponse> ResponsePtr;

// struct BackendRequest;
// typedef std::shared_ptr<BackendRequest> BackendRequestPtr;
// struct BackendResponse;
// typedef std::shared_ptr<BackendResponse> BackendResponsePtr;

// struct LibtorchRequest;
// typedef std::shared_ptr<LibtorchRequest> LibtorchRequestPtr;
// struct LibtorchResponse;
// typedef std::shared_ptr<LibtorchResponse> LibtorchResponsePtr;

/*
Maximum number of parameter elements to fetch ahead of use. Used by ZeRO3,
ZeRO3-Offload, ZeRO-Infinity, and ZeRO-Inference.
*/
#ifndef PREFETCH_BUCKET_SIZE
#define PREFETCH_BUCKET_SIZE 500000000LL
#endif

/*
The maximum number of parameters resident per GPU before releasing. Smaller
values use less memory, but perform more communication.
*/
#ifndef MAX_LIVE_PARAMETERS
#define MAX_LIVE_PARAMETERS 1500000000LL
#endif


/*
Do not release a parameter if it will be reused within this threshold of
parameters. Smaller values use less memory, but perform more communication.
*/
#ifndef MAX_REUSE_DISTANCE
#define MAX_REUSE_DISTANCE 1000000000LL
#endif
