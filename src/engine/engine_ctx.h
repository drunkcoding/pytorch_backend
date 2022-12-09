#pragma once

#include "muduo/base/noncopyable.h"
#include "flow_engine.h"
#include "libtorch_engine.h"
#include "utils/log_utils.h"

class EngineCtx : public muduo::noncopyable {
 public:
  STATIC_GET_INSTANCE(EngineCtx)
  DISABLE_COPY_AND_ASSIGN(EngineCtx)

  void ProcessLibtorchRequest(const LibtorchRequestPtr& request)
  {
    LOG_TRITON_VERBOSE("EngineCtx::ProcessRequest LibtorchRequest");
    libtorch_engine_->ProcessRequest(request);
  }
  void ProcessFlowRequest(const FlowOpRequestPtr& request)
  {
    LOG_TRITON_VERBOSE("EngineCtx::ProcessRequest FlowOpRequestPtr");
    flow_engine_->ProcessRequest(request);
  }

 private:
  EngineCtx()
  {
    libtorch_engine_ = std::make_shared<LibtorchEngine>();
    flow_engine_ = std::make_shared<FlowEngine>();
  }
  ~EngineCtx() {}

 private:
  std::shared_ptr<LibtorchEngine> libtorch_engine_;
  std::shared_ptr<FlowEngine> flow_engine_;
};

#define ENGINE_CTX EngineCtx::GetInstance()