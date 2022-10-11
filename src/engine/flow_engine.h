#include "engine_base.h"
#include "flow_op.h"

class FlowEngineHandle
    : public EngineHandleBase,
      public std::enable_shared_from_this<FlowEngineHandle> {
 public:
  DEFAULT_LOOPHANDLE_MEMBER(FlowEngineHandle)

 public:
  virtual void DispachRequest(const RequestPtr& request);

 private:
  void RegisterService();

  void RecordNode(const FlowOpRequestPtr& request);
  void PrefetchNode(const FlowOpRequestPtr& request);
};

class FlowEngine : public EngineBase,
                      public std::enable_shared_from_this<FlowEngine> {
 public:
  // DEFAULT_CLASS_MEMBER(FlowEngine)
  STATIC_GET_INSTANCE(FlowEngine)
  FlowEngine();
  ~FlowEngine();

  LoopHandle* GetHandle() { return loop_->GetLoopHandle(); }
};
