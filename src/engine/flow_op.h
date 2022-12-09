#pragma once

#include "dataflow/dag_node.h"
#include "dataflow/flow_controller.h"
#include "op_base.h"
#include "utils/enum_utils.h"
#include "utils/data_utils.h"

ENUM_MACRO(FlowOpType, kRecord, kPrefetch)

struct FlowOpRequest {
  FlowOpType op_type;
  FlowEnginePtr engine;
};
struct FlowOpResponse {
  FlowOpType op_type;
};

typedef std::shared_ptr<FlowOpRequest> FlowOpRequestPtr;
typedef std::shared_ptr<FlowOpResponse> FlowOpResponsePtr;

struct FlowRecordRequest : FlowOpRequest {
  DAGNodePtr node;
  InputIDPtr input_id;
  NodeMetaPtr node_meta;
  BackendEnginePtr engine;
  FlowRecordRequest() { op_type = FlowOpType::kRecord; }
};
struct FlowRecordResponse : FlowOpResponse {
  FlowRecordResponse() { op_type = FlowOpType::kRecord; }
};

struct FlowPrefetchRequest : FlowOpRequest {
  DAGNodePtr node;
  BackendEnginePtr engine;
  FlowPrefetchRequest() { op_type = FlowOpType::kPrefetch; }
};
struct FlowPrefetchResponse : FlowOpResponse {
  DAGNodePtr node;
  FlowPrefetchResponse() { op_type = FlowOpType::kPrefetch; }
};

class FlowOpManager : public OpBase {
 public:
  explicit FlowOpManager(muduo::net::EventLoop* loop);
  ~FlowOpManager();

  virtual void Process() {}

  void RecordNode(const FlowOpRequestPtr& request);
  void PrefetchNode(const FlowOpRequestPtr& request);
};