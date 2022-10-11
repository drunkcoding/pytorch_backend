#pragma once

#include "op_base.h"
#include "utils/enum_utils.h"
#include "dataflow/dag_node.h"
#include "dataflow/flow_controller.h"

ENUM_MACRO(FlowOpType, kRecord, kPrefetch)

struct FlowOpRequest : OpRequest {
  FlowOpType op_type;
};
struct FlowOpResponse : OpResponse {
  FlowOpType op_type;
};

typedef std::shared_ptr<FlowOpRequest> FlowOpRequestPtr;
typedef std::shared_ptr<FlowOpResponse> FlowOpResponsePtr;

struct FlowRecordRequest : FlowOpRequest {
  DAGNodePtr node;
  std::string request_id;
  NodeMetaPtr node_meta;
  LoopHandle* handle;
  FlowRecordRequest() { op_type = FlowOpType::kRecord; }
};
struct FlowRecordResponse : FlowOpResponse {
  FlowRecordResponse() { op_type = FlowOpType::kRecord; }
};

struct FlowPrefetchRequest : FlowOpRequest {
  DAGNodePtr node;
  LoopHandle* handle;
  FlowPrefetchRequest() { op_type = FlowOpType::kPrefetch; }
};
struct FlowPrefetchResponse : FlowOpResponse {
  DAGNodePtr node;
  FlowPrefetchResponse() { op_type = FlowOpType::kPrefetch; }
};