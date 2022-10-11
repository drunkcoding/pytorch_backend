#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "dataflow/dag_node.h"
#include "utils/class_utils.h"
#include "utils/lru_cache.h"

class NodeFlow;
typedef std::shared_ptr<NodeFlow> NodeFlowPtr;

typedef std::unordered_map<std::size_t, double> ModelProbabilityMap;
typedef std::vector<std::pair<DAGNodePtr, double>> ModelProbabilityVec;
typedef std::unordered_map<std::size_t, NodeFlowPtr> NodeFlowPtrMap;

struct NodeMeta {
  std::size_t node_id;
  std::size_t visit_cnt;
  std::size_t input_size_cnt;
  std::size_t exec_lat_us_cnt;
  std::size_t load_lat_us_cnt;
  std::size_t unload_lat_us_cnt;

  NodeMeta()
      : visit_cnt(0), input_size_cnt(0), exec_lat_us_cnt(0), load_lat_us_cnt(0),
        unload_lat_us_cnt(0)
  {
  }

  NodeMeta& operator+=(const NodeMeta& other)
  {
    visit_cnt += other.visit_cnt;
    input_size_cnt += other.input_size_cnt;
    exec_lat_us_cnt += other.exec_lat_us_cnt;
    load_lat_us_cnt += other.load_lat_us_cnt;
    unload_lat_us_cnt += other.unload_lat_us_cnt;
    return *this;
  }

  NodeMeta& operator-=(const NodeMeta& other)
  {
    visit_cnt -= other.visit_cnt;
    input_size_cnt -= other.input_size_cnt;
    exec_lat_us_cnt -= other.exec_lat_us_cnt;
    load_lat_us_cnt -= other.load_lat_us_cnt;
    unload_lat_us_cnt -= other.unload_lat_us_cnt;
    return *this;
  }
};
typedef std::shared_ptr<NodeMeta> NodeMetaPtr;
typedef std::list<NodeMetaPtr> NodeMetaPtrList;

class NodeFlow : std::enable_shared_from_this<NodeFlow> {
 public:
  explicit NodeFlow(const DAGNodePtr& node) : node_(node)
  {
    node_meta_ = std::make_shared<NodeMeta>();
    node_meta_->node_id = node->GetNodeID();
  }

  DAGNodePtr GetNode() { return node_; }
  std::size_t GetNodeID() { return node_->GetNodeID(); }
  NodeMetaPtr GetNodeMeta() { return node_meta_; }

  // Add a parent node, called at child when a new node is constructed
  void AddPrevNode(const NodeFlowPtr& prev_node);
  // Add a child node, called at parent when a new node is constructed
  // If the child node already exists, increase the visit count
  void AddNextNode(const NodeFlowPtr& next_node);
  // Remove a child node, called at parent when a request id is deleted
  // Decrease the visit count before actually removing the child node
  //   void RemoveNextNode(const NodeFlowPtr& node);

  // The request id is removed from cache, subtracting the node meta from
  // current record
  void DereferenceNode(const NodeMetaPtr& node_meta);

  const NodeFlowPtrMap& GetNextNodes() const { return next_nodes_; }
  const NodeFlowPtrMap& GetPrevNodes() const { return prev_nodes_; }

 private:
  DAGNodePtr node_;
  NodeFlowPtrMap next_nodes_;
  NodeFlowPtrMap prev_nodes_;
  NodeMetaPtr node_meta_;
};

class FlowController : public noncopyable {
 public:
  STATIC_GET_INSTANCE(FlowController)
  DISABLE_COPY_AND_ASSIGN(FlowController)

  void RecordNodeFlow(const std::string& request_id, const DAGNodePtr& node, const NodeMetaPtr& node_meta);
  ModelProbabilityVec GetChildernProbability(const DAGNodePtr& node);
  ModelProbabilityVec GetTreeProbability(const DAGNodePtr& node);
  //   DAGNodePtr GetDAGNode(const std::size_t& model_id);

  //   void RemoveModelFlow(
  //       const std::string& model_name, const std::string& model_version);
  //   void ClearModelFlow();

 private:
  FlowController() : request_trace_(100) {}
  virtual ~FlowController() = default;

  void RecursivelyUpdateProbability(
      const NodeFlowPtr& node_flow, ModelProbabilityVec& prob_map);

  LRUCache<std::size_t, NodeMetaPtrList> request_trace_;
  std::unordered_map<std::size_t, NodeFlowPtr> flow_graph_;
};
