#pragma once

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "muduo/base/noncopyable.h"
#include "utils/class_utils.h"
#include "utils/lru_cache.h"
#include "utils/topology.h"

class NodeFlow;
typedef std::shared_ptr<NodeFlow> NodeFlowPtr;

typedef std::unordered_map<std::size_t, double> ModelProbabilityMap;
typedef std::vector<std::pair<NodePtr, double>> ModelProbabilityVec;
typedef std::vector<NodePtr> NodePtrVec;
typedef std::vector<std::pair<NodePtr, Device>> NodeMoveVec;
typedef std::unordered_map<NodeID, NodePtr> NodePtrMap;
typedef std::unordered_map<std::size_t, NodeFlowPtr> NodeFlowPtrMap;

struct NodeMeta {
  std::size_t node_id;
  std::size_t visit_cnt;
  std::size_t input_size_cnt;
  std::size_t exec_lat_ns_cnt;
  std::size_t load_lat_us_cnt;
  std::size_t unload_lat_us_cnt;

  NodeMeta()
      : visit_cnt(0), input_size_cnt(0), exec_lat_ns_cnt(0), load_lat_us_cnt(0),
        unload_lat_us_cnt(0)
  {
  }

  NodeMeta& operator+=(const NodeMeta& other)
  {
    visit_cnt += other.visit_cnt;
    input_size_cnt += other.input_size_cnt;
    exec_lat_ns_cnt += other.exec_lat_ns_cnt;
    load_lat_us_cnt += other.load_lat_us_cnt;
    unload_lat_us_cnt += other.unload_lat_us_cnt;
    return *this;
  }

  NodeMeta& operator-=(const NodeMeta& other)
  {
    visit_cnt -= other.visit_cnt;
    input_size_cnt -= other.input_size_cnt;
    exec_lat_ns_cnt -= other.exec_lat_ns_cnt;
    load_lat_us_cnt -= other.load_lat_us_cnt;
    unload_lat_us_cnt -= other.unload_lat_us_cnt;
    return *this;
  }

  std::string ToString()
  {
    std::string str = "NodeMeta: ";
    str += "visit_cnt: " + std::to_string(visit_cnt) + ", ";
    str += "input_size_cnt: " + std::to_string(input_size_cnt) + ", ";
    str += "exec_lat_ns_cnt: " + std::to_string(exec_lat_ns_cnt) + ", ";
    str += "load_lat_us_cnt: " + std::to_string(load_lat_us_cnt) + ", ";
    str += "unload_lat_us_cnt: " + std::to_string(unload_lat_us_cnt);
    return str;
  }
};
typedef std::shared_ptr<NodeMeta> NodeMetaPtr;
typedef std::list<NodeMetaPtr> NodeMetaPtrList;

// class NodeFlow : std::enable_shared_from_this<NodeFlow> {
//  public:
//   explicit NodeFlow(const NodePtr& node) : node_(node)
//   {
//     node_meta_ = std::make_shared<NodeMeta>();
//     node_meta_->node_id = node->id;
//   }

//   NodePtr GetNode() { return node_; }
//   std::size_t GetNodeID() { return node_->id; }
//   NodeMetaPtr GetNodeMeta() { return node_meta_; }

//   // Add a parent node, called at child when a new node is constructed
//   void AddPrevNode(const NodeFlowPtr& prev_node);
//   // Add a child node, called at parent when a new node is constructed
//   // If the child node already exists, increase the visit count
//   void AddNextNode(const NodeFlowPtr& next_node);
//   // Remove a child node, called at parent when a request id is deleted
//   // Decrease the visit count before actually removing the child node
//   //   void RemoveNextNode(const NodeFlowPtr& node);

//   // The request id is removed from cache, subtracting the node meta from
//   // current record
//   void DereferenceNode(const NodeMetaPtr& node_meta);

//   const NodeFlowPtrMap& GetNextNodes() const { return next_nodes_; }
//   const NodeFlowPtrMap& GetPrevNodes() const { return prev_nodes_; }

//  private:
//   NodePtr node_;
//   NodeFlowPtrMap next_nodes_;
//   NodeFlowPtrMap prev_nodes_;
//   NodeMetaPtr node_meta_;
// };


class NodeTopology;
typedef std::shared_ptr<NodeTopology> NodeTopologyPtr;

typedef std::function<bool(const NodePtr&)> NodeFilterFunc;


typedef std::unordered_map<NodeID, NodeTopologyPtr> NodeTopoPtrMap;
class NodeTopology : public std::enable_shared_from_this<NodeTopology> {
 public:
  explicit NodeTopology(
      const NodePtr& node, const CorrelationID& correlation_id)
      : node_(node), correlation_id_(correlation_id)
  {
  }
  NodePtr GetNode() { return node_; }
  NodeID GetNodeID() { return node_->id; }
  CorrelationID GetCorrelationID() { return correlation_id_; }

  // Add a parent node, called at child when a new node is constructed
  void AddPrevNode(const NodeTopologyPtr& prev_node);
  // Add a child node, called at parent when a new node is constructed
  // If the child node already exists, increase the visit count
  void AddNextNode(const NodeTopologyPtr& next_node);

  const NodeTopoPtrMap& GetNextNodes() const { return next_nodes_; }
  const NodeTopoPtrMap& GetPrevNodes() const { return prev_nodes_; }

 private:
  NodePtr node_;
  CorrelationID correlation_id_;
  NodeTopoPtrMap next_nodes_;
  NodeTopoPtrMap
      next_sparse_nodes_;  // contains all sparse nodes in the next level
  NodeTopoPtrMap prev_nodes_;
};


typedef std::vector<NodePtr> NodePtrList;
typedef std::tuple<std::size_t, NodePtrList> FilterResult;
class FlowControllerFactory : public muduo::noncopyable {
 public:
  // DISABLE_COPY_AND_ASSIGN(FlowControllerFactory);
  virtual void RecordNode(
      const InputIDPtr& input_id, const NodePtr& node,
      const NodeMetaPtr& node_meta) = 0;
  virtual NodeMoveVec PrefetchNode(const NodePtr& node) = 0;

 protected:
  // void PutNodeTopology(
  //     const std::uint64_t& correlation_id, const NodePtr& node);
  void PutNodeToPipeline(
      const std::uint64_t& request_id, const std::uint64_t& correlation_id,
      const NodePtr& node);
  // NodeTopologyPtr GetNodeTopology(const NodeID& node_id);
  // NodePtrList GetNodesByFilter(
  //     const NodeFilterFunc& filter_func, const NodeID& node_id);

  void DispatchNodeMemoryInThread(const NodePtr& node, const Device& device);

  FilterResult GetTotalLiveParamSize();
  FilterResult GetTotalGPUParamSize();
  FilterResult GetRemovableNodes();
  FilterResult GetStandbyChildBySizeLimit(
      const NodePtr& node, std::size_t size_limit);

  bool MemorySizeFilter(const NodePtr& node, std::size_t* size)
  {
    if (node->byte_size > *size) {
      *size -= node->byte_size;
      return true;
    }
    return false;
  }

 private:
  void DispatchNodeMemory(const NodePtr& node, const Device& device);

 protected:
  // NodeTopologyPtr root_;
  // std::unordered_map<NodeID, NodeTopologyPtr> topology_;
  std::unordered_set<HashID> visited_;
  // std::size_t total_visit_cnt_{0};
  // std::unordered_map<NodeID, std::size_t> visit_cnt_;
  // std::unordered_map<NodeID, std::size_t> visit_time_;
  Pipeline pipeline_;
};

class DeepSpeedFlowController;
class CounterFlowController;
class PrefetchFlowController;

// #define FLOW_CONTROLLER GET_INSTANCE(DeepSpeedFlowController)
// #define FLOW_CONTROLLER GET_INSTANCE(CounterFlowController)
// #define FLOW_CONTROLLER GET_INSTANCE(PrefetchFlowController)

#ifdef ENABLE_DEEPSPEED_FLOW_CONTROLLER
#define FLOW_CONTROLLER GET_INSTANCE(DeepSpeedFlowController)
#endif

#ifdef ENABLE_COUNTER_FLOW_CONTROLLER
#define FLOW_CONTROLLER GET_INSTANCE(CounterFlowController)
#endif

#ifdef ENABLE_PREFETCH_FLOW_CONTROLLER
#define FLOW_CONTROLLER GET_INSTANCE(PrefetchFlowController)
#endif


#define BREAK_IF(node)                               \
  do {                                               \
    total_size += node->byte_size;                   \
    if (total_size >= size_limit) {                  \
      break;                                         \
    }                                                \
    if (node->memory_type == MemoryType::kStandBy) { \
      prefetch_node_list.push_back(node);            \
    }                                                \
  } while (0)

