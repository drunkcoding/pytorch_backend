#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "dag_node.h"
#include "libtorch_factory.h"
#include "libtorch_stats.h"
#include "lru_cache.h"
#include "model_meta.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace pytorch {

// struct Node;
// typedef std::shared_ptr<Node> NodePtr;
// typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;

// struct Node {
//   //   std::unordered_map<std::string, HistogramBuckets> children_stats;
//   std::unordered_map<std::size_t, NodePtr> children;
//   std::unordered_map<std::size_t, std::size_t> children_visited;
//   std::unordered_map<std::size_t, NodePtr> parents;
//   ModelMetaPtr checkpoint;

//   Node() {}
//   Node(const ModelMetaPtr& model_meta) : checkpoint(model_meta) {}
// };

// for each request id, we need to keep track of the nodes
// Use LRUCache to keep track of the most recent requests nodes

class ModelFlowRecorder : public SingletonFactory {
 public:
  FACTORY_STATIC_GET_INSTANCE(ModelFlowRecorder)
  DISABLE_COPY_AND_ASSIGN(ModelFlowRecorder)

  void RecordModelFlow(
      const std::string& request_id, const ModelMetaPtr& model_meta);
  std::unordered_map<std::size_t, double> GetChildernProbability(
      const std::size_t& model_id);
  std::unordered_map<std::size_t, double> GetTreeProbability(
      const std::size_t& model_id);
  DAGNodePtr GetDAGNode(const std::size_t& model_id);

  //   void RemoveModelFlow(
  //       const std::string& model_name, const std::string& model_version);
  //   void ClearModelFlow();

 private:
  ModelFlowRecorder() = default;
  virtual ~ModelFlowRecorder() = default;

  void RecursivelyUpdateProbability(
      const DAGNodePtr& node,
      std::unordered_map<std::size_t, double>& prob_map);


  LRUCache<std::string, std::list<DAGNodePtr>>
      request_trace_;  // request_id -> node tracing
  std::unordered_map<std::size_t, DAGNodePtr> model_graph_;
  std::mutex mutex_;
};

}}}  // namespace triton::backend::pytorch