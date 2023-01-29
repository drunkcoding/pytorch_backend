#pragma once
#include "muduo/base/noncopyable.h"
#include "utils/class_utils.h"
#include "utils/topology.h"
#include "mem_ctrl.h"

class TopologyPool : public muduo::noncopyable {
 public:
  static TopologyPool* GetInstance() { return new TopologyPool(); }

  void PutNodeToPipeline(
      const std::uint64_t& request_id, const std::uint64_t& correlation_id,
      const NodePtr& node);
  NodePtrList GetLFUNodes(const Device& device);
  NodePtrList GetTopKChildNodes(const NodePtr& node, const std::size_t& k, const std::size_t& skip);
  void TraceRequest(
      const std::uint64_t& request_id, const std::uint64_t& correlation_id,
      const NodePtr& node);

  void ReorderNodeLocations();
  

  std::uint64_t GetLastActivateStage(const HashID& hash_id);

 private:
  TopologyPool() = default;
  ~TopologyPool() = default;

 private:
  Pipeline pipeline_;
  std::unordered_set<HashID> visited_;
  std::unordered_map<HashID, std::uint64_t> last_active_stage_;
  std::vector<NodeBodyPtr> lfu_nodes_;
  std::unordered_map<std::size_t, std::size_t> request_time_;
  std::unordered_map<std::size_t, StagePtr> request_trace_;
  std::int64_t visit_count_ = 0;
  std::unordered_map<NodeID, Device> node_location_;
  std::mutex mutex_;
  std::int64_t free_cpu_memory_ = DEFAULT_SYSTEM_FREE_MEMORY;
  std::int64_t free_gpu_memory_ = DEFAULT_CUDA_FREE_MEMORY(0);
};

extern TopologyPool* kTopologyPool;