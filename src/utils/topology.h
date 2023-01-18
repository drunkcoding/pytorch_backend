#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <future>
#include <memory>

#include "log_utils.h"
#include "muduo/base/Mutex.h"
#include "state.h"
#include "time_utils.h"
#include "torch_utils.h"

/*
 * The datastructure for topology is as follow:
 * 1. Node: the basic uniot of topology, a node is equal to a model partition
 * 2. Stage: a list of nodes at the same level of topology, either all dense or
 * all sparse
 * 3. Pipeline: a list of stages that represents the whole topology
 */

inline void
RemoveModuleFromCache(torch::jit::script::Module* model)
{
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    (*it).unsafeReleaseIntrusivePtr().reset();
  }


  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    (*it).unsafeReleaseIntrusivePtr().reset();
  }
}

struct Node {
  ScriptModule model;      // always use raw pointer, since we need
                           // to manage the memory by ourselves
  MemoryType memory_type;  // indicating whether the node is executing or
                           // controlled by flow controller
  std::size_t id;
  std::size_t corr_id;
  std::int64_t byte_size;
  std::size_t last_access_time;
  std::size_t
      last_prefetch_time;  // the last time when the node is prefetched to GPU
  Device device;
  Device default_device;
  Device default_host = DISK_DEVICE;
  cudaStream_t stream;

  // for fetch thread synchronization
  // muduo::MutexLock mutex;
  std::mutex mutex;
  ScriptModule cpu_model;

 private:
  std::string model_path_;
  // bool waiting = false;
  // std::future<void> mem_future;
  bool is_loaded = false;

  // void Load(const Device device)
  // {
  //   model = new ScriptModule(torch::jit::load(model_path_, device));
  //   waiting = false;
  // }

 public:
  explicit Node(const std::string& model_path);
  const std::string GetModelInstanceInfo() noexcept;
  void SetDevice(const Device& target_device) noexcept;
};
typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodePtrList;
typedef std::tuple<std::int64_t, NodePtrList> FilterResult;

struct NodeBody;
typedef std::shared_ptr<NodeBody> NodeBodyPtr;

struct NodeBody {
  NodePtr node;
  std::vector<NodeBodyPtr> children;
  std::vector<std::size_t> children_visit_cnt;
  std::vector<std::deque<std::size_t>> children_visit_time;
  std::unordered_set<std::size_t> activate_request;
  std::size_t visit_cnt;
  std::deque<std::size_t> visit_time;
  explicit NodeBody(NodePtr node) : node(node), visit_cnt(0) {}

  std::string str() const noexcept
  {
    // std::stringstream ss;
    // ss << "NodeBody: " << node->GetModelInstanceInfo() << "visit_cnt "
    //    << visit_cnt << std::endl;

    // rewrite above in c style sprintf
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, "NodeBody: %s visit_cnt %ld;",
        node->GetModelInstanceInfo().c_str(), visit_cnt);

    return std::string(buffer);
  }
};

struct Stage {
  bool is_sparse;
  std::vector<NodeBodyPtr> nodes;
  std::size_t visit_cnt;
  std::int64_t byte_size;
  std::deque<std::size_t> visit_time;
  std::unordered_set<std::size_t> activate_request;
  Stage() : is_sparse(false), visit_cnt(0), byte_size(0) {}
  Stage(bool is_sparse) : is_sparse(is_sparse), visit_cnt(0), byte_size(0) {}

  std::string str() const noexcept
  {
    // std::stringstream ss;
    // ss << "Stage: " << nodes.size() << " nodes; visit_cnt " << visit_cnt
    //    << "; is_sparse " << is_sparse << std::endl;

    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(buffer, "Stage[%ld,%ld,%d]", nodes.size(), visit_cnt, is_sparse);

    return std::string(buffer);
  }
};
typedef std::shared_ptr<Stage> StagePtr;


struct Pipeline {
  std::vector<StagePtr> stages;
  std::size_t visit_cnt = 0;

  std::string str() const noexcept
  {
    std::stringstream ss;
    ss << "Pipeline: " << stages.size() << " stages; visit_cnt " << visit_cnt
       << std::endl;
    return ss.str();
  }
};
typedef std::shared_ptr<Pipeline> PipelinePtr;
