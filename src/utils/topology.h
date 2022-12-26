#pragma once

#include <memory>

#include "log_utils.h"
#include "state.h"
#include "time_utils.h"
#include "torch_utils.h"

#include "muduo/base/Mutex.h"

/*
 * The datastructure for topology is as follow:
 * 1. Node: the basic uniot of topology, a node is equal to a model partition
 * 2. Stage: a list of nodes at the same level of topology, either all dense or
 * all sparse
 * 3. Pipeline: a list of stages that represents the whole topology
 */

struct Node {
  torch::jit::script::Module* model;  // always use raw pointer, since we need
                                      // to manage the memory by ourselves
  MemoryType memory_type;  // indicating whether the node is executing or
                           // controlled by flow controller
  std::size_t id;
  std::size_t corr_id;
  std::int64_t byte_size;
  std::size_t last_access_time;
  Device device;


  // for fetch thread synchronization
  // muduo::MutexLock mutex;
  std::mutex mutex;

 private:
  std::string model_path_;

 public:
  explicit Node(const std::string& model_path)
      : id(std::hash<std::string>{}(model_path)), corr_id(0), byte_size(0),
        last_access_time(MCIROSECONDS_SINCE_EPOCH), device(DISK_DEVICE),
        model_path_(model_path)
  {
    model = new ScriptModule(torch::jit::load(model_path));
    std::int64_t param_size = 0;
    for (const auto& param : model->parameters()) {
      param_size += param.numel() * param.element_size();
    }

    // Iterate model buffers and calculate the total size of the model
    std::int64_t buffer_size = 0;
    for (const auto& buffer : model->buffers()) {
      buffer_size += buffer.numel() * buffer.element_size();
    }
    byte_size = param_size + buffer_size;
    // mutex.unlock();
  }

  const std::string GetModelInstanceInfo() noexcept
  {
    // return member value as string
    std::stringstream ss;
    ss << "model_path: " << model_path_ << " id: " << id
       << " corr_id: " << std::hex << corr_id << std::dec
       << " byte_size: " << byte_size
       << " last_access_time: " << last_access_time << "device: " << device
       << " memory_type: " << static_cast<int>(memory_type) << std::endl;
    return ss.str();
  }

  void SetDevice(const Device& target_device) noexcept
  {
    if (device == target_device)
      return;

    // InferenceMode should be used to guard all tensors operations including
    // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
    torch::InferenceMode infer_guard(true);

    // In our context, lazy device stays on disk
    if (target_device == DISK_DEVICE) {
      delete model;
      model = nullptr;
    } else {
      if (model == nullptr) {
        model = new ScriptModule(torch::jit::load(model_path_, target_device));
      } else {
        model->to(target_device);
      }
    }

    device = target_device;
  }
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
    std::stringstream ss;
    ss << "NodeBody: " << node->GetModelInstanceInfo() << "visit_cnt "
       << visit_cnt << std::endl;
    return ss.str();
  }
};

struct Stage {
  bool is_sparse;
  std::vector<NodeBodyPtr> nodes;
  std::size_t visit_cnt;
  std::deque<std::size_t> visit_time;
  std::unordered_set<std::size_t> activate_request;
  Stage() : is_sparse(false), visit_cnt(0) {}
  Stage(bool is_sparse) : is_sparse(is_sparse), visit_cnt(0) {}

  std::string str() const noexcept
  {
    std::stringstream ss;
    ss << "Stage: " << nodes.size() << " nodes; visit_cnt " << visit_cnt
       << "; is_sparse " << is_sparse << std::endl;
    return ss.str();
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
