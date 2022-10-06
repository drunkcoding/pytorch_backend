#pragma once

#include <mutex>
#include <unordered_map>

#include "dag_node.h"
#include "libtorch_factory.h"
#include "triton/backend/backend_common.h"
/*
A global registry for DAG nodes pointers.
*/
namespace triton { namespace backend { namespace pytorch {

class DAGRegistry : public SingletonFactory {
 public:
  DAGRegistry() = default;
  ~DAGRegistry() = default;

  FACTORY_STATIC_GET_INSTANCE(DAGRegistry);
  DISABLE_COPY_AND_ASSIGN(DAGRegistry);

  void AddNode(const std::size_t& id, const DAGNodePtr& node)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = node_map_.find(id);
    if (iter == node_map_.end()) {
      node_map_.insert({id, node});
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("DAGRegistry: Add node ") + std::to_string(id) +
           ", detail: " + node->GetModelInstanceInfo())
              .c_str());
      return;
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("DAG node with id ") + std::to_string(id) +
         std::string(" already exists"))
            .c_str());
  }

  DAGNodePtr GetNode(const std::size_t& id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = node_map_.find(id);
    if (iter == node_map_.end()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Cannot find node with id ") + std::to_string(id))
              .c_str());
      return nullptr;
    }
    return iter->second;
  }

  void RemoveNode(const std::size_t& id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = node_map_.find(id);
    if (iter == node_map_.end()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Cannot find node with id ") + std::to_string(id))
              .c_str());
      return;
    }
    node_map_.erase(id);
  }

  void Clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    node_map_.clear();
  }

  std::size_t Size()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_map_.size();
  }

 private:
  std::unordered_map<std::size_t, DAGNodePtr> node_map_;
  std::mutex mutex_;
};

}}}  // namespace triton::backend::pytorch