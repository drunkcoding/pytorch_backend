#include "topology_pool.h"

#include "dataflow/flow_controller.h"
#include "utils/log_utils.h"

TopologyPool* kTopologyPool = TopologyPool::GetInstance();

void
TopologyPool::TraceRequest(
    const std::uint64_t& request_id, const std::uint64_t& correlation_id,
    const NodePtr& node)
{
  auto low_id = correlation_id & 0xFFFFFFFF;

  auto now = MCIROSECONDS_SINCE_EPOCH;

  LOG_TRITON_VERBOSE(("TraceRequest: node: " + node->GetModelInstanceInfo() +
                      ", request_id: " + std::to_string(request_id) +
                      ", correlation_id: " + std::to_string(correlation_id))
                         .c_str());
  std::lock_guard<std::mutex> lock(mutex_);
  if (request_time_.find(request_id) == request_time_.end()) {
    request_time_.insert({request_id, now});
  }
  request_time_[request_id] = now;

  std::vector<std::size_t> delete_request;
  std::size_t microseconds = 60000000;
  for (auto& [request_id, time] : request_time_) {
    if (now - time > microseconds) {
      delete_request.push_back(request_id);
    }
  }

  for (auto& request_id : delete_request) {
    request_time_.erase(request_id);
    request_trace_.erase(request_id);
  }

  auto stage = pipeline_.stages[low_id];
  if (request_trace_.find(request_id) == request_trace_.end()) {
    request_trace_.insert({request_id, stage});
  }
  request_trace_[request_id] = stage;


  auto node_id = node->id;
  auto memory_size = node->byte_size;
  if (node_location_.find(node_id) == node_location_.end()) {
    if (free_cpu_memory_ > memory_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      node->default_host = CPU_DEVICE;
      free_cpu_memory_ -= memory_size;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
      node->default_host = DISK_DEVICE;
    }
  }

  visit_count_ += 1;
  ReorderNodeLocations();
}

void
TopologyPool::ReorderNodeLocations()
{
  // Sort node body according to the visit count
  std::vector<NodeBodyPtr> node_bodies;
  for (auto& stage : pipeline_.stages) {
    CONTINUE_IF_NULL(stage);
    for (auto& node_body : stage->nodes) {
      CONTINUE_IF_NULL(node_body);
      node_bodies.push_back(node_body);
    }
  }

  std::sort(
      node_bodies.begin(), node_bodies.end(),
      [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
        return a->visit_cnt > b->visit_cnt;
      });

  // Reorder node locations highest visit count first to CPU
  auto free_cpu_mem = DEFAULT_SYSTEM_FREE_MEMORY;
  node_location_.clear();
  for (auto& node_body : node_bodies) {
    auto node_id = node_body->node->id;
    if (free_cpu_mem > node_body->node->byte_size) {
      node_location_.insert({node_id, CPU_DEVICE});
      free_cpu_mem -= node_body->node->byte_size;
      node_body->node->default_host = CPU_DEVICE;
    } else {
      node_location_.insert({node_id, DISK_DEVICE});
      node_body->node->default_host = DISK_DEVICE;
    }
  }
}

void
TopologyPool::PutNodeToPipeline(
    const std::uint64_t& request_id, const std::uint64_t& correlation_id,
    const NodePtr& node)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::uint64_t high_corr_id =
      correlation_id >> 32;  // For childs in the same level
  std::uint64_t low_corr_id =
      correlation_id & 0xFFFFFFFF;  // For model inference pipeline

  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  LOG_TRITON_VERBOSE(
      (std::string("PutNodeToPipeline: request_id ") +
       std::to_string(request_id) + std::string(" correlation_id ") +
       std::to_string(correlation_id) + std::string(" high_corr_id ") +
       std::to_string(high_corr_id) + std::string(" low_corr_id ") +
       std::to_string(low_corr_id) + std::string(" is_last_node ") +
       std::to_string(is_last_node))
          .c_str());

  // node->corr_id = correlation_id;
  auto node_idx = (high_corr_id > 0) ? (high_corr_id - 1) : 0;

  if (visited_.find(correlation_id) == visited_.end()) {
    if (free_cpu_memory_ > node->byte_size) {
      free_cpu_memory_ -= node->byte_size;
      node_location_.insert({node->id, CPU_DEVICE});
    } else
      node_location_.insert({node->id, DISK_DEVICE});

    visited_.insert(correlation_id);
    auto node_body = std::make_shared<NodeBody>(node);

    // LOG_TRITON_VERBOSE(
    //     (std::string("PutNodeToPipeline: ") + pipeline_.str()).c_str());

    // assert(pipeline_.stages.size() >= low_corr_id); may skip some stages
    // since embed can run in parallel

    if (pipeline_.stages.size() < low_corr_id + 1)
      pipeline_.stages.resize(low_corr_id + 1, nullptr);

    if (pipeline_.stages[low_corr_id] == nullptr)
      pipeline_.stages[low_corr_id] = std::make_shared<Stage>();

    auto stage = pipeline_.stages[low_corr_id];

    if (stage->nodes.size() < node_idx + 1)
      stage->nodes.resize(node_idx + 1, nullptr);

    stage->nodes[node_idx] = node_body;

    if (high_corr_id > 0)
      stage->is_sparse = true;

    // LOG_TRITON_VERBOSE(
    //     (std::string("PutNodeToPipeline: Add new stage") + stage->str())
    //         .c_str());
    lfu_nodes_.push_back(node_body);
  }

  auto stage = pipeline_.stages[low_corr_id];
  auto node_body = stage->nodes[(stage->is_sparse) ? (high_corr_id - 1) : 0];
  auto now = MCIROSECONDS_SINCE_EPOCH;

  node_body->visit_time.push_back(now);
  node_body->visit_cnt += 1;
  node_body->activate_request.insert(request_id);
  node_body->node->last_access_time = now;


  // LOG_TRITON_VERBOSE(
  //     ("PutNodeToPipeline: node_body " + node_body->str()).c_str());

  // First update the visit count
  // 1. visit to sparse nodes only count once to total visit count
  // 2. find the last sparse layer and update the visit count of parent node


  if (stage->activate_request.find(request_id) ==
      stage->activate_request.end()) {
    // this is the first node in the stage
    // update the visit count of the stage
    stage->visit_cnt += 1;
  }

  stage->activate_request.insert(request_id);

  if (correlation_id == 0) {
    // this is the first node in the pipeline
    // update the visit count of the pipeline
    pipeline_.visit_cnt += 1;
  }

  // if (stage->is_sparse && high_corr_id == 0) {
  //   // this is the route node in the sparse layer
  //   // update the visit count of the stage
  //   stage->visit_cnt += 1;
  // }

  // if (!stage->is_sparse) {
  //   // this is the only node in the stage
  //   // update the visit count of the stage
  //   stage->visit_cnt += 1;
  // }

  // LOG_TRITON_VERBOSE(
  //     (std::string("PutNodeToPipeline: ") + pipeline_.str()).c_str());

  if (stage->is_sparse && high_corr_id > 0) {
    // this is the child node in the sparse layer
    // update the visit count of the parent node
    // LOG_TRITON_VERBOSE("PutNodeToPipeline: update sparse stage");
    StagePtr last_sparse_stage;
    for (auto i = low_corr_id - 1; i >= 0; i--) {
      if (stage->is_sparse) {
        last_sparse_stage = pipeline_.stages[i];
        break;
      }
    }

    for (auto& last_stage_node : last_sparse_stage->nodes) {
      if (last_stage_node->activate_request.find(request_id) !=
          last_stage_node->activate_request.end()) {
        if (last_stage_node->children_visit_cnt.size() < high_corr_id) {
          last_stage_node->children_visit_cnt.resize(high_corr_id);
        }
        if (last_stage_node->children.size() < high_corr_id) {
          last_stage_node->children.resize(high_corr_id, nullptr);
        }
        if (last_stage_node->children_visit_time.size() < high_corr_id) {
          last_stage_node->children_visit_time.resize(high_corr_id);
        }
        last_stage_node->children_visit_cnt[high_corr_id - 1] += 1;
        last_stage_node->children[high_corr_id - 1] = node_body;
        last_stage_node->children_visit_time[high_corr_id - 1].push_back(now);
      }
    }
  }

  if (is_last_node) {
    // this is the last node in the pipeline
    // remove request_id from all nodes
    // LOG_TRITON_VERBOSE("PutNodeToPipeline: is_last_node");
    for (auto& stage : pipeline_.stages) {
      if (stage == nullptr) {
        continue;
      }
      for (auto& node : stage->nodes) {
        if (node == nullptr) {
          continue;
        }
        node->activate_request.erase(request_id);
      }
      stage->activate_request.erase(request_id);
    }
    last_active_stage_.erase(request_id);
  }

  // for all nodes and stage and children in the pipeline, reduce count of nodes
  // that are not visited for 1 hour
  std::size_t microseconds = 60000000ULL * 60;
  for (auto& stage : pipeline_.stages) {
    if (stage == nullptr) {
      continue;
    }
    for (auto& node : stage->nodes) {
      if (node == nullptr) {
        continue;
      }
      auto visit = node->visit_time.begin();

      while (visit != node->visit_time.end() && *visit > 0 &&
             now - *visit > microseconds) {
        node->visit_time.pop_front();
        node->visit_cnt -= 1;
        visit = node->visit_time.begin();
      }

      int k = 0;
      for (auto& children : node->children_visit_time) {
        auto visit = children.begin();
        while (visit != children.end() && *visit > 0 &&
               now - *visit > microseconds) {
          children.pop_front();
          node->children_visit_cnt[k] -= 1;
          visit = children.begin();
        }
        k++;
      }
    }


    auto visit = stage->visit_time.begin();
    while (visit != stage->visit_time.end() && *visit > 0 &&
           now - *visit > microseconds) {
      stage->visit_time.pop_front();
      stage->visit_cnt -= 1;
      visit = stage->visit_time.begin();
    }
  }

  auto it = last_active_stage_.find(request_id);
  if (it == last_active_stage_.end()) {
    last_active_stage_.insert({request_id, low_corr_id});
  } else {
    it->second = low_corr_id;
  }

  std::size_t byte_size = 0;
  for (auto node_body : pipeline_.stages[low_corr_id]->nodes) {
    CONTINUE_IF_NULL(node_body);
    byte_size += node_body->node->byte_size;
  }
  pipeline_.stages[low_corr_id]->byte_size = byte_size;

  // sort lfu_nodes_ by visit count ascending
  std::sort(
      lfu_nodes_.begin(), lfu_nodes_.end(),
      [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
        return a->visit_cnt < b->visit_cnt;
      });
}

NodePtrList
TopologyPool::GetLFUNodes(const Device& device)
{
  NodePtrList nodes;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto node_body : lfu_nodes_) {
    CONTINUE_IF_NULL(node_body);
    if (node_body->node->device == device) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}

NodePtrList TopologyPool::GetTopKNodes(const Device& device, const NodePtr& node, const std::size_t& k) {
  NodePtrList nodes;
  auto root_stage_id = node->corr_id & 0xFFFFFFFF;
  auto root_node_id = node->corr_id >> 32;
  std::vector<NodeBodyPtr> candidates;
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::uint64_t stage_idx = root_stage_id;
       stage_idx < pipeline_.stages.size(); ++stage_idx) {
    auto stage = pipeline_.stages[stage_idx];
    BREAK_IF_NULL(stage)
    
    for (std::uint64_t node_idx = root_node_id; node_idx < stage->nodes.size();
         ++node_idx) {
      auto node_body = stage->nodes[node_idx];
      CONTINUE_IF_NULL(node_body)
      if (node_body->node->device == device) {
        candidates.push_back(node_body);
      }
    }
  }
  // sort candidates by visit count descending
  std::sort(
      candidates.begin(), candidates.end(),
      [](const NodeBodyPtr& a, const NodeBodyPtr& b) {
        return a->visit_cnt > b->visit_cnt;
      });
  for (std::size_t i = 0; i < k && i < candidates.size(); ++i) {
    nodes.push_back(candidates[i]->node);
  }
  return nodes;
}

NodePtrList
TopologyPool::GetConsecutiveNodes(const Device& device, const NodePtr& node)
{
  NodePtrList nodes;
  auto root_stage_id = node->corr_id & 0xFFFFFFFF;
  auto root_node_id = node->corr_id >> 32;
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::uint64_t stage_idx = root_stage_id;
       stage_idx < pipeline_.stages.size(); ++stage_idx) {
    auto stage = pipeline_.stages[stage_idx];
    BREAK_IF_NULL(stage)
    for (std::uint64_t node_idx = root_node_id; node_idx < stage->nodes.size();
         ++node_idx) {
      auto node_body = stage->nodes[node_idx];
      CONTINUE_IF_NULL(node_body)
      if (node_body->node->device == device) {
        nodes.push_back(node_body->node);
      }
    }
  }
  return nodes;
}

NodePtrList
TopologyPool::GetTopKChildNodes(
    const NodePtr& node, const std::size_t& k, const std::size_t& skip)
{
  NodePtrList nodes;
  std::size_t low_corr_id = node->corr_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = node->corr_id >> 32;        // node id
  std::lock_guard<std::mutex> lock(mutex_);

  if (low_corr_id >= pipeline_.stages.size()) {
    return nodes;
  }

  if (pipeline_.stages[low_corr_id] == nullptr) {
    return nodes;
  }
  high_corr_id = (high_corr_id > 0) ? (high_corr_id - 1) : (0);
  if (high_corr_id >= pipeline_.stages[low_corr_id]->nodes.size()) {
    return nodes;
  }

  auto root_node_body = pipeline_.stages[low_corr_id]->nodes[high_corr_id];
  if (root_node_body == nullptr) {
    return nodes;
  }
  // auto current_visit_cnt = root_node_body->visit_cnt;

  std::deque<std::tuple<NodeBodyPtr, double>> visited_sparse_nodes;
  visited_sparse_nodes.push_back(std::make_tuple(root_node_body, 1.0));

  // auto init_low_corr_id = low_corr_id;
  low_corr_id++;
  while (low_corr_id < pipeline_.stages.size()) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    BREAK_IF_NULL(stage);
    std::deque<std::tuple<NodeBodyPtr, double>> candidates;
    for (auto& [node_body, root_prob] : visited_sparse_nodes) {
      auto children = node_body->children;
      auto children_cnt = node_body->children_visit_cnt;

      // keep only not null from children and children_cnt
      children.erase(
          std::remove_if(
              children.begin(), children.end(),
              [](const NodeBodyPtr& node_body) {
                return node_body == nullptr;
              }),
          children.end());
      children_cnt.erase(
          std::remove_if(
              children_cnt.begin(), children_cnt.end(),
              [](const std::size_t& cnt) { return cnt == 0; }),
          children_cnt.end());
      // children_cnt divide by root cnt
      for (std::size_t i = 0; i < children_cnt.size(); i++) {
        candidates.push_back(std::make_tuple(
            children[i], children_cnt[i] / node_body->visit_cnt * root_prob));
      }
    }
    visited_sparse_nodes.clear();
    // sort candidates by second element in tuple decending
    std::sort(
        candidates.begin(), candidates.end(),
        [](const std::tuple<NodeBodyPtr, double>& a,
           const std::tuple<NodeBodyPtr, double>& b) {
          return std::get<1>(a) > std::get<1>(b);
        });
    // keep only top k and insert into visited_sparse_nodes
    for (std::size_t i = skip; i < std::min(k + skip, candidates.size()); i++) {
      visited_sparse_nodes.push_back(candidates[i]);
      nodes.push_back(std::get<0>(candidates[i])->node);
    }
    low_corr_id++;
    // if (low_corr_id >= init_low_corr_id + 32) {
    //   break;
    // }
  }
  return nodes;
}


std::uint64_t
TopologyPool::GetLastActivateStage(const HashID& hash_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = last_active_stage_.find(hash_id);
  if (it == last_active_stage_.end()) {
    return 0;
  }
  return it->second;
}