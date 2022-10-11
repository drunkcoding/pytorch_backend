#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

#include "event/eventloop.h"
#include "utils/class_utils.h"
#include "utils/enum_utils.h"

ENUM_MACRO(StatusType, kOK, kError, kOpTypeInsert, kOpTypeQuery, kOpTypeUpdate)

class OpBase;
struct OpRequest;
struct OpResponse;
typedef std::shared_ptr<OpRequest> RequestPtr;
typedef std::shared_ptr<OpResponse> ResponsePtr;

typedef std::function<OpBase*(EventLoop*)> CreateOpIns;
typedef std::function<void(const ResponsePtr&)> EngineCb;


class Status {
 public:
  Status() : status_(StatusType::kOK), err_() {}
  bool OK() const { return status_ == StatusType::kOK; }
  StatusType status() const { return status_; }
  const std::string& err() const { return err_; }
  void SetError(StatusType status, const std::string& msg)
  {
    status_ = status;
    err_ = StatusTypeToString(status) + " " + msg;
  }

 private:
  StatusType status_;
  std::string err_;
};

struct OpRequest {
  // LoopHandle* loop_handle;
  // std::vector<EngineCb> callbacks;
  // EngineCb cb;
  OpRequest() {}
};
struct OpResponse {
  Status status;
  OpResponse() {}
};


class OpBase : public std::enable_shared_from_this<OpBase> {
 public:
  explicit OpBase(EventLoop* loop);
  // virtual ~OpBase() = default;

 public:
  virtual void SendRequest(const RequestPtr& request)
  {
    request_ = request;
    auto task = SELF_BIND(OpBase, Process);
    loop_->RunInLoop(task);
  }

 protected:
  virtual void Process() = 0;

 protected:
  EventLoop* loop_;
  RequestPtr request_;
};

class OpRegistryBase {
 public:
  OpRegistryBase() = default;
  virtual ~OpRegistryBase() = default;

 public:
  virtual void RegisterOp(const std::string& name, CreateOpIns ins)
  {
    op_map_.insert(std::make_pair(name, ins));
  }
  virtual CreateOpIns GetOp(const std::string& name)
  {
    auto it = op_map_.find(name);
    if (it != op_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

 protected:
  std::unordered_map<std::string, CreateOpIns> op_map_;
};
