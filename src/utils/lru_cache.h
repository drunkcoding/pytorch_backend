#pragma once

#include <iostream>
#include <type_traits>

#include "log_utils.h"

template <typename Key, typename Value>
class LRUCache {
 public:
  typedef std::shared_ptr<Value> ValuePtr;

  LRUCache(size_t capacity) : capacity_(capacity) {}
  LRUCache() = delete;
  ~LRUCache() = default;

  ValuePtr Get(const Key& key)
  {
    auto cache_it = cache_.find(key);
    auto iter = cache_iter_.find(key);
    if (cache_it == cache_.end()) {
      return nullptr;
    }
    // Move the accessed item to the front of the list
    MoveToFront(iter);
    return cache_it->second;
  }

  ValuePtr Put(const Key& key, const ValuePtr& value)
  {
    auto cache_it = cache_.find(key);
    auto iter = cache_iter_.find(key);
    if (cache_it != cache_.end()) {
      // Move the accessed item to the front of the list
      LOG_VERBOSE((std::string("Key ") + std::to_string(key) +
                   std::string(" already exists, update value"))
                      .c_str());
      MoveToFront(iter);
      return nullptr;
    }

    if (cache_.size() == capacity_) {
      // Remove the least recently used item from the list
      LOG_VERBOSE(
          (std::string("Key ") + std::to_string(key) +
           std::string(" does not exist, remove the least recently used item"))
              .c_str());
      auto& back_key = lru_list_.back();
      lru_list_.pop_back();
      cache_.erase(back_key);
      cache_iter_.erase(back_key);
    }
    // Create a new item at the front of the list
    LOG_VERBOSE(
        (std::string("Key ") + std::to_string(key) +
         std::string(
             " does not exist, create a new item at the front of the list"))
            .c_str());
    lru_list_.emplace_front(key);
    cache_iter_.insert({key, lru_list_.begin()});
    cache_.insert({key, value});
    return nullptr;
  }

  ValuePtr Remove(const Key& key)
  {
    auto cache_it = cache_.find(key);
    auto iter = cache_iter_.find(key);
    if (cache_it == cache_.end()) {
      return nullptr;
    }
    ValuePtr value = cache_it->second;
    lru_list_.erase(iter->second);
    cache_iter_.erase(iter);
    cache_.erase(cache_it);
    return value;
  }

  std::pair<Key, ValuePtr> GetTail() const
  {
    auto cache_it = cache_.find(lru_list_.back());
    if (cache_it == cache_.end()) {
      return nullptr;
    }
    return *cache_it;
  }

  std::list<std::pair<Key, ValuePtr>> GetCacheInOrder() const {
    std::list<std::pair<Key, ValuePtr>> cache_in_order;
    for (auto iter = lru_list_.begin(); iter != lru_list_.end(); ++iter) {
      auto cache_it = cache_.find(*iter);
      if (cache_it != cache_.end()) {
        cache_in_order.push_back(*cache_it);
      }
    }
    return cache_in_order;
  }

  std::size_t Size() const { return cache_.size(); }

 private:
  using ListIterator = typename std::list<Key>::iterator;
  using CacheIterator =
      typename std::unordered_map<Key, ListIterator>::iterator;
  void MoveToFront(const CacheIterator& iter)
  {
    auto key = *iter->second;
    lru_list_.erase(iter->second);
    lru_list_.emplace_front(key);
    iter->second = lru_list_.begin();
  }

  size_t capacity_;
  std::unordered_map<Key, ValuePtr> cache_;
  std::unordered_map<Key, ListIterator> cache_iter_;
  std::list<Key> lru_list_;
};


// template <typename Key, typename Value>
// class LRUCache {
//  public:
//   LRUCache(size_t capacity) : capacity_(capacity) {}
//   LRUCache() = default;
//   ~LRUCache() = default;

//   std::shared_ptr<Value> Put(
//       const Key& key, const std::shared_ptr<Value>& value)
//   {
//     auto it = cache_.find(key);
//     if (it != cache_.end()) {
//       it->second = std::make_shared<CacheNode>(key, value);
//       MoveToHead(it->second);
//       LOG_MESSAGE(
//           TRITONSERVER_LOG_VERBOSE,
//           (std::string("Key ") + key +
//            std::string(" already exists, update the value"))
//               .c_str());
//       return nullptr;
//     }

//     std::shared_ptr<Value> evict_value = nullptr;
//     if (cache_.size() >= capacity_) {
//       LOG_MESSAGE(
//           TRITONSERVER_LOG_VERBOSE,
//           (std::string("Key ") + key + std::string(" causes eviction of key
//           ") +
//            tail_->key)
//               .c_str());
//       evict_value = tail_->value;
//       cache_.erase(tail_->key);
//       PopTail();
//     }

//     LOG_MESSAGE(
//         TRITONSERVER_LOG_VERBOSE,
//         (std::string("Key ") + key +
//          std::string(" does not exist, insert a new key"))
//             .c_str());
//     auto node = std::make_shared<CacheNode>(key, value);
//     cache_.insert({key, node});
//     MoveToHead(node);
//     LOG_MESSAGE(
//         TRITONSERVER_LOG_VERBOSE,
//         (std::string("Key ") + key + std::string(" is inserted")).c_str());
//     return evict_value;
//   }

//   bool Contains(const Key& key)
//   {
//     auto it = cache_.find(key);
//     return it != cache_.end();
//   }

//   std::shared_ptr<Value> Get(const Key& key)
//   {
//     auto it = cache_.find(key);
//     if (it == cache_.end()) {
//       return nullptr;
//     }

//     MoveToHead(it->second);
//     return it->second->value;
//   }

//   std::shared_ptr<Value> Remove(const Key& key)
//   {
//     auto it = cache_.find(key);
//     if (it == cache_.end()) {
//       return nullptr;
//     }

//     if (it->second == head_) {
//       PopHead();
//     } else if (it->second == tail_) {
//       PopTail();
//     } else {
//       it->second->prev->next = it->second->next;
//       it->second->next->prev = it->second->prev;
//     }

//     auto value = it->second->value;
//     cache_.erase(it);
//     return value;
//   }

//   size_t Size() const { return cache_.size(); }

//   void PopHead()
//   {
//     if (head_ == tail_) {  // only one node
//       head_ = nullptr;
//       tail_ = nullptr;
//     } else {
//       head_ = head_->next;
//       head_->prev = nullptr;
//     }
//   }

//   void PopTail()
//   {
//     if (head_ == tail_) {  // only one node
//       head_ = nullptr;
//       tail_ = nullptr;
//     } else {
//       tail_ = tail_->prev;
//       tail_->next = nullptr;
//     }
//   }

//  private:
//   struct CacheNode {
//     CacheNode(const Key& k, const std::shared_ptr<Value>& v)
//         : key(k), value(v), prev(nullptr), next(nullptr)
//     {
//     }
//     Key key;
//     std::shared_ptr<Value> value;
//     std::shared_ptr<CacheNode> prev;
//     std::shared_ptr<CacheNode> next;
//   };

//   void MoveToHead(const std::shared_ptr<CacheNode>& node)
//   {
//     if (head_ == nullptr) {
//       head_ = node;
//       tail_ = node;
//       return;
//     }

//     if (node == head_) {
//       return;
//     }

//     if (node == tail_) {
//       PopTail();
//     } else {
//       node->prev->next = node->next;
//       node->next->prev = node->prev;
//     }

//     node->next = head_;
//     head_->prev = node;
//     head_ = node;
//   }

//   size_t capacity_;
//   std::unordered_map<Key, std::shared_ptr<CacheNode>> cache_;
//   std::shared_ptr<CacheNode> head_;
//   std::shared_ptr<CacheNode> tail_;
// };