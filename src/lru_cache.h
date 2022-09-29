#pragma once

#include <type_traits>

// Primary template with a static assertion
// for a meaningful error message
// if it ever gets instantiated.
// We could leave it undefined if we didn't care.

#define DEFINE_HAS_MEMBER(member)                                \
  template <typename T>                                          \
  struct has_member_##member {                                   \
    template <typename U>                                        \
    static auto check(U*)                                        \
        -> decltype(std::declval<U>().member, std::true_type()); \
    template <typename>                                          \
    static std::false_type check(...);                           \
    static constexpr bool value = decltype(check<T>(0))::value;  \
  };

// template<typename, typename T>
// struct has_serialize {
//     static_assert(
//         std::integral_constant<T, false>::value,
//         "Second template parameter needs to be of function type.");
// };

// // specialization that does the checking
// template<typename C, typename Ret, typename... Args>
// struct has_serialize<C, Ret(Args...)> {
// private:
//     template<typename T>
//     static constexpr auto check(T*)
//     -> typename
//         std::is_same<
//             decltype( std::declval<T>().serialize( std::declval<Args>()... )
//             ), Ret    //
//             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//         >::type;  // attempt to call it and see if the return type is correct

//     template<typename>
//     static constexpr std::false_type check(...);

//     typedef decltype(check<C>(0)) type;

// public:
//     static constexpr bool value = type::value;
// };


DEFINE_HAS_MEMBER(size)

template <typename Key, typename Value>
class LRUCache {
 public:
  LRUCache(size_t capacity) : capacity_(capacity) {}
  LRUCache() = default;
  ~LRUCache() = default;

  std::shared_ptr<std::pair<Key, Value>> Put(const Key& key, const Value& value)
  {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second = std::make_shared<CacheNode>(key, value);
      MoveToHead(it->second);
      return nullptr;
    }

    // if value has member size, use it to calculate the size of cahce
    // if constexpr (has_member_size<Value>::value) {
    // std:;
    //   size_t size = 0;
    //   for (const auto& c : cache_) {
    //     size += c.second->value.size();
    //   }
    //   size += value.size();
    //   while (tail_!= nullptr && size > capacity_) {
    //     size -= cache_.back().second->value.size();
    //     cache_.erase(tail_->key);
    //     PopTail();
    //   }
    // } else {
    //   if (cache_.size() >= capacity_) {
    //     cache_.erase(tail_->key);
    //     PopTail();
    //   }
    // }

    std::shared_ptr<std::pair<Key, Value>> tail_del(nullptr);
    if (cache_.size() >= capacity_) {
      tail_del =
          std::make_shared<std::pair<Key, Value>>(tail_->key, tail_->value);
      cache_.erase(tail_->key);
      PopTail();
    }

    auto node = std::make_shared<CacheNode>(key, value);
    cache_[key] = node;
    MoveToHead(node);

    return tail_del;
  }

  bool Contains(const Key& key)
  {
    auto it = cache_.find(key);
    return it != cache_.end();
  }

  Value* Get(const Key& key)
  {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }

    MoveToHead(it->second);
    return &it->second->value;
  }

  int Remove(const Key& key)
  {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return -1;
    }

    if (it->second == head_) {
      PopHead();
    } else if (it->second == tail_) {
      PopTail();
    } else {
      it->second->prev->next = it->second->next;
      it->second->next->prev = it->second->prev;
    }

    cache_.erase(it);
    return 0;
  }

  size_t Size() const { return cache_.size(); }

  void PopHead()
  {
    if (head_ == tail_) {  // only one node
      head_ = nullptr;
      tail_ = nullptr;
    } else {
      head_ = head_->next;
      head_->prev = nullptr;
    }
  }

  void PopTail()
  {
    if (head_ == tail_) {  // only one node
      head_ = nullptr;
      tail_ = nullptr;
    } else {
      tail_ = tail_->prev;
      tail_->next = nullptr;
    }
  }

 private:
  struct CacheNode {
    CacheNode(const Key& k, const Value& v)
        : key(k), value(v), prev(nullptr), next(nullptr)
    {
    }
    Key key;
    Value value;
    std::shared_ptr<CacheNode> prev;
    std::shared_ptr<CacheNode> next;
  };

  void MoveToHead(const std::shared_ptr<CacheNode>& node)
  {
    if (node == head_) {
      return;
    }

    if (node == tail_) {
      PopTail();
    } else {
      node->prev->next = node->next;
      node->next->prev = node->prev;
    }

    node->next = head_;
    head_->prev = node;
    head_ = node;
  }

  size_t capacity_;
  std::unordered_map<Key, std::shared_ptr<CacheNode>> cache_;
  std::shared_ptr<CacheNode> head_;
  std::shared_ptr<CacheNode> tail_;
};