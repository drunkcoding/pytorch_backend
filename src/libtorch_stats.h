#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace triton { namespace backend { namespace pytorch {

// template <typename T, typename Compare>
// void
// compare_and_update(std::atomic<T>& value, T value)
// {
//   T old_value = value.load();
//   while (Compare()(value, old_value)) {
//     if (value.compare_exchange_weak(old_value, value)) {
//       break;
//     }
//   }
// }

struct Bucket {
  Bucket() : sum(0), count(0) {}
  void Clear()
  {
    sum = 0;
    count = 0;
  }
  void Add(std::int64_t s, std::uint64_t c)
  {
    sum += s;
    count += c;
  }
  std::int64_t Avg() const
  {
    if (count == 0)
      return 0;
    return static_cast<double>(sum) / static_cast<double>(count);
  }
  std::int64_t sum;
  std::uint64_t count;
};

class HistogramBuckets {
 public:
  HistogramBuckets(
      const std::string& name, std::size_t bucket_size, std::int64_t min,
      std::int64_t max);
  void AddValue(std::int64_t value);
  void AddValue(std::int64_t value, std::uint64_t count);
  void Clear();
  std::string OutputString() const;

  std::string GetName() { return name_; }
  std::uint64_t GetCount() const { return count_; };
  Bucket& GetByValue(std::int64_t value);
  const Bucket& GetByValue(std::int64_t value) const;
  std::size_t GetBucketIdx(std::int64_t value) const;
  std::int64_t GetPercentileEstimate(double pct) const;
  std::size_t GetPercentileBucketIdx(
      double pct, double* low_pct, double* high_pct) const;
  std::int64_t GetBucketMin(std::size_t idx) const;
  std::int64_t GetBucketMax(std::size_t idx) const;

 private:
  const std::string name_;
  const std::uint64_t bucket_size_;
  std::int64_t min_;
  std::int64_t max_;
  std::int64_t min_value_;
  std::int64_t max_value_;
  std::uint64_t count_;
  std::vector<Bucket> buckets_;
  // mutable std::mutex mutex_;
};

typedef std::shared_ptr<HistogramBuckets> HistogramBucketsPtr;

}}}  // namespace triton::backend::pytorch