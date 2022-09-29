#include "libtorch_stats.h"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace triton { namespace backend { namespace pytorch {

HistogramBuckets::HistogramBuckets(
    const std::string& name, std::size_t bucket_size, std::int64_t min, std::int64_t max)
    : name_(name), bucket_size_(bucket_size), min_(min), max_(max),
      min_value_(std::numeric_limits<std::int64_t>::max()),
      max_value_(std::numeric_limits<std::int64_t>::min()), count_(0)
{
  int64_t bucket_count = (max - min) / bucket_size;
  if (bucket_count * bucket_size < max - min)
    ++bucket_count;
  // for below min and above max
  bucket_count += 2;
  buckets_.assign(bucket_count, Bucket());
}

void
HistogramBuckets::AddValue(std::int64_t value)
{
  auto& bucket = GetByValue(value);
  bucket.count += 1;
  bucket.sum += value;
  if (value < min_value_)
    min_value_ = value;
  if (value > max_value_)
    max_value_ = value;
  count_ += 1;
}

void
HistogramBuckets::AddValue(std::int64_t value, std::uint64_t count)
{
  auto& bucket = GetByValue(value);
  bucket.count += count;
  bucket.sum += value * count;
  // atomic_compare_and_update<std::less>(min_value_, value);
  // atomic_compare_and_update<std::greater>(max_value_, value);
  if (value < min_value_)
    min_value_ = value;
  if (value > max_value_)
    max_value_ = value;
  count_ += count;
}

void
HistogramBuckets::Clear()
{
  for (auto it = buckets_.begin(); it != buckets_.end(); ++it) {
    it->Clear();
  }
  min_value_ = std::numeric_limits<std::int64_t>::max();
  max_value_ = std::numeric_limits<std::int64_t>::min();
  count_ = 0;
}

std::string
HistogramBuckets::OutputString() const
{
  std::ostringstream oss;

  oss << std::left << std::setw(36) << name_ << std::left << std::setw(24)
      << count_ << std::left << std::setw(24) << min_value_ << std::left
      << std::setw(24) << max_value_ << std::left << std::setw(24)
      << GetPercentileEstimate(0.5) << std::left << std::setw(24)
      << GetPercentileEstimate(0.9) << std::left << std::setw(24)
      << GetPercentileEstimate(0.95) << std::left << std::setw(24)
      << GetPercentileEstimate(0.99) << std::endl;

  if (count_ == 0) {
    return "";
  }
  return oss.str();
}

std::int64_t
HistogramBuckets::GetPercentileEstimate(double pct) const
{
  double low_pct = 0.0;
  double high_pct = 0.0;
  size_t idx = GetPercentileBucketIdx(pct, &low_pct, &high_pct);
  if (low_pct == 0.0 && high_pct == 0.0) {
    // means all buckets are empty
    return 0;
  }
  if (low_pct == high_pct) {
    return buckets_[idx].Avg();
  }

  std::int64_t avg = buckets_[idx].Avg();
  std::int64_t low;
  std::int64_t high;
  if (idx == 0) {
    if (avg > min_) {
      // Unlikely to happen except overflow happen
      return GetBucketMin(idx);
    }
    high = min_;
    low = high - (2 * (high - avg));
    if (low > avg)
      low = std::numeric_limits<std::int64_t>::min();
  } else if (idx == buckets_.size() - 1) {
    if (avg < max_) {
      // Unlikely to happen except overflow happen
      return GetBucketMax(idx);
    }
    low = max_;
    high = low + (2 * (avg - low));
    if (high < avg)
      high = std::numeric_limits<std::int64_t>::max();
  } else {
    low = GetBucketMin(idx);
    high = GetBucketMax(idx);
    if (avg < low || avg > high) {
      // Unlikely to happen except overflow happen
      return (low + high) / 2;
    }
  }

  double median_pct = (low_pct + high_pct) / 2;
  if (pct < median_pct) {
    double pct_through_section = (pct - low_pct) / (median_pct - low_pct);
    return low + (avg - low) * pct_through_section;
  } else {
    double pct_through_section = (pct - median_pct) / (high_pct - median_pct);
    return avg + (high - avg) * pct_through_section;
  }
}

size_t
HistogramBuckets::GetPercentileBucketIdx(
    double pct, double* low_pct, double* high_pct) const
{
  auto bucket_count = buckets_.size();
  std::vector<std::uint64_t> counts(bucket_count);
  std::uint64_t total_count = 0;
  for (size_t n = 0; n < bucket_count; ++n) {
    std::uint64_t count = buckets_[n].count;
    counts[n] = count;
    total_count += count;
  }

  if (total_count == 0) {
    *low_pct = 0.0;
    *high_pct = 0.0;
    return 1;
  }

  double prev_pct = 0.0;
  double cur_pct = 0.0;
  std::uint64_t cur_count = 0;
  size_t idx = 0;
  for (idx = 0; idx < bucket_count; ++idx) {
    if (counts[idx] == 0)
      continue;
    prev_pct = cur_pct;
    cur_count += counts[idx];
    cur_pct = static_cast<double>(cur_count) / total_count;
    if (pct < cur_pct)
      break;
  }
  *low_pct = prev_pct;
  *high_pct = cur_pct;
  return idx;
}

size_t
HistogramBuckets::GetBucketIdx(std::int64_t value) const
{
  if (value < min_) {
    return 0;
  } else if (value >= max_) {
    return buckets_.size() - 1;
  } else {
    return (value - min_) / bucket_size_ + 1;
  }
}

Bucket&
HistogramBuckets::GetByValue(std::int64_t value)
{
  return buckets_[GetBucketIdx(value)];
}

const Bucket&
HistogramBuckets::GetByValue(std::int64_t value) const
{
  return buckets_[GetBucketIdx(value)];
}

std::int64_t
HistogramBuckets::GetBucketMin(size_t idx) const
{
  if (idx == 0) {
    return std::numeric_limits<std::int64_t>::min();
  }
  if (idx == buckets_.size() - 1) {
    return max_;
  }
  return (min_ + (idx - 1) * bucket_size_);
}

std::int64_t
HistogramBuckets::GetBucketMax(size_t idx) const
{
  if (idx == buckets_.size() - 1) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return (min_ + idx * bucket_size_);
}

}}}  // namespace triton::backend::pytorch