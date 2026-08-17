//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// count_min_sketch.cpp
//
// Identification: src/primer/count_min_sketch.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "primer/count_min_sketch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bustub {

/**
 * Constructor for the count-min sketch.
 *
 * @param width The width of the sketch matrix.
 * @param depth The depth of the sketch matrix.
 * @throws std::invalid_argument if width or depth are zero.
 */
template <typename KeyType>
CountMinSketch<KeyType>::CountMinSketch(uint32_t width, uint32_t depth) : width_(width), depth_(depth) {
  if (width_ == 0 || depth_ == 0) {
    throw std::invalid_argument("CountMinSketch dimensions must be non-zero.");
  }
  matrix_ = std::vector<std::atomic<uint32_t>>(static_cast<size_t>(width_) * depth_);
  Clear();

  /** @spring2026 PLEASE DO NOT MODIFY THE FOLLOWING */
  // Initialize seeded hash functions
  hash_functions_.reserve(depth_);
  for (size_t i = 0; i < depth_; i++) {
    hash_functions_.push_back(this->HashFunction(i));
  }
}

template <typename KeyType>
CountMinSketch<KeyType>::CountMinSketch(CountMinSketch &&other) noexcept : width_(other.width_), depth_(other.depth_) {
  // Moving the vector only steals its buffer, so the counters themselves are never copied.
  matrix_ = std::move(other.matrix_);
  // The closures in other.hash_functions_ captured `&other`, so they cannot come along.
  ResetHashFunctions();

  // Leave `other` empty but valid: no counters, and no hash functions that could divide by
  // the now-zero width_.
  other.hash_functions_.clear();
  other.matrix_.clear();
  other.width_ = 0;
  other.depth_ = 0;
}

template <typename KeyType>
auto CountMinSketch<KeyType>::operator=(CountMinSketch &&other) noexcept -> CountMinSketch & {
  if (this == &other) {
    return *this;
  }
  width_ = other.width_;
  depth_ = other.depth_;
  matrix_ = std::move(other.matrix_);
  ResetHashFunctions();

  other.hash_functions_.clear();
  other.matrix_.clear();
  other.width_ = 0;
  other.depth_ = 0;
  return *this;
}

template <typename KeyType>
void CountMinSketch<KeyType>::Insert(const KeyType &item) {
  // Lock-free: each row's counter is bumped independently with an atomic read-modify-write.
  // Relaxed ordering is enough since the counters carry no happens-before relationship.
  for (size_t row = 0; row < depth_; row++) {
    matrix_[SlotOf(row, item)].fetch_add(1, std::memory_order_relaxed);
  }
}

template <typename KeyType>
void CountMinSketch<KeyType>::Merge(const CountMinSketch<KeyType> &other) {
  if (width_ != other.width_ || depth_ != other.depth_) {
    throw std::invalid_argument("Incompatible CountMinSketch dimensions for merge.");
  }
  // Equal dimensions and identical seeds mean both sketches map an item to the same cells,
  // so a cell-wise sum yields the sketch of the concatenated streams.
  for (size_t i = 0; i < matrix_.size(); i++) {
    matrix_[i].fetch_add(other.matrix_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
}

template <typename KeyType>
auto CountMinSketch<KeyType>::Count(const KeyType &item) const -> uint32_t {
  if (matrix_.empty()) {
    return 0;
  }
  // Every row overestimates by whatever collided into that cell, so the tightest bound is
  // the smallest of them.
  uint32_t estimate = std::numeric_limits<uint32_t>::max();
  for (size_t row = 0; row < depth_; row++) {
    estimate = std::min(estimate, matrix_[SlotOf(row, item)].load(std::memory_order_relaxed));
  }
  return estimate;
}

template <typename KeyType>
void CountMinSketch<KeyType>::Clear() {
  for (auto &counter : matrix_) {
    counter.store(0, std::memory_order_relaxed);
  }
}

template <typename KeyType>
auto CountMinSketch<KeyType>::TopK(uint16_t k, const std::vector<KeyType> &candidates)
    -> std::vector<std::pair<KeyType, uint32_t>> {
  std::vector<std::pair<KeyType, uint32_t>> ranked;
  ranked.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    ranked.emplace_back(candidate, Count(candidate));
  }

  // stable_sort so that equal estimates keep the caller's candidate order, making the
  // result deterministic instead of dependent on the sort's internals.
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const std::pair<KeyType, uint32_t> &lhs, const std::pair<KeyType, uint32_t> &rhs) -> bool {
                     return lhs.second > rhs.second;
                   });

  if (ranked.size() > static_cast<size_t>(k)) {
    ranked.erase(ranked.begin() + k, ranked.end());
  }
  return ranked;
}

// Explicit instantiations for all types used in tests
template class CountMinSketch<std::string>;
template class CountMinSketch<int64_t>;  // For int64_t tests
template class CountMinSketch<int>;      // This covers both int and int32_t
}  // namespace bustub
