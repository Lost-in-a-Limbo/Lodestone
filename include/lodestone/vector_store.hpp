#pragma once

#include "lodestone/types.hpp"

#include <cstddef>
#include <optional>
#include <span>

namespace lodestone {

/// Alignment every vector in the store starts on. One cache line.
inline constexpr std::size_t vector_alignment = 64;

/// Contiguous, 64-byte-aligned storage for fixed-dimension float32 vectors.
///
/// Phase 1 implements this; Phase 0 only fixes the shape. The layout rule is
/// load-bearing, not stylistic: one allocation, vector i living at offset
/// i * stride(). Never vector<vector<float>> — that turns every neighbour
/// visit into a pointer chase to an unrelated cache line, and a single query
/// visits hundreds of neighbours.
///
/// 64 bytes is the cache line on the target CPU. Aligning the base to it means
/// a 128-dim vector (512B) occupies exactly eight lines and straddles none,
/// and AVX2 can use aligned loads rather than paying for unaligned ones on
/// every single distance computation.
class VectorStore {
public:
  VectorStore() = default;
  ~VectorStore();

  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;
  VectorStore(VectorStore&&) noexcept;
  VectorStore& operator=(VectorStore&&) noexcept;

  /// Allocate room for `capacity` vectors of `dim` floats. Reserving up front
  /// matters: growth would move the whole block, and Phase 3's graph holds ids
  /// that index into it, so a reallocation mid-build is a correctness problem
  /// as much as a performance one.
  [[nodiscard]] Status reserve(std::size_t dim, std::size_t capacity);

  /// Append one vector, returning its id. `src` must hold dim() floats.
  [[nodiscard]] std::optional<VectorId> add(std::span<const float> src);

  /// Read-only view of one stored vector. No bounds check — this is the hot
  /// path, and ids come from the graph, which only ever holds ids it issued.
  [[nodiscard]] const float* get(VectorId id) const;

  [[nodiscard]] std::size_t dim() const { return dim_; }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }

  /// Floats between the start of one vector and the start of the next. Equals
  /// dim() when dim() * 4 is already a multiple of 64, and is padded up
  /// otherwise, so that *every* vector is line-aligned rather than only the
  /// first. At dim 128 no padding is needed; at dim 100 it is.
  [[nodiscard]] std::size_t stride() const { return stride_; }

  /// Bytes actually held, for the memory numbers every phase has to report.
  [[nodiscard]] std::size_t bytes() const { return capacity_ * stride_ * sizeof(float); }

private:
  float* data_ = nullptr;
  std::size_t dim_ = 0;
  std::size_t stride_ = 0;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

} // namespace lodestone
