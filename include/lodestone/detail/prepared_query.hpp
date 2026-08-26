#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace lodestone::detail {

/// A kernel's prepared query: `stride` floats, 64-byte aligned, and zero
/// everywhere past `dim`.
///
/// Extracted because every kernel needs exactly this and there are three of
/// them. But the alignment is the substantive reason, not the duplication.
///
/// `std::vector<float>` gets its storage from `operator new`, which on x86-64
/// guarantees 16-byte alignment and nothing more. That is enough for SSE's
/// 16-byte loads. It is **not** enough for AVX2: a 32-byte `_mm256_load_ps`
/// from a 16-byte-aligned base is undefined behaviour on half its offsets, and
/// even the unaligned form would have half its loads straddle a cache line —
/// a buffer 16 bytes into a line, read 32 bytes at a time, crosses the 64-byte
/// boundary on every other step.
///
/// So the buffer is aligned to a full cache line, which makes every load on
/// both sides of a distance computation aligned and line-contained: the store
/// is 64-byte aligned with a stride that is a multiple of 64 bytes
/// (`VectorStore`), and now so is this.
///
/// The padding tail is written once, at construction. `set()` overwrites only
/// the first `dim` floats, so a repeated query costs one memcpy of the payload
/// and nothing else — which matters because `prepare_query` runs once per
/// search and the padding is what lets kernels skip their tail loop entirely.
class PreparedQuery {
public:
  PreparedQuery() = default;

  /// `stride` must be a multiple of 16 floats — which `VectorStore::stride()`
  /// always is, since that is what makes every vector cache-line aligned.
  PreparedQuery(std::size_t dim, std::size_t stride) : dim_(dim), stride_(stride) {
    if (stride_ == 0) {
      return;
    }
    const std::size_t bytes = stride_ * sizeof(float);
    // aligned_alloc requires a size that is a multiple of the alignment. Free
    // here: stride is a multiple of 16 floats, which is exactly 64 bytes.
    void* block = std::aligned_alloc(alignment, bytes);
    if (block == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(block, 0, bytes);
    data_ = static_cast<float*>(block);
  }

  ~PreparedQuery() { std::free(data_); }

  PreparedQuery(const PreparedQuery&) = delete;
  PreparedQuery& operator=(const PreparedQuery&) = delete;

  PreparedQuery(PreparedQuery&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), dim_(std::exchange(other.dim_, 0)),
        stride_(std::exchange(other.stride_, 0)) {}

  PreparedQuery& operator=(PreparedQuery&& other) noexcept {
    if (this != &other) {
      std::free(data_);
      data_ = std::exchange(other.data_, nullptr);
      dim_ = std::exchange(other.dim_, 0);
      stride_ = std::exchange(other.stride_, 0);
    }
    return *this;
  }

  /// Copy `dim` floats in. The padding past `dim` is untouched and stays zero.
  void set(const float* query) { std::memcpy(data_, query, dim_ * sizeof(float)); }

  [[nodiscard]] const float* data() const { return data_; }
  [[nodiscard]] std::size_t stride() const { return stride_; }

  /// One cache line, matching `vector_alignment` in vector_store.hpp.
  static constexpr std::size_t alignment = 64;

private:
  float* data_ = nullptr;
  std::size_t dim_ = 0;
  std::size_t stride_ = 0;
};

} // namespace lodestone::detail
