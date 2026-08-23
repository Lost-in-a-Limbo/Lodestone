// Phase 1 — aligned contiguous vector storage.

#include "lodestone/vector_store.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace lodestone {

namespace {

/// Floats per cache line: 64 / 4 = 16.
constexpr std::size_t floats_per_line = vector_alignment / sizeof(float);

constexpr std::size_t round_up(std::size_t n, std::size_t multiple) {
  return ((n + multiple - 1) / multiple) * multiple;
}

} // namespace

VectorStore::~VectorStore() {
  // Paired with reserve()'s std::aligned_alloc: aligned_alloc memory is
  // released with plain free(), not a special call.
  std::free(data_);
}

VectorStore::VectorStore(VectorStore&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), dim_(std::exchange(other.dim_, 0)),
      stride_(std::exchange(other.stride_, 0)), size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)) {}

VectorStore& VectorStore::operator=(VectorStore&& other) noexcept {
  if (this != &other) {
    std::free(data_);
    data_ = std::exchange(other.data_, nullptr);
    dim_ = std::exchange(other.dim_, 0);
    stride_ = std::exchange(other.stride_, 0);
    size_ = std::exchange(other.size_, 0);
    capacity_ = std::exchange(other.capacity_, 0);
  }
  return *this;
}

const float* VectorStore::get(VectorId id) const {
  // Deliberately unchecked. This runs once per distance computation, and ids
  // only ever come from the graph, which only holds ids it issued itself.
  return data_ + (static_cast<std::size_t>(id) * stride_);
}

Status VectorStore::reserve(std::size_t dim, std::size_t capacity) {
  if (dim == 0 || capacity == 0) {
    return Status::invalid_argument;
  }

  // Ids are uint32_t (DECISIONS.md D2) and invalid_id is its maximum, which
  // doubles as the empty-slot marker in neighbour lists. A capacity beyond
  // that range would eventually hand out an id colliding with the marker —
  // corruption that surfaces in Phase 3 as inexplicably wrong recall rather
  // than as a crash anywhere near the cause.
  if (capacity > std::numeric_limits<VectorId>::max()) {
    return Status::invalid_argument;
  }

  // Round the dimension up so that *every* vector starts on a cache line, not
  // only the first. At dim 128 and dim 960 this is a no-op, which is precisely
  // why the tests also run at dim 100.
  const std::size_t stride = round_up(dim, floats_per_line);

  // Overflow guards. Without them a wrapped size makes aligned_alloc succeed
  // at some tiny value and every subsequent write runs off the end of the
  // block — a heap corruption whose first symptom is unrelated to its cause.
  if (stride > std::numeric_limits<std::size_t>::max() / capacity) {
    return Status::out_of_memory;
  }
  const std::size_t floats = stride * capacity;
  if (floats > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Status::out_of_memory;
  }
  const std::size_t bytes = floats * sizeof(float);

  // std::aligned_alloc requires the size to be a multiple of the alignment.
  // glibc tolerates a violation but it is UB, so it has to be guaranteed
  // rather than hoped for — and it is, for free: stride is a multiple of 16
  // floats, which is exactly 64 bytes.
  assert(bytes % vector_alignment == 0);

  void* block = std::aligned_alloc(vector_alignment, bytes);
  if (block == nullptr) {
    return Status::out_of_memory;
  }

  // Zero the whole block, padding included. This is a contract the distance
  // kernels rely on, not tidiness: with padding at zero, a squared-L2 term is
  // (0-0)^2 = 0 and an inner-product term is 0*0 = 0, so a kernel may process
  // all `stride` floats instead of exactly `dim`. That lets Phase 2's AVX2
  // kernel run whole 8-wide iterations with no scalar tail and no masked final
  // load — the tail being the fiddliest, buggiest part of a hand-written SIMD
  // kernel. It also makes the reported RSS honest, since touching every page
  // here means VmHWM reflects resident memory rather than lazily-mapped
  // address space.
  std::memset(block, 0, bytes);

  // Only now release the old block, so a failed allocation above leaves the
  // store exactly as it was rather than empty.
  std::free(data_);

  data_ = static_cast<float*>(block);
  dim_ = dim;
  stride_ = stride;
  capacity_ = capacity;
  size_ = 0; // reserve is a reset, not a grow

  return Status::ok;
}

std::optional<VectorId> VectorStore::add(std::span<const float> src) {
  if (src.size() != dim_) {
    return std::nullopt;
  }
  if (size_ >= capacity_) {
    return std::nullopt;
  }

  // Copies dim_ floats, never stride_ — the padding was zeroed by reserve()
  // and must stay that way.
  std::memcpy(data_ + (size_ * stride_), src.data(), dim_ * sizeof(float));

  const auto id = static_cast<VectorId>(size_);
  ++size_;
  return id;
}

} // namespace lodestone
