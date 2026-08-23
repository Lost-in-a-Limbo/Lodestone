// Phase 1 — aligned contiguous vector storage.
//
// Phase 0 defines only the special members, so the class is not a link-time
// landmine for anything that merely constructs one. The two functions carrying
// the actual design work — the stride/alignment maths in reserve() and the
// append path — are Phase 1.

#include "lodestone/vector_store.hpp"

#include <cstdlib>
#include <utility>

namespace lodestone {

VectorStore::~VectorStore() {
  // Paired with the std::aligned_alloc that reserve() will use in Phase 1:
  // aligned_alloc memory is released with plain free(), not a special call.
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

Status VectorStore::reserve(std::size_t /*dim*/, std::size_t /*capacity*/) {
  return Status::not_implemented; // Phase 1
}

std::optional<VectorId> VectorStore::add(std::span<const float> /*src*/) {
  return std::nullopt; // Phase 1
}

} // namespace lodestone
