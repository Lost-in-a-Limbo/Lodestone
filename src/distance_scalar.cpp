// Phase 1/2 — scalar squared-L2 kernel.
//
// Phase 1 implements L2 only, because brute force cannot compute a single
// distance without one concrete kernel existing and architecture rule 1
// forbids reaching around the interface to get it. Inner product, the SIMD
// variants, and runtime dispatch are Phase 2.

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

// This file is the *baseline* in "AVX2 is >=3x scalar". Under the release
// preset's -O3 -march=native, GCC will happily auto-vectorise a plain L2 loop
// into AVX2, at which point Phase 2's headline number is AVX2-vs-AVX2 and
// means nothing at all. CMakeLists.txt pins -fno-tree-vectorize on this one
// file and passes this define from the same property, so the guard below fails
// the build if the flag is ever dropped.
#ifndef LODESTONE_SCALAR_NOVEC
#error "distance_scalar.cpp must be compiled with -fno-tree-vectorize. See the \
set_source_files_properties() call in CMakeLists.txt — removing it silently \
turns the scalar baseline into a vectorised one and invalidates every speedup \
number this project publishes."
#endif

namespace lodestone {

namespace {

/// Scalar distance against a prepared query, for one metric.
///
/// Templated on the metric rather than branching at run time: the metric is
/// fixed for the lifetime of an index, so a runtime test in the innermost loop
/// would be a perfectly predictable branch that still costs an instruction and
/// blocks reassociation. `if constexpr` compiles each variant down to exactly
/// the loop it needs.
///
/// **L2 is squared, never rooted.** sqrt is monotone so it cannot change a
/// ranking, and it would cost a transcendental on the hottest path in the
/// system. Every "distance" this project reports is a squared one, including in
/// the recall and benchmark output, so nothing downstream un-squares it.
///
/// **Inner product is negated.** See `Metric` in distance.hpp for why.
template <Metric metric_v>
class ScalarComputer final : public DistanceComputer {
public:
  explicit ScalarComputer(const VectorStore& store)
      : store_(store), dim_(store.dim()), stride_(store.stride()), query_(store.stride(), 0.0F) {}

  void prepare_query(const float* query) override {
    // Copy exactly dim_ floats into a buffer that is stride_ long and was
    // zero-filled at construction. The padding tail is therefore written once,
    // in the constructor, and never touched again — so repeated queries cost
    // one memcpy of the payload and nothing else.
    std::memcpy(query_.data(), query, dim_ * sizeof(float));
  }

  [[nodiscard]] float distance_to(VectorId id) const override { return compute(id); }

  void distances_to(std::span<const VectorId> ids, std::span<float> out) const override {
    assert(ids.size() == out.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      out[i] = compute(ids[i]);
    }
  }

  [[nodiscard]] std::size_t dim() const override { return dim_; }

  [[nodiscard]] Metric metric() const override { return metric_v; }

  [[nodiscard]] KernelKind kernel() const override { return KernelKind::scalar; }

private:
  /// Non-virtual, so the batch loop above makes one indirect call in total
  /// rather than one per id. Relying on the compiler to devirtualise through a
  /// `final` class would work today and stop working the day someone profiles
  /// a debug build and wonders where the time went.
  [[nodiscard]] float compute(VectorId id) const {
    const float* stored = store_.get(id);
    const float* query = query_.data();

    // Runs the full stride, not just dim. Safe because the store zeroes its
    // padding (VectorStore::reserve) and the query buffer is zero-padded
    // above, so every term past dim is (0-0)^2 = 0. stride is always a
    // multiple of 16 floats, which is what lets Phase 2's AVX2 kernel run
    // whole 8-wide iterations with no tail loop and no masked final load.
    //
    // At the two dimensions that actually get benchmarked, 128 and 960, stride
    // equals dim and this is exactly dim iterations — so the baseline is not
    // inflated by padding at any measured point.
    float sum = 0.0F;
    for (std::size_t i = 0; i < stride_; ++i) {
      if constexpr (metric_v == Metric::l2) {
        const float diff = stored[i] - query[i];
        sum += diff * diff;
      } else {
        sum += stored[i] * query[i];
      }
    }

    if constexpr (metric_v == Metric::inner_product) {
      // Negated once, here, so every consumer can assume smaller is closer.
      return -sum;
    } else {
      return sum;
    }
  }

  const VectorStore& store_;
  std::size_t dim_;
  std::size_t stride_;

  /// Padded to stride_ and zero beyond dim_. Not 64-byte aligned: the query
  /// stays resident in L1 across a whole search, so an unaligned load on it is
  /// far less interesting than on the streaming store side. Phase 2 should
  /// measure that rather than assume it — logged in IDEAS.md.
  std::vector<float> query_;
};

} // namespace

namespace detail {

std::unique_ptr<DistanceComputer> make_scalar_l2(const VectorStore& store) {
  return std::make_unique<ScalarComputer<Metric::l2>>(store);
}

std::unique_ptr<DistanceComputer> make_scalar_ip(const VectorStore& store) {
  return std::make_unique<ScalarComputer<Metric::inner_product>>(store);
}

} // namespace detail

} // namespace lodestone
