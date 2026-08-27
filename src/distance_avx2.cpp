// Phase 2 — 8-wide AVX2 squared-L2 and inner-product kernels.
//
// **Single accumulator, on purpose.** Task 5 of .claude/plans/phase2.md is the
// experiment that varies the accumulator count and publishes the whole curve,
// and that experiment needs a naive baseline to move. Writing the tuned version
// first would hide the most instructive number in the phase.
//
// The SSE row already told us something that changes what to expect here: at
// 4 wide, `stream` saturates single-core memory bandwidth at 15.5 GiB/s. So AVX2
// should roughly halve the `l1` times and leave `stream` almost untouched. That
// is recorded in BENCHMARKS.md as a prediction to be judged, not assumed.

#include "lodestone/detail/prepared_query.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <span>

// AVX2 and FMA are scoped to this single translation unit, never enabled
// globally. Global -mavx2 would let the compiler emit AVX2 anywhere — in the
// scalar baseline, and in the dispatch code that is supposed to run on a
// machine that lacks it, which would defeat runtime feature detection by
// crashing before it could dispatch.
//
// distance_sse.cpp carries the mirror of these: it #errors when __AVX2__ is
// PRESENT. Between the two, a mislabelled measurement cannot be produced
// quietly.
#ifndef __AVX2__
#error "distance_avx2.cpp must be compiled with -mavx2. See CMakeLists.txt."
#endif
#ifndef __FMA__
#error "distance_avx2.cpp must be compiled with -mfma. See CMakeLists.txt."
#endif

namespace lodestone {

namespace {

/// Floats per AVX2 register.
constexpr std::size_t lanes = 8;

/// Collapse eight lanes to one float.
///
/// Fold the upper 128 bits onto the lower half first, then finish in SSE. The
/// cast is free — it is a register view, not an instruction — while the
/// `extractf128` is the only real cross-lane move needed.
///
/// Paid once per distance, not per element. At dim 128 the main loop runs 16
/// iterations against this one reduction; at dim 960 it runs 120. So the
/// reduction's share of the work shrinks as the dimension grows, which predicts
/// a *larger* speedup at dim 960 than at dim 128 — and SSE already showed
/// exactly that (3.95x versus 3.84x).
inline float horizontal_sum(__m256 v) {
  const __m128 low = _mm256_castps256_ps128(v);
  const __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(low, high); // 8 -> 4

  const __m128 swapped = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
  sum = _mm_add_ps(sum, swapped); // 4 -> 2
  const __m128 upper = _mm_movehl_ps(swapped, sum);
  return _mm_cvtss_f32(_mm_add_ss(sum, upper)); // 2 -> 1
}

template <Metric metric_v>
class Avx2Computer final : public DistanceComputer {
public:
  explicit Avx2Computer(const VectorStore& store)
      : store_(store), dim_(store.dim()), stride_(store.stride()),
        query_(store.dim(), store.stride()) {}

  void prepare_query(const float* query) override { query_.set(query); }

  [[nodiscard]] float distance_to(VectorId id) const override { return compute(id); }

  void distances_to(std::span<const VectorId> ids, std::span<float> out) const override {
    assert(ids.size() == out.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      out[i] = compute(ids[i]);
    }
  }

  [[nodiscard]] std::size_t dim() const override { return dim_; }
  [[nodiscard]] Metric metric() const override { return metric_v; }
  [[nodiscard]] KernelKind kernel() const override { return KernelKind::avx2; }

private:
  [[nodiscard]] float compute(VectorId id) const {
    const float* stored = store_.get(id);
    const float* query = query_.data();

    __m256 acc = _mm256_setzero_ps();

    // No tail loop and no masked final load. stride_ is always a multiple of 16
    // floats — so also a multiple of 8 — the store zeroes its own padding, and
    // PreparedQuery zero-pads the query. Every padding term is therefore
    // (0-0)^2 = 0 for L2 and 0*0 = 0 for inner product, and the loop simply
    // runs to the end of the stride. The tail is the fiddliest and buggiest
    // part of writing a SIMD kernel by hand, and Phase 1's D13 deleted it.
    //
    // Both loads are aligned and neither straddles a cache line: the store is
    // 64-byte aligned with a 64-byte-multiple stride, and PreparedQuery is
    // 64-byte aligned for exactly this reason — std::vector<float> would only
    // have guaranteed 16, making _mm256_load_ps undefined on half its offsets.
    for (std::size_t i = 0; i < stride_; i += lanes) {
      const __m256 a = _mm256_load_ps(stored + i);
      const __m256 b = _mm256_load_ps(query + i);
      if constexpr (metric_v == Metric::l2) {
        const __m256 diff = _mm256_sub_ps(a, b);
        // One sub plus one FMA per 8 elements. Every FMA depends on the
        // previous one through `acc` — that single chain is what task 5
        // measures the cost of.
        acc = _mm256_fmadd_ps(diff, diff, acc);
      } else {
        acc = _mm256_fmadd_ps(a, b, acc);
      }
    }

    const float sum = horizontal_sum(acc);

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
  detail::PreparedQuery query_;
};

} // namespace

namespace detail {

std::unique_ptr<DistanceComputer> make_avx2_l2(const VectorStore& store) {
  return std::make_unique<Avx2Computer<Metric::l2>>(store);
}

std::unique_ptr<DistanceComputer> make_avx2_ip(const VectorStore& store) {
  return std::make_unique<Avx2Computer<Metric::inner_product>>(store);
}

} // namespace detail

} // namespace lodestone
