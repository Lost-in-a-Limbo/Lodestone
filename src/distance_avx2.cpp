// Phase 2 — 8-wide AVX2 squared-L2 and inner-product kernels.
//
// The kernel is templated on its accumulator count because task 5 of
// .claude/plans/phase2.md is an experiment, not an implementation: measure 1, 2,
// 4 and 8 independent accumulators, publish the whole curve including the
// losers, and keep the winner.
//
// The textbook argument for more accumulators: `vfmadd231ps` on Zen 3 has ~4
// cycles of latency and 2/cycle of throughput, so a single accumulator makes
// every FMA wait on the previous one and the loop runs at one FMA per 4 cycles
// instead of two per cycle. Saturation needs latency x throughput = 8 chains.
//
// Measured before this task ran, and it says the textbook argument may not
// apply here: the single-accumulator kernel already hits 8.64x scalar at
// dim 128, taking 2.55 cycles per iteration where a 4-cycle dependent chain
// should floor it at 4. The chain is already broken — `distances_to()` computes
// independent distances back to back, so the tail of one overlaps the head of
// the next, and batching supplies the parallelism extra accumulators would.
//
// See BENCHMARKS.md for the resulting curve.

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

/// The accumulator count the shipped kernel uses.
///
/// Chosen by measurement, not by the latency argument — see BENCHMARKS.md. The
/// curve is nearly flat, because `distances_to()` already overlaps independent
/// distances and supplies the instruction-level parallelism that extra
/// accumulators would otherwise provide.
constexpr std::size_t default_accumulators = 4;

/// Collapse eight lanes to one float.
///
/// Fold the upper 128 bits onto the lower half first, then finish in SSE. The
/// cast is free — a register view, not an instruction — while `extractf128` is
/// the only real cross-lane move needed.
///
/// Paid once per distance, not per element. At dim 128 the main loop runs 16
/// eight-lane iterations against this one reduction; at dim 960 it runs 120. So
/// the reduction's share of the work shrinks as the dimension grows.
inline float horizontal_sum(__m256 v) {
  const __m128 low = _mm256_castps256_ps128(v);
  const __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(low, high); // 8 -> 4

  const __m128 swapped = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
  sum = _mm_add_ps(sum, swapped); // 4 -> 2
  const __m128 upper = _mm_movehl_ps(swapped, sum);
  return _mm_cvtss_f32(_mm_add_ss(sum, upper)); // 2 -> 1
}

template <Metric metric_v, std::size_t accumulators>
class Avx2Computer final : public DistanceComputer {
  static_assert(accumulators >= 1);

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
  /// One fused multiply-add step against accumulator `acc` at offset `offset`.
  inline void step(__m256& acc, const float* stored, const float* query, std::size_t offset) const {
    const __m256 a = _mm256_load_ps(stored + offset);
    const __m256 b = _mm256_load_ps(query + offset);
    if constexpr (metric_v == Metric::l2) {
      const __m256 diff = _mm256_sub_ps(a, b);
      acc = _mm256_fmadd_ps(diff, diff, acc);
    } else {
      acc = _mm256_fmadd_ps(a, b, acc);
    }
  }

  [[nodiscard]] float compute(VectorId id) const {
    const float* stored = store_.get(id);
    const float* query = query_.data();

    __m256 acc[accumulators];
    for (std::size_t a = 0; a < accumulators; ++a) {
      acc[a] = _mm256_setzero_ps();
    }

    // No tail loop over *elements* and no masked load anywhere. stride_ is
    // always a multiple of 16 floats, the store zeroes its own padding, and
    // PreparedQuery zero-pads the query — so every padding term is (0-0)^2 = 0
    // for L2 and 0*0 = 0 for inner product (D13).
    //
    // There is still a remainder loop, but over *accumulator groups*, not
    // elements: with 4 accumulators the group is 32 floats and stride 112
    // (dim 100) leaves 16 floats over. Those go through acc[0] eight at a time,
    // which is still whole vectors — never a partial one.
    //
    // Both loads are aligned and neither straddles a cache line: the store is
    // 64-byte aligned with a 64-byte-multiple stride, and PreparedQuery is
    // 64-byte aligned for the same reason.
    constexpr std::size_t group = lanes * accumulators;
    std::size_t i = 0;
    for (; i + group <= stride_; i += group) {
      for (std::size_t a = 0; a < accumulators; ++a) {
        step(acc[a], stored, query, i + (a * lanes));
      }
    }
    for (; i < stride_; i += lanes) {
      step(acc[0], stored, query, i);
    }

    __m256 total = acc[0];
    for (std::size_t a = 1; a < accumulators; ++a) {
      total = _mm256_add_ps(total, acc[a]);
    }

    const float sum = horizontal_sum(total);

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

template <std::size_t accumulators>
std::unique_ptr<DistanceComputer> make_with(Metric metric, const VectorStore& store) {
  if (metric == Metric::l2) {
    return std::make_unique<Avx2Computer<Metric::l2, accumulators>>(store);
  }
  return std::make_unique<Avx2Computer<Metric::inner_product, accumulators>>(store);
}

} // namespace

namespace detail {

std::unique_ptr<DistanceComputer> make_avx2_l2(const VectorStore& store) {
  return make_with<default_accumulators>(Metric::l2, store);
}

std::unique_ptr<DistanceComputer> make_avx2_ip(const VectorStore& store) {
  return make_with<default_accumulators>(Metric::inner_product, store);
}

std::unique_ptr<DistanceComputer> make_avx2_experiment(Metric metric, const VectorStore& store,
                                                       std::size_t accumulators) {
  switch (accumulators) {
  case 1:
    return make_with<1>(metric, store);
  case 2:
    return make_with<2>(metric, store);
  case 4:
    return make_with<4>(metric, store);
  case 8:
    return make_with<8>(metric, store);
  default:
    return nullptr;
  }
}

} // namespace detail

} // namespace lodestone
