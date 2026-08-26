// Phase 2 — 4-wide SSE squared-L2 and inner-product kernels.
//
// SSE's real job in this project is being the *control*. If SSE lands near 4x
// scalar and AVX2 near 8x, the scaling is real. If SSE lands at 4x and AVX2 also
// at 4x, something is wrong with the AVX2 kernel — and without an SSE row in the
// table there is nothing to notice that against.
//
// It is also the fallback that needs no feature check: every x86-64 CPU has
// SSE2 by definition of the architecture, and every intrinsic used below is
// SSE2. Nothing here requires runtime detection to be safe.

#include "lodestone/detail/prepared_query.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <emmintrin.h> // SSE2
#include <xmmintrin.h> // SSE

#include <cassert>
#include <cstddef>
#include <memory>
#include <span>

// The mirror image of the guard in distance_avx2.cpp, and it exists for the
// same reason D5 exists.
//
// Under the release preset's -march=native, GCC compiles SSE intrinsics to
// VEX-encoded 128-bit instructions. That alone does not widen them, so a
// 4-wide measurement stays 4-wide — but any scalar code the compiler decides to
// vectorise in this file *would* become AVX2, and the result would be an "SSE"
// row that is partly AVX2. CMakeLists.txt appends -mno-avx -mno-avx2 -mno-fma
// after -march=native so the negations win; these guards fail the build loudly
// if that is ever removed.
//
// distance_avx2.cpp #errors when __AVX2__ is ABSENT. This one #errors when it is
// PRESENT. Between them, a mislabelled number cannot be produced quietly.
#ifdef __AVX2__
#error "distance_sse.cpp must be compiled with -mno-avx2. See CMakeLists.txt — \
without it this file's numbers are not an SSE measurement."
#endif
#ifdef __AVX__
#error "distance_sse.cpp must be compiled with -mno-avx. See CMakeLists.txt."
#endif

namespace lodestone {

namespace {

/// Floats per SSE register.
constexpr std::size_t lanes = 4;

/// Collapse four lanes to one float.
///
/// Shuffles rather than two `_mm_hadd_ps` calls, for two reasons. It is SSE2
/// only — `hadd` is SSE3 — which keeps this kernel a fallback that needs no
/// feature check at all. And `hadd` has notably worse throughput than a shuffle
/// plus an add on every microarchitecture this would run on, despite reading
/// like the obvious instruction for the job.
///
/// Paid once per distance, not per element: at dim 128 that is 32 loop
/// iterations against this one reduction, at dim 960 it is 240. So the
/// reduction's share of the work shrinks as the dimension grows, which predicts
/// a *larger* SIMD speedup at dim 960 than at dim 128.
inline float horizontal_sum(__m128 v) {
  // v = [a b c d] -> [b a d c], add -> [a+b a+b c+d c+d]
  const __m128 swapped = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128 pairs = _mm_add_ps(v, swapped);
  // Move the high pair into the low lane and add once more.
  const __m128 high = _mm_movehl_ps(swapped, pairs);
  return _mm_cvtss_f32(_mm_add_ss(pairs, high));
}

template <Metric metric_v>
class SseComputer final : public DistanceComputer {
public:
  explicit SseComputer(const VectorStore& store)
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
  [[nodiscard]] KernelKind kernel() const override { return KernelKind::sse; }

private:
  [[nodiscard]] float compute(VectorId id) const {
    const float* stored = store_.get(id);
    const float* query = query_.data();

    __m128 acc = _mm_setzero_ps();

    // No tail loop and no masked final load. stride_ is always a multiple of 16
    // floats, the store zeroes its own padding, and PreparedQuery zero-pads the
    // query — so every padding term is (0-0)^2 = 0 for L2 and 0*0 = 0 for inner
    // product, and the loop can simply run to the end of the stride. That is
    // the single biggest simplification Phase 1 handed this phase (D13).
    //
    // Both loads are aligned and neither crosses a cache line: the store is
    // 64-byte aligned with a 64-byte-multiple stride, and PreparedQuery is
    // 64-byte aligned too.
    for (std::size_t i = 0; i < stride_; i += lanes) {
      const __m128 a = _mm_load_ps(stored + i);
      const __m128 b = _mm_load_ps(query + i);
      if constexpr (metric_v == Metric::l2) {
        const __m128 diff = _mm_sub_ps(a, b);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
      } else {
        acc = _mm_add_ps(acc, _mm_mul_ps(a, b));
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

std::unique_ptr<DistanceComputer> make_sse_l2(const VectorStore& store) {
  return std::make_unique<SseComputer<Metric::l2>>(store);
}

std::unique_ptr<DistanceComputer> make_sse_ip(const VectorStore& store) {
  return std::make_unique<SseComputer<Metric::inner_product>>(store);
}

} // namespace detail

} // namespace lodestone
