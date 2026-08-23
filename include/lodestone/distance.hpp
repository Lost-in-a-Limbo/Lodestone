#pragma once

#include "lodestone/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace lodestone {

/// Which distance the index is built and queried under. Chosen once per index;
/// mixing metrics between build and query silently destroys recall.
enum class Metric : std::uint8_t {
  l2,
  inner_product,
};

/// THE INJECTABLE INTERFACE. Graph code holds a `DistanceComputer&` and
/// nothing else — it never names a concrete kernel. Phase 2 swaps in SIMD
/// through this seam and Phase 5 swaps in quantised distances through the same
/// one. If `l2_distance(` ever appears inside hnsw.cpp, this rule has broken.
///
/// Three requirements shaped this into an object rather than a free function,
/// and the third is why it has a batch method:
///
/// 1. It needs per-query state. Phase 5's product quantisation uses
///    *asymmetric* distance: the query stays full-precision float while the
///    stored vectors are one-byte-per-subspace codes. You project the query
///    into each subspace once, build a 256 x m lookup table, and every
///    subsequent distance is m table lookups and adds. That table is per-query
///    state, and a free function f(a, b, dim) has nowhere to put it. Hence
///    prepare_query() — for plain L2 it just stashes the pointer.
///
/// 2. It must not cost an indirect call per distance. A virtual call on the
///    single hottest operation in the system is exactly the wrong instinct,
///    which is what distances_to() is for. HNSW's inner loop expands one
///    node's neighbour list, which is naturally a batch of M ids: one virtual
///    call, M distances computed inside it. The dispatch amortises to nothing
///    and the kernel stays free to run AVX2 across the whole batch, including
///    prefetching the next vector while computing the current one.
///
/// 3. The kernel must be selectable at *runtime*, because Phase 2 dispatches
///    on CPU feature detection. That is what rules out the faster-looking
///    alternative — a C++20 concept plus templating the index on the computer
///    type, which inlines fully and dispatches not at all, but bakes the
///    choice in at compile time. See DECISIONS.md for the full trade.
class DistanceComputer {
public:
  DistanceComputer() = default;
  virtual ~DistanceComputer() = default;

  // Not copied or moved: implementations own per-query scratch buffers, and a
  // silent copy of a prepared computer is a bug that shows up as wrong
  // distances rather than as a crash.
  DistanceComputer(const DistanceComputer&) = delete;
  DistanceComputer& operator=(const DistanceComputer&) = delete;
  DistanceComputer(DistanceComputer&&) = delete;
  DistanceComputer& operator=(DistanceComputer&&) = delete;

  /// Bind the query. Called once per search, before any distance call.
  /// `query` must point to dim() floats and stay valid until the next
  /// prepare_query() or destruction — the computer does not copy it.
  virtual void prepare_query(const float* query) = 0;

  /// Distance from the prepared query to one stored vector. Present for
  /// non-batchable callers (the entry-point descent visits a single node per
  /// layer); the hot path uses distances_to() instead.
  [[nodiscard]] virtual float distance_to(VectorId id) const = 0;

  /// Distance from the prepared query to each of `ids`, written to `out`.
  /// `out.size()` must equal `ids.size()`. This is the hot path.
  virtual void distances_to(std::span<const VectorId> ids, std::span<float> out) const = 0;

  /// Dimensionality this computer expects. Used to validate queries at the
  /// boundary, not on the hot path.
  [[nodiscard]] virtual std::size_t dim() const = 0;

  /// Which metric this computer implements. Serialised with the index so a
  /// reload cannot pair an L2 graph with an inner-product computer.
  [[nodiscard]] virtual Metric metric() const = 0;
};

} // namespace lodestone
