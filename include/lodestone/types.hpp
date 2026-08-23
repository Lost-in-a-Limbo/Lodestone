#pragma once

#include <cstddef>
#include <cstdint>

namespace lodestone {

/// Identifier for one stored vector.
///
/// 32-bit, deliberately not std::size_t. Neighbour lists dominate index memory
/// and are the hottest read in the whole system: search expands one node's
/// neighbours, then the next, then the next. At M = 16 a 32-bit list is
/// 16 x 4B = 64B — exactly one cache line, one miss. With 64-bit ids the same
/// list is 128B: two lines, two potential misses, on every node expansion.
/// 1M vectors needs 20 bits, so 32 leaves three orders of magnitude of room.
using VectorId = std::uint32_t;

/// "No such vector". Also the empty-slot marker inside neighbour lists, so an
/// unused edge costs no side-table and no extra byte.
inline constexpr VectorId invalid_id = static_cast<VectorId>(-1);

/// Error channel.
///
/// CLAUDE.md forbids exceptions on the search hot path. Operations that can
/// fail return this; operations that also carry a value return
/// std::optional<T> alongside it. std::expected would be the nicer shape, but
/// it is C++23 and libstdc++ gates <expected> behind __cplusplus > 202002L —
/// see the probe in CMakeLists.txt and the entry in DECISIONS.md.
enum class Status : std::uint8_t {
  ok = 0,
  io_error,
  invalid_argument,
  dimension_mismatch,
  out_of_memory,
  not_implemented,
};

/// Graph construction parameters. Field names match Malkov & Yashunin
/// (arXiv 1603.09320) so the code can be read against the paper.
struct HnswConfig {
  /// Edges per node on layers above 0. The paper's M.
  std::size_t m = 16;

  /// Edge budget on layer 0. Conventionally 2*M: layer 0 contains every node
  /// and carries the last, accuracy-critical hops, so it gets the wider budget.
  std::size_t m_max0 = 32;

  /// Candidate list size during construction. Larger builds a better graph and
  /// takes longer; it is a build-time-only knob and does not affect query cost
  /// once the index exists.
  std::size_t ef_construction = 200;

  /// Seed for the layer-assignment RNG. Fixed, not time-based: a benchmark you
  /// cannot rerun and reproduce is not a measurement.
  std::uint64_t seed = 100;
};

/// Query parameters.
struct SearchParams {
  /// Dynamic candidate list size — the recall/QPS dial. Larger ef explores
  /// more of the graph for more recall at lower throughput. Must be >= k;
  /// Phase 4 sweeps it over {8, 16, 32, 64, 128, 256} to draw the curve.
  std::size_t ef = 64;
};

} // namespace lodestone
