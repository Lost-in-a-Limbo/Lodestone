#pragma once

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace lodestone {

/// Hierarchical Navigable Small World index.
///
/// Follows Malkov & Yashunin, arXiv 1603.09320. The method names below are the
/// paper's algorithm names, so the code can be read side by side with it:
///
///   Algorithm 1  INSERT                     -> add()
///   Algorithm 2  SEARCH-LAYER               -> search_layer()
///   Algorithm 3  SELECT-NEIGHBORS-SIMPLE    -> select_neighbours_simple()
///   Algorithm 4  SELECT-NEIGHBORS-HEURISTIC -> select_neighbours_heuristic()
///   Algorithm 5  K-NN-SEARCH                -> search()
///
/// **Distances go through `DistanceComputer` and nowhere else.** The graph
/// never names a kernel; that is architecture rule 1, and it is what lets
/// Phase 5 swap in quantised distances without touching this file.
class HnswIndex {
public:
  virtual ~HnswIndex() = default;

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;
  HnswIndex(HnswIndex&&) = delete;
  HnswIndex& operator=(HnswIndex&&) = delete;

  /// Insert one stored vector into the graph. Ids must already exist in the
  /// store this index was built against.
  virtual Status add(VectorId id) = 0;

  /// Algorithm 5. `out.size()` is k. Results are sorted nearest-first under the
  /// same `(distance, id)` total order brute force uses, so the two are
  /// directly comparable.
  ///
  /// `params.ef` bounds the layer-0 candidate list and must be >= k. Larger ef
  /// explores more of the graph: more recall, less throughput.
  ///
  /// **Const but not thread-safe** — per-query scratch (the visited set and the
  /// heaps) lives in the index. One index per thread, or add a scratch
  /// parameter, if Phase 4 ever threads this.
  [[nodiscard]] virtual Status search(const float* query, const SearchParams& params,
                                      std::span<Neighbor> out) const = 0;

  /// How many nodes the last search() actually examined. The number that says
  /// whether the graph is doing its job: brute force visits every vector, and
  /// the whole point of an index is that this is orders of magnitude smaller.
  [[nodiscard]] virtual std::size_t last_visited() const = 0;

  [[nodiscard]] virtual std::size_t size() const = 0;
  [[nodiscard]] virtual std::size_t max_level() const = 0;
  [[nodiscard]] virtual VectorId entry_point() const = 0;

  /// Level assigned to a node, 0 for the vast majority.
  [[nodiscard]] virtual std::size_t level_of(VectorId id) const = 0;

  /// Neighbours of `id` at `level`, for tests and for Phase 6's traversal.
  [[nodiscard]] virtual std::span<const VectorId> neighbours(VectorId id,
                                                             std::size_t level) const = 0;

  /// Bytes the graph occupies, excluding the vector store. The memory figure
  /// the phase has to report.
  [[nodiscard]] virtual std::size_t graph_bytes() const = 0;

  /// Write the graph — not the vectors. The store is reloaded from its `.fvecs`
  /// separately; duplicating 488 MiB into an index file to make one API
  /// prettier is not a trade worth making.
  [[nodiscard]] virtual Status save(const std::filesystem::path& path) const = 0;

protected:
  HnswIndex() = default;
};

/// Which neighbour-selection rule construction uses.
///
/// Both exist because the *comparison* is the point. `heuristic` is Algorithm 4
/// and is what the paper recommends; `simple` is Algorithm 3, kept so the
/// recall gap between them can be measured rather than asserted.
enum class NeighbourSelection : std::uint8_t {
  heuristic,
  simple,
};

/// Build an empty index over `store`. Returns nullptr for a config that cannot
/// produce a usable graph (m == 0, ef_construction < m_max0, empty store).
///
/// `store` and the metric are fixed for the index's life. `store` must outlive
/// the index — it is referenced, not copied.
[[nodiscard]] std::unique_ptr<HnswIndex>
make_hnsw_index(const VectorStore& store, Metric metric, const HnswConfig& config,
                NeighbourSelection selection = NeighbourSelection::heuristic);

/// Produces a fresh `DistanceComputer` on each call. The index needs two — one
/// bound to the query being inserted, one for the neighbour-selection heuristic
/// — so it asks for them rather than being handed one.
using ComputerFactory = std::function<std::unique_ptr<DistanceComputer>()>;

/// Build an index over `store` using computers the caller supplies.
///
/// **This is how Phase 5's quantized distances reach the graph**, and it is the
/// entire accommodation the graph made for them: `make_hnsw_index` above simply
/// delegates here with `make_distance_computer`. No algorithm changed — not
/// `SEARCH-LAYER`, not `INSERT`, not the neighbour heuristic, not the search
/// path. See DECISIONS.md D33 for the honest accounting of that claim, which is
/// narrower than "without touching hnsw.cpp".
///
/// The factory must return computers over the same `store` and metric. Handing
/// back a computer bound to different data would produce a graph whose edges
/// mean nothing, with no symptom but poor recall.
[[nodiscard]] std::unique_ptr<HnswIndex>
make_hnsw_index_with(const VectorStore& store, ComputerFactory factory, const HnswConfig& config,
                     NeighbourSelection selection = NeighbourSelection::heuristic);

/// Reload a graph written by `save()`, re-binding it to a freshly loaded store.
/// Returns nullptr if the file is malformed, or if its recorded shape does not
/// match the store handed in — pairing an index with the wrong corpus would
/// otherwise show up as mysteriously poor recall.
[[nodiscard]] std::unique_ptr<HnswIndex> load_hnsw_index(const std::filesystem::path& path,
                                                         const VectorStore& store, Metric metric);

namespace detail {

/// Level assignment, exposed for testing.
///
/// `floor(-ln(U(0,1)) * mL)` with `mL = 1/ln(M)`, which makes the level
/// distribution geometric: about 1/M of the nodes reach each successive level.
/// That is what makes the top layers sparse enough to skip across the corpus in
/// a few hops, and layer 0 dense enough to be accurate.
[[nodiscard]] std::size_t assign_level(std::uint64_t& rng_state, double level_multiplier);

/// The `mL` above, from M.
[[nodiscard]] double level_multiplier_for(std::size_t m);

} // namespace detail

} // namespace lodestone
