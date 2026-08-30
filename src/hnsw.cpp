// Phase 3 — Hierarchical Navigable Small World index.
//
// Malkov & Yashunin, arXiv 1603.09320. Algorithms 1-5, named as the paper names
// them so the two can be read side by side.
//
// Standing rule for this file, from CLAUDE.md: it holds a DistanceComputer& and
// never names a concrete kernel. If `l2_distance(` ever appears here, that is
// architecture rule 1 breaking.

#include "lodestone/hnsw.hpp"

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace lodestone {

namespace detail {

double level_multiplier_for(std::size_t m) {
  // mL = 1 / ln(M). The paper's section 4.1: this is the value that makes the
  // expected number of layers minimal while keeping the overlap between
  // adjacent layers small enough that the descent stays cheap.
  if (m <= 1) {
    return 1.0;
  }
  return 1.0 / std::log(static_cast<double>(m));
}

std::size_t assign_level(std::uint64_t& state, double level_multiplier) {
  // splitmix64, inlined: a fixed, dependency-free generator so a rebuild with
  // the same seed produces the same graph. std::mt19937 would do, but its state
  // is 2.5 KB and this is called once per insert; splitmix64 is 8 bytes and
  // passes the statistical tests that matter for "pick a level".
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);

  // Uniform in (0, 1]. Never exactly 0, because ln(0) is -infinity and the
  // level would be unbounded — a single such draw would create a node hundreds
  // of layers high and every later descent would walk all of them.
  const double u = (static_cast<double>(z >> 11) + 1.0) / 9007199254740993.0;

  const double level = -std::log(u) * level_multiplier;
  return static_cast<std::size_t>(level);
}

} // namespace detail

namespace {

/// One entry in a search heap.
struct Candidate {
  float distance;
  VectorId id;
};

/// The project's total order, matching brute_force.cpp: `(distance, id)`.
/// Distance alone would leave equal-distance nodes ordered by whatever the heap
/// happened to do, and a rebuild could then return a different-but-equally-
/// correct answer. This index is measured against brute force, so the two must
/// break ties the same way.
struct Closer {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    return a.id < b.id;
  }
};

/// Reversed, for a min-heap via std::push_heap (which builds max-heaps).
struct Farther {
  bool operator()(const Candidate& a, const Candidate& b) const { return Closer{}(b, a); }
};

/// "Have I seen this node in *this* query?"
///
/// Asked thousands of times per query, so the answer has to be O(1) and so does
/// the reset. A vector<bool> cleared per query is O(N) — at a million nodes
/// that clearing alone would dwarf the search it serves. Epoch stamping makes
/// the reset a single increment.
class VisitedSet {
public:
  void resize(std::size_t n) {
    marks_.assign(n, 0);
    epoch_ = 0;
  }

  void begin_query() {
    ++epoch_;
    if (epoch_ == 0) {
      // Wrapped after 4 billion queries. Every stale mark would now read as
      // "visited" and searches would silently return nothing. Costs one O(N)
      // clear every 2^32 queries, which is never, but the alternative is a bug
      // that only appears after days of continuous load.
      std::fill(marks_.begin(), marks_.end(), 0);
      epoch_ = 1;
    }
  }

  [[nodiscard]] bool test_and_set(VectorId id) {
    auto& mark = marks_[id];
    if (mark == epoch_) {
      return true;
    }
    mark = epoch_;
    return false;
  }

  [[nodiscard]] std::size_t bytes() const { return marks_.size() * sizeof(std::uint32_t); }

private:
  std::vector<std::uint32_t> marks_;
  std::uint32_t epoch_ = 0;
};

class HnswIndexImpl final : public HnswIndex {
public:
  HnswIndexImpl(const VectorStore& store, Metric metric, const HnswConfig& config,
                NeighbourSelection selection, std::unique_ptr<DistanceComputer> computer,
                std::unique_ptr<DistanceComputer> build_computer)
      : store_(store), config_(config), selection_(selection), metric_(metric),
        computer_(std::move(computer)), build_computer_(std::move(build_computer)),
        level_multiplier_(detail::level_multiplier_for(config.m)), rng_state_(config.seed) {
    const std::size_t capacity = store_.size();
    level0_stride_ = config_.m_max0 + 1;
    upper_stride_ = config_.m + 1;

    level0_.assign(capacity * level0_stride_, 0);
    levels_.assign(capacity, 0);
    upper_.resize(capacity);
    inserted_.assign(capacity, 0);
    visited_.resize(capacity);
  }

  Status add(VectorId id) override;

  [[nodiscard]] Status search(const float* query, const SearchParams& params,
                              std::span<Neighbor> out) const override;

  [[nodiscard]] std::size_t last_visited() const override { return last_visited_; }
  [[nodiscard]] std::size_t size() const override { return count_; }
  [[nodiscard]] std::size_t max_level() const override { return max_level_; }
  [[nodiscard]] VectorId entry_point() const override { return entry_point_; }

  [[nodiscard]] std::size_t level_of(VectorId id) const override { return levels_[id]; }

  [[nodiscard]] std::span<const VectorId> neighbours(VectorId id,
                                                     std::size_t level) const override {
    return neighbours_view(id, level);
  }

  [[nodiscard]] std::size_t graph_bytes() const override {
    std::size_t bytes = level0_.capacity() * sizeof(VectorId);
    bytes += levels_.capacity() * sizeof(std::uint8_t);
    bytes += inserted_.capacity() * sizeof(std::uint8_t);
    bytes += visited_.bytes();
    for (const auto& block : upper_) {
      bytes += block.capacity() * sizeof(VectorId);
    }
    return bytes;
  }

  [[nodiscard]] Status save(const std::filesystem::path& path) const override;

  // Used by load_hnsw_index to restore state without replaying inserts.
  void restore(std::vector<VectorId> level0, std::vector<std::vector<VectorId>> upper,
               std::vector<std::uint8_t> levels, std::size_t count, std::size_t max_level,
               VectorId entry, std::uint64_t rng_state) {
    level0_ = std::move(level0);
    upper_ = std::move(upper);
    levels_ = std::move(levels);
    inserted_.assign(levels_.size(), 1);
    count_ = count;
    max_level_ = max_level;
    entry_point_ = entry;
    rng_state_ = rng_state;
    visited_.resize(levels_.size());
  }

  [[nodiscard]] const HnswConfig& config() const { return config_; }

private:
  // --- neighbour list access -----------------------------------------------

  [[nodiscard]] VectorId* slot(VectorId id, std::size_t level) {
    if (level == 0) {
      return level0_.data() + (static_cast<std::size_t>(id) * level0_stride_);
    }
    return upper_[id].data() + ((level - 1) * upper_stride_);
  }

  [[nodiscard]] const VectorId* slot(VectorId id, std::size_t level) const {
    if (level == 0) {
      return level0_.data() + (static_cast<std::size_t>(id) * level0_stride_);
    }
    return upper_[id].data() + ((level - 1) * upper_stride_);
  }

  /// Slot 0 of every list holds the degree; the neighbours follow it.
  [[nodiscard]] std::span<const VectorId> neighbours_view(VectorId id, std::size_t level) const {
    if (level > levels_[id] || inserted_[id] == 0) {
      return {};
    }
    const VectorId* base = slot(id, level);
    return {base + 1, static_cast<std::size_t>(base[0])};
  }

  void set_neighbours(VectorId id, std::size_t level, std::span<const VectorId> ids) {
    VectorId* base = slot(id, level);
    const auto budget = static_cast<VectorId>(level == 0 ? config_.m_max0 : config_.m);
    const auto n = static_cast<VectorId>(std::min<std::size_t>(ids.size(), budget));
    base[0] = n;
    std::memcpy(base + 1, ids.data(), static_cast<std::size_t>(n) * sizeof(VectorId));
  }

  // --- the algorithms ------------------------------------------------------

  /// Algorithm 2. Results left in `results_`, a max-heap under Closer so
  /// results_.front() is the furthest kept node.
  void search_layer(const DistanceComputer& computer, VectorId entry, float entry_distance,
                    std::size_t ef, std::size_t level) const;

  /// Algorithm 3. The M nearest, nothing cleverer.
  static void select_neighbours_simple(std::vector<Candidate>& candidates, std::size_t m);

  /// Algorithm 4, the one that matters for recall.
  void select_neighbours_heuristic(DistanceComputer& computer, std::vector<Candidate>& candidates,
                                   std::size_t m) const;

  void select_neighbours(DistanceComputer& computer, std::vector<Candidate>& candidates,
                         std::size_t m) const {
    if (selection_ == NeighbourSelection::simple) {
      select_neighbours_simple(candidates, m);
    } else {
      select_neighbours_heuristic(computer, candidates, m);
    }
  }

  /// Greedy descent used by both insert and search: one step per layer with
  /// ef = 1, which is the cheap part of the algorithm and the reason the top
  /// layers exist at all.
  void descend_to(const DistanceComputer& computer, VectorId& current, float& distance,
                  std::size_t from_level, std::size_t to_level) const;

  const VectorStore& store_;
  HnswConfig config_;
  NeighbourSelection selection_;
  Metric metric_;

  /// Query-time computer, and a separate one for construction. Two because a
  /// single computer holds one prepared query, and INSERT interleaves "distance
  /// to the node being inserted" with "distance between two existing nodes" —
  /// sharing one would mean re-preparing the query on every heuristic check.
  std::unique_ptr<DistanceComputer> computer_;
  std::unique_ptr<DistanceComputer> build_computer_;

  double level_multiplier_;
  std::uint64_t rng_state_;

  std::vector<VectorId> level0_;
  std::vector<std::vector<VectorId>> upper_;
  std::vector<std::uint8_t> levels_;
  std::vector<std::uint8_t> inserted_;
  std::size_t level0_stride_ = 0;
  std::size_t upper_stride_ = 0;

  std::size_t count_ = 0;
  std::size_t max_level_ = 0;
  VectorId entry_point_ = invalid_id;

  // Per-query scratch. Mutable because search() is const and these carry no
  // observable state between calls — see the thread-safety note in the header.
  mutable VisitedSet visited_;
  mutable std::vector<Candidate> candidates_;
  mutable std::vector<Candidate> results_;
  mutable std::vector<VectorId> batch_ids_;
  mutable std::vector<float> batch_distances_;
  mutable std::vector<Candidate> working_;
  mutable std::size_t last_visited_ = 0;
};

void HnswIndexImpl::search_layer(const DistanceComputer& computer, VectorId entry,
                                 float entry_distance, std::size_t ef, std::size_t level) const {
  candidates_.clear();
  results_.clear();

  // Mark the entry visited before anything else. Without this it can be
  // rediscovered as a neighbour later and pushed into the result heap a second
  // time — a duplicate id in the top-k, silently costing one result slot and
  // therefore recall. Nothing crashes; the number just comes out low.
  (void)visited_.test_and_set(entry);

  candidates_.push_back({entry_distance, entry});
  results_.push_back({entry_distance, entry});
  std::push_heap(candidates_.begin(), candidates_.end(), Farther{});
  std::push_heap(results_.begin(), results_.end(), Closer{});
  ++last_visited_;

  while (!candidates_.empty()) {
    const Candidate nearest = candidates_.front();

    // The paper's termination condition. Once the closest thing left to explore
    // is further than the worst thing already kept, nothing reachable through
    // it can improve the result — every step from here moves away.
    if (!results_.empty() && nearest.distance > results_.front().distance &&
        results_.size() >= ef) {
      break;
    }

    std::pop_heap(candidates_.begin(), candidates_.end(), Farther{});
    candidates_.pop_back();

    const auto links = neighbours_view(nearest.id, level);
    if (links.empty()) {
      continue;
    }

    // Collect the unvisited neighbours first, then take their distances in ONE
    // batched call. Phase 2 established that batching independent distances is
    // where the instruction-level parallelism comes from (D22) — the tail of
    // one distance overlaps the head of the next. This is that mechanism's
    // first consumer at graph-sized batches.
    batch_ids_.clear();
    for (const VectorId neighbour : links) {
      if (!visited_.test_and_set(neighbour)) {
        batch_ids_.push_back(neighbour);
      }
    }
    if (batch_ids_.empty()) {
      continue;
    }

    batch_distances_.resize(batch_ids_.size());
    computer.distances_to(batch_ids_, batch_distances_);
    last_visited_ += batch_ids_.size();

    for (std::size_t i = 0; i < batch_ids_.size(); ++i) {
      const Candidate found{batch_distances_[i], batch_ids_[i]};
      const bool room = results_.size() < ef;
      if (!room && !Closer{}(found, results_.front())) {
        continue;
      }

      candidates_.push_back(found);
      std::push_heap(candidates_.begin(), candidates_.end(), Farther{});

      results_.push_back(found);
      std::push_heap(results_.begin(), results_.end(), Closer{});
      if (results_.size() > ef) {
        std::pop_heap(results_.begin(), results_.end(), Closer{});
        results_.pop_back();
      }
    }
  }
}

void HnswIndexImpl::select_neighbours_simple(std::vector<Candidate>& candidates, std::size_t m) {
  std::sort(candidates.begin(), candidates.end(), Closer{});
  if (candidates.size() > m) {
    candidates.resize(m);
  }
}

void HnswIndexImpl::select_neighbours_heuristic(DistanceComputer& computer,
                                                std::vector<Candidate>& candidates,
                                                std::size_t m) const {
  // Algorithm 4. The line that does the work is the `closer_to_query` test.
  //
  // Keeping the M nearest (Algorithm 3) produces a cluster of mutually
  // redundant edges: if a, b and c are all near q and near each other, all
  // three edges point into the same small region and the graph has no way out
  // of it. Recall collapses because greedy search gets stuck in a local
  // neighbourhood it can see perfectly and cannot leave.
  //
  // The heuristic keeps `e` only when e is closer to q than to anything already
  // selected — so each new edge has to open a *direction* not already covered.
  // That is what produces the long-range links that make the graph navigable,
  // and it is the single biggest recall lever in the whole construction.
  std::sort(candidates.begin(), candidates.end(), Closer{});

  working_.clear();
  working_.reserve(candidates.size());

  for (const Candidate& e : candidates) {
    if (working_.size() >= m) {
      break;
    }

    // The query for `computer` is `e`, so distance_to(r) is dist(e, r).
    computer.prepare_query(store_.get(e.id));
    bool closer_to_query = true;
    for (const Candidate& kept : working_) {
      if (computer.distance_to(kept.id) < e.distance) {
        closer_to_query = false;
        break;
      }
    }
    if (closer_to_query) {
      working_.push_back(e);
    }
  }

  candidates = working_;
}

void HnswIndexImpl::descend_to(const DistanceComputer& computer, VectorId& current, float& distance,
                               std::size_t from_level, std::size_t to_level) const {
  for (std::size_t level = from_level; level > to_level; --level) {
    bool moved = true;
    while (moved) {
      moved = false;
      const auto links = neighbours_view(current, level);
      if (links.empty()) {
        break;
      }
      batch_ids_.assign(links.begin(), links.end());
      batch_distances_.resize(batch_ids_.size());
      computer.distances_to(batch_ids_, batch_distances_);
      last_visited_ += batch_ids_.size();

      for (std::size_t i = 0; i < batch_ids_.size(); ++i) {
        if (Closer{}({batch_distances_[i], batch_ids_[i]}, {distance, current})) {
          distance = batch_distances_[i];
          current = batch_ids_[i];
          moved = true;
        }
      }
    }
  }
}

Status HnswIndexImpl::add(VectorId id) {
  if (static_cast<std::size_t>(id) >= levels_.size()) {
    return Status::invalid_argument;
  }
  if (inserted_[id] != 0) {
    return Status::invalid_argument;
  }

  const std::size_t level = detail::assign_level(rng_state_, level_multiplier_);
  levels_[id] = static_cast<std::uint8_t>(std::min<std::size_t>(level, 255));

  if (levels_[id] > 0) {
    upper_[id].assign(static_cast<std::size_t>(levels_[id]) * upper_stride_, 0);
  }

  // First node: it is the entry point and has nothing to link to.
  if (count_ == 0) {
    inserted_[id] = 1;
    entry_point_ = id;
    max_level_ = levels_[id];
    count_ = 1;
    return Status::ok;
  }

  DistanceComputer& computer = *build_computer_;
  computer.prepare_query(store_.get(id));

  visited_.begin_query();
  last_visited_ = 0;

  VectorId current = entry_point_;
  float distance = computer.distance_to(current);
  const std::size_t node_level = levels_[id];

  // Phase 1 of INSERT: greedy descent through the layers above this node's own
  // level, ef = 1. Cheap, and it is the whole reason the hierarchy exists.
  if (max_level_ > node_level) {
    descend_to(computer, current, distance, max_level_, node_level);
  }

  // Phase 2: from this node's level down to 0, search properly and link.
  const std::size_t start = std::min(node_level, max_level_);
  for (std::size_t lc = start + 1; lc-- > 0;) {
    visited_.begin_query();
    search_layer(computer, current, distance, config_.ef_construction, lc);

    std::vector<Candidate> selected(results_.begin(), results_.end());
    const std::size_t budget = (lc == 0) ? config_.m_max0 : config_.m;
    select_neighbours(*computer_, selected, config_.m);

    batch_ids_.clear();
    for (const Candidate& c : selected) {
      batch_ids_.push_back(c.id);
    }
    inserted_[id] = 1; // visible to neighbours_view from here on
    set_neighbours(id, lc, batch_ids_);

    // Bidirectional. An edge that only points one way is invisible to every
    // search arriving from the other side, and the graph quietly becomes a
    // collection of one-way streets.
    for (const Candidate& c : selected) {
      const auto existing = neighbours_view(c.id, lc);
      std::vector<VectorId> updated(existing.begin(), existing.end());
      if (std::find(updated.begin(), updated.end(), id) != updated.end()) {
        continue;
      }
      updated.push_back(id);

      if (updated.size() <= budget) {
        set_neighbours(c.id, lc, updated);
        continue;
      }

      // Over budget: re-select rather than dropping the newest. Truncating
      // arbitrarily would let a node's edge set drift towards whichever
      // neighbours happened to arrive first, which is exactly the clustering
      // the heuristic exists to prevent.
      computer_->prepare_query(store_.get(c.id));
      std::vector<Candidate> pool;
      pool.reserve(updated.size());
      for (const VectorId n : updated) {
        pool.push_back({computer_->distance_to(n), n});
      }
      select_neighbours(*build_computer_, pool, budget);

      std::vector<VectorId> pruned;
      pruned.reserve(pool.size());
      for (const Candidate& p : pool) {
        pruned.push_back(p.id);
      }
      set_neighbours(c.id, lc, pruned);

      // The build computer's query was clobbered by the pruning above; restore.
      computer.prepare_query(store_.get(id));
    }

    if (!results_.empty()) {
      const auto best = std::min_element(results_.begin(), results_.end(), Closer{});
      current = best->id;
      distance = best->distance;
    }
  }

  inserted_[id] = 1;
  ++count_;
  if (node_level > max_level_) {
    max_level_ = node_level;
    entry_point_ = id;
  }
  return Status::ok;
}

Status HnswIndexImpl::search(const float* query, const SearchParams& params,
                             std::span<Neighbor> out) const {
  const std::size_t k = out.size();
  if (k == 0 || query == nullptr) {
    return Status::invalid_argument;
  }
  if (count_ == 0) {
    return Status::invalid_argument;
  }
  if (params.ef < k) {
    // ef bounds the candidate list, so ef < k cannot produce k results. Failing
    // loudly beats returning a short list that a caller then divides by k.
    return Status::invalid_argument;
  }
  if (k > count_) {
    return Status::invalid_argument;
  }

  DistanceComputer& computer = *computer_;
  computer.prepare_query(query);

  visited_.begin_query();
  last_visited_ = 0;

  VectorId current = entry_point_;
  float distance = computer.distance_to(current);
  ++last_visited_;

  // Algorithm 5: greedy descent with ef = 1 through every layer above 0, then
  // one proper search at layer 0 with the caller's ef.
  descend_to(computer, current, distance, max_level_, 0);

  visited_.begin_query();
  search_layer(computer, current, distance, params.ef, 0);

  std::sort(results_.begin(), results_.end(), Closer{});
  const std::size_t n = std::min(k, results_.size());
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = Neighbor{results_[i].id, results_[i].distance};
  }
  for (std::size_t i = n; i < k; ++i) {
    out[i] = Neighbor{invalid_id, std::numeric_limits<float>::max()};
  }
  return n == k ? Status::ok : Status::not_implemented;
}

constexpr std::uint32_t save_magic = 0x4C4F4448; // "LODH"
constexpr std::uint32_t save_version = 1;

Status HnswIndexImpl::save(const std::filesystem::path& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status::io_error;
  }

  const auto put32 = [&out](std::uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };
  const auto put64 = [&out](std::uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };

  put32(save_magic);
  put32(save_version);
  put64(static_cast<std::uint64_t>(store_.dim()));
  put64(static_cast<std::uint64_t>(levels_.size()));
  put64(static_cast<std::uint64_t>(count_));
  put64(static_cast<std::uint64_t>(config_.m));
  put64(static_cast<std::uint64_t>(config_.m_max0));
  put64(static_cast<std::uint64_t>(config_.ef_construction));
  put64(config_.seed);
  put64(rng_state_);
  put64(static_cast<std::uint64_t>(max_level_));
  put32(entry_point_);
  put32(static_cast<std::uint32_t>(metric_));

  out.write(reinterpret_cast<const char*>(levels_.data()),
            static_cast<std::streamsize>(levels_.size()));
  out.write(reinterpret_cast<const char*>(level0_.data()),
            static_cast<std::streamsize>(level0_.size() * sizeof(VectorId)));

  for (std::size_t i = 0; i < upper_.size(); ++i) {
    put64(static_cast<std::uint64_t>(upper_[i].size()));
    if (!upper_[i].empty()) {
      out.write(reinterpret_cast<const char*>(upper_[i].data()),
                static_cast<std::streamsize>(upper_[i].size() * sizeof(VectorId)));
    }
  }

  out.flush();
  return out.good() ? Status::ok : Status::io_error;
}

} // namespace

std::unique_ptr<HnswIndex> make_hnsw_index_with(const VectorStore& store, ComputerFactory factory,
                                                const HnswConfig& config,
                                                NeighbourSelection selection) {
  if (store.size() == 0 || store.dim() == 0) {
    return nullptr;
  }
  if (config.m == 0 || config.m_max0 < config.m) {
    return nullptr;
  }
  if (config.ef_construction < config.m_max0) {
    // A construction candidate list narrower than the edge budget cannot fill
    // it, so the graph would be starved of edges before pruning even ran.
    return nullptr;
  }
  if (!factory) {
    return nullptr;
  }

  auto computer = factory();
  auto build_computer = factory();
  if (computer == nullptr || build_computer == nullptr) {
    return nullptr;
  }
  // The metric is taken from the computer rather than passed alongside it, so
  // an index can never record a metric its own kernel does not implement.
  const Metric metric = computer->metric();

  return std::make_unique<HnswIndexImpl>(store, metric, config, selection, std::move(computer),
                                         std::move(build_computer));
}

std::unique_ptr<HnswIndex> make_hnsw_index(const VectorStore& store, Metric metric,
                                           const HnswConfig& config, NeighbourSelection selection) {
  return make_hnsw_index_with(
      store, [&store, metric] { return make_distance_computer(metric, store); }, config, selection);
}

std::unique_ptr<HnswIndex> load_hnsw_index(const std::filesystem::path& path,
                                           const VectorStore& store, Metric metric) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return nullptr;
  }

  const auto get32 = [&in]() {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };
  const auto get64 = [&in]() {
    std::uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };

  if (get32() != save_magic || get32() != save_version) {
    return nullptr;
  }

  const auto dim = static_cast<std::size_t>(get64());
  const auto capacity = static_cast<std::size_t>(get64());
  const auto count = static_cast<std::size_t>(get64());

  HnswConfig config;
  config.m = static_cast<std::size_t>(get64());
  config.m_max0 = static_cast<std::size_t>(get64());
  config.ef_construction = static_cast<std::size_t>(get64());
  config.seed = get64();
  const std::uint64_t rng_state = get64();
  const auto max_level = static_cast<std::size_t>(get64());
  const auto entry = static_cast<VectorId>(get32());
  const auto saved_metric = static_cast<Metric>(get32());

  if (!in) {
    return nullptr;
  }

  // Checked, not assumed. An index paired with the wrong corpus, or with a
  // different metric, would otherwise surface as mysteriously poor recall and
  // get blamed on the algorithm.
  if (dim != store.dim() || capacity != store.size() || saved_metric != metric) {
    return nullptr;
  }

  std::vector<std::uint8_t> levels(capacity);
  in.read(reinterpret_cast<char*>(levels.data()), static_cast<std::streamsize>(capacity));

  auto index = make_hnsw_index(store, metric, config);
  if (index == nullptr) {
    return nullptr;
  }
  auto* impl = static_cast<HnswIndexImpl*>(index.get());

  std::vector<VectorId> level0(capacity * (config.m_max0 + 1));
  in.read(reinterpret_cast<char*>(level0.data()),
          static_cast<std::streamsize>(level0.size() * sizeof(VectorId)));

  std::vector<std::vector<VectorId>> upper(capacity);
  for (std::size_t i = 0; i < capacity; ++i) {
    const auto n = static_cast<std::size_t>(get64());
    if (!in) {
      return nullptr;
    }
    upper[i].resize(n);
    if (n != 0) {
      in.read(reinterpret_cast<char*>(upper[i].data()),
              static_cast<std::streamsize>(n * sizeof(VectorId)));
    }
  }

  if (!in) {
    return nullptr;
  }

  impl->restore(std::move(level0), std::move(upper), std::move(levels), count, max_level, entry,
                rng_state);
  return index;
}

} // namespace lodestone
