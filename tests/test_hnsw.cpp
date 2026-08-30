// HNSW tests.
//
// The structural ones matter as much as the recall ones. A graph with
// one-directional edges, or a node over its degree budget, still *searches* —
// it just quietly returns worse answers, which is the failure mode this whole
// project is built to catch rather than average away.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/hnsw.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace lodestone;

namespace {

VectorStore random_store(std::size_t dim, std::size_t count, std::uint32_t seed) {
  VectorStore store;
  REQUIRE(store.reserve(dim, count) == Status::ok);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  std::vector<float> row(dim);
  for (std::size_t i = 0; i < count; ++i) {
    for (auto& x : row) {
      x = values(rng);
    }
    REQUIRE(store.add(row).has_value());
  }
  return store;
}

std::unique_ptr<HnswIndex> build(const VectorStore& store, const HnswConfig& cfg,
                                 NeighbourSelection sel = NeighbourSelection::heuristic) {
  auto index = make_hnsw_index(store, Metric::l2, cfg, sel);
  REQUIRE(index != nullptr);
  for (std::size_t i = 0; i < store.size(); ++i) {
    REQUIRE(index->add(static_cast<VectorId>(i)) == Status::ok);
  }
  return index;
}

HnswConfig small_config() {
  HnswConfig cfg;
  cfg.m = 8;
  cfg.m_max0 = 16;
  cfg.ef_construction = 64;
  return cfg;
}

} // namespace

TEST_CASE("level assignment is geometric with ratio 1/M", "[hnsw]") {
  // The distribution is the whole reason the hierarchy works: about 1/M of the
  // nodes reach each successive level, so the top is sparse enough to cross the
  // corpus in a few hops and layer 0 is dense enough to be accurate. Get the
  // multiplier wrong and you either build a flat graph (no speedup) or a tower
  // (every descent walks hundreds of empty layers).
  const std::size_t m = 16;
  const double multiplier = detail::level_multiplier_for(m);
  CHECK(multiplier == Catch::Approx(1.0 / std::log(16.0)).epsilon(1e-9));

  std::uint64_t state = 42;
  std::map<std::size_t, std::size_t> histogram;
  const std::size_t draws = 200000;
  for (std::size_t i = 0; i < draws; ++i) {
    ++histogram[detail::assign_level(state, multiplier)];
  }

  // ~15/16 of nodes at level 0, and each level up roughly 1/16 of the previous.
  CHECK(histogram[0] > draws * 90 / 100);
  CHECK(histogram[0] < draws * 96 / 100);

  const double ratio = static_cast<double>(histogram[1]) / static_cast<double>(histogram[0]);
  CHECK(ratio > 0.04);
  CHECK(ratio < 0.10);

  // No runaway tower — ln(0) would have produced one.
  CHECK(histogram.rbegin()->first < 12);
}

TEST_CASE("level assignment is reproducible from the seed", "[hnsw]") {
  // A graph you cannot rebuild identically is a benchmark you cannot repeat.
  const double multiplier = detail::level_multiplier_for(16);
  std::uint64_t a = 12345;
  std::uint64_t b = 12345;
  for (int i = 0; i < 1000; ++i) {
    CHECK(detail::assign_level(a, multiplier) == detail::assign_level(b, multiplier));
  }
}

TEST_CASE("the factory rejects a config that cannot build a graph", "[hnsw]") {
  const auto store = random_store(8, 4, 1);

  HnswConfig zero_m = small_config();
  zero_m.m = 0;
  CHECK(make_hnsw_index(store, Metric::l2, zero_m) == nullptr);

  HnswConfig narrow = small_config();
  narrow.ef_construction = 2; // narrower than the edge budget
  CHECK(make_hnsw_index(store, Metric::l2, narrow) == nullptr);

  const VectorStore empty;
  CHECK(make_hnsw_index(empty, Metric::l2, small_config()) == nullptr);
}

TEST_CASE("edges are overwhelmingly reciprocal", "[hnsw]") {
  // Insertion links both directions, so a *mostly* reciprocal graph is the
  // signature of that having happened. It is deliberately not "always": when a
  // neighbour is already at its degree budget, Algorithm 1 re-selects its edge
  // set, and that can legitimately drop the brand-new back-edge. The paper
  // permits it and hnswlib does the same.
  //
  // So the assertion is on the rate. A rate near 1.0 with a few exceptions is
  // healthy pruning; a rate near 0.5 would mean the back-edges were never added
  // at all, which searches perfectly well from one side and returns nonsense
  // from the other.
  const auto store = random_store(16, 300, 7);
  auto index = build(store, small_config());

  std::size_t checked = 0;
  std::size_t reciprocal = 0;
  for (std::size_t i = 0; i < store.size(); ++i) {
    const auto id = static_cast<VectorId>(i);
    for (std::size_t level = 0; level <= index->level_of(id); ++level) {
      for (const VectorId n : index->neighbours(id, level)) {
        const auto back = index->neighbours(n, level);
        if (std::find(back.begin(), back.end(), id) != back.end()) {
          ++reciprocal;
        }
        ++checked;
      }
    }
  }
  REQUIRE(checked > 1000); // the test is worth nothing if it examined nothing
  const double rate = static_cast<double>(reciprocal) / static_cast<double>(checked);
  CHECK(rate > 0.90);
}

TEST_CASE("no node exceeds its degree budget", "[hnsw]") {
  // m_max0 at layer 0, m above. Exceeding it means the pruning path never ran,
  // and memory grows without bound as the graph fills in.
  const auto cfg = small_config();
  const auto store = random_store(16, 400, 11);
  auto index = build(store, cfg);

  for (std::size_t i = 0; i < store.size(); ++i) {
    const auto id = static_cast<VectorId>(i);
    CHECK(index->neighbours(id, 0).size() <= cfg.m_max0);
    for (std::size_t level = 1; level <= index->level_of(id); ++level) {
      CHECK(index->neighbours(id, level).size() <= cfg.m);
    }
  }
}

TEST_CASE("every node is reachable at layer 0", "[hnsw]") {
  // A disconnected component is unreachable by definition: no query will ever
  // find those vectors, whatever ef is set to. This is the structural property
  // Phase 6 goes on to break deliberately with a filter.
  const auto store = random_store(16, 500, 13);
  auto index = build(store, small_config());

  std::vector<bool> seen(store.size(), false);
  std::vector<VectorId> stack{index->entry_point()};
  seen[index->entry_point()] = true;
  std::size_t reached = 1;

  while (!stack.empty()) {
    const VectorId current = stack.back();
    stack.pop_back();
    for (const VectorId n : index->neighbours(current, 0)) {
      if (!seen[n]) {
        seen[n] = true;
        ++reached;
        stack.push_back(n);
      }
    }
  }

  CHECK(reached == store.size());
}

TEST_CASE("the entry point sits at the top of the hierarchy", "[hnsw]") {
  const auto store = random_store(16, 400, 17);
  auto index = build(store, small_config());

  CHECK(index->level_of(index->entry_point()) == index->max_level());
  for (std::size_t i = 0; i < store.size(); ++i) {
    CHECK(index->level_of(static_cast<VectorId>(i)) <= index->max_level());
  }
}

TEST_CASE("search rejects requests it cannot serve", "[hnsw]") {
  const auto store = random_store(16, 100, 19);
  auto index = build(store, small_config());
  const std::vector<float> query(16, 1.0F);

  SearchParams params;
  params.ef = 64;

  std::vector<Neighbor> none;
  CHECK(index->search(query.data(), params, none) == Status::invalid_argument);

  // ef below k cannot produce k results. Failing loudly beats returning a short
  // list that a caller then divides by k.
  std::vector<Neighbor> ten(10);
  SearchParams narrow;
  narrow.ef = 5;
  CHECK(index->search(query.data(), narrow, ten) == Status::invalid_argument);
}

TEST_CASE("search finds the exact answer on a small corpus", "[hnsw]") {
  // With ef large relative to the corpus, HNSW should agree with brute force
  // outright. If it does not at this size, the graph is broken rather than
  // merely approximate.
  const std::size_t dim = 24;
  const std::size_t count = 600;
  const auto store = random_store(dim, count, 23);
  auto index = build(store, small_config());

  auto exact = make_distance_computer(Metric::l2, store);
  REQUIRE(exact != nullptr);

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);

  SearchParams params;
  params.ef = 200;

  double recall_total = 0.0;
  const std::size_t queries = 40;
  for (std::size_t q = 0; q < queries; ++q) {
    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }

    std::vector<Neighbor> truth(10);
    REQUIRE(brute_force_knn(*exact, query.data(), count, truth) == Status::ok);

    std::vector<Neighbor> got(10);
    REQUIRE(index->search(query.data(), params, got) == Status::ok);

    std::vector<std::int32_t> truth_ids;
    truth_ids.reserve(truth.size());
    for (const auto& n : truth) {
      truth_ids.push_back(static_cast<std::int32_t>(n.id));
    }
    recall_total += recall_at_k(got, truth_ids);
  }

  const double recall = recall_total / static_cast<double>(queries);
  CHECK(recall > 0.99);
}

TEST_CASE("the graph visits far fewer nodes than brute force", "[hnsw]") {
  // The entire justification for the index. If this is not orders of magnitude
  // below the corpus size, the hierarchy is not doing its job and no amount of
  // kernel tuning will save it.
  const auto store = random_store(24, 2000, 29);
  auto index = build(store, small_config());

  const std::vector<float> query(24, 128.0F);
  SearchParams params;
  params.ef = 32;
  std::vector<Neighbor> got(10);
  REQUIRE(index->search(query.data(), params, got) == Status::ok);

  CHECK(index->last_visited() > 0);
  CHECK(index->last_visited() < store.size() / 2);
}

TEST_CASE("larger ef does not lower recall", "[hnsw]") {
  // ef is the recall/throughput dial and it must be monotone, or the curve
  // Phase 4 plots is meaningless.
  const std::size_t dim = 24;
  const std::size_t count = 1500;
  const auto store = random_store(dim, count, 31);
  auto index = build(store, small_config());

  auto exact = make_distance_computer(Metric::l2, store);
  REQUIRE(exact != nullptr);

  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  std::vector<std::vector<float>> queries;
  for (std::size_t q = 0; q < 25; ++q) {
    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }
    queries.push_back(std::move(query));
  }

  double previous = 0.0;
  for (const std::size_t ef : {std::size_t{10}, std::size_t{32}, std::size_t{128}}) {
    SearchParams params;
    params.ef = ef;
    double total = 0.0;
    for (const auto& query : queries) {
      std::vector<Neighbor> truth(10);
      REQUIRE(brute_force_knn(*exact, query.data(), count, truth) == Status::ok);
      std::vector<std::int32_t> truth_ids;
      for (const auto& n : truth) {
        truth_ids.push_back(static_cast<std::int32_t>(n.id));
      }
      std::vector<Neighbor> got(10);
      REQUIRE(index->search(query.data(), params, got) == Status::ok);
      total += recall_at_k(got, truth_ids);
    }
    const double recall = total / static_cast<double>(queries.size());
    CHECK(recall >= previous - 0.02); // monotone within noise
    previous = recall;
  }
  CHECK(previous > 0.95);
}

TEST_CASE("the neighbour heuristic and simple selection both build usable graphs", "[hnsw]") {
  // Both stay in the tree so the recall gap between them can be *measured*
  // rather than asserted — it is the most-asked question about an HNSW
  // implementation. Phase 3 task 7 records the number on SIFT1M; this only
  // checks that neither is broken.
  const std::size_t dim = 24;
  const std::size_t count = 1200;
  const auto store = random_store(dim, count, 37);

  auto exact = make_distance_computer(Metric::l2, store);
  REQUIRE(exact != nullptr);

  std::mt19937 rng(555);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  std::vector<std::vector<float>> queries;
  for (std::size_t q = 0; q < 30; ++q) {
    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }
    queries.push_back(std::move(query));
  }

  for (const auto selection : {NeighbourSelection::heuristic, NeighbourSelection::simple}) {
    auto index = build(store, small_config(), selection);
    SearchParams params;
    params.ef = 64;
    double total = 0.0;
    for (const auto& query : queries) {
      std::vector<Neighbor> truth(10);
      REQUIRE(brute_force_knn(*exact, query.data(), count, truth) == Status::ok);
      std::vector<std::int32_t> truth_ids;
      for (const auto& n : truth) {
        truth_ids.push_back(static_cast<std::int32_t>(n.id));
      }
      std::vector<Neighbor> got(10);
      REQUIRE(index->search(query.data(), params, got) == Status::ok);
      total += recall_at_k(got, truth_ids);
    }
    CHECK(total / static_cast<double>(queries.size()) > 0.90);
  }
}

TEST_CASE("a rebuild with the same seed produces the same graph", "[hnsw]") {
  const auto store = random_store(16, 400, 41);
  auto a = build(store, small_config());
  auto b = build(store, small_config());

  REQUIRE(a->max_level() == b->max_level());
  REQUIRE(a->entry_point() == b->entry_point());
  for (std::size_t i = 0; i < store.size(); ++i) {
    const auto id = static_cast<VectorId>(i);
    REQUIRE(a->level_of(id) == b->level_of(id));
    const auto na = a->neighbours(id, 0);
    const auto nb = b->neighbours(id, 0);
    REQUIRE(na.size() == nb.size());
    CHECK(std::equal(na.begin(), na.end(), nb.begin()));
  }
}

TEST_CASE("a saved index reloads bit-identically", "[hnsw]") {
  // "Round-trips" has to mean identical *answers*, not merely similar recall.
  // A graph that reloads with subtly different edges still searches, still
  // scores plausibly, and is silently not the index that was measured.
  const std::size_t dim = 24;
  const std::size_t count = 800;
  const auto store = random_store(dim, count, 61);
  auto original = build(store, small_config());

  const auto path = std::filesystem::temp_directory_path() /
                    ("lodestone_hnsw_" + std::to_string(std::random_device{}()) + ".idx");
  REQUIRE(original->save(path) == Status::ok);

  auto reloaded = load_hnsw_index(path, store, Metric::l2);
  REQUIRE(reloaded != nullptr);

  CHECK(reloaded->size() == original->size());
  CHECK(reloaded->max_level() == original->max_level());
  CHECK(reloaded->entry_point() == original->entry_point());

  for (std::size_t i = 0; i < count; ++i) {
    const auto id = static_cast<VectorId>(i);
    REQUIRE(reloaded->level_of(id) == original->level_of(id));
    for (std::size_t level = 0; level <= original->level_of(id); ++level) {
      const auto a = original->neighbours(id, level);
      const auto b = reloaded->neighbours(id, level);
      REQUIRE(a.size() == b.size());
      CHECK(std::equal(a.begin(), a.end(), b.begin()));
    }
  }

  // And the observable behaviour, which is what actually matters.
  std::mt19937 rng(7777);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  SearchParams params;
  params.ef = 64;
  for (std::size_t q = 0; q < 20; ++q) {
    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }
    std::vector<Neighbor> a(10);
    std::vector<Neighbor> b(10);
    REQUIRE(original->search(query.data(), params, a) == Status::ok);
    REQUIRE(reloaded->search(query.data(), params, b) == Status::ok);
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].id == b[i].id);
      CHECK(a[i].distance == b[i].distance);
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("loading refuses an index paired with the wrong corpus", "[hnsw]") {
  // Checked rather than assumed. An index reloaded against a different store
  // would otherwise search happily and return ids that mean nothing, which
  // reads as an algorithm problem rather than a plumbing one.
  const auto store = random_store(24, 400, 67);
  auto index = build(store, small_config());

  const auto path = std::filesystem::temp_directory_path() /
                    ("lodestone_hnsw_bad_" + std::to_string(std::random_device{}()) + ".idx");
  REQUIRE(index->save(path) == Status::ok);

  const auto wrong_size = random_store(24, 401, 67);
  CHECK(load_hnsw_index(path, wrong_size, Metric::l2) == nullptr);

  const auto wrong_dim = random_store(32, 400, 67);
  CHECK(load_hnsw_index(path, wrong_dim, Metric::l2) == nullptr);

  // Metric is recorded too: an L2 graph traversed with inner-product distances
  // is a different index entirely.
  CHECK(load_hnsw_index(path, store, Metric::inner_product) == nullptr);

  CHECK(load_hnsw_index(path / "nope", store, Metric::l2) == nullptr);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
