// Brute-force k-NN and the recall metric.
//
// This is the code the entire phase's exit criterion runs through, so the tests
// lean on two things that are easy to get subtly wrong: the ordering of equal
// distances, and what recall does with a truth row longer than k.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace lodestone;

namespace {

/// A dim-1 store holding 0.0, 1.0, ... n-1. Distances from a query are then
/// trivially hand-computable, which is what makes the expectations in these
/// tests checkable by eye rather than by rerunning the code.
VectorStore line_store(std::size_t n) {
  VectorStore store;
  REQUIRE(store.reserve(1, n) == Status::ok);
  for (std::size_t i = 0; i < n; ++i) {
    const std::vector<float> v = {static_cast<float>(i)};
    REQUIRE(store.add(v).has_value());
  }
  return store;
}

std::vector<VectorId> ids_of(std::span<const Neighbor> ns) {
  std::vector<VectorId> out;
  out.reserve(ns.size());
  for (const auto& n : ns) {
    out.push_back(n.id);
  }
  return out;
}

std::vector<Neighbor> neighbors_from(const std::vector<VectorId>& ids) {
  std::vector<Neighbor> out;
  out.reserve(ids.size());
  for (const auto id : ids) {
    out.push_back(Neighbor{id, 0.0F});
  }
  return out;
}

} // namespace

TEST_CASE("brute force finds the exact nearest neighbours", "[brute_force]") {
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 0.0F;
  std::vector<Neighbor> out(3);
  REQUIRE(brute_force_knn(*computer, &query, store.size(), out) == Status::ok);

  CHECK(ids_of(out) == std::vector<VectorId>{0, 1, 2});
  CHECK(out[0].distance == 0.0F); // (0-0)^2
  CHECK(out[1].distance == 1.0F); // (1-0)^2
  CHECK(out[2].distance == 4.0F); // (2-0)^2, squared — not 2
}

TEST_CASE("equal distances order by id, and do so reproducibly", "[brute_force]") {
  // Query at 5.0 on the line: ids 4 and 6 are both at distance 1. Sorting on
  // distance alone would leave their arrangement to whatever the heap did, so
  // a rerun could return a different-but-equally-correct answer. This function
  // is the ground truth everything else is measured against — it has to be
  // reproducible down to the id.
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 5.0F;
  std::vector<Neighbor> first(3);
  REQUIRE(brute_force_knn(*computer, &query, store.size(), first) == Status::ok);

  CHECK(ids_of(first) == std::vector<VectorId>{5, 4, 6});
  CHECK(first[1].distance == first[2].distance);

  // Same inputs, same answer — including the tie arrangement.
  std::vector<Neighbor> second(3);
  REQUIRE(brute_force_knn(*computer, &query, store.size(), second) == Status::ok);
  CHECK(ids_of(second) == ids_of(first));
}

TEST_CASE("results come back sorted nearest-first", "[brute_force]") {
  const auto store = line_store(64);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 31.5F;
  std::vector<Neighbor> out(20);
  REQUIRE(brute_force_knn(*computer, &query, store.size(), out) == Status::ok);

  for (std::size_t i = 1; i < out.size(); ++i) {
    CHECK(out[i - 1].distance <= out[i].distance);
  }
}

TEST_CASE("k of one and k of the whole corpus both work", "[brute_force]") {
  const auto store = line_store(8);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);
  const float query = 3.0F;

  SECTION("k = 1") {
    std::vector<Neighbor> out(1);
    REQUIRE(brute_force_knn(*computer, &query, store.size(), out) == Status::ok);
    CHECK(out[0].id == 3U);
    CHECK(out[0].distance == 0.0F);
  }

  SECTION("k = count returns every id exactly once") {
    std::vector<Neighbor> out(8);
    REQUIRE(brute_force_knn(*computer, &query, store.size(), out) == Status::ok);

    auto ids = ids_of(out);
    std::sort(ids.begin(), ids.end());
    CHECK(ids == std::vector<VectorId>{0, 1, 2, 3, 4, 5, 6, 7});
  }
}

TEST_CASE("brute force rejects impossible requests", "[brute_force]") {
  const auto store = line_store(4);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);
  const float query = 0.0F;

  SECTION("k of zero") {
    std::vector<Neighbor> none;
    CHECK(brute_force_knn(*computer, &query, store.size(), none) == Status::invalid_argument);
  }

  SECTION("empty corpus") {
    std::vector<Neighbor> out(1);
    CHECK(brute_force_knn(*computer, &query, 0, out) == Status::invalid_argument);
  }

  SECTION("k larger than the corpus") {
    // Returning fewer than k while reporting ok would let a caller compute
    // recall against a short result and get a number that looks fine.
    std::vector<Neighbor> out(5);
    CHECK(brute_force_knn(*computer, &query, store.size(), out) == Status::invalid_argument);
  }
}

TEST_CASE("brute force matches an independent full-scan reference", "[brute_force]") {
  // The reference sorts *every* id by the computer's own distance, so this
  // isolates the selection logic — the bounded heap and the block batching —
  // from kernel precision, which Task 3 already covers.
  std::mt19937 rng(9876);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  const std::size_t dim = 128;
  const std::size_t count = 200;

  VectorStore store;
  REQUIRE(store.reserve(dim, count) == Status::ok);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<float> row(dim);
    for (auto& x : row) {
      x = dist(rng);
    }
    REQUIRE(store.add(row).has_value());
  }

  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  std::vector<float> query(dim);
  for (auto& x : query) {
    x = dist(rng);
  }

  // Reference: every distance, sorted by (distance, id) — the same total order
  // brute_force_knn promises.
  computer->prepare_query(query.data());
  std::vector<Neighbor> all(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto id = static_cast<VectorId>(i);
    all[i] = Neighbor{id, computer->distance_to(id)};
  }
  std::sort(all.begin(), all.end(), [](const Neighbor& a, const Neighbor& b) {
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    return a.id < b.id;
  });

  for (const std::size_t k : {std::size_t{1}, std::size_t{10}, std::size_t{99}}) {
    std::vector<Neighbor> got(k);
    REQUIRE(brute_force_knn(*computer, query.data(), count, got) == Status::ok);

    for (std::size_t i = 0; i < k; ++i) {
      CHECK(got[i].id == all[i].id);
      CHECK(got[i].distance == all[i].distance);
    }
  }
}

TEST_CASE("brute force scans past a single batch", "[brute_force]") {
  // The scan is blocked so one virtual call covers many distances. A count
  // that is not a whole multiple of the block size is where an off-by-one in
  // the final partial block would live, and the nearest vector is placed at
  // the very end so a dropped tail cannot pass.
  const std::size_t count = 1000;
  VectorStore store;
  REQUIRE(store.reserve(1, count) == Status::ok);
  for (std::size_t i = 0; i < count; ++i) {
    const std::vector<float> v = {static_cast<float>(count - i)};
    REQUIRE(store.add(v).has_value());
  }

  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 1.0F; // matches the last vector exactly
  std::vector<Neighbor> out(2);
  REQUIRE(brute_force_knn(*computer, &query, count, out) == Status::ok);

  CHECK(out[0].id == static_cast<VectorId>(count - 1));
  CHECK(out[0].distance == 0.0F);
}

TEST_CASE("recall_at_k counts set overlap", "[brute_force]") {
  const std::vector<std::int32_t> truth = {10, 11, 12, 13};

  SECTION("perfect") {
    const auto got = neighbors_from({10, 11, 12, 13});
    CHECK(recall_at_k(got, truth) == 1.0);
  }

  SECTION("half") {
    const auto got = neighbors_from({10, 11, 98, 99});
    CHECK(recall_at_k(got, truth) == 0.5);
  }

  SECTION("none") {
    const auto got = neighbors_from({90, 91, 92, 93});
    CHECK(recall_at_k(got, truth) == 0.0);
  }

  SECTION("order within k is irrelevant") {
    const auto got = neighbors_from({13, 10, 12, 11});
    CHECK(recall_at_k(got, truth) == 1.0);
  }
}

TEST_CASE("recall_at_k reads only the first k of a longer truth row", "[brute_force]") {
  // TEXMEX ground truth holds the top 100 per query, so recall@10 must compare
  // against the first 10 and ignore the rest. Consulting all 100 would score a
  // wrong answer as correct whenever it happened to land in ranks 11-100 —
  // inflating recall, which is the direction that matters.
  std::vector<std::int32_t> truth(100);
  for (std::size_t i = 0; i < truth.size(); ++i) {
    truth[i] = static_cast<std::int32_t>(i);
  }

  SECTION("a true top-10 answer scores 1.0") {
    const auto got = neighbors_from({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    CHECK(recall_at_k(got, truth) == 1.0);
  }

  SECTION("an id from rank 50 does not count") {
    const auto got = neighbors_from({0, 1, 2, 3, 4, 5, 6, 7, 8, 50});
    CHECK(recall_at_k(got, truth) == 0.9);
  }
}

TEST_CASE("recall_at_k_tied forgives a distance tie but not a real miss", "[brute_force]") {
  // The case this exists for, in miniature. On SIFT1M it is caused by the
  // 14,538 byte-identical duplicate vectors in the corpus: when a duplicate
  // lands on the k-th boundary, several id sets are equally correct answers and
  // strict set comparison measures the tie-break convention instead of the
  // search.
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  SECTION("a boundary tie scores 1.0, where strict recall scores 0.5") {
    const float query = 5.0F; // ids 4 and 6 are equidistant
    computer->prepare_query(&query);

    const std::vector<Neighbor> got = {{5, 0.0F}, {4, 1.0F}};
    const std::vector<std::int32_t> truth = {5, 6};

    CHECK(recall_at_k(got, truth) == 0.5);
    CHECK(recall_at_k_tied(*computer, got, truth) == 1.0);
  }

  SECTION("a genuinely worse neighbour is still a miss") {
    const float query = 0.0F;
    computer->prepare_query(&query);

    // Truth is {0,1,2}; we returned 5, which is far outside the boundary.
    const std::vector<Neighbor> got = {{0, 0.0F}, {1, 1.0F}, {5, 25.0F}};
    const std::vector<std::int32_t> truth = {0, 1, 2};

    CHECK(recall_at_k(got, truth) == 2.0 / 3.0);
    CHECK(recall_at_k_tied(*computer, got, truth) == 2.0 / 3.0);
  }

  SECTION("with no ties it agrees with strict recall") {
    const float query = 0.0F;
    computer->prepare_query(&query);

    const std::vector<Neighbor> got = {{0, 0.0F}, {1, 1.0F}, {2, 4.0F}};
    const std::vector<std::int32_t> truth = {0, 1, 2};

    CHECK(recall_at_k(got, truth) == 1.0);
    CHECK(recall_at_k_tied(*computer, got, truth) == 1.0);
  }
}

TEST_CASE("diagnose_recall names what was missed and what replaced it", "[brute_force]") {
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 0.0F;
  computer->prepare_query(&query);

  // Pretend the search returned id 5 where the truth says id 2.
  const std::vector<Neighbor> got = {{0, 0.0F}, {1, 1.0F}, {5, 25.0F}};
  const std::vector<std::int32_t> truth = {0, 1, 2};

  RecallDiagnosis diag;
  REQUIRE(diagnose_recall(*computer, got, truth, diag) == Status::ok);

  CHECK(diag.recall == 2.0 / 3.0);
  CHECK(diag.missed == std::vector<std::int32_t>{2});
  CHECK(diag.extra == std::vector<VectorId>{5});
  CHECK(diag.worst_kept == 25.0F); // id 5
  CHECK(diag.best_missed == 4.0F); // id 2, (2-0)^2

  // A gap of 25 against 4 is not a tie — this is a real miss, and saying so is
  // the whole point of the diagnostic.
  CHECK_FALSE(diag.boundary_tie);
}

TEST_CASE("diagnose_recall recognises a boundary tie", "[brute_force]") {
  // Query at 5.0: ids 4 and 6 are equidistant. A reference generator that
  // broke the tie the other way produces recall < 1.0 through no fault of the
  // search, and this is what tells the difference.
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const float query = 5.0F;
  computer->prepare_query(&query);

  const std::vector<Neighbor> got = {{5, 0.0F}, {4, 1.0F}};
  const std::vector<std::int32_t> truth = {5, 6};

  RecallDiagnosis diag;
  REQUIRE(diagnose_recall(*computer, got, truth, diag) == Status::ok);

  CHECK(diag.recall == 0.5);
  CHECK(diag.missed == std::vector<std::int32_t>{6});
  CHECK(diag.extra == std::vector<VectorId>{4});
  CHECK(diag.worst_kept == diag.best_missed);
  CHECK(diag.boundary_tie);
}

TEST_CASE("diagnose_recall on a perfect result finds nothing to report", "[brute_force]") {
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);
  const float query = 0.0F;
  computer->prepare_query(&query);

  const std::vector<Neighbor> got = {{0, 0.0F}, {1, 1.0F}};
  const std::vector<std::int32_t> truth = {0, 1};

  RecallDiagnosis diag;
  REQUIRE(diagnose_recall(*computer, got, truth, diag) == Status::ok);
  CHECK(diag.recall == 1.0);
  CHECK(diag.missed.empty());
  CHECK(diag.extra.empty());
  CHECK_FALSE(diag.boundary_tie);
}

TEST_CASE("the search and diagnostics work with negative distances", "[brute_force]") {
  // Metric::inner_product negates the dot product so that smaller still means
  // closer, which makes every distance negative. Nothing in the bounded heap or
  // the diagnostics may assume distances are non-negative — `worst_kept` seeded
  // from 0.0F would report a value no neighbour has, and every boundary-tie
  // verdict downstream of it would be wrong.
  const auto store = line_store(10); // dim 1, values 0..9
  auto computer = make_distance_computer(Metric::inner_product, store);
  REQUIRE(computer != nullptr);

  const float query = 1.0F;
  std::vector<Neighbor> out(3);
  REQUIRE(brute_force_knn(*computer, &query, store.size(), out) == Status::ok);

  // Largest dot product is the largest value, id 9, and negation puts it first.
  CHECK(out[0].id == 9U);
  CHECK(out[0].distance == -9.0F);
  CHECK(out[1].id == 8U);
  CHECK(out[2].id == 7U);
  for (std::size_t i = 1; i < out.size(); ++i) {
    CHECK(out[i - 1].distance <= out[i].distance);
  }

  computer->prepare_query(&query);
  const std::vector<std::int32_t> truth = {9, 8, 6};

  RecallDiagnosis diag;
  REQUIRE(diagnose_recall(*computer, out, truth, diag) == Status::ok);
  CHECK(diag.recall == 2.0 / 3.0);
  CHECK(diag.worst_kept == -7.0F);  // the worst we kept, id 7 — not 0.0F
  CHECK(diag.best_missed == -6.0F); // id 6
  CHECK_FALSE(diag.boundary_tie);
}

TEST_CASE("ground-truth ids are checked against the corpus size", "[brute_force]") {
  // A ground-truth file paired with the wrong base file otherwise shows up as
  // a low recall number that gets blamed on the algorithm.
  SECTION("in range") {
    const std::vector<std::int32_t> ids = {0, 5, 9};
    CHECK(validate_ground_truth_ids(ids, 10) == Status::ok);
  }

  SECTION("one past the end") {
    const std::vector<std::int32_t> ids = {0, 5, 10};
    CHECK(validate_ground_truth_ids(ids, 10) == Status::invalid_argument);
  }

  SECTION("negative") {
    const std::vector<std::int32_t> ids = {0, -1, 5};
    CHECK(validate_ground_truth_ids(ids, 10) == Status::invalid_argument);
  }
}

TEST_CASE("ground-truth row order is checked against our own metric", "[brute_force]") {
  const auto store = line_store(10);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);
  const float query = 0.0F;
  computer->prepare_query(&query);

  SECTION("ascending under our kernel") {
    const std::vector<std::int32_t> row = {0, 1, 2, 3};
    CHECK(validate_ground_truth_order(*computer, row) == Status::ok);
  }

  SECTION("shuffled") {
    // If the file's order disagrees with our metric, our metric disagrees with
    // the one the file was generated under — a squared-versus-plain L2 slip,
    // say. Better caught here than as an unexplained 0.97.
    const std::vector<std::int32_t> row = {0, 3, 1, 2};
    CHECK(validate_ground_truth_order(*computer, row) == Status::invalid_argument);
  }

  SECTION("equal adjacent distances are fine") {
    const float centre = 5.0F;
    computer->prepare_query(&centre);
    const std::vector<std::int32_t> row = {5, 4, 6};
    CHECK(validate_ground_truth_order(*computer, row) == Status::ok);
  }
}
