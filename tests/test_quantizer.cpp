// Product quantization tests.
//
// The interesting assertions are not "the numbers are close". They are that
// error *falls monotonically as m rises* — the accuracy/memory frontier this
// phase exists to measure — and that PQ reaches the graph through the same
// `DistanceComputer` seam every other kernel uses.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/hnsw.hpp"
#include "lodestone/quantizer.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace lodestone;

namespace {

/// Vectors drawn from a small number of well-separated clusters, so a correct
/// k-means has something findable to find and reconstruction error should be
/// near zero once there are more centroids than clusters.
VectorStore clustered_store(std::size_t dim, std::size_t count, std::size_t clusters,
                            std::uint32_t seed) {
  VectorStore store;
  REQUIRE(store.reserve(dim, count) == Status::ok);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> centre_dist(0.0F, 255.0F);
  std::normal_distribution<float> jitter(0.0F, 1.0F);

  std::vector<std::vector<float>> centres;
  for (std::size_t c = 0; c < clusters; ++c) {
    std::vector<float> centre(dim);
    for (auto& x : centre) {
      x = centre_dist(rng);
    }
    centres.push_back(std::move(centre));
  }

  std::vector<float> row(dim);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& centre = centres[i % clusters];
    for (std::size_t d = 0; d < dim; ++d) {
      row[d] = centre[d] + jitter(rng);
    }
    REQUIRE(store.add(row).has_value());
  }
  return store;
}

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

} // namespace

TEST_CASE("training rejects a config that cannot work", "[quantizer]") {
  const auto store = random_store(32, 400, 1);

  PqConfig bad_subspaces;
  bad_subspaces.subspaces = 7; // does not divide 32
  CHECK(train_product_quantizer(store, bad_subspaces) == nullptr);

  PqConfig zero;
  zero.subspaces = 0;
  CHECK(train_product_quantizer(store, zero) == nullptr);

  // Fewer training points than centroids guarantees empty clusters, which cost
  // the same 8 bits while representing less.
  const auto tiny = random_store(32, 100, 2);
  PqConfig fine;
  fine.subspaces = 4;
  CHECK(train_product_quantizer(tiny, fine) == nullptr);

  const VectorStore empty;
  CHECK(train_product_quantizer(empty, fine) == nullptr);
}

TEST_CASE("codes are one byte per subspace and the arithmetic holds", "[quantizer]") {
  const std::size_t dim = 32;
  const auto store = clustered_store(dim, 1000, 20, 3);

  PqConfig config;
  config.subspaces = 8;
  config.iterations = 8;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);
  REQUIRE(pq->encode(store) == Status::ok);

  CHECK(pq->subspaces() == 8);
  CHECK(pq->sub_dim() == dim / 8);
  CHECK(pq->code_bytes_per_vector() == 8);
  CHECK(pq->size() == store.size());

  // The whole point: 32 floats become 8 bytes.
  CHECK(pq->code_bytes() == store.size() * 8);
  CHECK(pq->codebook_bytes() == 8 * 256 * (dim / 8) * sizeof(float));

  for (std::size_t i = 0; i < store.size(); ++i) {
    CHECK(pq->code(static_cast<VectorId>(i)).size() == 8);
  }
}

TEST_CASE("reconstruction error falls as subspaces rise", "[quantizer]") {
  // The accuracy/memory frontier, which is the phase's actual deliverable.
  // More subspaces means fewer dimensions per centroid to represent, so each
  // code carries more information and the reconstruction is closer.
  const std::size_t dim = 32;
  const auto store = random_store(dim, 2000, 5);

  double previous = std::numeric_limits<double>::max();
  for (const std::size_t m : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
    PqConfig config;
    config.subspaces = m;
    config.iterations = 10;
    auto pq = train_product_quantizer(store, config);
    REQUIRE(pq != nullptr);
    REQUIRE(pq->encode(store) == Status::ok);

    const double error = pq->reconstruction_error(store);
    CHECK(error >= 0.0);
    CHECK(error < previous);
    previous = error;
  }
}

TEST_CASE("well-separated clusters reconstruct almost exactly", "[quantizer]") {
  // With far more centroids than clusters, k-means should place a centre on
  // each cluster and the residual is only the jitter. A large error here means
  // the initialisation collapsed centres into the same region.
  const std::size_t dim = 16;
  const auto store = clustered_store(dim, 1200, 12, 7);

  PqConfig config;
  config.subspaces = 4;
  config.iterations = 15;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);
  REQUIRE(pq->encode(store) == Status::ok);

  // Jitter is unit-variance per dimension, so the irreducible error is about
  // dim * 1.0. Anything near that means the codebook found the clusters.
  const double error = pq->reconstruction_error(store);
  CHECK(error < static_cast<double>(dim) * 4.0);
}

TEST_CASE("the same seed produces the same codebook", "[quantizer]") {
  const auto store = clustered_store(16, 800, 10, 11);
  PqConfig config;
  config.subspaces = 4;
  config.iterations = 6;

  auto a = train_product_quantizer(store, config);
  auto b = train_product_quantizer(store, config);
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  REQUIRE(a->encode(store) == Status::ok);
  REQUIRE(b->encode(store) == Status::ok);

  for (std::size_t i = 0; i < store.size(); ++i) {
    const auto ca = a->code(static_cast<VectorId>(i));
    const auto cb = b->code(static_cast<VectorId>(i));
    REQUIRE(ca.size() == cb.size());
    CHECK(std::equal(ca.begin(), ca.end(), cb.begin()));
  }
  CHECK(a->reconstruction_error(store) == b->reconstruction_error(store));
}

TEST_CASE("ADC approximates exact L2 and improves with more subspaces", "[quantizer]") {
  // Asymmetric: the query is *not* quantized, only the stored vectors are. So
  // the error is the stored side's alone, which is exactly why ADC beats SDC
  // at the same code size.
  const std::size_t dim = 32;
  const auto store = random_store(dim, 1500, 13);

  auto exact = make_distance_computer(Metric::l2, store);
  REQUIRE(exact != nullptr);

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  std::vector<float> query(dim);
  for (auto& x : query) {
    x = values(rng);
  }
  exact->prepare_query(query.data());

  double previous_error = std::numeric_limits<double>::max();
  for (const std::size_t m : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
    PqConfig config;
    config.subspaces = m;
    config.iterations = 10;
    auto pq = train_product_quantizer(store, config);
    REQUIRE(pq != nullptr);
    REQUIRE(pq->encode(store) == Status::ok);

    auto approx = make_pq_distance_computer(*pq);
    REQUIRE(approx != nullptr);
    CHECK(approx->dim() == dim);
    CHECK(approx->metric() == Metric::l2);
    approx->prepare_query(query.data());

    double total_relative = 0.0;
    for (std::size_t i = 0; i < store.size(); ++i) {
      const auto id = static_cast<VectorId>(i);
      const double truth = exact->distance_to(id);
      const double got = approx->distance_to(id);
      CHECK(got >= 0.0); // squared distances are never negative
      total_relative += std::abs(got - truth) / std::max(truth, 1.0);
    }
    const double mean_relative = total_relative / static_cast<double>(store.size());
    CHECK(mean_relative < previous_error);
    previous_error = mean_relative;
  }

  // Even the coarsest setting should be in the right ballpark, not noise.
  CHECK(previous_error < 0.25);
}

TEST_CASE("the batch path agrees with the single path", "[quantizer]") {
  const auto store = random_store(32, 500, 17);
  PqConfig config;
  config.subspaces = 8;
  config.iterations = 6;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);
  REQUIRE(pq->encode(store) == Status::ok);

  auto computer = make_pq_distance_computer(*pq);
  REQUIRE(computer != nullptr);

  std::vector<float> query(32, 100.0F);
  computer->prepare_query(query.data());

  std::vector<VectorId> ids(64);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<VectorId>(i);
  }
  std::vector<float> batch(ids.size(), -1.0F);
  computer->distances_to(ids, batch);

  for (std::size_t i = 0; i < ids.size(); ++i) {
    CHECK(batch[i] == computer->distance_to(ids[i]));
  }
}

TEST_CASE("prepare_query rebuilds the table with no stale state", "[quantizer]") {
  // The failure D1's no-copy rule exists to prevent, in its most literal form:
  // the entire per-query state here *is* a table, and a second query that did
  // not fully overwrite it would return distances computed against the first.
  const auto store = clustered_store(16, 600, 8, 19);
  PqConfig config;
  config.subspaces = 4;
  config.iterations = 8;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);
  REQUIRE(pq->encode(store) == Status::ok);

  auto computer = make_pq_distance_computer(*pq);
  REQUIRE(computer != nullptr);

  const std::vector<float> a(16, 10.0F);
  const std::vector<float> b(16, 200.0F);

  computer->prepare_query(a.data());
  const float first = computer->distance_to(0);
  computer->prepare_query(b.data());
  const float second = computer->distance_to(0);
  computer->prepare_query(a.data());
  const float again = computer->distance_to(0);

  CHECK(first != second);
  CHECK(first == again);
}

TEST_CASE("make_pq_distance_computer refuses what it cannot serve", "[quantizer]") {
  const auto store = random_store(32, 400, 23);
  PqConfig config;
  config.subspaces = 8;
  config.iterations = 4;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);

  // Nothing encoded yet: there are no codes to look up.
  CHECK(make_pq_distance_computer(*pq) == nullptr);

  REQUIRE(pq->encode(store) == Status::ok);
  CHECK(make_pq_distance_computer(*pq) != nullptr);

  // Inner product under PQ needs a table of dot products, not squared
  // distances. Refused rather than silently answering with the wrong metric.
  CHECK(make_pq_distance_computer(*pq, Metric::inner_product) == nullptr);
}

TEST_CASE("encode rejects a corpus of the wrong dimension", "[quantizer]") {
  const auto store = random_store(32, 400, 29);
  PqConfig config;
  config.subspaces = 8;
  config.iterations = 4;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);

  const auto wrong = random_store(64, 400, 31);
  CHECK(pq->encode(wrong) == Status::dimension_mismatch);
}

TEST_CASE("PQ drives HNSW through the same seam as every other kernel", "[quantizer][hnsw]") {
  // The claim DECISIONS.md D1 made in Phase 0, finally exercised: a quantized
  // computer plugs into the graph through DistanceComputer, and the graph's
  // algorithms are untouched.
  const std::size_t dim = 32;
  const std::size_t count = 3000;
  const auto store = random_store(dim, count, 37);

  PqConfig config;
  config.subspaces = 16;
  config.iterations = 10;
  auto pq = train_product_quantizer(store, config);
  REQUIRE(pq != nullptr);
  REQUIRE(pq->encode(store) == Status::ok);

  HnswConfig hnsw;
  hnsw.m = 8;
  hnsw.m_max0 = 16;
  hnsw.ef_construction = 64;

  auto index = make_hnsw_index_with(
      store, [&] { return make_pq_distance_computer(*pq); }, hnsw);
  REQUIRE(index != nullptr);
  for (std::size_t i = 0; i < count; ++i) {
    REQUIRE(index->add(static_cast<VectorId>(i)) == Status::ok);
  }

  // Graded against *exact* ground truth, because the question is what the
  // compression costs end to end, not whether the graph is self-consistent.
  auto exact = make_distance_computer(Metric::l2, store);
  REQUIRE(exact != nullptr);

  std::mt19937 rng(41);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  SearchParams params;
  params.ef = 64;

  double recall_total = 0.0;
  const std::size_t queries = 30;
  for (std::size_t q = 0; q < queries; ++q) {
    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }
    std::vector<Neighbor> truth(10);
    REQUIRE(brute_force_knn(*exact, query.data(), count, truth) == Status::ok);
    std::vector<std::int32_t> truth_ids;
    for (const auto& n : truth) {
      truth_ids.push_back(static_cast<std::int32_t>(n.id));
    }
    std::vector<Neighbor> got(10);
    REQUIRE(index->search(query.data(), params, got) == Status::ok);
    recall_total += recall_at_k(got, truth_ids);
  }

  // Uniform random vectors in 32 dimensions are close to the worst case for
  // PQ — there is no cluster structure for k-means to exploit — so this bound
  // is deliberately loose. The number that matters is measured on real SIFT
  // data and lives in BENCHMARKS.md.
  CHECK(recall_total / static_cast<double>(queries) > 0.30);
}
