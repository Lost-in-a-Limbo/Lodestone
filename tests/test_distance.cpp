// Distance kernel tests.
//
// Two of these are architectural rather than numerical, and they matter more
// than the arithmetic: the factory must never hand back a concrete type, and
// the zeroed-padding contract must hold from the *kernel* side, not just from
// the store's side. Phase 2's tail-free AVX2 kernel is built on the second one.

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

using namespace lodestone;

namespace {

constexpr std::size_t sift_dim = 128;
constexpr std::size_t padded_dim = 100; // stride 112

/// Independent reference, in double, over exactly `dim` terms. Deliberately
/// the dumbest possible implementation — if it and the kernel agree, they are
/// unlikely to be wrong in the same way.
double reference_sq_l2(const float* a, const float* b, std::size_t dim) {
  double sum = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

VectorStore make_store(std::size_t dim, const std::vector<std::vector<float>>& rows) {
  VectorStore store;
  REQUIRE(store.reserve(dim, rows.size()) == Status::ok);
  for (const auto& row : rows) {
    REQUIRE(store.add(row).has_value());
  }
  return store;
}

} // namespace

TEST_CASE("the factory hands back the interface, never a concrete kernel",
          "[distance]") {
  // This is architecture rule 1 expressed as a compile-time assertion. If the
  // factory's return type ever narrows to a concrete kernel, callers can name
  // that kernel, and Phase 2's runtime CPU dispatch and Phase 5's quantised
  // distances both stop being drop-in.
  VectorStore store;
  REQUIRE(store.reserve(4, 1) == Status::ok);

  STATIC_REQUIRE(std::is_same_v<decltype(make_distance_computer(Metric::l2, store)),
                                std::unique_ptr<DistanceComputer>>);
}

TEST_CASE("squared L2 is exact, and never takes a square root", "[distance]") {
  // The sqrt is omitted on purpose: it is monotone, so it cannot change a
  // ranking, and it would cost a transcendental per distance on the hottest
  // path in the system. Every "distance" in this project is therefore squared,
  // and this test is what stops someone helpfully adding the sqrt back.
  const auto store = make_store(4, {
                                       {3.0F, 0.0F, 0.0F, 0.0F},
                                       {0.0F, 4.0F, 0.0F, 0.0F},
                                       {1.0F, 2.0F, 3.0F, 4.0F},
                                   });

  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const std::vector<float> origin = {0.0F, 0.0F, 0.0F, 0.0F};
  computer->prepare_query(origin.data());

  CHECK(computer->distance_to(0) == 9.0F);  // 3^2, not 3
  CHECK(computer->distance_to(1) == 16.0F); // 4^2, not 4
  CHECK(computer->distance_to(2) == 30.0F); // 1+4+9+16

  const std::vector<float> query = {3.0F, 4.0F, 0.0F, 0.0F};
  computer->prepare_query(query.data());

  CHECK(computer->distance_to(0) == 16.0F); // (0)^2 + (4)^2
  CHECK(computer->distance_to(1) == 9.0F);  // (3)^2 + (0)^2
  CHECK(computer->distance_to(2) == 4.0F + 4.0F + 9.0F + 16.0F);
}

TEST_CASE("a vector is at zero distance from itself", "[distance]") {
  const auto store = make_store(4, {{1.5F, -2.25F, 3.0F, 0.125F}});
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  computer->prepare_query(store.get(0));
  CHECK(computer->distance_to(0) == 0.0F);
}

TEST_CASE("padding contributes nothing to a distance", "[distance]") {
  // The contract, stated from the kernel side: at dim 100 the store's stride is
  // 112, and a kernel is permitted to process all 112 floats because the store
  // zeroes its padding and prepare_query() zero-pads the query. This test
  // asserts the *result* is unchanged either way, which is the property Phase 2
  // depends on — it deletes the scalar tail loop and the masked final load from
  // the AVX2 kernel, the fiddliest part of writing one by hand.
  std::vector<float> a(padded_dim);
  std::vector<float> q(padded_dim);
  for (std::size_t i = 0; i < padded_dim; ++i) {
    a[i] = static_cast<float>(i) * 0.5F;
    q[i] = static_cast<float>(padded_dim - i) * 0.25F;
  }

  const auto store = make_store(padded_dim, {a});
  REQUIRE(store.stride() == 112);

  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);
  computer->prepare_query(q.data());

  const double over_dim = reference_sq_l2(a.data(), q.data(), padded_dim);

  // The same sum taken stride-wide, against a query padded with zeros exactly
  // the way prepare_query() pads it. Must land on the identical value.
  std::vector<float> q_padded(store.stride(), 0.0F);
  for (std::size_t i = 0; i < padded_dim; ++i) {
    q_padded[i] = q[i];
  }
  const double over_stride = reference_sq_l2(store.get(0), q_padded.data(), store.stride());

  CHECK(over_stride == over_dim);
  CHECK(static_cast<double>(computer->distance_to(0)) == Catch::Approx(over_dim).epsilon(1e-5));
}

TEST_CASE("distances_to agrees with distance_to elementwise", "[distance]") {
  // The batch method exists so that one virtual call covers M distances
  // (DECISIONS.md D1). It must be an optimisation, not a second implementation
  // that can drift from the scalar one.
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  const std::size_t count = 40;
  std::vector<std::vector<float>> rows;
  rows.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<float> row(sift_dim);
    for (auto& x : row) {
      x = dist(rng);
    }
    rows.push_back(std::move(row));
  }

  const auto store = make_store(sift_dim, rows);
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  std::vector<float> query(sift_dim);
  for (auto& x : query) {
    x = dist(rng);
  }
  computer->prepare_query(query.data());

  std::vector<VectorId> ids(count);
  for (std::size_t i = 0; i < count; ++i) {
    ids[i] = static_cast<VectorId>(i);
  }

  std::vector<float> batch(count, -1.0F);
  computer->distances_to(ids, batch);

  for (std::size_t i = 0; i < count; ++i) {
    // Bit-exact: same kernel, same order of operations, so any difference is a
    // real divergence rather than float accumulation noise.
    CHECK(batch[i] == computer->distance_to(ids[i]));

    // And both agree with the independent double reference.
    const double expected = reference_sq_l2(rows[i].data(), query.data(), sift_dim);
    CHECK(static_cast<double>(batch[i]) == Catch::Approx(expected).epsilon(1e-5));
  }
}

TEST_CASE("distances_to handles a partial and an empty batch", "[distance]") {
  const auto store = make_store(4, {
                                       {1.0F, 0.0F, 0.0F, 0.0F},
                                       {0.0F, 1.0F, 0.0F, 0.0F},
                                       {0.0F, 0.0F, 1.0F, 0.0F},
                                   });
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const std::vector<float> origin = {0.0F, 0.0F, 0.0F, 0.0F};
  computer->prepare_query(origin.data());

  // Out of order and with a repeat — the graph will hand this method whatever
  // a neighbour list happens to contain, in whatever order it is stored.
  const std::vector<VectorId> ids = {2, 0, 2};
  std::vector<float> out(ids.size(), -1.0F);
  computer->distances_to(ids, out);
  CHECK(out[0] == 1.0F);
  CHECK(out[1] == 1.0F);
  CHECK(out[2] == 1.0F);

  // An empty batch must be a no-op, not a crash. SEARCH-LAYER will hit this
  // whenever a node has no unvisited neighbours.
  computer->distances_to({}, {});
}

TEST_CASE("prepare_query rebinds without leaving stale state", "[distance]") {
  // The failure mode D1's no-copy rule exists to prevent: a computer holding
  // per-query state that a second query does not fully overwrite. It surfaces
  // as subtly wrong distances, never as a crash, which is the worst possible
  // shape for a bug in a recall benchmark.
  const auto store = make_store(4, {{10.0F, 0.0F, 0.0F, 0.0F}});
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  const std::vector<float> first = {0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> second = {10.0F, 0.0F, 0.0F, 0.0F};

  computer->prepare_query(first.data());
  REQUIRE(computer->distance_to(0) == 100.0F);

  computer->prepare_query(second.data());
  CHECK(computer->distance_to(0) == 0.0F);

  computer->prepare_query(first.data());
  CHECK(computer->distance_to(0) == 100.0F);
}

TEST_CASE("the query buffer is owned, so the caller's may go away",
          "[distance]") {
  // prepare_query() copies into an internal zero-padded buffer rather than
  // stashing the pointer. That is what makes a stride-wide kernel safe — it
  // would otherwise read past the end of the caller's dim-sized array — and it
  // removes a lifetime requirement from the interface.
  const auto store = make_store(4, {{1.0F, 2.0F, 3.0F, 4.0F}});
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  {
    const std::vector<float> transient = {1.0F, 2.0F, 3.0F, 4.0F};
    computer->prepare_query(transient.data());
  } // transient is gone here

  // ASan would report a use-after-scope if the pointer had merely been stashed.
  CHECK(computer->distance_to(0) == 0.0F);
}

TEST_CASE("a computer reports its shape and metric", "[distance]") {
  const auto store = make_store(padded_dim, {std::vector<float>(padded_dim, 1.0F)});
  auto computer = make_distance_computer(Metric::l2, store);
  REQUIRE(computer != nullptr);

  // dim(), not stride() — the dimension is what a caller's query must supply.
  CHECK(computer->dim() == padded_dim);
  CHECK(computer->metric() == Metric::l2);
}

TEST_CASE("the factory refuses what Phase 1 does not implement", "[distance]") {
  VectorStore store;
  REQUIRE(store.reserve(4, 1) == Status::ok);

  SECTION("inner product arrives in Phase 2") {
    CHECK(make_distance_computer(Metric::inner_product, store) == nullptr);
  }

  SECTION("an unreserved store has no dimension to compute over") {
    const VectorStore empty;
    CHECK(make_distance_computer(Metric::l2, empty) == nullptr);
  }
}
