// Distance kernel tests.
//
// Two of these are architectural rather than numerical, and they matter more
// than the arithmetic: the factory must never hand back a concrete type, and
// the zeroed-padding contract must hold from the *kernel* side, not just from
// the store's side. Phase 2's tail-free AVX2 kernel is built on the second one.

#include "lodestone/detail/prepared_query.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

/// The SIMD kernels this build and this CPU can actually construct. Iterating
/// over it rather than hardcoding one means each new kernel is covered by every
/// cross-kernel test the moment it lands, and a kernel this machine lacks is
/// skipped rather than failing.
std::vector<KernelKind> available_simd_kernels() {
  VectorStore probe;
  REQUIRE(probe.reserve(16, 1) == Status::ok);
  std::vector<KernelKind> out;
  for (const KernelKind k : {KernelKind::sse, KernelKind::avx2}) {
    if (make_distance_computer(Metric::l2, probe, k) != nullptr) {
      out.push_back(k);
    }
  }
  return out;
}

} // namespace

TEST_CASE("the factory hands back the interface, never a concrete kernel", "[distance]") {
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

TEST_CASE("the query buffer is owned, so the caller's may go away", "[distance]") {
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

TEST_CASE("the factory refuses a request it cannot serve", "[distance]") {
  SECTION("an unreserved store has no dimension to compute over") {
    const VectorStore empty;
    CHECK(make_distance_computer(Metric::l2, empty) == nullptr);
  }

  SECTION("a kernel this CPU cannot run") {
    VectorStore store;
    REQUIRE(store.reserve(4, 1) == Status::ok);
    // Whatever the factory does hand back must be runnable. A kernel full of
    // instructions the machine lacks is an illegal-instruction crash, not a
    // wrong number, and a crash inside a benchmark loop is far worse than a
    // nullptr at the factory. So: every kernel it offers must compute.
    for (const KernelKind kind : {KernelKind::scalar, KernelKind::sse, KernelKind::avx2}) {
      auto computer = make_distance_computer(Metric::l2, store, kind);
      if (computer == nullptr) {
        continue; // unavailable here — acceptable
      }
      CHECK(computer->kernel() == kind);
      const std::vector<float> query = {1.0F, 0.0F, 0.0F, 0.0F};
      computer->prepare_query(query.data());
      CHECK(computer->distance_to(0) >= 0.0F);
    }
  }
}

TEST_CASE("every SIMD kernel agrees with scalar across dimensions and metrics", "[distance]") {
  // The real cross-kernel gate. Relative, not absolute: SIFT squared-L2
  // distances run to ~5e4, and reordering a float32 sum of n terms — which is
  // exactly what vectorising does — perturbs it by roughly n*eps*magnitude. At
  // dim 960 that is an expected absolute difference of order 1e-2 on entirely
  // correct code, so the PRD's "within 1e-4" can only sensibly mean relative.
  std::mt19937 rng(31337);
  // 0..255, matching SIFT's magnitudes: real SIFT vectors are uint8 histograms
  // widened to float, and testing at a realistic magnitude is what exposes
  // accumulation error at a realistic scale.
  std::uniform_real_distribution<float> values(0.0F, 255.0F);

  // dim 100 is in the list because its stride is 112: it is the only case here
  // where the kernels run past `dim` into padding, so it is the only one that
  // can catch a kernel that mishandles the tail-free loop.
  for (const std::size_t dim : {std::size_t{100}, std::size_t{128}, std::size_t{960}}) {
    const std::size_t count = 24;
    std::vector<std::vector<float>> rows;
    rows.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      std::vector<float> row(dim);
      for (auto& x : row) {
        x = values(rng);
      }
      rows.push_back(std::move(row));
    }
    const auto store = make_store(dim, rows);

    std::vector<float> query(dim);
    for (auto& x : query) {
      x = values(rng);
    }

    for (const Metric metric : {Metric::l2, Metric::inner_product}) {
      auto reference = make_distance_computer(metric, store, KernelKind::scalar);
      REQUIRE(reference != nullptr);
      reference->prepare_query(query.data());

      for (const KernelKind kind : available_simd_kernels()) {
        auto simd = make_distance_computer(metric, store, kind);
        REQUIRE(simd != nullptr);
        CHECK(simd->kernel() == kind);
        simd->prepare_query(query.data());

        for (std::size_t i = 0; i < count; ++i) {
          const auto id = static_cast<VectorId>(i);
          const double want = static_cast<double>(reference->distance_to(id));
          const double got = static_cast<double>(simd->distance_to(id));
          CHECK(got == Catch::Approx(want).epsilon(1e-5));
        }
      }
    }
  }
}

TEST_CASE("every SIMD kernel ranks identically to scalar, which is the property "
          "that matters",
          "[distance]") {
  // Distance agreement is the weak test. Two kernels can agree to 1e-5 and
  // still return different neighbours, because a reordering only needs the gap
  // between two adjacent ranks to be smaller than the error — the same float32
  // effect Phase 1 met on SIFT1M. So compare the induced ordering directly.
  std::mt19937 rng(777);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);

  const std::size_t dim = 128;
  const std::size_t count = 400;
  std::vector<std::vector<float>> rows;
  rows.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<float> row(dim);
    for (auto& x : row) {
      x = values(rng);
    }
    rows.push_back(std::move(row));
  }
  const auto store = make_store(dim, rows);

  std::vector<float> query(dim);
  for (auto& x : query) {
    x = values(rng);
  }

  const auto sort_by = [&](KernelKind kind) {
    auto computer = make_distance_computer(Metric::l2, store, kind);
    REQUIRE(computer != nullptr);
    computer->prepare_query(query.data());
    std::vector<VectorId> ids(count);
    for (std::size_t i = 0; i < count; ++i) {
      ids[i] = static_cast<VectorId>(i);
    }
    std::sort(ids.begin(), ids.end(), [&](VectorId a, VectorId b) {
      const float da = computer->distance_to(a);
      const float db = computer->distance_to(b);
      return da != db ? da < db : a < b;
    });
    return ids;
  };

  const auto by_reference = sort_by(KernelKind::scalar);
  for (const KernelKind kind : available_simd_kernels()) {
    CHECK(sort_by(kind) == by_reference);
  }
}

TEST_CASE("the prepared query buffer is cache-line aligned", "[distance]") {
  // Not cosmetic. A 32-byte _mm256_load_ps from a 16-byte-aligned base — which
  // is all std::vector<float> guarantees — is undefined behaviour on half its
  // offsets, so Task 4's AVX2 kernel depends on this holding. Checked through
  // the only observable route: that both kernels agree at a dimension whose
  // stride includes padding.
  detail::PreparedQuery query(100, 112);
  CHECK(reinterpret_cast<std::uintptr_t>(query.data()) % 64 == 0);
  CHECK(query.stride() == 112);

  // Padding is zero at construction and stays zero after a set().
  const std::vector<float> payload(100, 3.5F);
  query.set(payload.data());
  for (std::size_t i = 100; i < 112; ++i) {
    CHECK(query.data()[i] == 0.0F);
  }
  CHECK(query.data()[0] == 3.5F);
  CHECK(query.data()[99] == 3.5F);
}

TEST_CASE("inner product is negated so that smaller still means closer", "[distance]") {
  // Larger dot product means *more* similar, which is backwards from L2. Every
  // consumer — the bounded heap in brute_force, Phase 3's candidate queues — is
  // written around "smaller is closer", so the kernel negates rather than
  // asking each of them to flip a comparator per metric.
  const auto store = make_store(4, {
                                       {1.0F, 1.0F, 1.0F, 1.0F}, // dot 4 with the query
                                       {2.0F, 2.0F, 2.0F, 2.0F}, // dot 8  — most similar
                                       {0.0F, 0.0F, 0.0F, 0.0F}, // dot 0  — least similar
                                   });

  auto computer = make_distance_computer(Metric::inner_product, store);
  REQUIRE(computer != nullptr);
  CHECK(computer->metric() == Metric::inner_product);

  const std::vector<float> query = {1.0F, 1.0F, 1.0F, 1.0F};
  computer->prepare_query(query.data());

  CHECK(computer->distance_to(0) == -4.0F);
  CHECK(computer->distance_to(1) == -8.0F);
  CHECK(computer->distance_to(2) == 0.0F);

  // The property that actually matters: the most similar vector has the
  // smallest "distance", so an unmodified nearest-neighbour search finds it.
  CHECK(computer->distance_to(1) < computer->distance_to(0));
  CHECK(computer->distance_to(0) < computer->distance_to(2));
}

TEST_CASE("under inner product a vector's distance to itself is not zero", "[distance]") {
  // -‖x‖², a direct consequence of the negation convention. Asserted so that
  // nobody "fixes" it into 0 and breaks the ordering in the process.
  const auto store = make_store(4, {{1.0F, 2.0F, 3.0F, 4.0F}});
  auto computer = make_distance_computer(Metric::inner_product, store);
  REQUIRE(computer != nullptr);

  computer->prepare_query(store.get(0));
  CHECK(computer->distance_to(0) == -(1.0F + 4.0F + 9.0F + 16.0F));
}

TEST_CASE("inner-product padding contributes nothing either", "[distance]") {
  // The same contract as L2, and it holds for the same reason: a padding term
  // is 0 * 0. Tested separately because a kernel could get L2 right and inner
  // product wrong — squared L2 tolerates garbage padding far less obviously
  // than a product does.
  std::vector<float> a(padded_dim);
  std::vector<float> q(padded_dim);
  for (std::size_t i = 0; i < padded_dim; ++i) {
    a[i] = static_cast<float>(i) * 0.25F;
    q[i] = 0.5F;
  }

  const auto store = make_store(padded_dim, {a});
  REQUIRE(store.stride() == 112);

  auto computer = make_distance_computer(Metric::inner_product, store);
  REQUIRE(computer != nullptr);
  computer->prepare_query(q.data());

  double dot = 0.0;
  for (std::size_t i = 0; i < padded_dim; ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(q[i]);
  }
  CHECK(static_cast<double>(computer->distance_to(0)) == Catch::Approx(-dot).epsilon(1e-5));
}

TEST_CASE("an explicit kernel request is honoured, not downgraded", "[distance]") {
  VectorStore store;
  REQUIRE(store.reserve(4, 1) == Status::ok);

  for (const Metric m : {Metric::l2, Metric::inner_product}) {
    auto computer = make_distance_computer(m, store, KernelKind::scalar);
    REQUIRE(computer != nullptr);
    CHECK(computer->kernel() == KernelKind::scalar);
    CHECK(computer->metric() == m);
  }
}

TEST_CASE("detected_kernel resolves to something concrete", "[distance]") {
  // `automatic` is a request, never an answer. If this ever returns automatic,
  // make_distance_computer would recurse or return nullptr for every caller.
  CHECK(detected_kernel() != KernelKind::automatic);

  VectorStore store;
  REQUIRE(store.reserve(4, 1) == Status::ok);

  auto automatic = make_distance_computer(Metric::l2, store, KernelKind::automatic);
  REQUIRE(automatic != nullptr);
  CHECK(automatic->kernel() == detected_kernel());

  // Whatever detection claims is available must actually be constructible.
  // A dispatch table that selects a kernel this machine cannot run is an
  // illegal-instruction crash, not a wrong number.
  auto explicit_same = make_distance_computer(Metric::l2, store, detected_kernel());
  REQUIRE(explicit_same != nullptr);
  CHECK(explicit_same->kernel() == detected_kernel());
}

TEST_CASE("dispatch picks the widest kernel the CPU can run", "[distance]") {
  VectorStore store;
  REQUIRE(store.reserve(4, 1) == Status::ok);

  const KernelKind picked = detected_kernel();
  CHECK(picked != KernelKind::automatic);

  // Whatever detection picks must be the widest thing available. Silently
  // selecting scalar on a machine with AVX2 would cost 10x and show up as
  // nothing worse than a disappointing benchmark.
  const bool avx2_available =
      make_distance_computer(Metric::l2, store, KernelKind::avx2) != nullptr;
  const bool sse_available = make_distance_computer(Metric::l2, store, KernelKind::sse) != nullptr;

  if (avx2_available) {
    CHECK(picked == KernelKind::avx2);
  } else if (sse_available) {
    CHECK(picked == KernelKind::sse);
  } else {
    CHECK(picked == KernelKind::scalar);
  }

  // And `automatic` must actually deliver it.
  auto computer = make_distance_computer(Metric::l2, store, KernelKind::automatic);
  REQUIRE(computer != nullptr);
  CHECK(computer->kernel() == picked);
}

TEST_CASE("dispatch never selects a kernel it cannot construct", "[distance]") {
  // The failure this guards is not a wrong number, it is an
  // illegal-instruction crash: a dispatch table that names a kernel the machine
  // cannot execute takes the process down at the first distance.
  VectorStore store;
  REQUIRE(store.reserve(128, 2) == Status::ok);
  const std::vector<float> row(128, 1.0F);
  REQUIRE(store.add(row).has_value());
  REQUIRE(store.add(row).has_value());

  for (const Metric metric : {Metric::l2, Metric::inner_product}) {
    auto computer = make_distance_computer(metric, store, detected_kernel());
    REQUIRE(computer != nullptr);
    computer->prepare_query(row.data());
    // Executes the kernel's actual instruction mix, which is the only way to
    // find out that it runs here.
    CHECK(computer->distance_to(0) == computer->distance_to(1));
  }
}

TEST_CASE("every KernelKind has a name", "[distance]") {
  // These strings end up in results.json and benchmark output, so a missing
  // one is a hole in the record of what was measured.
  for (const KernelKind k :
       {KernelKind::automatic, KernelKind::scalar, KernelKind::sse, KernelKind::avx2}) {
    CHECK_FALSE(kernel_name(k).empty());
    CHECK(kernel_name(k) != "unknown");
  }
}
