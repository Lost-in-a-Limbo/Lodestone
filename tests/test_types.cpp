// A test asserting 1 == 1 proves the harness runs. These assert the design
// invariants later phases are built on, so they also prove the harness keeps
// running — and they catch the day someone "tidies up" a type.

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

using namespace lodestone;

TEST_CASE("VectorId stays 4 bytes", "[types]") {
  // Widening this to 64 bits doubles the size of every neighbour list, which
  // is the hottest read in search: at M = 16 a list goes from exactly one
  // cache line to two. If this assertion ever fails, the change was not a
  // cleanup, and the memory numbers in BENCHMARKS.md no longer describe it.
  STATIC_REQUIRE(sizeof(VectorId) == 4);
  STATIC_REQUIRE(std::is_unsigned_v<VectorId>);

  // 1M vectors is the stated working size; leave real headroom above it.
  STATIC_REQUIRE(std::numeric_limits<VectorId>::max() > 100'000'000U);
}

TEST_CASE("invalid_id cannot collide with a real id", "[types]") {
  // invalid_id doubles as the empty-slot marker inside neighbour lists, so it
  // has to be a value no real vector can ever take.
  STATIC_REQUIRE(invalid_id == std::numeric_limits<VectorId>::max());
  REQUIRE(invalid_id > 1'000'000U);
}

TEST_CASE("Status::ok is zero", "[types]") {
  // So that `if (s != Status::ok)` and a zero-initialised Status agree.
  STATIC_REQUIRE(static_cast<std::uint8_t>(Status::ok) == 0);
  STATIC_REQUIRE(sizeof(Status) == 1);
}

TEST_CASE("HnswConfig defaults match the paper", "[types]") {
  const HnswConfig cfg;

  CHECK(cfg.m == 16);

  // Layer 0 holds every node and carries the final, accuracy-critical hops,
  // so it gets twice the edge budget. Malkov & Yashunin, section 4.
  CHECK(cfg.m_max0 == 2 * cfg.m);

  // A construction candidate list narrower than the edge budget cannot fill
  // it, so the graph would be starved of edges before pruning even runs.
  CHECK(cfg.ef_construction >= cfg.m_max0);

  // Fixed, not time-seeded. A build you cannot reproduce is not a measurement.
  CHECK(cfg.seed != 0);
}

TEST_CASE("SearchParams default can serve recall@10", "[types]") {
  const SearchParams params;
  // ef bounds the result list, so ef < k cannot return k neighbours at all.
  // The headline metric is recall@10, so the default must clear 10.
  CHECK(params.ef >= 10);
}

TEST_CASE("the distance seam is a runtime-dispatched interface", "[distance]") {
  // These four assertions *are* the Phase 2 and Phase 5 plan. If
  // DistanceComputer stops being abstract, or loses its virtual destructor,
  // the kernel choice has moved to compile time and runtime CPU dispatch is no
  // longer possible without instantiating the whole index once per kernel.
  STATIC_REQUIRE(std::is_abstract_v<DistanceComputer>);
  STATIC_REQUIRE(std::has_virtual_destructor_v<DistanceComputer>);

  // Implementations own per-query scratch (Phase 5's 256 x m lookup table). A
  // silent copy of a prepared computer would surface as wrong distances, not
  // as a crash — the worst possible failure mode in a recall benchmark.
  STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<DistanceComputer>);
  STATIC_REQUIRE_FALSE(std::is_move_assignable_v<DistanceComputer>);
}

TEST_CASE("vectors align to a cache line", "[vector_store]") {
  STATIC_REQUIRE(vector_alignment == 64);

  // A 128-dim float32 vector must tile the alignment exactly, or every vector
  // after the first straddles a line boundary and AVX2 loses aligned loads.
  STATIC_REQUIRE((128 * sizeof(float)) % vector_alignment == 0);
}

TEST_CASE("VectorStore moves but does not copy", "[vector_store]") {
  // It owns one large aligned allocation, and Phase 3's graph holds ids that
  // index into it. An accidental copy would double peak RSS and quietly
  // invalidate every memory number the project reports.
  STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<VectorStore>);
  STATIC_REQUIRE(std::is_move_constructible_v<VectorStore>);
  STATIC_REQUIRE(std::is_nothrow_move_constructible_v<VectorStore>);

  const VectorStore store;
  CHECK(store.size() == 0);
  CHECK(store.dim() == 0);
  CHECK(store.bytes() == 0);
}
