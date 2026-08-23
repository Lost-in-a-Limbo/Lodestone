// Every test here that can run at more than one dimension runs at dim 100 as
// well as dim 128, and that is the whole point.
//
// The stride rule rounds a dimension up to a multiple of 16 floats. SIFT is
// dim 128 and GIST is dim 960 — both already multiples of 16, so both pad to
// nothing. A stride bug is therefore invisible on every dataset this project
// uses. dim 100 (stride 112) is the case that catches it.

#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using namespace lodestone;

namespace {

/// A vector of `n` floats with distinct, exactly-representable values, so a
/// round-trip comparison can use == without any tolerance hand-waving.
std::vector<float> ramp(std::size_t n, float base = 1.0F) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = base + static_cast<float>(i);
  }
  return v;
}

bool is_line_aligned(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % vector_alignment == 0;
}

/// The two dimensions every dimension-sensitive test runs at: the one the
/// datasets use, and the one that exposes padding.
constexpr std::size_t sift_dim = 128;
constexpr std::size_t padded_dim = 100;

} // namespace

TEST_CASE("stride rounds a dimension up to 16 floats", "[vector_store]") {
  // 16 floats is 64 bytes is one cache line. The two dimensions this project
  // actually uses, 128 and 960, are the two rows that need no padding — which
  // is exactly why the others are here.
  struct Case {
    std::size_t dim;
    std::size_t expected_stride;
  };
  const Case cases[] = {
      {1, 16},    // smallest possible: a whole line for one float
      {16, 16},   // exact fit, no padding
      {17, 32},   // one float over, costs a whole line
      {100, 112}, // the case SIFT and GIST both hide
      {128, 128}, // SIFT — exact
      {129, 144}, {960, 960}, // GIST — exact
  };

  for (const auto& c : cases) {
    VectorStore store;
    REQUIRE(store.reserve(c.dim, 4) == Status::ok);
    CHECK(store.dim() == c.dim);
    CHECK(store.stride() == c.expected_stride);
    CHECK(store.stride() % (vector_alignment / sizeof(float)) == 0);
    CHECK(store.stride() >= c.dim);
  }
}

TEST_CASE("every vector starts on a cache line, not just the first",
          "[vector_store]") {
  // Aligning only the base pointer is the easy mistake. It looks correct at
  // dim 128, where the stride happens to preserve alignment for free, and
  // silently misaligns every vector after the first at dim 100.
  for (const std::size_t dim : {padded_dim, sift_dim}) {
    VectorStore store;
    REQUIRE(store.reserve(dim, 8) == Status::ok);

    for (std::size_t i = 0; i < store.capacity(); ++i) {
      CHECK(is_line_aligned(store.get(static_cast<VectorId>(i))));
    }
  }
}

TEST_CASE("the whole allocation is zeroed, padding included", "[vector_store]") {
  // Zeroed padding is a contract, not tidiness: it is what lets Phase 2's AVX2
  // kernel run whole 8-wide iterations across the full stride with no scalar
  // tail loop, because every padding term contributes (0-0)^2 = 0 to squared
  // L2 and 0*0 = 0 to an inner product. tests/test_distance.cpp depends on it.
  VectorStore store;
  REQUIRE(store.reserve(100, 4) == Status::ok);

  SECTION("before anything is added") {
    for (std::size_t i = 0; i < store.capacity(); ++i) {
      const float* v = store.get(static_cast<VectorId>(i));
      for (std::size_t j = 0; j < store.stride(); ++j) {
        CHECK(v[j] == 0.0F);
      }
    }
  }

  SECTION("add() fills the payload and leaves the padding alone") {
    const std::vector<float> src = ramp(100);
    REQUIRE(store.add(src).has_value());

    const float* v = store.get(0);
    for (std::size_t j = 0; j < 100; ++j) {
      CHECK(v[j] == src[j]);
    }
    // The 12 padding floats between dim 100 and stride 112.
    for (std::size_t j = 100; j < store.stride(); ++j) {
      CHECK(v[j] == 0.0F);
    }
  }
}

TEST_CASE("add returns sequential ids and round-trips the payload",
          "[vector_store]") {
  for (const std::size_t dim : {padded_dim, sift_dim}) {
    VectorStore store;
    REQUIRE(store.reserve(dim, 3) == Status::ok);

    const std::vector<float> a = ramp(dim, 1.0F);
    const std::vector<float> b = ramp(dim, 1000.0F);
    const std::vector<float> c = ramp(dim, -500.0F);

    const auto id_a = store.add(a);
    const auto id_b = store.add(b);
    const auto id_c = store.add(c);

    REQUIRE(id_a.has_value());
    REQUIRE(id_b.has_value());
    REQUIRE(id_c.has_value());
    CHECK(*id_a == 0U);
    CHECK(*id_b == 1U);
    CHECK(*id_c == 2U);
    CHECK(store.size() == 3);

    // Bit-exact: the store copies what it is given and must not transform it.
    for (std::size_t j = 0; j < dim; ++j) {
      CHECK(store.get(*id_a)[j] == a[j]);
      CHECK(store.get(*id_b)[j] == b[j]);
      CHECK(store.get(*id_c)[j] == c[j]);
    }
  }
}

TEST_CASE("add rejects a wrong-sized vector", "[vector_store]") {
  VectorStore store;
  REQUIRE(store.reserve(128, 4) == Status::ok);

  const std::vector<float> too_short = ramp(127);
  const std::vector<float> too_long = ramp(129);

  CHECK_FALSE(store.add(too_short).has_value());
  CHECK_FALSE(store.add(too_long).has_value());
  CHECK_FALSE(store.add({}).has_value());

  // A rejected add must not consume a slot.
  CHECK(store.size() == 0);
}

TEST_CASE("add refuses to exceed capacity", "[vector_store]") {
  VectorStore store;
  REQUIRE(store.reserve(4, 2) == Status::ok);

  const std::vector<float> v = ramp(4);
  REQUIRE(store.add(v).has_value());
  REQUIRE(store.add(v).has_value());
  REQUIRE(store.size() == 2);

  CHECK_FALSE(store.add(v).has_value());
  CHECK(store.size() == 2); // unchanged
}

TEST_CASE("reserve validates its arguments", "[vector_store]") {
  VectorStore store;

  CHECK(store.reserve(0, 10) == Status::invalid_argument);
  CHECK(store.reserve(10, 0) == Status::invalid_argument);

  // A rejected reserve leaves the store exactly as it was.
  CHECK(store.dim() == 0);
  CHECK(store.capacity() == 0);
  CHECK(store.bytes() == 0);
}

TEST_CASE("reserve rejects a capacity VectorId cannot address",
          "[vector_store]") {
  // Ids are uint32_t (DECISIONS.md D2), and invalid_id is its maximum. A
  // capacity beyond that would hand out an id that collides with the
  // empty-slot marker in a neighbour list — a corruption that would surface
  // in Phase 3 as mysteriously wrong recall, not as a crash here.
  VectorStore store;
  const auto too_many = static_cast<std::size_t>(std::numeric_limits<VectorId>::max()) + 1;
  CHECK(store.reserve(128, too_many) != Status::ok);
}

TEST_CASE("reserve rejects a size that would overflow", "[vector_store]") {
  VectorStore store;
  // stride * capacity * sizeof(float) must not wrap. If it does, the
  // allocation succeeds at a tiny size and every write runs off the end.
  const auto huge = std::numeric_limits<std::size_t>::max() / 8;
  CHECK(store.reserve(128, huge) != Status::ok);
  CHECK(store.capacity() == 0);
}

TEST_CASE("reserve twice resets the store and frees the old block",
          "[vector_store]") {
  // ASan is on in the debug preset, so a leaked first allocation fails here.
  VectorStore store;
  REQUIRE(store.reserve(4, 2) == Status::ok);
  REQUIRE(store.add(ramp(4)).has_value());
  REQUIRE(store.size() == 1);

  REQUIRE(store.reserve(100, 3) == Status::ok);
  CHECK(store.dim() == 100);
  CHECK(store.stride() == 112);
  CHECK(store.capacity() == 3);
  CHECK(store.size() == 0); // reserve is a reset, not a grow
  CHECK(store.bytes() == 3 * 112 * sizeof(float));

  // And the fresh block is zeroed like any other.
  CHECK(store.get(0)[0] == 0.0F);
}

TEST_CASE("bytes() reports the real allocation, padding included",
          "[vector_store]") {
  VectorStore store;
  REQUIRE(store.reserve(100, 10) == Status::ok);
  // 10 vectors x 112 floats x 4 bytes — the 12 padding floats per vector are
  // memory actually spent, so they must show up in the reported number.
  CHECK(store.bytes() == 4480);
  CHECK(store.bytes() % vector_alignment == 0);
}

TEST_CASE("moving a store transfers the allocation", "[vector_store]") {
  SECTION("move construction") {
    VectorStore source;
    REQUIRE(source.reserve(4, 2) == Status::ok);
    REQUIRE(source.add(ramp(4)).has_value());

    const VectorStore moved(std::move(source));

    CHECK(moved.dim() == 4);
    CHECK(moved.size() == 1);
    CHECK(moved.get(0)[0] == 1.0F);

    // The moved-from store must be empty, not merely unspecified: its
    // destructor is about to run, and a double free is the failure mode.
    CHECK(source.size() == 0);       // NOLINT(bugprone-use-after-move)
    CHECK(source.capacity() == 0);   // NOLINT(bugprone-use-after-move)
    CHECK(source.bytes() == 0);      // NOLINT(bugprone-use-after-move)
  }

  SECTION("move assignment frees the destination's old block") {
    VectorStore source;
    REQUIRE(source.reserve(4, 2) == Status::ok);
    REQUIRE(source.add(ramp(4)).has_value());

    VectorStore dest;
    REQUIRE(dest.reserve(128, 100) == Status::ok); // must not leak

    dest = std::move(source);

    CHECK(dest.dim() == 4);
    CHECK(dest.capacity() == 2);
    CHECK(dest.size() == 1);
    CHECK(dest.get(0)[0] == 1.0F);
  }

  SECTION("self-move-assignment does not destroy the store") {
    VectorStore store;
    REQUIRE(store.reserve(4, 2) == Status::ok);
    REQUIRE(store.add(ramp(4)).has_value());

    auto& alias = store;
    store = std::move(alias); // NOLINT(clang-diagnostic-self-move)

    CHECK(store.size() == 1);
    CHECK(store.dim() == 4);
    CHECK(store.get(0)[0] == 1.0F);
  }
}
