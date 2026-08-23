// Parser tests. Every file here is synthetic and written to a temp directory
// inside the test, so the suite needs no downloaded data and runs in CI.
//
// The bias of these tests is deliberate: most of them feed the parser a *bad*
// file. A parser that only handles good input is the exact failure this phase
// is built to prevent, because a misread .fvecs still yields plausible 128-dim
// vectors and a recall of 0.98 that looks like an algorithm problem.

#include "lodestone/io.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace lodestone;

namespace {

/// A temp directory that removes itself, so tests leave nothing behind even
/// when they fail. Unique per instance, so a parallel `ctest -j` cannot have
/// two cases collide on the same path.
class TempDir {
public:
  TempDir() {
    static std::atomic<unsigned> counter{0};
    std::random_device rd;
    const auto unique = (static_cast<std::uint64_t>(rd()) << 16U) ^ counter.fetch_add(1);
    path_ = std::filesystem::temp_directory_path() / ("lodestone_test_" + std::to_string(unique));
    std::error_code ec;
    std::filesystem::create_directories(path_, ec);
    REQUIRE_FALSE(ec);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  TempDir(TempDir&&) = delete;
  TempDir& operator=(TempDir&&) = delete;

  /// Write raw bytes to `name` inside the directory and return the path.
  [[nodiscard]] std::filesystem::path write(const std::string& name,
                                            std::span<const char> bytes) const {
    const auto p = path_ / name;
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    REQUIRE(out.good());
    return p;
  }

  [[nodiscard]] std::filesystem::path missing() const { return path_ / "does_not_exist.fvecs"; }

private:
  std::filesystem::path path_;
};

void append_bytes(std::vector<char>& bytes, const void* src, std::size_t n) {
  const auto* p = static_cast<const char*>(src);
  bytes.insert(bytes.end(), p, p + n);
}

/// Encode `flat` as .fvecs / .ivecs bytes: per record, a 4-byte little-endian
/// dimension followed by that many 4-byte elements. `T` is float or int32_t.
template <typename T>
std::vector<char> encode(std::int32_t dim, std::span<const T> flat) {
  const auto d = static_cast<std::size_t>(dim);
  std::vector<char> bytes;
  for (std::size_t offset = 0; offset < flat.size(); offset += d) {
    append_bytes(bytes, &dim, sizeof(dim));
    append_bytes(bytes, flat.data() + offset, d * sizeof(T));
  }
  return bytes;
}

/// `count` records of `dim` floats, each value distinct so that a
/// one-record-off read cannot accidentally produce the right answer.
std::vector<float> float_ramp(std::size_t dim, std::size_t count) {
  std::vector<float> v(dim * count);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>(i) + 0.5F;
  }
  return v;
}

constexpr std::size_t sift_dim = 128;
constexpr std::size_t padded_dim = 100; // stride 112 — the case SIFT hides

} // namespace

TEST_CASE("probe_vecs reports the shape from the header and the file size",
          "[io]") {
  const TempDir dir;

  // 3 records of 4 floats: 3 * (4 + 16) = 60 bytes.
  const auto flat = float_ramp(4, 3);
  const auto path = dir.write("small.fvecs", encode<float>(4, flat));

  VecsInfo info;
  REQUIRE(probe_vecs(path, info) == Status::ok);
  CHECK(info.dim == 4);
  CHECK(info.count == 3);
}

TEST_CASE("fvecs round-trips bit-exactly", "[io]") {
  const TempDir dir;

  for (const std::size_t dim : {padded_dim, sift_dim}) {
    const std::size_t count = 5;
    const auto flat = float_ramp(dim, count);
    const auto path =
        dir.write("rt_" + std::to_string(dim) + ".fvecs",
                  encode<float>(static_cast<std::int32_t>(dim), flat));

    VectorStore store;
    REQUIRE(load_fvecs(path, store) == Status::ok);

    REQUIRE(store.dim() == dim);
    REQUIRE(store.size() == count);
    REQUIRE(store.capacity() == count);

    for (std::size_t i = 0; i < count; ++i) {
      const float* v = store.get(static_cast<VectorId>(i));
      for (std::size_t j = 0; j < dim; ++j) {
        CHECK(v[j] == flat[(i * dim) + j]);
      }
    }
  }
}

TEST_CASE("loading does not disturb the store's zeroed padding", "[io]") {
  // Cross-checks the contract Task 1 established from the other side: the
  // loader must write exactly `dim` floats per vector, never `stride`.
  // Phase 2's tail-free AVX2 kernel depends on this holding after a real load,
  // not merely after a fresh reserve().
  const TempDir dir;
  const auto flat = float_ramp(padded_dim, 4);
  const auto path = dir.write("pad.fvecs",
                              encode<float>(static_cast<std::int32_t>(padded_dim), flat));

  VectorStore store;
  REQUIRE(load_fvecs(path, store) == Status::ok);
  REQUIRE(store.stride() == 112);

  for (std::size_t i = 0; i < store.size(); ++i) {
    const float* v = store.get(static_cast<VectorId>(i));
    for (std::size_t j = padded_dim; j < store.stride(); ++j) {
      CHECK(v[j] == 0.0F);
    }
  }
}

TEST_CASE("a truncated file is rejected", "[io]") {
  // The likeliest real-world corruption: an interrupted download. The record
  // size is fixed at 4 + 4*dim, so a file whose length is not a whole multiple
  // of it cannot be valid, and saying so here is what stops it becoming a
  // mysterious recall number three tasks later.
  const TempDir dir;
  auto bytes = encode<float>(4, float_ramp(4, 3));

  SECTION("cut mid-record") {
    bytes.resize(bytes.size() - 3);
    const auto path = dir.write("cut.fvecs", bytes);
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
    VectorStore store;
    CHECK(load_fvecs(path, store) == Status::io_error);
  }

  SECTION("cut leaving only a partial header") {
    bytes.resize(2);
    const auto path = dir.write("hdr.fvecs", bytes);
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
  }
}

TEST_CASE("a record whose dimension prefix disagrees is rejected", "[io]") {
  // The dimension is repeated on every record, which is redundant — and
  // redundancy is free validation. This file has a valid size and a valid
  // first record, so only a parser that checks *every* prefix catches it.
  const TempDir dir;
  auto bytes = encode<float>(4, float_ramp(4, 4));

  // Record 2 starts at byte 2 * (4 + 16) = 40. Rewrite its prefix to 5.
  const std::int32_t wrong = 5;
  std::memcpy(bytes.data() + 40, &wrong, sizeof(wrong));

  const auto path = dir.write("mismatch.fvecs", bytes);

  // The shape still probes fine — that is the point of the test.
  VecsInfo info;
  REQUIRE(probe_vecs(path, info) == Status::ok);
  CHECK(info.dim == 4);
  CHECK(info.count == 4);

  VectorStore store;
  CHECK(load_fvecs(path, store) == Status::dimension_mismatch);
}

TEST_CASE("malformed headers are rejected", "[io]") {
  const TempDir dir;

  SECTION("dimension zero") {
    std::vector<char> bytes;
    const std::int32_t dim = 0;
    append_bytes(bytes, &dim, sizeof(dim));
    const auto path = dir.write("zero.fvecs", bytes);
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
  }

  SECTION("negative dimension") {
    std::vector<char> bytes;
    const std::int32_t dim = -128;
    append_bytes(bytes, &dim, sizeof(dim));
    append_bytes(bytes, &dim, sizeof(dim));
    const auto path = dir.write("neg.fvecs", bytes);
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
  }

  SECTION("absurd dimension") {
    // A garbage header — a wrong file type, or a read at the wrong offset.
    std::vector<char> bytes;
    const std::int32_t dim = max_vecs_dim + 1;
    append_bytes(bytes, &dim, sizeof(dim));
    const auto path = dir.write("absurd.fvecs", bytes);
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
  }

  SECTION("empty file") {
    const auto path = dir.write("empty.fvecs", std::span<const char>{});
    VecsInfo info;
    CHECK(probe_vecs(path, info) == Status::io_error);
  }

  SECTION("nonexistent path") {
    VecsInfo info;
    CHECK(probe_vecs(dir.missing(), info) == Status::io_error);
    VectorStore store;
    CHECK(load_fvecs(dir.missing(), store) == Status::io_error);
  }
}

TEST_CASE("ivecs round-trips, negatives included", "[io]") {
  // The parser treats payload elements as opaque int32, not as indices, so a
  // negative value is its caller's problem to interpret. Ground-truth id
  // validation belongs in Task 4, against a known corpus size.
  const TempDir dir;
  const std::vector<std::int32_t> flat = {0,
                                          -1,
                                          7,
                                          std::numeric_limits<std::int32_t>::max(),
                                          3,
                                          std::numeric_limits<std::int32_t>::min(),
                                          100,
                                          42};
  const auto path = dir.write("gt.ivecs", encode<std::int32_t>(4, flat));

  IvecsData data;
  REQUIRE(load_ivecs(path, data) == Status::ok);
  REQUIRE(data.dim == 4);
  REQUIRE(data.count == 2);
  REQUIRE(data.data.size() == flat.size());

  for (std::size_t i = 0; i < flat.size(); ++i) {
    CHECK(data.data[i] == flat[i]);
  }
}

TEST_CASE("IvecsData::row addresses the right slice", "[io]") {
  const TempDir dir;
  // Ground-truth shaped: rows of 100, which is also the padded dimension, so
  // an off-by-one in row() would be caught rather than masked by a round dim.
  const std::size_t dim = 100;
  const std::size_t count = 5;
  std::vector<std::int32_t> flat(dim * count);
  for (std::size_t i = 0; i < flat.size(); ++i) {
    flat[i] = static_cast<std::int32_t>(i);
  }
  const auto path =
      dir.write("rows.ivecs", encode<std::int32_t>(static_cast<std::int32_t>(dim), flat));

  IvecsData data;
  REQUIRE(load_ivecs(path, data) == Status::ok);
  REQUIRE(data.count == count);

  for (std::size_t i = 0; i < count; ++i) {
    const auto row = data.row(i);
    REQUIRE(row.size() == dim);
    CHECK(row[0] == static_cast<std::int32_t>(i * dim));
    CHECK(row[dim - 1] == static_cast<std::int32_t>(((i + 1) * dim) - 1));
  }
}

TEST_CASE("ivecs rejects the same malformed files fvecs does", "[io]") {
  const TempDir dir;
  auto bytes = encode<std::int32_t>(4, std::vector<std::int32_t>(12, 1));
  bytes.resize(bytes.size() - 5);
  const auto path = dir.write("bad.ivecs", bytes);

  IvecsData data;
  CHECK(load_ivecs(path, data) == Status::io_error);
}

TEST_CASE("a failed load leaves its output untouched", "[io]") {
  // The caller may reasonably keep using a store that a failed load did not
  // fill. Half-populating it and returning an error is the worst of both.
  const TempDir dir;
  const auto good = dir.write("good.fvecs", encode<float>(4, float_ramp(4, 2)));

  VectorStore store;
  REQUIRE(load_fvecs(good, store) == Status::ok);
  REQUIRE(store.size() == 2);

  CHECK(load_fvecs(dir.missing(), store) == Status::io_error);

  // Still the previously loaded data, not a cleared or half-written store.
  CHECK(store.dim() == 4);
  CHECK(store.size() == 2);
  CHECK(store.get(0)[0] == 0.5F);
}
