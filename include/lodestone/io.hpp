#pragma once

#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace lodestone {

/// Readers for the TEXMEX `.fvecs` / `.ivecs` format used by SIFT and GIST.
///
/// A record is a 4-byte little-endian `int32` dimension followed by that many
/// 4-byte elements — `float` for `.fvecs`, `int32` for `.ivecs`. Records repeat
/// to end of file with no header and no footer, so the record size
/// `4 + 4*dim` is the only structure the format gives you.
///
/// That dimension prefix is repeated on every record and is therefore
/// redundant, which makes it free validation. These readers check it on every
/// record rather than only the first, because the failure they exist to prevent
/// is silent: a misread file still produces plausible 128-dimensional vectors
/// and a recall figure that looks like an algorithm problem rather than a
/// parsing one.
///
/// One corruption the format cannot detect: a file truncated by a *whole*
/// number of records is indistinguishable from a shorter valid file, since the
/// count is derived from the length. That is why `tools/download_sift.sh`
/// verifies exact file sizes and the upstream MD5s instead of trusting this.

/// Sanity ceiling for a dimension read out of a file header. GIST, the widest
/// dataset this project touches, is 960. A value above this is a garbage
/// header — a wrong file type, or a read at the wrong offset — not a real
/// dimension, and treating it as one would mean sizing an allocation from
/// noise.
inline constexpr std::int32_t max_vecs_dim = 65536;

/// Shape of a `.fvecs` / `.ivecs` file: the dimension from record 0, and the
/// record count derived from the file length.
struct VecsInfo {
  std::size_t dim = 0;
  std::size_t count = 0;
};

/// Read the shape without reading any payload, so a caller can size an
/// allocation before committing to the load.
///
/// Returns `io_error` if the file is missing, empty, shorter than a header, has
/// a non-positive or absurd dimension, or has a length that is not a whole
/// multiple of its record size.
Status probe_vecs(const std::filesystem::path& path, VecsInfo& out);

/// Load a `.fvecs` into `store`, calling `store.reserve()` itself — which is
/// why the store must not grow: Phase 3's graph will hold ids that index into
/// this exact allocation.
///
/// On `io_error` from the initial probe, `store` is untouched. On a failure
/// *during* the read (`dimension_mismatch`, or a short read), `reserve()` has
/// already run, so the store is left reset and partially filled; a caller that
/// ignores the returned status and uses `size()` will see only the records that
/// were read successfully.
Status load_fvecs(const std::filesystem::path& path, VectorStore& store);

/// A loaded `.ivecs`, held flat.
///
/// Flat rather than `vector<vector<int32_t>>` even though ground truth is read
/// once and never touched on the hot path. Consistency is cheaper than an
/// exception to the layout rule that someone later cites as precedent.
struct IvecsData {
  std::vector<std::int32_t> data;
  std::size_t dim = 0;
  std::size_t count = 0;

  /// Row `i`. No bounds check — callers iterate over `count`.
  [[nodiscard]] std::span<const std::int32_t> row(std::size_t i) const {
    return {data.data() + (i * dim), dim};
  }
};

/// Load an `.ivecs`. Payload elements are treated as opaque `int32`, not as
/// indices: validating that ground-truth ids fall inside a corpus is the
/// caller's job, because only the caller knows the corpus size.
Status load_ivecs(const std::filesystem::path& path, IvecsData& out);

} // namespace lodestone
