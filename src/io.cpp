// Phase 1 — .fvecs / .ivecs readers. Format notes live in the header.

#include "lodestone/io.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <new>
#include <vector>

namespace lodestone {

// The format is little-endian on disk, and every read here reinterprets bytes
// as int32/float directly rather than assembling them byte by byte. On a
// big-endian host that silently produces garbage — plausible-looking garbage,
// which is the worst kind. Fail at compile time instead.
static_assert(std::endian::native == std::endian::little,
              "The .fvecs/.ivecs readers assume a little-endian host. Porting to "
              "big-endian means byte-swapping every dimension prefix and every "
              "payload element, not just relaxing this assertion.");

namespace {

/// Bytes per payload element. float and int32_t are both 4, which is why one
/// probe serves both formats.
constexpr std::size_t element_size = 4;
static_assert(sizeof(float) == element_size);
static_assert(sizeof(std::int32_t) == element_size);

/// A read buffer large enough that the per-record `read()` calls below are
/// memcpys out of the stream buffer rather than syscalls. The default filebuf
/// buffer is a few KB, which at two reads per record over a million records is
/// a lot of round trips for no reason.
constexpr std::streamsize stream_buffer_bytes = 1 << 20;

/// Open for binary reading with an enlarged buffer.
///
/// `pubsetbuf` must be called before the file is opened — libstdc++'s filebuf
/// ignores it once I/O has begun — hence the construct-then-open dance instead
/// of the usual one-line constructor.
bool open_buffered(std::ifstream& in, std::vector<char>& buffer,
                   const std::filesystem::path& path) {
  buffer.resize(static_cast<std::size_t>(stream_buffer_bytes));
  in.rdbuf()->pubsetbuf(buffer.data(), stream_buffer_bytes);
  in.open(path, std::ios::binary);
  return in.is_open();
}

/// Read one record's dimension prefix and confirm it matches what the probe
/// found. This is the check that turns a misaligned read into a loud error.
Status read_and_check_prefix(std::ifstream& in, std::size_t expected_dim) {
  std::int32_t d = 0;
  in.read(reinterpret_cast<char*>(&d), sizeof(d));
  if (!in) {
    return Status::io_error;
  }
  if (d <= 0 || static_cast<std::size_t>(d) != expected_dim) {
    return Status::dimension_mismatch;
  }
  return Status::ok;
}

} // namespace

Status probe_vecs(const std::filesystem::path& path, VecsInfo& out) {
  std::error_code ec;
  const auto file_bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    return Status::io_error; // missing, a directory, or unreadable
  }
  if (file_bytes < sizeof(std::int32_t)) {
    return Status::io_error; // empty, or too short to even hold a header
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io_error;
  }

  std::int32_t dim = 0;
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!in) {
    return Status::io_error;
  }
  if (dim <= 0 || dim > max_vecs_dim) {
    return Status::io_error;
  }

  const auto record_bytes = sizeof(std::int32_t) + (static_cast<std::size_t>(dim) * element_size);

  // The only structural invariant the format offers. A length that is not a
  // whole multiple of the record size means truncation — the likeliest
  // real-world corruption, since it is what an interrupted download leaves.
  if (file_bytes % record_bytes != 0) {
    return Status::io_error;
  }

  out.dim = static_cast<std::size_t>(dim);
  out.count = static_cast<std::size_t>(file_bytes / record_bytes);
  return Status::ok;
}

Status load_fvecs(const std::filesystem::path& path, VectorStore& store) {
  VecsInfo info;
  if (const Status s = probe_vecs(path, info); s != Status::ok) {
    return s; // store deliberately untouched
  }

  if (const Status s = store.reserve(info.dim, info.count); s != Status::ok) {
    return s;
  }

  std::ifstream in;
  std::vector<char> buffer;
  if (!open_buffered(in, buffer, path)) {
    return Status::io_error;
  }

  // One record's payload, staged before handing it to the store.
  //
  // This is a second copy of every vector: file -> scratch -> store. It could
  // be avoided by exposing writable uninitialised slots on VectorStore, but
  // that would let a caller leave a vector half-written and would put the
  // padding contract in the caller's hands. At roughly 0.05 s per 512 MB the
  // copy is far below the disk read it accompanies, so it stays until a
  // measured load time says otherwise.
  std::vector<float> scratch(info.dim);

  for (std::size_t i = 0; i < info.count; ++i) {
    if (const Status s = read_and_check_prefix(in, info.dim); s != Status::ok) {
      return s;
    }

    in.read(reinterpret_cast<char*>(scratch.data()),
            static_cast<std::streamsize>(info.dim * element_size));
    if (!in) {
      return Status::io_error;
    }

    // Cannot fail: capacity came from the same probe, and the span is exactly
    // dim floats wide. Checked anyway, because an unchecked [[nodiscard]] here
    // would be the one place a store invariant could break unnoticed.
    if (!store.add(scratch).has_value()) {
      return Status::io_error;
    }
  }

  return Status::ok;
}

Status load_ivecs(const std::filesystem::path& path, IvecsData& out) {
  VecsInfo info;
  if (const Status s = probe_vecs(path, info); s != Status::ok) {
    return s;
  }

  std::vector<std::int32_t> data;
  try {
    data.resize(info.dim * info.count);
  } catch (const std::bad_alloc&) {
    // Loading is not the search hot path, so an allocation here is allowed to
    // throw — but the function's contract is a Status, and letting an
    // exception escape a Status-returning API would make the error channel
    // mean two different things depending on which failure occurred.
    return Status::out_of_memory;
  }

  std::ifstream in;
  std::vector<char> buffer;
  if (!open_buffered(in, buffer, path)) {
    return Status::io_error;
  }

  for (std::size_t i = 0; i < info.count; ++i) {
    if (const Status s = read_and_check_prefix(in, info.dim); s != Status::ok) {
      return s;
    }

    in.read(reinterpret_cast<char*>(data.data() + (i * info.dim)),
            static_cast<std::streamsize>(info.dim * element_size));
    if (!in) {
      return Status::io_error;
    }
  }

  out.data = std::move(data);
  out.dim = info.dim;
  out.count = info.count;
  return Status::ok;
}

} // namespace lodestone
