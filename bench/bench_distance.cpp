// Phase 2 — distance kernel microbenchmarks.
//
// Built before any kernel is optimised, on purpose. Optimising before the
// measurement exists is how you come to believe an improvement that was noise.
//
// The design point: **"ns per distance" is two different numbers**, and
// publishing only one of them is how a SIMD speedup gets overstated.
//
//   l1        the store fits in L1d, so the same few vectors are re-read for the
//             whole run. Compute-bound. This is what the kernel can do.
//   stream    the store is 16x L3, walked linearly, so every distance touches a
//             cold line. Memory-bound. This is what predicts brute-force QPS.
//
// A kernel that is 6x faster in `l1` and 3x faster in `stream` is not a
// contradiction — it is the memory wall, and the gap between those two numbers
// is the answer to Phase 2's "explain why the speedup isn't exactly 8x".
//
// Run:
//   ./build/release/bench/bench_distance
//   ./build/release/bench/bench_distance --benchmark_filter='stream.*dim128'
//   ./build/release/bench/bench_distance --benchmark_repetitions=3
//       --benchmark_report_aggregates_only=true

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using lodestone::DistanceComputer;
using lodestone::KernelKind;
using lodestone::Metric;
using lodestone::Status;
using lodestone::VectorId;
using lodestone::VectorStore;

/// Half of Zen 3's 32 KiB L1d, leaving room for the query, the id array and the
/// output buffer so the vectors are not evicted by our own scaffolding.
constexpr std::size_t l1_target_bytes = 16 * 1024;

/// 16x the 16 MiB L3, so a linear walk cannot be served by any cache level and
/// the measurement is genuinely bound by memory.
constexpr std::size_t stream_target_bytes = 256UL * 1024 * 1024;

/// Matches `scan_block` in brute_force.cpp, so the batch size the kernel sees
/// here is the batch size it will see in the real scan.
constexpr std::size_t scan_batch = 256;

constexpr std::size_t round_up_16(std::size_t n) { return ((n + 15) / 16) * 16; }

enum class Residency : std::uint8_t { l1, stream };

/// Cached across benchmark entries: Google Benchmark re-enters each function
/// many times, and building a 256 MiB store costs far more than the
/// measurement it supports.
const VectorStore& shared_store(std::size_t dim, Residency residency) {
  static std::map<std::pair<std::size_t, std::uint8_t>, VectorStore> cache;

  auto& store = cache[{dim, static_cast<std::uint8_t>(residency)}];
  if (store.size() != 0) {
    return store;
  }

  const std::size_t target =
      (residency == Residency::l1) ? l1_target_bytes : stream_target_bytes;
  const std::size_t bytes_each = round_up_16(dim) * sizeof(float);
  const std::size_t count = std::max<std::size_t>(1, target / bytes_each);

  if (store.reserve(dim, count) != Status::ok) {
    std::abort();
  }

  // Values in 0..255, matching SIFT's magnitudes: real SIFT vectors are uint8
  // histograms widened to float. Keeping the magnitude realistic keeps the
  // arithmetic realistic, and keeps denormals — which would distort timings
  // badly — out of the picture entirely.
  //
  // Only a 1024-vector tile is genuinely random; it is then repeated to fill.
  // Legitimate here because distance cost does not depend on the *values*, only
  // on the dimension and the memory traffic, and both of those are unchanged by
  // repetition. It turns a multi-second fill into a negligible one.
  const std::size_t tile_count = std::min<std::size_t>(1024, count);
  std::vector<float> tile(tile_count * dim);
  std::mt19937 rng(20260827);
  std::uniform_real_distribution<float> values(0.0F, 255.0F);
  for (auto& x : tile) {
    x = values(rng);
  }

  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t t = i % tile_count;
    const auto row = std::span<const float>(tile).subspan(t * dim, dim);
    if (!store.add(row).has_value()) {
      std::abort();
    }
  }

  return store;
}

void run(benchmark::State& state, Metric metric, KernelKind kernel, std::size_t dim,
         Residency residency) {
  const VectorStore& store = shared_store(dim, residency);

  auto computer = lodestone::make_distance_computer(metric, store, kernel);
  if (computer == nullptr) {
    state.SkipWithError("kernel unavailable on this build or CPU");
    return;
  }

  std::vector<float> query(dim);
  {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> values(0.0F, 255.0F);
    for (auto& x : query) {
      x = values(rng);
    }
  }
  computer->prepare_query(query.data());

  const std::size_t count = store.size();
  const std::size_t batch = std::min(scan_batch, count);

  // Ids are built once, outside the timed region. brute_force.cpp fills them
  // per batch, but that is one integer store per distance which does not shrink
  // when the kernel gets 8x faster — leaving it in would dilute exactly the
  // speedup this benchmark exists to measure.
  std::vector<VectorId> ids(count);
  for (std::size_t i = 0; i < count; ++i) {
    ids[i] = static_cast<VectorId>(i);
  }
  std::vector<float> out(batch);

  for (auto unused : state) {
    benchmark::DoNotOptimize(unused);
    for (std::size_t base = 0; base < count; base += batch) {
      const std::size_t n = std::min(batch, count - base);
      computer->distances_to(std::span<const VectorId>(ids).subspan(base, n),
                             std::span<float>(out).first(n));
      benchmark::DoNotOptimize(out.data());
    }
    benchmark::ClobberMemory();
  }

  const auto per_iteration = static_cast<std::int64_t>(count);
  state.SetItemsProcessed(state.iterations() * per_iteration);

  // Payload bytes only — dim floats, not stride floats — so the reported
  // bandwidth is comparable across dimensions and against the brute-force
  // figure in BENCHMARKS.md.
  state.SetBytesProcessed(state.iterations() * per_iteration *
                          static_cast<std::int64_t>(dim * sizeof(float)));

  // Seconds per distance, which Google Benchmark prints with an SI prefix — so
  // "12.3n" reads as 12.3 ns per distance. items_per_second carries the same
  // information; this is here because ns/distance is what PRD Phase 2 asks to
  // record and deriving it by hand invites arithmetic slips.
  state.counters["s/dist"] =
      benchmark::Counter(static_cast<double>(count),
                         benchmark::Counter::kIsIterationInvariantRate |
                             benchmark::Counter::kInvert);
}

std::string_view metric_name(Metric metric) {
  return metric == Metric::l2 ? "l2" : "ip";
}

std::string_view residency_name(Residency residency) {
  return residency == Residency::l1 ? "l1" : "stream";
}

/// Only kernels that can actually be built get registered, so the table grows
/// on its own as tasks 3 and 4 land, and a missing kernel is an absent row
/// rather than a misleading one.
bool kernel_available(KernelKind kernel) {
  VectorStore probe;
  if (probe.reserve(16, 1) != Status::ok) {
    return false;
  }
  return lodestone::make_distance_computer(Metric::l2, probe, kernel) != nullptr;
}

void register_all() {
  for (const KernelKind kernel :
       {KernelKind::scalar, KernelKind::sse, KernelKind::avx2}) {
    if (!kernel_available(kernel)) {
      continue;
    }
    for (const Metric metric : {Metric::l2, Metric::inner_product}) {
      for (const Residency residency : {Residency::l1, Residency::stream}) {
        // 128 is SIFT. 960 is GIST's dimension, on synthetic data — this is
        // ns/distance, not recall, so no GIST corpus is needed. Flagged in the
        // custom context so a row is never mistaken for a GIST result.
        for (const std::size_t dim : {std::size_t{128}, std::size_t{960}}) {
          std::string name;
          name += lodestone::kernel_name(kernel);
          name += "/";
          name += metric_name(metric);
          name += "/";
          name += residency_name(residency);
          name += "/dim";
          name += std::to_string(dim);

          benchmark::RegisterBenchmark(name, [=](benchmark::State& state) {
            run(state, metric, kernel, dim, residency);
          });
        }
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  register_all();

  benchmark::AddCustomContext("detected_kernel",
                              std::string(lodestone::kernel_name(
                                  lodestone::detected_kernel())));
  benchmark::AddCustomContext("l1_fixture", "store ~16 KiB, compute-bound");
  benchmark::AddCustomContext("stream_fixture", "store 256 MiB (16x L3), memory-bound");
  benchmark::AddCustomContext("dim960_data", "synthetic — GIST dimension, not GIST data");
  benchmark::AddCustomContext("bytes_reported", "dim floats per distance, excluding stride padding");

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
