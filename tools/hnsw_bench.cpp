// Phase 3's proof command.
//
//   hnsw_bench <base.fvecs> <query.fvecs> <groundtruth.ivecs> [--ef=LIST]
//              [--m=N] [--ef-construction=N] [--selection=heuristic|simple|both]
//              [--min-recall=X] [--queries=N] [--save=PATH]
//
// Builds an HNSW index, then sweeps ef and reports the recall/QPS curve the
// phase has to record. Exits non-zero if the best recall@10 falls below
// --min-recall.
//
// Not the benchmark harness — no JSON, no hnswlib, no latency percentiles.
// Those are Phase 4. This exists to produce Phase 3's numbers and to assert the
// exit criterion.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/hnsw.hpp"
#include "lodestone/io.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace lodestone;
using clock_type = std::chrono::steady_clock;

constexpr std::size_t k_neighbors = 10;
constexpr int exit_ok = 0;
constexpr int exit_fail = 1;
constexpr int exit_usage = 2;
constexpr int exit_skip = 4;

double seconds_since(clock_type::time_point t) {
  return std::chrono::duration<double>(clock_type::now() - t).count();
}

std::optional<std::size_t> parse_size(std::string_view text) {
  std::size_t v = 0;
  const auto* end = text.data() + text.size();
  if (std::from_chars(text.data(), end, v).ec != std::errc{}) {
    return std::nullopt;
  }
  return v;
}

std::string mib(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
  return buf;
}

std::optional<std::size_t> peak_rss_bytes() {
  std::ifstream status("/proc/self/status");
  if (!status) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::size_t kib = 0;
      const char* begin = line.data() + 6;
      const char* end = line.data() + line.size();
      while (begin != end && (*begin == ' ' || *begin == '\t')) {
        ++begin;
      }
      if (std::from_chars(begin, end, kib).ec == std::errc{}) {
        return kib * 1024;
      }
    }
  }
  return std::nullopt;
}

struct SweepRow {
  std::size_t ef;
  double recall_strict;
  double recall_tied;
  double qps;
  double mean_visited;
};

void usage() {
  std::fprintf(stderr,
               "usage: hnsw_bench <base.fvecs> <query.fvecs> <groundtruth.ivecs>\n"
               "         [--ef=8,16,32,64,128,256] [--m=16] [--ef-construction=200]\n"
               "         [--selection=heuristic|simple|both] [--min-recall=0.95]\n"
               "         [--queries=N] [--save=PATH]\n");
}

/// One full sweep for a given index. Returns the best tie-aware recall seen.
double sweep(const HnswIndex& index, const VectorStore& base, const VectorStore& queries,
             const IvecsData& truth, const std::vector<std::size_t>& ef_values,
             std::size_t query_count, DistanceComputer& exact,
             std::vector<SweepRow>& rows) {
  double best = 0.0;
  std::vector<Neighbor> got(k_neighbors);

  for (const std::size_t ef : ef_values) {
    // ef bounds the layer-0 candidate list, so ef < k cannot return k results.
    // PRD section 6 suggests sweeping ef from 8, which is below the k=10 this
    // phase reports — skipped rather than silently clamped, because a row
    // labelled ef=8 that actually ran at ef=10 would be a lie in a table.
    if (ef < k_neighbors) {
      std::printf("  ef=%-4zu  skipped: ef < k=%zu cannot return k results\n", ef,
                  k_neighbors);
      continue;
    }

    SearchParams params;
    params.ef = ef;

    double strict_total = 0.0;
    double tied_total = 0.0;
    std::size_t visited_total = 0;

    const auto start = clock_type::now();
    for (std::size_t q = 0; q < query_count; ++q) {
      const float* query = queries.get(static_cast<VectorId>(q));
      if (index.search(query, params, got) != Status::ok) {
        std::fprintf(stderr, "error: search failed at query %zu, ef=%zu\n", q, ef);
        return -1.0;
      }
      visited_total += index.last_visited();

      const auto row = truth.row(q).first(k_neighbors);
      strict_total += recall_at_k(got, row);

      // The exact computer needs this query prepared to threshold on the k-th
      // true distance (D17): SIFT1M has 14,538 duplicate vectors, so strict id
      // comparison charges the index for tie-break convention rather than for
      // anything it did.
      exact.prepare_query(query);
      tied_total += recall_at_k_tied(exact, got, row);
    }
    const double elapsed = seconds_since(start);

    SweepRow out{};
    out.ef = ef;
    out.recall_strict = strict_total / static_cast<double>(query_count);
    out.recall_tied = tied_total / static_cast<double>(query_count);
    out.qps = static_cast<double>(query_count) / elapsed;
    out.mean_visited = static_cast<double>(visited_total) / static_cast<double>(query_count);
    rows.push_back(out);

    std::printf("  ef=%-4zu  recall@10 %.4f (strict %.4f)  %8.1f QPS  %7.0f visited  "
                "%.3f%% of corpus\n",
                ef, out.recall_tied, out.recall_strict, out.qps, out.mean_visited,
                100.0 * out.mean_visited / static_cast<double>(base.size()));

    best = std::max(best, out.recall_tied);
  }
  return best;
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  std::vector<std::string_view> positional;
  std::vector<std::size_t> ef_values;
  HnswConfig config;
  std::string selection = "heuristic";
  double min_recall = 0.95;
  std::size_t query_limit = 0;
  std::string save_path;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto value_of = [&](std::string_view prefix) {
      return arg.substr(prefix.size());
    };
    if (arg.starts_with("--ef=")) {
      std::string list{value_of("--ef=")};
      std::size_t pos = 0;
      while (pos <= list.size()) {
        const auto comma = list.find(',', pos);
        const auto token = list.substr(pos, comma - pos);
        if (const auto v = parse_size(token); v && *v > 0) {
          ef_values.push_back(*v);
        }
        if (comma == std::string::npos) {
          break;
        }
        pos = comma + 1;
      }
    } else if (arg.starts_with("--m=")) {
      if (const auto v = parse_size(value_of("--m="))) {
        config.m = *v;
        config.m_max0 = *v * 2;
      }
    } else if (arg.starts_with("--ef-construction=")) {
      if (const auto v = parse_size(value_of("--ef-construction="))) {
        config.ef_construction = *v;
      }
    } else if (arg.starts_with("--selection=")) {
      selection = std::string{value_of("--selection=")};
    } else if (arg.starts_with("--min-recall=")) {
      min_recall = std::stod(std::string{value_of("--min-recall=")});
    } else if (arg.starts_with("--queries=")) {
      if (const auto v = parse_size(value_of("--queries="))) {
        query_limit = *v;
      }
    } else if (arg.starts_with("--save=")) {
      save_path = std::string{value_of("--save=")};
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return exit_ok;
    } else if (arg.starts_with("--")) {
      std::fprintf(stderr, "error: unknown option '%s'\n", std::string(arg).c_str());
      return exit_usage;
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() != 3) {
    usage();
    return exit_usage;
  }
  if (ef_values.empty()) {
    ef_values = {8, 16, 32, 64, 128, 256};
  }

  const std::filesystem::path base_path{positional[0]};
  const std::filesystem::path query_path{positional[1]};
  const std::filesystem::path truth_path{positional[2]};

  for (const auto* p : {&base_path, &query_path, &truth_path}) {
    if (!std::filesystem::exists(*p)) {
      std::fprintf(stderr, "skip: %s not found — run ./tools/download_sift.sh\n",
                   p->string().c_str());
      return exit_skip;
    }
  }

  VectorStore base;
  VectorStore queries;
  IvecsData truth;
  if (load_fvecs(base_path, base) != Status::ok ||
      load_fvecs(query_path, queries) != Status::ok ||
      load_ivecs(truth_path, truth) != Status::ok) {
    std::fprintf(stderr, "error: could not load the dataset\n");
    return exit_fail;
  }
  if (truth.dim < k_neighbors || truth.count != queries.size()) {
    std::fprintf(stderr, "error: ground truth does not match the query set\n");
    return exit_fail;
  }

  const std::size_t query_count =
      (query_limit == 0) ? queries.size() : std::min(query_limit, queries.size());

  auto exact = make_distance_computer(Metric::l2, base);
  if (exact == nullptr) {
    std::fprintf(stderr, "error: no distance computer\n");
    return exit_fail;
  }

  std::printf("corpus %zu x %zu, %zu queries, kernel %s\n", base.size(), base.dim(),
              query_count, std::string(kernel_name(detected_kernel())).c_str());
  std::printf("config M=%zu M_max0=%zu ef_construction=%zu seed=%llu\n", config.m,
              config.m_max0, config.ef_construction,
              static_cast<unsigned long long>(config.seed));

  const std::vector<std::string> selections =
      (selection == "both") ? std::vector<std::string>{"heuristic", "simple"}
                            : std::vector<std::string>{selection};

  double best_overall = 0.0;
  for (const auto& sel : selections) {
    const auto kind = (sel == "simple") ? NeighbourSelection::simple
                                        : NeighbourSelection::heuristic;

    auto index = make_hnsw_index(base, Metric::l2, config, kind);
    if (index == nullptr) {
      std::fprintf(stderr, "error: config rejected\n");
      return exit_fail;
    }

    std::printf("\nselection: %s\n", sel.c_str());
    const auto build_start = clock_type::now();
    const std::size_t report_every = std::max<std::size_t>(1, base.size() / 10);
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (index->add(static_cast<VectorId>(i)) != Status::ok) {
        std::fprintf(stderr, "error: insert failed at %zu\n", i);
        return exit_fail;
      }
      if (base.size() > 100000 && (i + 1) % report_every == 0) {
        std::fprintf(stderr, "\r  building %zu%% (%.0f s)",
                     100 * (i + 1) / base.size(), seconds_since(build_start));
      }
    }
    const double build_seconds = seconds_since(build_start);
    if (base.size() > 100000) {
      std::fprintf(stderr, "\r%*s\r", 40, "");
    }

    std::printf("  build %.1f s (%.0f vectors/s), graph %s, max level %zu\n", build_seconds,
                static_cast<double>(base.size()) / build_seconds,
                mib(index->graph_bytes()).c_str(), index->max_level());
    if (const auto rss = peak_rss_bytes()) {
      std::printf("  peak RSS %s (store %s)\n", mib(*rss).c_str(), mib(base.bytes()).c_str());
    }

    std::vector<SweepRow> rows;
    const double best = sweep(*index, base, queries, truth, ef_values, query_count, *exact,
                              rows);
    if (best < 0.0) {
      return exit_fail;
    }
    best_overall = std::max(best_overall, best);

    if (!save_path.empty() && sel == selections.front()) {
      if (index->save(save_path) != Status::ok) {
        std::fprintf(stderr, "error: could not save the index\n");
        return exit_fail;
      }
      std::printf("  saved to %s\n", save_path.c_str());
    }
  }

  std::printf("\nbest tie-aware recall@%zu = %.4f (required %.4f)\n", k_neighbors,
              best_overall, min_recall);
  if (best_overall + 1e-9 < min_recall) {
    std::printf("FAIL: no ef reached the required recall\n");
    return exit_fail;
  }
  std::printf("PASS\n");
  return exit_ok;
}
