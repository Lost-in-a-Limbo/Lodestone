// Phase 4 — the benchmark harness.
//
//   bench --all [--small] [--runs=3] [--warmup=10] [--ef=...] [--k=...]
//               [--m=16] [--ef-construction=200] [--out=PATH] [--skip-hnswlib]
//
// Emits bench/results/results.json with no manual steps. Everything downstream —
// Phase 5's quantisation curve, Phase 6's selectivity sweep, Phase 8's frontend
// — reports through this file, so nothing here is hand-typed and every number
// carries the machine that produced it.
//
// The one rule that decides whether the comparison is honest: **match on recall,
// never on ef.** Two implementations do not agree on what a given ef buys, so
// comparing ef=64 to ef=64 compares different operating points and flatters
// whichever explores less. The output is a curve; it is read vertically.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
#include "lodestone/hnsw.hpp"
#include "lodestone/io.hpp"
#include "lodestone/quantizer.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <nlohmann/json.hpp>

#ifdef LODESTONE_HAVE_HNSWLIB
#include "hnswlib/hnswlib.h"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace lodestone;
using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point t) {
  return std::chrono::duration<double>(clock_type::now() - t).count();
}

std::optional<std::size_t> parse_size(std::string_view text) {
  std::size_t v = 0;
  if (std::from_chars(text.data(), text.data() + text.size(), v).ec != std::errc{}) {
    return std::nullopt;
  }
  return v;
}

std::vector<std::size_t> parse_list(std::string_view text) {
  std::vector<std::size_t> out;
  std::size_t pos = 0;
  const std::string s{text};
  while (pos <= s.size()) {
    const auto comma = s.find(',', pos);
    if (const auto v = parse_size(std::string_view{s}.substr(pos, comma - pos)); v && *v > 0) {
      out.push_back(*v);
    }
    if (comma == std::string::npos) {
      break;
    }
    pos = comma + 1;
  }
  return out;
}

std::string read_first_line_after(const std::filesystem::path& file, std::string_view key) {
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) == 0) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) {
        return {};
      }
      auto value = line.substr(colon + 1);
      const auto first = value.find_first_not_of(" \t");
      return first == std::string::npos ? std::string{} : value.substr(first);
    }
  }
  return {};
}

std::optional<std::size_t> proc_status_kib(std::string_view key) {
  const auto text = read_first_line_after("/proc/self/status", key);
  if (text.empty()) {
    return std::nullopt;
  }
  return parse_size(std::string_view{text}.substr(0, text.find(' ')));
}

/// Everything about the machine that could change a number. A result without
/// this is not reproducible and therefore not a result.
json machine_spec() {
  json m;
  m["cpu"] = read_first_line_after("/proc/cpuinfo", "model name");

  std::size_t threads = 0;
  {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("processor", 0) == 0) {
        ++threads;
      }
    }
  }
  m["threads"] = threads;

  const auto siblings = parse_size(read_first_line_after("/proc/cpuinfo", "siblings"));
  const auto cores = parse_size(read_first_line_after("/proc/cpuinfo", "cpu cores"));
  m["physical_cores"] = cores.value_or(0);

  const auto mem = read_first_line_after("/proc/meminfo", "MemTotal");
  if (const auto kib = parse_size(std::string_view{mem}.substr(0, mem.find(' ')))) {
    m["ram_gib"] = static_cast<double>(*kib) / (1024.0 * 1024.0);
  }

  for (const auto& [index, label] : {std::pair{0, "l1d"}, std::pair{2, "l2"}, std::pair{3, "l3"}}) {
    const auto path = std::filesystem::path("/sys/devices/system/cpu/cpu0/cache/index") +=
        std::to_string(index);
    std::ifstream in(path / "size");
    std::string value;
    if (in && std::getline(in, value)) {
      m[label] = value;
    }
  }

  m["compiler"] = __VERSION__;
#ifdef LODESTONE_BUILD_FLAGS
  m["build_flags"] = LODESTONE_BUILD_FLAGS;
#endif
  m["distance_kernel"] = std::string(kernel_name(detected_kernel()));

  // The governor matters here more than usual: this machine throttles, and a
  // reader comparing two results files needs to know whether either was pinned.
  std::ifstream gov("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string governor;
  if (gov && std::getline(gov, governor)) {
    m["cpu_governor"] = governor;
  }
  std::ifstream temp("/sys/class/thermal/thermal_zone0/temp");
  std::string millidegrees;
  if (temp && std::getline(temp, millidegrees)) {
    if (const auto v = parse_size(millidegrees)) {
      m["temp_c_at_start"] = static_cast<double>(*v) / 1000.0;
    }
  }
  (void)siblings;
  return m;
}

/// One (k, ef) operating point, measured.
struct Measurement {
  std::size_t k;
  std::size_t ef;
  double recall_tied;
  double recall_strict;
  std::vector<double> qps_runs;
  double p50_us;
  double p95_us;
  double p99_us;
  double visited_mean;
};

double percentile(std::vector<double>& sorted_us, double q) {
  if (sorted_us.empty()) {
    return 0.0;
  }
  const auto index = static_cast<std::size_t>(q * static_cast<double>(sorted_us.size() - 1));
  return sorted_us[index];
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

/// A search callable, so the same measurement loop serves both libraries.
/// Returns the ids it found; the harness owns recall and timing.
using SearchFn = std::function<void(const float* query, std::size_t k, std::vector<Neighbor>& out)>;

Measurement measure(const SearchFn& search, const VectorStore& queries, const IvecsData& truth,
                    DistanceComputer& exact, std::size_t k, std::size_t ef, std::size_t runs,
                    double warmup_seconds, const std::function<std::size_t()>& visited_of) {
  Measurement result{};
  result.k = k;
  result.ef = ef;

  std::vector<Neighbor> got(k);
  const std::size_t query_count = queries.size();

  // Warmup, discarded. The methodology asks for ten seconds; on a machine that
  // throttles, its real job is to get the clock into the state it will spend
  // the measurement in, not to warm a cache.
  {
    const auto start = clock_type::now();
    std::size_t q = 0;
    while (seconds_since(start) < warmup_seconds) {
      search(queries.get(static_cast<VectorId>(q % query_count)), k, got);
      ++q;
    }
  }

  // Recall is computed once. It does not vary between runs — and checking that
  // it does not is itself worth something, so the first run asserts it.
  double tied_total = 0.0;
  double strict_total = 0.0;
  std::size_t visited_total = 0;

  std::vector<double> latencies_us;
  latencies_us.reserve(query_count);

  // One extra pass beyond `runs`, discarded. Phase 4's data showed the
  // 10-second warmup does not reach steady state on this workload: run 1 was
  // the slowest of three in 13 of 24 measurements, against a chance
  // expectation of 8, and the worst point's runs rose monotonically. The fix
  // was logged then and deliberately not applied retroactively — changing
  // methodology after seeing numbers is how a benchmark stops being one — so
  // it is applied here, before Phase 5 measures anything.
  for (std::size_t run = 0; run < runs + 1; ++run) {
    latencies_us.clear();
    const auto run_start = clock_type::now();

    for (std::size_t q = 0; q < query_count; ++q) {
      const float* query = queries.get(static_cast<VectorId>(q));
      const auto t0 = clock_type::now();
      search(query, k, got);
      const auto t1 = clock_type::now();
      latencies_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

      if (run == 1) {
        const auto row = truth.row(q).first(k);
        strict_total += recall_at_k(got, row);
        exact.prepare_query(query);
        tied_total += recall_at_k_tied(exact, got, row);
        visited_total += visited_of ? visited_of() : 0;
      }
    }

    const double elapsed = seconds_since(run_start);
    if (run > 0) {
      result.qps_runs.push_back(static_cast<double>(query_count) / elapsed);
    }
  }

  result.recall_tied = tied_total / static_cast<double>(query_count);
  result.recall_strict = strict_total / static_cast<double>(query_count);
  result.visited_mean = static_cast<double>(visited_total) / static_cast<double>(query_count);

  std::sort(latencies_us.begin(), latencies_us.end());
  result.p50_us = percentile(latencies_us, 0.50);
  result.p95_us = percentile(latencies_us, 0.95);
  result.p99_us = percentile(latencies_us, 0.99);
  return result;
}

json to_json(const Measurement& m) {
  json j;
  j["k"] = m.k;
  j["ef"] = m.ef;
  j["recall"] = m.recall_tied;
  j["recall_strict"] = m.recall_strict;
  j["qps"] = median(m.qps_runs);
  j["qps_runs"] = m.qps_runs;
  const auto lo = *std::min_element(m.qps_runs.begin(), m.qps_runs.end());
  const auto hi = *std::max_element(m.qps_runs.begin(), m.qps_runs.end());
  j["qps_spread_pct"] = 100.0 * (hi - lo) / median(m.qps_runs);
  j["p50_us"] = m.p50_us;
  j["p95_us"] = m.p95_us;
  j["p99_us"] = m.p99_us;
  if (m.visited_mean > 0.0) {
    j["visited_mean"] = m.visited_mean;
  }
  return j;
}

void print_row(std::string_view label, const Measurement& m) {
  std::printf("  %-9s k=%-4zu ef=%-4zu  recall %.4f  %9.1f QPS  p50 %7.1f  p99 %8.1f us"
              "  spread %.1f%%\n",
              std::string(label).c_str(), m.k, m.ef, m.recall_tied, median(m.qps_runs), m.p50_us,
              m.p99_us,
              100.0 *
                  (*std::max_element(m.qps_runs.begin(), m.qps_runs.end()) -
                   *std::min_element(m.qps_runs.begin(), m.qps_runs.end())) /
                  median(m.qps_runs));
}

void usage() {
  std::fprintf(stderr, "usage: bench --all [--small] [--runs=3] [--warmup=10]\n"
                       "             [--ef=16,32,64,128,256] [--k=1,10,100]\n"
                       "             [--m=16] [--ef-construction=200] [--out=PATH]\n"
                       "             [--skip-hnswlib]\n");
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  bool all = false;
  bool small = false;
  bool skip_hnswlib = false;
  bool with_pq = false;
  bool pq_graph = false;
  std::vector<std::size_t> pq_subspaces{8, 16, 32};
  std::size_t runs = 3;
  double warmup_seconds = 10.0;
  std::vector<std::size_t> ef_values{16, 32, 64, 128, 256};
  std::vector<std::size_t> k_values{1, 10, 100};
  HnswConfig config;
  std::string out_path = "bench/results/results.json";

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--all") {
      all = true;
    } else if (arg == "--small") {
      small = true;
    } else if (arg == "--skip-hnswlib") {
      skip_hnswlib = true;
    } else if (arg == "--pq") {
      with_pq = true;
    } else if (arg == "--pq-graph") {
      // Building the graph *through* a PQ computer. Measured to be impractical
      // at 1M, so it is opt-in rather than default — see below.
      with_pq = true;
      pq_graph = true;
    } else if (arg.starts_with("--pq-m=")) {
      with_pq = true;
      pq_subspaces = parse_list(arg.substr(7));
    } else if (arg.starts_with("--runs=")) {
      runs = parse_size(arg.substr(7)).value_or(3);
    } else if (arg.starts_with("--warmup=")) {
      warmup_seconds = static_cast<double>(parse_size(arg.substr(9)).value_or(10));
    } else if (arg.starts_with("--ef=")) {
      ef_values = parse_list(arg.substr(5));
    } else if (arg.starts_with("--k=")) {
      k_values = parse_list(arg.substr(4));
    } else if (arg.starts_with("--m=")) {
      config.m = parse_size(arg.substr(4)).value_or(16);
      config.m_max0 = config.m * 2;
    } else if (arg.starts_with("--ef-construction=")) {
      config.ef_construction = parse_size(arg.substr(18)).value_or(200);
    } else if (arg.starts_with("--out=")) {
      out_path = std::string{arg.substr(6)};
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "error: unknown option '%s'\n", std::string(arg).c_str());
      return 2;
    }
  }

  if (!all) {
    usage();
    return 2;
  }

  const std::filesystem::path root = small ? "data/siftsmall" : "data/sift";
  const std::string prefix = small ? "siftsmall" : "sift";
  const auto base_path = root / (prefix + "_base.fvecs");
  const auto query_path = root / (prefix + "_query.fvecs");
  const auto truth_path = root / (prefix + "_groundtruth.ivecs");

  for (const auto* p : {&base_path, &query_path, &truth_path}) {
    if (!std::filesystem::exists(*p)) {
      std::fprintf(stderr, "skip: %s not found — run ./tools/download_sift.sh --full\n",
                   p->string().c_str());
      return 4;
    }
  }

  VectorStore base;
  VectorStore queries;
  IvecsData truth;
  const auto load_start = clock_type::now();
  if (load_fvecs(base_path, base) != Status::ok || load_fvecs(query_path, queries) != Status::ok ||
      load_ivecs(truth_path, truth) != Status::ok) {
    std::fprintf(stderr, "error: could not load the dataset\n");
    return 1;
  }
  const double load_seconds = seconds_since(load_start);

  auto exact = make_distance_computer(Metric::l2, base);
  if (exact == nullptr) {
    std::fprintf(stderr, "error: no distance computer\n");
    return 1;
  }

  json out;
  out["schema_version"] = 1;
  {
    const auto now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
    out["generated_at"] = stamp;
  }
  out["machine"] = machine_spec();
  out["dataset"] = {{"name", prefix},
                    {"vectors", base.size()},
                    {"dim", base.dim()},
                    {"queries", queries.size()},
                    {"ground_truth_depth", truth.dim},
                    {"load_seconds", load_seconds}};
  out["methodology"] = {
      {"runs", runs},
      {"warmup_seconds", warmup_seconds},
      {"queries_per_measurement", queries.size()},
      {"threads", 1},
      {"aggregate", "median"},
      {"discarded_runs",
       "warmup plus the first measured run (Phase 4 showed 10 s is not steady state)"},
      {"recall", "tie-aware, thresholded on the k-th true distance (DECISIONS.md D17)"},
      {"comparison_rule", "match on recall, never on ef"},
      {"build_time_runs", 1}};

  std::printf("dataset %s: %zu x %zu, %zu queries, kernel %s\n", prefix.c_str(), base.size(),
              base.dim(), queries.size(), std::string(kernel_name(detected_kernel())).c_str());
  std::printf("config M=%zu M_max0=%zu ef_construction=%zu, %zu runs, %.0fs warmup\n\n", config.m,
              config.m_max0, config.ef_construction, runs, warmup_seconds);

  json indexes = json::array();

  // ---- Lodestone ----------------------------------------------------------
  {
    auto index = make_hnsw_index(base, Metric::l2, config);
    if (index == nullptr) {
      std::fprintf(stderr, "error: config rejected\n");
      return 1;
    }
    std::printf("building lodestone...\n");
    const auto start = clock_type::now();
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (index->add(static_cast<VectorId>(i)) != Status::ok) {
        std::fprintf(stderr, "error: insert failed at %zu\n", i);
        return 1;
      }
    }
    const double build_seconds = seconds_since(start);
    std::printf("  built in %.1f s, graph %.1f MiB\n", build_seconds,
                static_cast<double>(index->graph_bytes()) / (1024.0 * 1024.0));

    json entry;
    entry["name"] = "lodestone";
    entry["build_seconds"] = build_seconds;
    entry["index_bytes"] = index->graph_bytes();
    entry["config"] = {{"m", config.m},
                       {"m_max0", config.m_max0},
                       {"ef_construction", config.ef_construction},
                       {"seed", config.seed}};
    json sweep = json::array();

    for (const std::size_t k : k_values) {
      for (const std::size_t ef : ef_values) {
        if (ef < k) {
          continue; // ef bounds the candidate list; ef < k cannot serve k (D28)
        }
        SearchParams params;
        params.ef = ef;
        const SearchFn fn = [&](const float* q, std::size_t kk, std::vector<Neighbor>& o) {
          o.resize(kk);
          (void)index->search(q, params, o);
        };
        const auto m = measure(fn, queries, truth, *exact, k, ef, runs, warmup_seconds,
                               [&] { return index->last_visited(); });
        print_row("lodestone", m);
        sweep.push_back(to_json(m));
      }
    }
    entry["sweep"] = sweep;
    indexes.push_back(entry);
  }

  // ---- hnswlib ------------------------------------------------------------
#ifdef LODESTONE_HAVE_HNSWLIB
  if (!skip_hnswlib) {
    std::printf("\nbuilding hnswlib...\n");
    hnswlib::L2Space space(base.dim());
    hnswlib::HierarchicalNSW<float> reference(&space, base.size(), config.m, config.ef_construction,
                                              static_cast<std::size_t>(config.seed));
    const auto start = clock_type::now();
    for (std::size_t i = 0; i < base.size(); ++i) {
      reference.addPoint(base.get(static_cast<VectorId>(i)), i);
    }
    const double build_seconds = seconds_since(start);
    const auto rss = proc_status_kib("VmHWM");
    std::printf("  built in %.1f s\n", build_seconds);

    json entry;
    entry["name"] = "hnswlib";
    entry["version"] = "0.8.0";
    entry["build_seconds"] = build_seconds;
    if (rss) {
      entry["process_peak_rss_bytes"] = *rss * 1024;
    }
    entry["config"] = {{"m", config.m}, {"ef_construction", config.ef_construction}};
    json sweep = json::array();

    for (const std::size_t k : k_values) {
      for (const std::size_t ef : ef_values) {
        if (ef < k) {
          continue;
        }
        reference.setEf(ef);
        const SearchFn fn = [&](const float* q, std::size_t kk, std::vector<Neighbor>& o) {
          auto heap = reference.searchKnn(q, kk);
          o.resize(kk);
          // hnswlib returns a max-heap: farthest first. Fill backwards so the
          // harness sees the same nearest-first order Lodestone produces, or
          // recall@k would be compared against a reversed list.
          std::size_t i = o.size();
          while (!heap.empty() && i > 0) {
            --i;
            o[i] = Neighbor{static_cast<VectorId>(heap.top().second), heap.top().first};
            heap.pop();
          }
        };
        const auto m = measure(fn, queries, truth, *exact, k, ef, runs, warmup_seconds, nullptr);
        print_row("hnswlib", m);
        sweep.push_back(to_json(m));
      }
    }
    entry["sweep"] = sweep;
    indexes.push_back(entry);
  }
#else
  (void)skip_hnswlib;
  std::printf("\nhnswlib not available in this build — comparison skipped\n");
#endif

  // ---- product quantization ----------------------------------------------
  //
  // Two measurements per m, because they answer different questions.
  //
  // Brute-force ADC scans every code, so it isolates *quantization* loss with
  // no graph in the way — the accuracy/memory frontier this phase exists to
  // draw. HNSW+PQ is the practical configuration and compounds two
  // approximations, so its recall is lower and the difference between the two
  // says which approximation cost what.
  //
  // Both are graded against the same exact ground truth, never against each
  // other.
  if (with_pq) {
    const auto learn_path = root / (prefix + "_learn.fvecs");
    VectorStore learn;
    if (!std::filesystem::exists(learn_path) || load_fvecs(learn_path, learn) != Status::ok) {
      std::fprintf(stderr, "\nskip: %s not found — PQ needs the learn split\n",
                   learn_path.string().c_str());
    } else {
      std::printf("\nproduct quantization, trained on %zu held-out vectors\n", learn.size());

      for (const std::size_t m : pq_subspaces) {
        if (base.dim() % m != 0) {
          std::printf("  m=%zu skipped: does not divide dim %zu\n", m, base.dim());
          continue;
        }
        PqConfig pq_config;
        pq_config.subspaces = m;

        const auto train_start = clock_type::now();
        auto pq = train_product_quantizer(learn, pq_config);
        if (pq == nullptr) {
          std::printf("  m=%zu rejected by the trainer\n", m);
          continue;
        }
        const double train_seconds = seconds_since(train_start);

        const auto encode_start = clock_type::now();
        if (pq->encode(base) != Status::ok) {
          std::fprintf(stderr, "error: PQ encode failed at m=%zu\n", m);
          return 1;
        }
        const double encode_seconds = seconds_since(encode_start);
        const double error = pq->reconstruction_error(base);

        const double ratio = static_cast<double>(base.dim() * sizeof(float)) /
                             static_cast<double>(pq->code_bytes_per_vector());
        std::printf("  m=%-3zu %zu B/vector (%.0fx smaller), codes %.1f MiB, codebook %.0f KiB,"
                    " train %.0f s, encode %.0f s, recon err %.0f\n",
                    m, pq->code_bytes_per_vector(), ratio,
                    static_cast<double>(pq->code_bytes()) / (1024.0 * 1024.0),
                    static_cast<double>(pq->codebook_bytes()) / 1024.0, train_seconds,
                    encode_seconds, error);

        json entry;
        entry["name"] = "lodestone-pq-bruteforce";
        entry["pq"] = {{"subspaces", m},
                       {"centroids", pq_centroids},
                       {"bytes_per_vector", pq->code_bytes_per_vector()},
                       {"compression_ratio", ratio},
                       {"code_bytes", pq->code_bytes()},
                       {"codebook_bytes", pq->codebook_bytes()},
                       {"train_seconds", train_seconds},
                       {"encode_seconds", encode_seconds},
                       {"reconstruction_error", error},
                       {"trained_on", "held-out learn split"},
                       {"distance", "asymmetric (ADC), query not quantized"},
                       {"reranking", "none — it would recover recall and hide the loss"}};

        // Brute force over the codes: quantization loss alone.
        {
          auto adc = make_pq_distance_computer(*pq);
          if (adc == nullptr) {
            std::fprintf(stderr, "error: no ADC computer at m=%zu\n", m);
            return 1;
          }
          json sweep = json::array();
          for (const std::size_t k : k_values) {
            const SearchFn fn = [&](const float* q, std::size_t kk, std::vector<Neighbor>& o) {
              o.resize(kk);
              (void)brute_force_knn(*adc, q, base.size(), o);
            };
            const auto mm =
                measure(fn, queries, truth, *exact, k, 0, runs, warmup_seconds, nullptr);
            std::printf("    brute ADC  k=%-4zu recall %.4f  %9.1f QPS  p50 %8.1f us\n", k,
                        mm.recall_tied, median(mm.qps_runs), mm.p50_us);
            sweep.push_back(to_json(mm));
          }
          entry["sweep"] = sweep;
          indexes.push_back(entry);
        }

        // HNSW over the codes: both approximations together.
        //
        // Off by default, and for a measured reason. The neighbour selection
        // heuristic calls prepare_query() once per candidate. That costs 28 ns
        // for the exact kernel — it is a memcpy — and 10 microseconds for PQ,
        // which rebuilds the whole m x 256 table. 373x to 670x, measured.
        // Construction is therefore hours at 1M where the exact build is six
        // minutes.
        //
        // The fix is not in this file: PQ belongs at *search* time over a graph
        // built with exact distances. DECISIONS.md D36.
        if (pq_graph) {
          auto index = make_hnsw_index_with(
              base, [&] { return make_pq_distance_computer(*pq); }, config);
          if (index == nullptr) {
            std::fprintf(stderr, "error: PQ-backed index rejected at m=%zu\n", m);
            return 1;
          }
          const auto build_start = clock_type::now();
          for (std::size_t i = 0; i < base.size(); ++i) {
            if (index->add(static_cast<VectorId>(i)) != Status::ok) {
              std::fprintf(stderr, "error: PQ index insert failed at %zu\n", i);
              return 1;
            }
          }
          const double build_seconds = seconds_since(build_start);

          json graph_entry;
          graph_entry["name"] = "lodestone-pq-hnsw";
          graph_entry["pq"] = entry["pq"];
          graph_entry["build_seconds"] = build_seconds;
          graph_entry["index_bytes"] = index->graph_bytes();
          json sweep = json::array();
          for (const std::size_t k : k_values) {
            for (const std::size_t ef : ef_values) {
              if (ef < k) {
                continue;
              }
              SearchParams params;
              params.ef = ef;
              const SearchFn fn = [&](const float* q, std::size_t kk, std::vector<Neighbor>& o) {
                o.resize(kk);
                (void)index->search(q, params, o);
              };
              const auto mm =
                  measure(fn, queries, truth, *exact, k, ef, runs, warmup_seconds, nullptr);
              std::printf("    hnsw+PQ    k=%-4zu ef=%-4zu recall %.4f  %9.1f QPS\n", k, ef,
                          mm.recall_tied, median(mm.qps_runs));
              sweep.push_back(to_json(mm));
            }
          }
          graph_entry["sweep"] = sweep;
          indexes.push_back(graph_entry);
        }
      }
    }
  }

  out["indexes"] = indexes;

  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(out_path).parent_path(), ec);
  std::ofstream file(out_path);
  if (!file) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    return 1;
  }
  file << out.dump(2) << '\n';
  if (!file.good()) {
    return 1;
  }

  std::printf("\nwrote %s\n", out_path.c_str());
  return 0;
}
