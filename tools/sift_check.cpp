// Phase 1's proof command.
//
//   sift_check [--kernel=NAME] <base.fvecs> <query.fvecs> <groundtruth.ivecs>
//              [max_queries]
//
// Loads a corpus, brute-forces exact k-NN for every query, and checks the
// result against the provided ground truth. Prints the three numbers Phase 1
// has to record — load time, peak RSS, brute-force QPS — and the one it has to
// assert: mean recall@10 must be exactly 1.000000.
//
// This is deliberately NOT the benchmark harness. No JSON, no parameter sweep,
// no latency percentiles, no hnswlib. Those are Phase 4, and building them here
// would be the scope creep PRD section 3 names as this project's primary
// failure mode.
//
// On the QPS figure and the fixed methodology: this is a single run with no
// warmup, which the project's benchmark rules would otherwise forbid. Stated
// rather than hidden, because both rules are meaningless for this particular
// workload. A brute-force query streams the entire 512 MB corpus, so there is
// no working set for a warmup to warm; and at roughly eight minutes per pass
// over SIFT1M, a median of three costs half an hour to produce a number Phase 4
// will measure properly anyway. Phase 1's job is to prove recall is 1.000.

#include "lodestone/brute_force.hpp"
#include "lodestone/distance.hpp"
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

using lodestone::IvecsData;
using lodestone::Metric;
using lodestone::Neighbor;
using lodestone::RecallDiagnosis;
using lodestone::Status;
using lodestone::VecsInfo;
using lodestone::VectorStore;

/// recall@10 is the headline metric. recall@1 and @100 arrive with the Phase 4
/// harness that can sweep them properly.
constexpr std::size_t k_neighbors = 10;

/// How many failing queries to explain before going quiet. Enough to see a
/// pattern; not enough to bury the summary under ten thousand lines.
constexpr std::size_t max_reported_failures = 10;

constexpr int exit_ok = 0;
constexpr int exit_fail = 1;
constexpr int exit_usage = 2;
/// Matches the SKIP_RETURN_CODE the build sets on this tool's ctest entry, so a
/// checkout without data/ reports skipped rather than failed. CI has no
/// dataset and must stay green.
constexpr int exit_skip = 4;

std::string_view status_name(Status s) {
  switch (s) {
  case Status::ok:
    return "ok";
  case Status::io_error:
    return "io_error";
  case Status::invalid_argument:
    return "invalid_argument";
  case Status::dimension_mismatch:
    return "dimension_mismatch";
  case Status::out_of_memory:
    return "out_of_memory";
  case Status::not_implemented:
    return "not_implemented";
  }
  return "unknown";
}

/// Peak resident set size, from VmHWM.
///
/// The *peak*, not the current value: it is the number that says whether this
/// fits in memory, and it is unaffected by whatever the allocator has since
/// returned to the OS. Linux-only; reported as unavailable elsewhere.
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
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::string mib(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f MiB",
                static_cast<double>(bytes) / (1024.0 * 1024.0));
  return buf;
}

bool parse_size(std::string_view text, std::size_t& out) {
  const auto* end = text.data() + text.size();
  return std::from_chars(text.data(), end, out).ec == std::errc{} && out > 0;
}

void usage() {
  std::fprintf(stderr,
               "usage: sift_check [--kernel=auto|scalar|sse|avx2]\n"
               "                  <base.fvecs> <query.fvecs> <groundtruth.ivecs>\n"
               "                  [max_queries]\n\n"
               "Brute-forces exact k-NN (k=%zu) for every query and checks it\n"
               "against the provided ground truth. Exits non-zero unless mean\n"
               "tie-aware recall@%zu is exactly 1.000000.\n\n"
               "--kernel exists so every distance kernel can be checked end to end.\n"
               "Agreeing with scalar to 1e-5 is the weak test; producing the same\n"
               "ranking is the one that decides recall.\n",
               k_neighbors, k_neighbors);
}

/// Parse --kernel=NAME. Returns false on an unknown name.
bool parse_kernel(std::string_view text, lodestone::KernelKind& out) {
  if (text == "auto" || text == "automatic") {
    out = lodestone::KernelKind::automatic;
  } else if (text == "scalar") {
    out = lodestone::KernelKind::scalar;
  } else if (text == "sse") {
    out = lodestone::KernelKind::sse;
  } else if (text == "avx2") {
    out = lodestone::KernelKind::avx2;
  } else {
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  // Line-buffered, because a full SIFT1M pass takes minutes and the default
  // block buffering on a redirected stdout shows nothing at all until the end,
  // which is indistinguishable from a hang.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // Flags first, then positionals, so --kernel can appear anywhere before them.
  auto requested_kernel = lodestone::KernelKind::automatic;
  std::vector<std::string_view> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    constexpr std::string_view kernel_flag = "--kernel=";
    if (arg.starts_with(kernel_flag)) {
      if (!parse_kernel(arg.substr(kernel_flag.size()), requested_kernel)) {
        std::fprintf(stderr, "error: unknown kernel '%s'\n",
                     std::string(arg.substr(kernel_flag.size())).c_str());
        return exit_usage;
      }
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

  if (positional.size() < 3 || positional.size() > 4) {
    usage();
    return exit_usage;
  }

  const std::filesystem::path base_path{positional[0]};
  const std::filesystem::path query_path{positional[1]};
  const std::filesystem::path truth_path{positional[2]};

  std::size_t query_limit = 0; // 0 means "all"
  if (positional.size() == 4 && !parse_size(positional[3], query_limit)) {
    std::fprintf(stderr, "error: max_queries must be a positive integer\n");
    return exit_usage;
  }

  // A missing dataset is a skip, not a failure — that is the difference between
  // "nobody ran ./tools/download_sift.sh" and "the data is wrong".
  for (const auto* p : {&base_path, &query_path, &truth_path}) {
    if (!std::filesystem::exists(*p)) {
      std::fprintf(stderr, "skip: %s not found — run ./tools/download_sift.sh\n",
                   p->string().c_str());
      return exit_skip;
    }
  }

  // ---- load ---------------------------------------------------------------

  VecsInfo base_info;
  if (const Status s = lodestone::probe_vecs(base_path, base_info); s != Status::ok) {
    std::fprintf(stderr, "error: probing %s failed: %s\n", base_path.string().c_str(),
                 std::string(status_name(s)).c_str());
    return exit_fail;
  }

  VectorStore base;
  const auto load_start = std::chrono::steady_clock::now();
  if (const Status s = lodestone::load_fvecs(base_path, base); s != Status::ok) {
    std::fprintf(stderr, "error: loading %s failed: %s\n", base_path.string().c_str(),
                 std::string(status_name(s)).c_str());
    return exit_fail;
  }
  const auto load_end = std::chrono::steady_clock::now();
  const double load_seconds =
      std::chrono::duration<double>(load_end - load_start).count();

  VectorStore queries;
  if (const Status s = lodestone::load_fvecs(query_path, queries); s != Status::ok) {
    std::fprintf(stderr, "error: loading %s failed: %s\n", query_path.string().c_str(),
                 std::string(status_name(s)).c_str());
    return exit_fail;
  }

  IvecsData truth;
  if (const Status s = lodestone::load_ivecs(truth_path, truth); s != Status::ok) {
    std::fprintf(stderr, "error: loading %s failed: %s\n", truth_path.string().c_str(),
                 std::string(status_name(s)).c_str());
    return exit_fail;
  }

  const std::size_t bytes_per_vector_on_disk =
      sizeof(std::int32_t) + (base.dim() * sizeof(float));

  std::printf("loaded %zu x %zu from %s in %.2f s\n", base.size(), base.dim(),
              base_path.filename().string().c_str(), load_seconds);
  if (const auto rss = peak_rss_bytes()) {
    std::printf("peak RSS %s   (store %s, %zu B/vector on disk)\n", mib(*rss).c_str(),
                mib(base.bytes()).c_str(), bytes_per_vector_on_disk);
  } else {
    std::printf("peak RSS unavailable   (store %s, %zu B/vector on disk)\n",
                mib(base.bytes()).c_str(), bytes_per_vector_on_disk);
  }

  // ---- shape checks -------------------------------------------------------

  if (queries.dim() != base.dim()) {
    std::fprintf(stderr, "error: query dim %zu != base dim %zu\n", queries.dim(),
                 base.dim());
    return exit_fail;
  }
  if (truth.count != queries.size()) {
    std::fprintf(stderr, "error: %zu ground-truth rows for %zu queries\n", truth.count,
                 queries.size());
    return exit_fail;
  }
  if (truth.dim < k_neighbors) {
    std::fprintf(stderr, "error: ground truth has %zu neighbours per row, need %zu\n",
                 truth.dim, k_neighbors);
    return exit_fail;
  }

  // Checked, not assumed (plan assumption A5). A ground-truth file paired with
  // the wrong base file otherwise surfaces as a low recall number that gets
  // blamed on the search.
  if (const Status s = lodestone::validate_ground_truth_ids(truth.data, base.size());
      s != Status::ok) {
    std::fprintf(stderr, "error: ground truth contains ids outside [0, %zu)\n",
                 base.size());
    return exit_fail;
  }

  auto computer = lodestone::make_distance_computer(Metric::l2, base, requested_kernel);
  if (computer == nullptr) {
    // A kernel this CPU cannot run is a skip, not a failure: the ctest matrix
    // asks for all three, and a machine without AVX2 should report "not
    // applicable" rather than red.
    std::fprintf(stderr, "skip: kernel '%s' is unavailable on this CPU\n",
                 std::string(lodestone::kernel_name(requested_kernel)).c_str());
    return exit_skip;
  }
  std::printf("kernel: %s (requested %s)\n",
              std::string(lodestone::kernel_name(computer->kernel())).c_str(),
              std::string(lodestone::kernel_name(requested_kernel)).c_str());

  const std::size_t query_count =
      (query_limit == 0) ? queries.size() : std::min(query_limit, queries.size());

  // ---- ground-truth ordering, untimed ------------------------------------
  //
  // Separate pass so it stays out of the QPS measurement. If the file's order
  // disagrees with our own metric, our metric disagrees with the one the file
  // was generated under — a squared-versus-plain-L2 slip, say — and that is
  // worth knowing before a recall number gets interpreted.
  std::size_t order_violations = 0;
  for (std::size_t i = 0; i < query_count; ++i) {
    computer->prepare_query(queries.get(static_cast<lodestone::VectorId>(i)));
    if (lodestone::validate_ground_truth_order(*computer, truth.row(i)) != Status::ok) {
      ++order_violations;
    }
  }
  if (order_violations != 0) {
    std::printf("ground truth: %zu of %zu rows are not sorted under our metric\n",
                order_violations, query_count);
  } else {
    std::printf("ground truth: %zu rows, ids in range, order agrees with our metric\n",
                query_count);
  }

  // ---- the timed search --------------------------------------------------

  std::vector<Neighbor> got(k_neighbors);
  std::vector<std::size_t> failing;
  double recall_total = 0.0;
  double tied_recall_total = 0.0;

  // One progress line per ~5% for long runs; silent for short ones.
  const std::size_t progress_every = (query_count >= 1000) ? query_count / 20 : 0;

  const auto search_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < query_count; ++i) {
    const Status s = lodestone::brute_force_knn(
        *computer, queries.get(static_cast<lodestone::VectorId>(i)), base.size(), got);
    if (s != Status::ok) {
      std::fprintf(stderr, "error: query %zu failed: %s\n", i,
                   std::string(status_name(s)).c_str());
      return exit_fail;
    }

    const auto row = truth.row(i).first(k_neighbors);

    // Strict set recall: how often our tie-break matched the reference's.
    const double recall = lodestone::recall_at_k(got, row);
    recall_total += recall;
    if (recall < 1.0) {
      failing.push_back(i);
    }

    // Tie-aware recall: whether the search was actually right. The query is
    // still prepared from the call above, which is what this needs.
    tied_recall_total += lodestone::recall_at_k_tied(*computer, got, row);

    if (progress_every != 0 && (i + 1) % progress_every == 0) {
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - search_start)
              .count();
      const double done = static_cast<double>(i + 1);
      std::fprintf(stderr, "\r  %zu/%zu queries, %.0f s elapsed, ~%.0f s left",
                   i + 1, query_count, elapsed,
                   elapsed * (static_cast<double>(query_count) - done) / done);
    }
  }
  if (progress_every != 0) {
    std::fprintf(stderr, "\r%*s\r", 70, "");
  }
  const auto search_end = std::chrono::steady_clock::now();
  const double search_seconds =
      std::chrono::duration<double>(search_end - search_start).count();

  const double qps =
      (search_seconds > 0.0) ? static_cast<double>(query_count) / search_seconds : 0.0;
  const double mean_recall = recall_total / static_cast<double>(query_count);
  const double mean_tied_recall = tied_recall_total / static_cast<double>(query_count);

  std::printf("brute force: %zu queries, k=%zu, %.2f QPS (%.1f s total)\n", query_count,
              k_neighbors, qps, search_seconds);
  std::printf("mean recall@%zu = %.6f   (strict id-set match)\n", k_neighbors,
              mean_recall);
  std::printf("mean recall@%zu = %.6f   (tie-aware, thresholded on the k-th true "
              "distance)\n",
              k_neighbors, mean_tied_recall);

  if (query_count < 10000) {
    std::printf("note: %zu queries is below the project's 10,000-query minimum, so the\n"
                "      QPS above is indicative only. Phase 4 measures this properly.\n",
                query_count);
  }

  // ---- explain any shortfall ---------------------------------------------

  // The assertion is on the tie-aware figure, because that is the one that
  // answers "was the search correct". It reaches 1.000000 exactly when every
  // returned neighbour is at most as far as the k-th true neighbour — which is
  // equivalent to there being no real miss anywhere.
  //
  // The strict figure can sit below 1.0 without anything being wrong: SIFT1M
  // contains byte-identical duplicate vectors, so when one lands on the k-th
  // boundary several id sets are equally correct and the strict number is
  // reporting whose tie-break convention won. It is printed, never asserted on.
  if (mean_tied_recall >= 1.0) {
    if (failing.empty()) {
      std::printf("PASS: recall@%zu is exactly 1.000000 by both measures over %zu "
                  "queries\n",
                  k_neighbors, query_count);
    } else {
      std::printf("PASS: tie-aware recall@%zu is exactly 1.000000 over %zu queries.\n",
                  k_neighbors, query_count);
      std::printf("      %zu queries differ from the reference id set, every one of "
                  "them\n"
                  "      on the k-th boundary — a tie, not a miss. Details below.\n",
                  failing.size());
    }
  } else {
    std::printf("\nFAIL: tie-aware recall@%zu is %.6f, not 1.000000 — at least one\n"
                "      returned neighbour is genuinely farther than the k-th true one.\n",
                k_neighbors, mean_tied_recall);
  }

  if (failing.empty()) {
    return exit_ok;
  }

  std::printf("\n%zu of %zu queries differ from the reference id set.\n", failing.size(),
              query_count);
  std::printf("The distances tell a tie from a defect: a near-zero gap between the\n"
              "worst neighbour kept and the best one missed means the disagreement sits\n"
              "exactly on the k-th boundary — a duplicate vector, or two distances\n"
              "closer together than float32 can separate. A large gap is a real bug.\n\n");

  std::size_t ties = 0;
  const std::size_t to_report = std::min(failing.size(), max_reported_failures);
  for (std::size_t n = 0; n < failing.size(); ++n) {
    const std::size_t i = failing[n];

    // Re-run the query so the diagnosis sees its own result. Only failing
    // queries pay for this, so it stays outside the timed loop above.
    if (lodestone::brute_force_knn(*computer,
                                  queries.get(static_cast<lodestone::VectorId>(i)),
                                  base.size(), got) != Status::ok) {
      continue;
    }

    RecallDiagnosis diag;
    if (lodestone::diagnose_recall(*computer, got, truth.row(i).first(k_neighbors),
                                   diag) != Status::ok) {
      continue;
    }
    if (diag.boundary_tie) {
      ++ties;
    }

    if (n >= to_report) {
      continue;
    }

    std::printf("  query %zu: recall %.4f  worst_kept %.6g  best_missed %.6g  %s\n", i,
                diag.recall, static_cast<double>(diag.worst_kept),
                static_cast<double>(diag.best_missed),
                diag.boundary_tie ? "<- boundary tie" : "<- REAL MISS");
    std::printf("    missed:");
    for (const auto id : diag.missed) {
      std::printf(" %d", id);
    }
    std::printf("\n    extra: ");
    for (const auto id : diag.extra) {
      std::printf(" %u", id);
    }
    std::printf("\n");
  }

  if (failing.size() > to_report) {
    std::printf("  ... and %zu more\n", failing.size() - to_report);
  }
  std::printf("\n%zu of %zu differences are boundary ties; %zu are real misses.\n", ties,
              failing.size(), failing.size() - ties);

  // Ties are not failures. A real miss is.
  return (mean_tied_recall >= 1.0) ? exit_ok : exit_fail;
}
