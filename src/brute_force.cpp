// Phase 1 — exact k-NN, the recall metric, and the diagnostics that make an
// imperfect recall figure explainable rather than merely disappointing.

#include "lodestone/brute_force.hpp"

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lodestone {

namespace {

/// Ids handed to distances_to() per call.
///
/// The batch exists to amortise one virtual call over many distances
/// (DECISIONS.md D1), and 256 is far past the point where that overhead
/// disappears. It is not tuned: the scan is bound by streaming vector data out
/// of memory, not by the call, so there is nothing here worth tuning until
/// Phase 2 has a kernel fast enough for the difference to be measurable.
constexpr std::size_t scan_block = 256;

/// The total order results are sorted by, and the order the bounded heap keeps.
///
/// `(distance, id)` lexicographic, not distance alone. Distance alone leaves
/// equal-distance neighbours arranged however the heap happened to leave them,
/// so a rerun can produce a different-but-equally-correct answer. This function
/// is the ground truth everything else is measured against, so it has to be
/// reproducible to the id.
bool closer(const Neighbor& a, const Neighbor& b) {
  if (a.distance != b.distance) {
    return a.distance < b.distance;
  }
  return a.id < b.id;
}

/// Relative tolerance for calling two distances equal.
///
/// Squared L2 accumulated in float32 over 128 terms carries roughly this much
/// relative error, so two distances agreeing to within it are indistinguishable
/// to this arithmetic — whether they are exactly equal or merely very close.
constexpr float tie_tolerance = 1e-5F;

bool within_tolerance(float a, float b) {
  const float scale = std::max({1.0F, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= tie_tolerance * scale;
}

} // namespace

Status brute_force_knn(DistanceComputer& computer, const float* query, std::size_t count,
                       std::span<Neighbor> out) {
  const std::size_t k = out.size();

  if (k == 0 || count == 0 || query == nullptr) {
    return Status::invalid_argument;
  }
  // Returning fewer than k while reporting ok would let a caller divide by k
  // and get a recall figure that looks fine.
  if (k > count) {
    return Status::invalid_argument;
  }

  computer.prepare_query(query);

  std::vector<VectorId> ids(scan_block);
  std::vector<float> distances(scan_block);

  // `out` doubles as the heap, so there is no second allocation for results.
  // Once it holds k entries it is a max-heap under `closer`, meaning out[0] is
  // the *worst* kept neighbour — the one a new candidate has to beat.
  std::size_t filled = 0;

  for (std::size_t base = 0; base < count; base += scan_block) {
    const std::size_t n = std::min(scan_block, count - base);

    for (std::size_t j = 0; j < n; ++j) {
      ids[j] = static_cast<VectorId>(base + j);
    }
    computer.distances_to(std::span<const VectorId>(ids).first(n),
                          std::span<float>(distances).first(n));

    for (std::size_t j = 0; j < n; ++j) {
      const Neighbor candidate{ids[j], distances[j]};

      if (filled < k) {
        out[filled] = candidate;
        ++filled;
        if (filled == k) {
          std::make_heap(out.begin(), out.end(), closer);
        }
        continue;
      }

      // The common case by an enormous margin: after the first few hundred
      // candidates almost everything loses to the heap top immediately, so the
      // steady state is one float comparison per vector. That is why a bounded
      // heap beats sorting all `count` distances — at k=10 over a million
      // vectors, sorting does 1M log(1M) comparisons to answer a question that
      // 1M comparisons answer.
      if (!closer(candidate, out[0])) {
        continue;
      }

      std::pop_heap(out.begin(), out.end(), closer);
      out[k - 1] = candidate;
      std::push_heap(out.begin(), out.end(), closer);
    }
  }

  // k <= count was checked, so the heap is full and this leaves `out` sorted
  // nearest-first.
  std::sort_heap(out.begin(), out.end(), closer);
  return Status::ok;
}

double recall_at_k(std::span<const Neighbor> got, std::span<const std::int32_t> truth) {
  const std::size_t k = got.size();
  if (k == 0 || truth.size() < k) {
    return 0.0;
  }

  // O(k^2), with k at most 100. Deliberately the simple version: this runs
  // once per query off the measured path, and a hash set would cost an
  // allocation per query to save comparisons that do not matter.
  std::size_t hits = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const auto want = truth[i];
    for (const auto& n : got) {
      if (static_cast<std::int64_t>(n.id) == static_cast<std::int64_t>(want)) {
        ++hits;
        break;
      }
    }
  }

  return static_cast<double>(hits) / static_cast<double>(k);
}

double recall_at_k_tied(const DistanceComputer& computer, std::span<const Neighbor> got,
                        std::span<const std::int32_t> truth) {
  const std::size_t k = got.size();
  if (k == 0 || truth.size() < k) {
    return 0.0;
  }

  // The k-th true distance: the largest among the first k truth entries. Rows
  // arrive sorted, but taking the max rather than truth[k-1] means a row that
  // is not perfectly sorted cannot silently lower the bar.
  float threshold = 0.0F;
  bool first = true;
  for (std::size_t i = 0; i < k; ++i) {
    if (truth[i] < 0) {
      return 0.0;
    }
    const float d = computer.distance_to(static_cast<VectorId>(truth[i]));
    if (first || d > threshold) {
      threshold = d;
      first = false;
    }
  }

  // A neighbour at most as far as the k-th true neighbour is a correct answer,
  // whether or not it is the particular id the reference generator chose.
  std::size_t hits = 0;
  for (const auto& n : got) {
    if (n.distance <= threshold || within_tolerance(n.distance, threshold)) {
      ++hits;
    }
  }

  return static_cast<double>(hits) / static_cast<double>(k);
}

Status diagnose_recall(const DistanceComputer& computer, std::span<const Neighbor> got,
                       std::span<const std::int32_t> truth, RecallDiagnosis& out) {
  const std::size_t k = got.size();
  if (k == 0 || truth.size() < k) {
    return Status::invalid_argument;
  }

  out = RecallDiagnosis{};
  out.recall = recall_at_k(got, truth);

  const auto in_got = [&](std::int32_t id) {
    return std::any_of(got.begin(), got.end(), [&](const Neighbor& n) {
      return static_cast<std::int64_t>(n.id) == static_cast<std::int64_t>(id);
    });
  };
  const auto in_truth = [&](VectorId id) {
    for (std::size_t i = 0; i < k; ++i) {
      if (static_cast<std::int64_t>(truth[i]) == static_cast<std::int64_t>(id)) {
        return true;
      }
    }
    return false;
  };

  for (std::size_t i = 0; i < k; ++i) {
    if (!in_got(truth[i])) {
      out.missed.push_back(truth[i]);
    }
  }
  for (const auto& n : got) {
    if (!in_truth(n.id)) {
      out.extra.push_back(n.id);
    }
  }

  for (const auto& n : got) {
    out.worst_kept = std::max(out.worst_kept, n.distance);
  }

  if (!out.missed.empty()) {
    bool first = true;
    for (const auto id : out.missed) {
      if (id < 0) {
        return Status::invalid_argument;
      }
      const float d = computer.distance_to(static_cast<VectorId>(id));
      if (first || d < out.best_missed) {
        out.best_missed = d;
        first = false;
      }
    }

    // The crisp test. If the worst neighbour we kept is the same distance as
    // the best one we missed, the disagreement sits exactly on the k-th
    // boundary: either a genuine tie, or two distances closer together than
    // float32 can separate. Neither is a defect in the search. A gap that is
    // large relative to the distances involved is.
    out.boundary_tie = within_tolerance(out.worst_kept, out.best_missed);
  }

  return Status::ok;
}

Status validate_ground_truth_ids(std::span<const std::int32_t> ids, std::size_t corpus_size) {
  for (const auto id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= corpus_size) {
      return Status::invalid_argument;
    }
  }
  return Status::ok;
}

Status validate_ground_truth_order(const DistanceComputer& computer,
                                   std::span<const std::int32_t> row) {
  if (row.empty()) {
    return Status::invalid_argument;
  }

  float previous = 0.0F;
  for (std::size_t i = 0; i < row.size(); ++i) {
    if (row[i] < 0) {
      return Status::invalid_argument;
    }
    const float d = computer.distance_to(static_cast<VectorId>(row[i]));

    // Non-decreasing, but tolerant of a decrease inside float tolerance:
    // float32 accumulation can invert two neighbours that are genuinely that
    // close, and flagging that as a corrupt file would be wrong.
    if (i > 0 && d < previous && !within_tolerance(d, previous)) {
      return Status::invalid_argument;
    }
    previous = d;
  }

  return Status::ok;
}

} // namespace lodestone
