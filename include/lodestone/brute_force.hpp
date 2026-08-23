#pragma once

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lodestone {

/// One result: a vector id and its **squared** distance from the query.
///
/// Squared throughout the project — sqrt is monotone so it cannot change a
/// ranking, and it would cost a transcendental per distance. Nothing
/// downstream un-squares it, including the benchmark output.
struct Neighbor {
  VectorId id = invalid_id;
  float distance = 0.0F;
};

/// Exact k-nearest-neighbour search by scanning ids `[0, count)`.
///
/// `out.size()` **is** k — there is no separate parameter, so the two cannot
/// disagree. Results are written sorted nearest-first.
///
/// The ordering is the total order `(distance, id)` lexicographically, not
/// distance alone. Sorting on distance alone leaves the arrangement of equal
/// distances up to whatever the heap happened to do, which makes a rerun
/// capable of producing a different-but-equally-correct answer. That is
/// intolerable here: this function *is* the ground truth other things are
/// measured against, so it has to be reproducible to the id.
///
/// `computer` has `prepare_query()` called on it; `count` must be the number of
/// vectors in the store the computer was built against.
///
/// Phase 6 will want a variant scanning an explicit candidate subset rather
/// than a contiguous range — that is the pre-filter strategy. Not built yet;
/// see IDEAS.md.
Status brute_force_knn(DistanceComputer& computer, const float* query, std::size_t count,
                       std::span<Neighbor> out);

/// recall@k = |{ids we returned} ∩ {the k true nearest}| / k.
///
/// k is `got.size()`, and only the first k entries of `truth` are consulted —
/// TEXMEX ground truth rows hold the top 100, so recall@10 reads the first 10.
/// `truth` must therefore be at least as long as `got`.
///
/// A set measure: the order within the returned k does not affect it.
[[nodiscard]] double recall_at_k(std::span<const Neighbor> got,
                                std::span<const std::int32_t> truth);

/// recall@k measured against the k-th true *distance* rather than the k-th true
/// *id set*.
///
/// A returned neighbour counts as a hit when its distance is no greater than
/// the largest distance in `truth[0, k)`. This is the ANN-Benchmarks
/// convention, and on SIFT1M it is not optional: 14,538 of the corpus's
/// 1,000,000 vectors are byte-identical duplicates of another vector (1.45%,
/// 985,462 distinct). When a duplicate lands on the k-th boundary, "the k
/// nearest neighbours" is not a well-defined *set* — several equally correct
/// answers exist — so comparing ids against whichever one the reference
/// generator happened to pick measures the tie-break convention, not the search.
///
/// This is deliberately not the same thing as loosening the metric with an
/// epsilon to make a number look better, which `RecallDiagnosis` exists to
/// avoid. The distinction is that this was reached for only *after*
/// `diagnose_recall` proved every shortfall sat exactly on the boundary and the
/// tied vectors were confirmed bit-identical. Report both numbers: the strict
/// set recall says how often the tie-break differed, and this one says whether
/// the search was actually right.
///
/// `computer` must already have the query prepared that produced `got`.
[[nodiscard]] double recall_at_k_tied(const DistanceComputer& computer,
                                     std::span<const Neighbor> got,
                                     std::span<const std::int32_t> truth);

/// Why a recall figure came out below 1.000.
///
/// The honest response to imperfect recall is not to loosen the metric into an
/// epsilon comparison — that would hide real bugs alongside benign ties. It is
/// to look at the numbers and decide which one it is, which is what this
/// carries.
///
/// Two distinct causes produce the same symptom, and the distance values are
/// what separate them:
///
///  * A genuine tie at the k-th position. Two vectors are equidistant, both
///    answers are correct, and whichever the reference generator happened to
///    pick differs from ours. `boundary_tie` is set.
///  * float32 accumulation. Our sum over 128 terms is not bit-identical to the
///    reference implementation's, so two neighbours whose true distances differ
///    by less than float32 can be *ordered differently* by us. This is not a
///    bug in the search — it is the precision floor of the arithmetic — and it
///    also shows as a near-zero gap.
///
/// Anything else, with a gap that is large relative to the distances involved,
/// is a real defect. Hence both raw values are reported rather than a verdict.
struct RecallDiagnosis {
  double recall = 0.0;

  /// In `truth[0, k)` but absent from what we returned.
  std::vector<std::int32_t> missed;

  /// Returned by us but absent from `truth[0, k)`.
  std::vector<VectorId> extra;

  /// Largest distance among the neighbours we kept.
  float worst_kept = 0.0F;

  /// Smallest distance among the ids we missed, recomputed with our own
  /// kernel. Only meaningful when `missed` is non-empty.
  float best_missed = 0.0F;

  /// `worst_kept` and `best_missed` agree to within float tolerance, meaning
  /// the disagreement sits exactly on the k-th boundary and is a tie or a
  /// precision artefact rather than a wrong answer.
  bool boundary_tie = false;
};

/// Explain a recall figure. `computer` must already have the same query
/// prepared that produced `got`, since the missed ids' distances are recomputed
/// against it.
Status diagnose_recall(const DistanceComputer& computer, std::span<const Neighbor> got,
                       std::span<const std::int32_t> truth, RecallDiagnosis& out);

/// Every ground-truth id must index into the corpus.
///
/// Checked rather than assumed: a ground-truth file paired with the wrong base
/// file is a mistake that otherwise shows up as a low recall number and gets
/// blamed on the algorithm.
Status validate_ground_truth_ids(std::span<const std::int32_t> ids, std::size_t corpus_size);

/// One ground-truth row's distances must be non-decreasing under *our* kernel.
///
/// If the provided ordering disagrees with our own metric, our metric disagrees
/// with the one the file was generated under — a squared-versus-plain L2 slip,
/// say. Catching that here beats discovering it as an unexplained 0.97.
///
/// `computer` must already have that row's query prepared. Tolerates equal
/// adjacent distances, and tolerates a decrease within float tolerance, since
/// float32 accumulation can invert two neighbours that are genuinely that
/// close.
Status validate_ground_truth_order(const DistanceComputer& computer,
                                   std::span<const std::int32_t> row);

} // namespace lodestone
