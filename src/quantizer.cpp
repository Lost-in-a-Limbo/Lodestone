// Phase 5 — product quantization.
//
// Per-subspace k-means with 256 centroids, and asymmetric distance through a
// per-query lookup table. Format and reasoning in quantizer.hpp.
//
// This file is the test of DECISIONS.md D1. The seam was chosen in Phase 0 with
// this exact use named in its justification, and nothing in the graph changes
// to accommodate it.

#include "lodestone/quantizer.hpp"

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace lodestone {

namespace {

/// splitmix64, the same generator the HNSW level assignment uses. Fixed seed,
/// fixed codebook, repeatable benchmark.
std::uint64_t next_random(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double uniform01(std::uint64_t& state) {
  return static_cast<double>(next_random(state) >> 11) / 9007199254740992.0;
}

float squared_distance(const float* a, const float* b, std::size_t n) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/// k-means for one subspace.
///
/// `points` is `count` contiguous rows of `sub_dim` floats. Writes
/// `pq_centroids * sub_dim` floats into `centroids`.
void train_subspace(const float* points, std::size_t count, std::size_t sub_dim,
                    std::size_t iterations, std::uint64_t& rng, std::vector<float>& centroids) {
  centroids.assign(pq_centroids * sub_dim, 0.0F);

  // ---- k-means++ initialisation ------------------------------------------
  //
  // Not random-pick: k-means++ chooses each new centre with probability
  // proportional to its squared distance from the nearest existing one, which
  // spreads the initial centres out. Plain random initialisation on SIFT data
  // routinely puts several centres inside the same dense cluster, and Lloyd's
  // iterations cannot recover from that — they only ever move a centre to the
  // mean of what it already owns, so a centre that starts in a crowded region
  // stays there and the sparse regions go unrepresented.
  std::vector<float> nearest(count, std::numeric_limits<float>::max());
  {
    const auto first = static_cast<std::size_t>(next_random(rng) % count);
    std::copy_n(points + (first * sub_dim), sub_dim, centroids.begin());

    for (std::size_t c = 1; c < pq_centroids; ++c) {
      const float* previous = centroids.data() + ((c - 1) * sub_dim);
      double total = 0.0;
      for (std::size_t i = 0; i < count; ++i) {
        const float d = squared_distance(points + (i * sub_dim), previous, sub_dim);
        nearest[i] = std::min(nearest[i], d);
        total += static_cast<double>(nearest[i]);
      }

      // Sample proportional to squared distance.
      std::size_t chosen = count - 1;
      if (total > 0.0) {
        double target = uniform01(rng) * total;
        for (std::size_t i = 0; i < count; ++i) {
          target -= static_cast<double>(nearest[i]);
          if (target <= 0.0) {
            chosen = i;
            break;
          }
        }
      } else {
        // Every point is already a centre (fewer distinct points than
        // centroids). Cycle rather than looping forever.
        chosen = c % count;
      }
      std::copy_n(points + (chosen * sub_dim), sub_dim, centroids.begin() + (c * sub_dim));
    }
  }

  // ---- Lloyd's iterations -------------------------------------------------
  std::vector<std::uint8_t> assignment(count, 0);
  std::vector<double> sums(pq_centroids * sub_dim, 0.0);
  std::vector<std::size_t> counts(pq_centroids, 0);
  std::vector<float> worst_distance(pq_centroids, 0.0F);
  std::vector<std::size_t> worst_point(pq_centroids, 0);

  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);

    float global_worst = -1.0F;
    std::size_t global_worst_point = 0;

    for (std::size_t i = 0; i < count; ++i) {
      const float* point = points + (i * sub_dim);
      float best = std::numeric_limits<float>::max();
      std::size_t best_c = 0;
      for (std::size_t c = 0; c < pq_centroids; ++c) {
        const float d = squared_distance(point, centroids.data() + (c * sub_dim), sub_dim);
        if (d < best) {
          best = d;
          best_c = c;
        }
      }
      assignment[i] = static_cast<std::uint8_t>(best_c);
      ++counts[best_c];
      double* target = sums.data() + (best_c * sub_dim);
      for (std::size_t d = 0; d < sub_dim; ++d) {
        target[d] += static_cast<double>(point[d]);
      }
      if (best > global_worst) {
        global_worst = best;
        global_worst_point = i;
      }
    }

    for (std::size_t c = 0; c < pq_centroids; ++c) {
      if (counts[c] > 0) {
        float* centre = centroids.data() + (c * sub_dim);
        const double* sum = sums.data() + (c * sub_dim);
        for (std::size_t d = 0; d < sub_dim; ++d) {
          centre[d] = static_cast<float>(sum[d] / static_cast<double>(counts[c]));
        }
        continue;
      }

      // Empty cluster. Re-seed it on the point currently worst served by its
      // own centre. Leaving it empty would waste a code and silently shrink the
      // effective codebook — 255 usable centroids while still paying 8 bits.
      std::copy_n(points + (global_worst_point * sub_dim), sub_dim,
                  centroids.begin() + (c * sub_dim));
      global_worst = -1.0F; // one repair per iteration; the next finds the next
    }
  }
  (void)worst_distance;
  (void)worst_point;
}

class ProductQuantizerImpl final : public ProductQuantizer {
public:
  ProductQuantizerImpl(std::size_t dim, std::size_t subspaces, std::vector<float> codebook)
      : dim_(dim), subspaces_(subspaces), sub_dim_(dim / subspaces),
        codebook_(std::move(codebook)) {}

  [[nodiscard]] Status encode(const VectorStore& corpus) override {
    if (corpus.dim() != dim_) {
      return Status::dimension_mismatch;
    }
    if (corpus.size() == 0) {
      return Status::invalid_argument;
    }

    codes_.assign(corpus.size() * subspaces_, 0);
    for (std::size_t i = 0; i < corpus.size(); ++i) {
      const float* vector = corpus.get(static_cast<VectorId>(i));
      std::uint8_t* out = codes_.data() + (i * subspaces_);
      for (std::size_t s = 0; s < subspaces_; ++s) {
        const float* slice = vector + (s * sub_dim_);
        const float* book = codebook_.data() + (s * pq_centroids * sub_dim_);
        float best = std::numeric_limits<float>::max();
        std::size_t best_c = 0;
        for (std::size_t c = 0; c < pq_centroids; ++c) {
          const float d = squared_distance(slice, book + (c * sub_dim_), sub_dim_);
          if (d < best) {
            best = d;
            best_c = c;
          }
        }
        out[s] = static_cast<std::uint8_t>(best_c);
      }
    }
    count_ = corpus.size();
    return Status::ok;
  }

  [[nodiscard]] std::size_t subspaces() const override { return subspaces_; }
  [[nodiscard]] std::size_t sub_dim() const override { return sub_dim_; }
  [[nodiscard]] std::size_t dim() const override { return dim_; }
  [[nodiscard]] std::size_t size() const override { return count_; }
  [[nodiscard]] std::size_t code_bytes_per_vector() const override { return subspaces_; }
  [[nodiscard]] std::size_t codebook_bytes() const override {
    return codebook_.size() * sizeof(float);
  }
  [[nodiscard]] std::size_t code_bytes() const override { return codes_.size(); }

  [[nodiscard]] std::span<const std::uint8_t> code(VectorId id) const override {
    return {codes_.data() + (static_cast<std::size_t>(id) * subspaces_), subspaces_};
  }

  void reconstruct(VectorId id, std::span<float> out) const override {
    const std::uint8_t* c = codes_.data() + (static_cast<std::size_t>(id) * subspaces_);
    for (std::size_t s = 0; s < subspaces_; ++s) {
      const float* centre =
          codebook_.data() + (s * pq_centroids * sub_dim_) + (std::size_t{c[s]} * sub_dim_);
      std::copy_n(centre, sub_dim_, out.begin() + static_cast<std::ptrdiff_t>(s * sub_dim_));
    }
  }

  [[nodiscard]] double reconstruction_error(const VectorStore& corpus) const override {
    if (corpus.dim() != dim_ || corpus.size() != count_) {
      return -1.0;
    }
    std::vector<float> approximation(dim_);
    double total = 0.0;
    for (std::size_t i = 0; i < count_; ++i) {
      reconstruct(static_cast<VectorId>(i), approximation);
      total += static_cast<double>(
          squared_distance(corpus.get(static_cast<VectorId>(i)), approximation.data(), dim_));
    }
    return total / static_cast<double>(count_);
  }

  [[nodiscard]] const std::vector<float>& codebook() const { return codebook_; }

private:
  std::size_t dim_;
  std::size_t subspaces_;
  std::size_t sub_dim_;
  std::vector<float> codebook_;
  std::vector<std::uint8_t> codes_;
  std::size_t count_ = 0;
};

/// The ADC computer. This is what D1's `prepare_query()` was designed for.
class PqComputer final : public DistanceComputer {
public:
  explicit PqComputer(const ProductQuantizerImpl& quantizer)
      : quantizer_(quantizer), subspaces_(quantizer.subspaces()), sub_dim_(quantizer.sub_dim()),
        dim_(quantizer.dim()), table_(quantizer.subspaces() * pq_centroids, 0.0F) {}

  void prepare_query(const float* query) override {
    // The per-query state that made D1 an object rather than a free function:
    // subspaces x 256 squared distances from the query's slices to every
    // centroid. 4,096 distance computations at m = 16 — real work, but it is
    // paid once and then amortised over however many vectors the search visits.
    const float* book = quantizer_.codebook().data();
    for (std::size_t s = 0; s < subspaces_; ++s) {
      const float* slice = query + (s * sub_dim_);
      const float* sub_book = book + (s * pq_centroids * sub_dim_);
      float* row = table_.data() + (s * pq_centroids);
      for (std::size_t c = 0; c < pq_centroids; ++c) {
        row[c] = squared_distance(slice, sub_book + (c * sub_dim_), sub_dim_);
      }
    }
  }

  [[nodiscard]] float distance_to(VectorId id) const override { return lookup(id); }

  void distances_to(std::span<const VectorId> ids, std::span<float> out) const override {
    assert(ids.size() == out.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      out[i] = lookup(ids[i]);
    }
  }

  [[nodiscard]] std::size_t dim() const override { return dim_; }
  [[nodiscard]] Metric metric() const override { return Metric::l2; }

  /// PQ is not an instruction-set variant, so it reports the kernel the *table
  /// build* used. Phase 4's results.json records this field, and claiming
  /// "avx2" for a lookup-table path would misattribute where the time went.
  [[nodiscard]] KernelKind kernel() const override { return KernelKind::scalar; }

private:
  [[nodiscard]] float lookup(VectorId id) const {
    const std::uint8_t* code = quantizer_.code(id).data();
    const float* table = table_.data();
    float sum = 0.0F;
    // m lookups and m adds, against 128 multiply-adds for the exact kernel.
    // The table is m x 256 floats — 16 KiB at m = 16, so it lives in L1 and the
    // gather is cheap. The real win is upstream: the code is m bytes where the
    // vector was 512, so a scan moves 32x less memory.
    for (std::size_t s = 0; s < subspaces_; ++s) {
      sum += table[(s * pq_centroids) + code[s]];
    }
    return sum;
  }

  const ProductQuantizerImpl& quantizer_;
  std::size_t subspaces_;
  std::size_t sub_dim_;
  std::size_t dim_;
  std::vector<float> table_;
};

} // namespace

std::unique_ptr<ProductQuantizer> train_product_quantizer(const VectorStore& training,
                                                          const PqConfig& config) {
  const std::size_t dim = training.dim();
  if (dim == 0 || training.size() == 0) {
    return nullptr;
  }
  if (config.subspaces == 0 || dim % config.subspaces != 0) {
    return nullptr;
  }
  if (training.size() < pq_centroids) {
    // Fewer training points than centroids means empty clusters by
    // construction, and a codebook with unusable entries costs the same bits
    // while representing less.
    return nullptr;
  }

  const std::size_t sub_dim = dim / config.subspaces;
  std::vector<float> codebook(config.subspaces * pq_centroids * sub_dim, 0.0F);

  // Gather each subspace's slice contiguously before training it. The store is
  // strided by the full vector, so a k-means inner loop reading slice s would
  // touch one cache line per point and use sub_dim floats of it. Repacking
  // costs one pass and makes the training loop sequential.
  std::vector<float> slice(training.size() * sub_dim);
  std::uint64_t rng = config.seed;

  for (std::size_t s = 0; s < config.subspaces; ++s) {
    for (std::size_t i = 0; i < training.size(); ++i) {
      const float* vector = training.get(static_cast<VectorId>(i)) + (s * sub_dim);
      std::copy_n(vector, sub_dim, slice.begin() + static_cast<std::ptrdiff_t>(i * sub_dim));
    }
    std::vector<float> centroids;
    train_subspace(slice.data(), training.size(), sub_dim, config.iterations, rng, centroids);
    std::copy(centroids.begin(), centroids.end(),
              codebook.begin() + static_cast<std::ptrdiff_t>(s * pq_centroids * sub_dim));
  }

  return std::make_unique<ProductQuantizerImpl>(dim, config.subspaces, std::move(codebook));
}

std::unique_ptr<DistanceComputer> make_pq_distance_computer(const ProductQuantizer& quantizer,
                                                            Metric metric) {
  if (metric != Metric::l2) {
    return nullptr; // inner product under PQ needs a different table
  }
  if (quantizer.size() == 0) {
    return nullptr; // nothing encoded yet
  }
  return std::make_unique<PqComputer>(static_cast<const ProductQuantizerImpl&>(quantizer));
}

} // namespace lodestone
