#pragma once

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace lodestone {

/// Product quantization.
///
/// Split each vector into `subspaces` contiguous slices of `dim/subspaces`
/// dimensions, run k-means with 256 centroids independently in each slice, and
/// store one byte per slice: the index of the nearest centroid. A 128-dim
/// float32 vector (512 bytes) becomes `subspaces` bytes.
///
/// **Asymmetric distance (ADC) is what this implements, and the asymmetry is
/// the whole trick: only one side of the comparison is approximated.** The
/// query stays full precision. `prepare_query()` computes, for every subspace
/// and every centroid, the squared distance from the query's slice to that
/// centroid — a `subspaces × 256` table. Every distance after that is
/// `subspaces` lookups and adds:
///
///     d(q, x) ≈ Σ_s table[s][code_x[s]]
///
/// **Symmetric distance (SDC) is the alternative and is not implemented.** It
/// quantizes the query too and uses a 256×256 inter-centroid table per
/// subspace, precomputed once at training time — so `prepare_query()` becomes
/// free. The cost is that the query's own quantization error is added to the
/// stored vector's, and accuracy is meaningfully worse for the same code size.
/// ADC's table costs `subspaces × 256` distance computations per query — 4,096
/// at m = 16 — which is real but amortises over a million distances.
struct PqConfig {
  /// m. Must divide the store's dimension exactly.
  std::size_t subspaces = 16;

  /// Lloyd's iterations per subspace. Fixed rather than convergence-tested:
  /// a codebook you cannot rebuild identically is a benchmark you cannot repeat.
  std::size_t iterations = 20;

  /// Seeds k-means++ initialisation. Same seed, same codebook, same numbers.
  std::uint64_t seed = 100;
};

/// Centroids per subspace. Fixed at 256 because that is exactly what makes a
/// code one byte — 512 would need 9 bits and the packing would cost more than
/// the extra precision saves.
inline constexpr std::size_t pq_centroids = 256;

class ProductQuantizer {
public:
  virtual ~ProductQuantizer() = default;

  ProductQuantizer(const ProductQuantizer&) = delete;
  ProductQuantizer& operator=(const ProductQuantizer&) = delete;
  ProductQuantizer(ProductQuantizer&&) = delete;
  ProductQuantizer& operator=(ProductQuantizer&&) = delete;

  /// Encode every vector in `corpus` to its `subspaces`-byte code. The corpus
  /// must have the dimension the quantizer was trained at.
  [[nodiscard]] virtual Status encode(const VectorStore& corpus) = 0;

  [[nodiscard]] virtual std::size_t subspaces() const = 0;
  [[nodiscard]] virtual std::size_t sub_dim() const = 0;
  [[nodiscard]] virtual std::size_t dim() const = 0;

  /// How many vectors have been encoded.
  [[nodiscard]] virtual std::size_t size() const = 0;

  /// Bytes per encoded vector. Equals `subspaces()`.
  [[nodiscard]] virtual std::size_t code_bytes_per_vector() const = 0;

  /// The codebook: `subspaces × 256 × sub_dim` floats. Fixed cost, independent
  /// of corpus size — 128 KiB at m = 16, which is why it is worth reporting
  /// separately from the codes.
  [[nodiscard]] virtual std::size_t codebook_bytes() const = 0;

  /// Total encoded corpus, `size() * subspaces()` bytes.
  [[nodiscard]] virtual std::size_t code_bytes() const = 0;

  [[nodiscard]] virtual std::span<const std::uint8_t> code(VectorId id) const = 0;

  /// The centroid a code maps back to. For tests and for measuring
  /// reconstruction error directly, which is the cleanest way to see what the
  /// compression actually cost before any search is involved.
  virtual void reconstruct(VectorId id, std::span<float> out) const = 0;

  /// Mean squared distance between the original vectors and their
  /// reconstructions. The compression's error, with no search in the way.
  [[nodiscard]] virtual double reconstruction_error(const VectorStore& corpus) const = 0;

protected:
  ProductQuantizer() = default;
};

/// Train a codebook on `training`.
///
/// Train on a *held-out* set, not on the corpus being encoded. The dataset
/// ships `sift_learn.fvecs` for exactly this reason; training a codebook on the
/// vectors it will encode is a mild form of testing on the training set.
///
/// Returns nullptr if the config cannot work: subspaces not dividing the
/// dimension, an empty training set, or fewer training vectors than centroids.
[[nodiscard]] std::unique_ptr<ProductQuantizer> train_product_quantizer(const VectorStore& training,
                                                                        const PqConfig& config);

/// An ADC `DistanceComputer` over an encoded corpus.
///
/// This is the seam D1 was designed for, four phases ago, with this exact use
/// named in its justification: `prepare_query()` exists because PQ needs
/// per-query state that a free function `f(a, b, dim)` has nowhere to put.
///
/// L2 only for now. Inner product under PQ needs a different table (dot
/// products rather than squared distances) and nothing in the project asks for
/// it yet.
[[nodiscard]] std::unique_ptr<DistanceComputer>
make_pq_distance_computer(const ProductQuantizer& quantizer, Metric metric = Metric::l2);

} // namespace lodestone
