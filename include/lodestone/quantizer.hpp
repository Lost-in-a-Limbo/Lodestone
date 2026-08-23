#pragma once

// Phase 5 — product quantisation.
//
// Placeholder. What lands here: per-subspace k-means with 256 centroids (so
// one byte per subspace), codebook training, and asymmetric distance with
// precomputed lookup tables.
//
// The whole point of the exercise is that it plugs in as a DistanceComputer
// and hnsw.cpp is not touched at all. If Phase 5 ends up editing the graph,
// the seam in distance.hpp was wrong and that is the thing to fix.

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"

namespace lodestone {

class ProductQuantizer;

} // namespace lodestone
