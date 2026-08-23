#pragma once

// Phase 3 — Hierarchical Navigable Small World index.
//
// Placeholder. The implementation follows Malkov & Yashunin, arXiv 1603.09320,
// Algorithms 1-5, one algorithm per task, each with its own test:
//
//   Alg 1  INSERT
//   Alg 2  SEARCH-LAYER
//   Alg 3  SELECT-NEIGHBORS-SIMPLE   (kept only as a comparison baseline)
//   Alg 4  SELECT-NEIGHBORS-HEURISTIC (the one that matters for recall)
//   Alg 5  K-NN-SEARCH
//
// Nothing here yet on purpose: writing the class shape before the algorithms
// are understood is how the distance seam gets bypassed. The one thing already
// decided is that the constructor takes a DistanceComputer& and the class
// never names a kernel.

#include "lodestone/distance.hpp"
#include "lodestone/types.hpp"
#include "lodestone/vector_store.hpp"

namespace lodestone {

class HnswIndex;

} // namespace lodestone
