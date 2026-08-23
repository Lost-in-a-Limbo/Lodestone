// Phase 1 — kernel selection.
//
// This file exists so that the choice of kernel lives in exactly one function.
// It is compiled with the project's ordinary flags: no -fno-tree-vectorize, no
// -mavx2. Dispatch code must run on any machine, including one that lacks the
// instruction set it is about to select — which is precisely why Phase 2's CPU
// feature detection belongs here and not in distance_avx2.cpp.

#include "lodestone/distance.hpp"
#include "lodestone/vector_store.hpp"

#include <memory>

namespace lodestone {

std::unique_ptr<DistanceComputer> make_distance_computer(Metric metric,
                                                         const VectorStore& store) {
  // A store with no dimension has nothing to compute over, and a computer
  // built against one would produce zero for every pair — a plausible number,
  // which is the failure mode this project keeps trying to eliminate.
  if (store.dim() == 0) {
    return nullptr;
  }

  switch (metric) {
  case Metric::l2:
    // Phase 2 replaces this line with a feature check choosing between the
    // scalar, SSE and AVX2 kernels. No caller changes when it does.
    return detail::make_scalar_l2(store);

  case Metric::inner_product:
    return nullptr; // Phase 2
  }

  return nullptr;
}

} // namespace lodestone
