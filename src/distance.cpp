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
#include <string_view>

namespace lodestone {

KernelKind detected_kernel() {
  // Task 6 of the Phase 2 plan replaces this with __builtin_cpu_supports()
  // checks for avx2 and fma. Until the SIMD kernels exist there is nothing to
  // detect: reporting avx2 here while make_distance_computer() could only
  // build scalar would make `automatic` return nullptr for every caller.
  return KernelKind::scalar;
}

std::string_view kernel_name(KernelKind kind) {
  switch (kind) {
  case KernelKind::automatic:
    return "automatic";
  case KernelKind::scalar:
    return "scalar";
  case KernelKind::sse:
    return "sse";
  case KernelKind::avx2:
    return "avx2";
  }
  return "unknown";
}

std::unique_ptr<DistanceComputer> make_distance_computer(Metric metric,
                                                         const VectorStore& store,
                                                         KernelKind kind) {
  // A store with no dimension has nothing to compute over, and a computer
  // built against one would produce zero for every pair — a plausible number,
  // which is the failure mode this project keeps trying to eliminate.
  if (store.dim() == 0) {
    return nullptr;
  }

  if (kind == KernelKind::automatic) {
    kind = detected_kernel();
  }

  switch (kind) {
  case KernelKind::scalar:
    switch (metric) {
    case Metric::l2:
      return detail::make_scalar_l2(store);
    case Metric::inner_product:
      return detail::make_scalar_ip(store);
    }
    return nullptr;

  case KernelKind::sse:
  case KernelKind::avx2:
    // Tasks 3 and 4. nullptr rather than a silent downgrade to scalar: a
    // benchmark that asked for AVX2 and quietly got scalar would report a
    // speedup of 1.0 and read as a slow kernel rather than a missing one.
    return nullptr;

  case KernelKind::automatic:
    // Unreachable — resolved above. Present so the switch stays exhaustive and
    // a new KernelKind cannot be added without the compiler pointing here.
    return nullptr;
  }

  return nullptr;
}

} // namespace lodestone
