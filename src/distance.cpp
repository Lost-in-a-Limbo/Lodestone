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

namespace {

/// AVX2 *and* FMA, because they are separate CPUID feature bits and the kernel
/// uses `_mm256_fmadd_ps`. A machine with AVX2 but no FMA is exotic, and
/// checking one bit for a kernel that needs two is exactly the sort of thing
/// that works on every machine you own and crashes on someone else's.
///
/// `__builtin_cpu_supports` rather than hand-rolled CPUID: it needs no inline
/// asm, and its AVX path already accounts for the OS having enabled XSAVE state
/// for the YMM registers. A raw CPUID feature bit does not, and would happily
/// select AVX2 on a kernel that does not preserve those registers across a
/// context switch.
bool cpu_supports_avx2() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
#else
  return false;
#endif
}

} // namespace

KernelKind detected_kernel() {
  // The whole of Phase 2's runtime dispatch, in one function body, with no
  // caller changes anywhere — which is what D15's seam was shaped to deliver.
  if (cpu_supports_avx2()) {
    return KernelKind::avx2;
  }

  // SSE needs no check. Every x86-64 CPU has SSE2 by definition of the
  // architecture, and distance_sse.cpp uses nothing beyond it.
#if defined(__x86_64__) || defined(_M_X64)
  return KernelKind::sse;
#else
  // Not x86-64 at all: the SIMD files would not have compiled, so scalar is
  // the only kernel that exists.
  return KernelKind::scalar;
#endif
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
    // No feature check: every x86-64 CPU has SSE2 by definition of the
    // architecture, and distance_sse.cpp uses nothing beyond it.
    switch (metric) {
    case Metric::l2:
      return detail::make_sse_l2(store);
    case Metric::inner_product:
      return detail::make_sse_ip(store);
    }
    return nullptr;

  case KernelKind::avx2:
    // Gated on the CPU, even for an *explicit* request. Handing back a kernel
    // full of instructions this machine cannot execute would be an
    // illegal-instruction crash rather than a wrong number — and a crash inside
    // a benchmark loop is a much worse failure than a nullptr at the factory.
    //
    // Task 6 reuses this check in detected_kernel(); it lives here first
    // because correctness cannot wait for the dispatch task.
    if (!cpu_supports_avx2()) {
      return nullptr;
    }
    switch (metric) {
    case Metric::l2:
      return detail::make_avx2_l2(store);
    case Metric::inner_product:
      return detail::make_avx2_ip(store);
    }
    return nullptr;

  case KernelKind::automatic:
    // Unreachable — resolved above. Present so the switch stays exhaustive and
    // a new KernelKind cannot be added without the compiler pointing here.
    return nullptr;
  }

  return nullptr;
}

} // namespace lodestone
