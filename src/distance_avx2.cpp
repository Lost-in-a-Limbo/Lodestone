// Phase 2 — AVX2 L2 and inner-product kernels.
//
// Stub. The intrinsics land in Phase 2; this file exists in Phase 0 to prove
// that its per-file -mavx2 -mfma actually applies.

#include "lodestone/distance.hpp"

#include <immintrin.h>

// AVX2 and FMA are scoped to this single translation unit, never enabled
// globally. Global -mavx2 would let the compiler emit AVX2 anywhere — in the
// scalar baseline, and in the dispatch code that is supposed to run on a
// machine that lacks it, which would defeat Phase 2's runtime feature
// detection by crashing before it could dispatch.
#ifndef __AVX2__
#error "distance_avx2.cpp must be compiled with -mavx2. See CMakeLists.txt."
#endif
#ifndef __FMA__
#error "distance_avx2.cpp must be compiled with -mfma. See CMakeLists.txt."
#endif

namespace lodestone {

// Nothing yet.

} // namespace lodestone
