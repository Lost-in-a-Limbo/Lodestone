// Phase 2 — scalar L2 and inner-product baseline.
//
// Stub. The kernels land in Phase 2; this file exists in Phase 0 to prove the
// library links and that the per-file compile flags actually reach it.

#include "lodestone/distance.hpp"

// This file is the *baseline* in "AVX2 is >=3x scalar". Under the release
// preset's -O3 -march=native, GCC will happily auto-vectorise a plain L2 loop
// into AVX2, at which point Phase 2's headline number is AVX2-vs-AVX2 and
// means nothing at all. CMakeLists.txt pins -fno-tree-vectorize on this one
// file and passes this define from the same property, so the guard below fails
// the build if the flag is ever dropped.
#ifndef LODESTONE_SCALAR_NOVEC
#error "distance_scalar.cpp must be compiled with -fno-tree-vectorize. See the \
set_source_files_properties() call in CMakeLists.txt — removing it silently \
turns the scalar baseline into a vectorised one and invalidates every speedup \
number this project publishes."
#endif

namespace lodestone {

// Nothing yet.

} // namespace lodestone
