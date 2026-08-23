cmake_minimum_required(VERSION 3.21)

project(
  lodestone
  VERSION 0.1.0
  DESCRIPTION "A filtered approximate nearest neighbour index"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -std=c++20, not -std=gnu++20. A GNU extension that sneaks in unnoticed is a
# porting bug found months later, in a phase that has nothing to do with it.
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LODESTONE_BUILD_TESTS "Build the Catch2 test suite" ON)
option(LODESTONE_BUILD_BENCH "Build Google Benchmark microbenchmarks" OFF)

include(CPM)

# ---------------------------------------------------------------------------
# Probe: is std::expected usable?
# ---------------------------------------------------------------------------
# CLAUDE.md says fallible operations return std::expected or an error enum.
# std::expected is C++23, and libstdc++ gates <expected> behind
# __cplusplus > 202002L — so under -std=c++20 it is unavailable on *every* GCC,
# not merely old ones. Probing rather than assuming means the configure log
# records the answer, instead of Phase 3 discovering it the hard way.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles(
  "
  #include <expected>
  int main() { std::expected<int, int> e{42}; return e.value() - 42; }
  "
  LODESTONE_HAS_STD_EXPECTED)
unset(CMAKE_REQUIRED_FLAGS)

if(LODESTONE_HAS_STD_EXPECTED)
  message(STATUS "lodestone: <expected> is available under C++20")
else()
  message(STATUS "lodestone: no <expected> under C++20 — the error channel is "
                 "lodestone::Status plus std::optional")
endif()

# ---------------------------------------------------------------------------
# Warnings, carried by one interface target
# ---------------------------------------------------------------------------
# Every target links this, so the warning set is defined in exactly one place.
# -Wconversion earns its keep in this project specifically: the codebase mixes
# size_t indices, uint32_t ids, int SIMD lane counts and float distances, and a
# silent narrowing in a distance kernel is a recall bug that no test names.
add_library(lodestone_options INTERFACE)
target_compile_options(
  lodestone_options
  INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
# Sources are listed explicitly, never globbed: a glob is evaluated at
# configure time, so a newly added file silently does not build until someone
# reconfigures for an unrelated reason.
add_library(
  lodestone_core STATIC
  src/distance.cpp
  src/distance_scalar.cpp
  src/distance_avx2.cpp
  src/io.cpp
  src/vector_store.cpp
  src/hnsw.cpp
  src/filter.cpp
  src/quantizer.cpp)

target_include_directories(lodestone_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(lodestone_core PRIVATE lodestone_options)

if(LODESTONE_HAS_STD_EXPECTED)
  target_compile_definitions(lodestone_core PUBLIC LODESTONE_HAS_STD_EXPECTED=1)
endif()

# ---------------------------------------------------------------------------
# Per-file compile flags — this is what protects Phase 2's headline number
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The release preset builds with -O3 -march=native, under which GCC happily
  # auto-vectorises the "scalar" L2 loop into AVX2. Phase 2's exit criterion is
  # "AVX2 is >=3x scalar" — measured against an auto-vectorised baseline that
  # comparison is AVX2-vs-AVX2 and means nothing. Isolating the file now means
  # Phase 2's measurement is valid the first time instead of being re-run after
  # someone notices the speedup is suspiciously close to 1.0.
  #
  # The -D rides along in COMPILE_OPTIONS rather than COMPILE_DEFINITIONS on
  # purpose: same property, so the two cannot drift apart. distance_scalar.cpp
  # #errors if the define is missing, which means deleting the flag breaks the
  # build loudly instead of quietly inflating Phase 2's baseline.
  set_source_files_properties(
    src/distance_scalar.cpp PROPERTIES COMPILE_OPTIONS
                                       "-fno-tree-vectorize;-DLODESTONE_SCALAR_NOVEC=1")

  # AVX2 + FMA are scoped to the one file that uses the intrinsics, not applied
  # globally. Global -mavx2 would let the compiler emit AVX2 anywhere,
  # including in the scalar baseline and in code paths that are supposed to run
  # on a machine without it — which defeats Phase 2's runtime dispatch entirely.
  set_source_files_properties(
    src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# ---------------------------------------------------------------------------
# Tests and benchmarks
# ---------------------------------------------------------------------------
if(LODESTONE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(LODESTONE_BUILD_BENCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt")
  add_subdirectory(bench)
endif()
