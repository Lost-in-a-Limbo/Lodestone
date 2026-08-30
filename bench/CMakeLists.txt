CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.9.0
  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
          "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_ENABLE_GTEST_TESTS OFF")

CPMAddPackage(
  NAME nlohmann_json
  GITHUB_REPOSITORY nlohmann/json
  VERSION 3.11.3
  OPTIONS "JSON_BuildTests OFF")

# hnswlib is the comparison baseline and is header-only. DOWNLOAD_ONLY so its
# CMakeLists never runs: no inherited flags, no example targets, no test
# targets. Added SYSTEM so this project's -Wconversion has no opinions about
# somebody else's headers.
#
# Architecture rule 3: this appears in bench/ and nowhere else. It is never
# linked into src/, and lodestone_core does not know it exists.
CPMAddPackage(
  NAME hnswlib
  GITHUB_REPOSITORY nmslib/hnswlib
  GIT_TAG v0.8.0
  DOWNLOAD_ONLY YES)

add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE lodestone_core lodestone_options
                                             benchmark::benchmark)

# Google Benchmark's headers are not compiled under this project's warning set —
# they are somebody else's code and -Wconversion has opinions about their macros.
if(benchmark_SOURCE_DIR)
  target_include_directories(bench_distance SYSTEM
                             PRIVATE "${benchmark_SOURCE_DIR}/include")
endif()

# PRD section 7 names this binary `bench`. It is the one the exit criterion
# invokes: `./bench --all` must produce results.json with no manual steps.
add_executable(bench bench_main.cpp)
target_link_libraries(bench PRIVATE lodestone_core lodestone_options
                                    nlohmann_json::nlohmann_json)
if(hnswlib_SOURCE_DIR)
  target_include_directories(bench SYSTEM PRIVATE "${hnswlib_SOURCE_DIR}")
  target_compile_definitions(bench PRIVATE LODESTONE_HAVE_HNSWLIB=1)
endif()

# The exact flags this binary was compiled with, so results.json can record them
# rather than a human guessing. A number without its build flags is not a result.
target_compile_definitions(
  bench PRIVATE "LODESTONE_BUILD_FLAGS=\"${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}\"")

# Deliberately NOT registered with add_test(). CI must never time anything —
# GitHub's runners are shared Intel hosts and a performance gate there would be
# measuring a machine that is not the one publishing numbers (DECISIONS.md D11).
