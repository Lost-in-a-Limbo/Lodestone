CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.9.0
  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
          "BENCHMARK_ENABLE_WERROR OFF" "BENCHMARK_ENABLE_GTEST_TESTS OFF")

add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE lodestone_core lodestone_options
                                             benchmark::benchmark)

# Google Benchmark's headers are not compiled under this project's warning set —
# they are somebody else's code and -Wconversion has opinions about their macros.
if(benchmark_SOURCE_DIR)
  target_include_directories(bench_distance SYSTEM
                             PRIVATE "${benchmark_SOURCE_DIR}/include")
endif()

# Deliberately NOT registered with add_test(). CI must never time anything —
# GitHub's runners are shared Intel hosts and a performance gate there would be
# measuring a machine that is not the one publishing numbers (DECISIONS.md D11).
# This target is built by the `release` preset and run by hand.
