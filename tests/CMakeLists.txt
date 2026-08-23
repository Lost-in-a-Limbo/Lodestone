CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.5.2
  OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF")

# catch_discover_tests() lives in Catch2's extras/, which CPM leaves in the
# source tree whether or not it is installed.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

# One executable that accumulates test files across phases, rather than one
# binary per file. Catch2 v3 is a compiled library, so N binaries means N links
# against it; catch_discover_tests() still registers every TEST_CASE as its own
# ctest entry, so nothing is lost in granularity.
add_executable(lodestone_tests test_types.cpp test_distance.cpp test_io.cpp
                               test_vector_store.cpp)

target_link_libraries(lodestone_tests PRIVATE lodestone_core lodestone_options
                                              Catch2::Catch2WithMain)

# Registers each TEST_CASE as its own ctest entry, keyed on the test's *name*.
# Catch2 [tags] do not become ctest labels — Catch.cmake has no option for it
# as of 3.7.1, and passing one is silently ignored rather than an error. So to
# run a single area, filter on the binary instead of through ctest:
#
#   ./build/debug/tests/lodestone_tests "[vector_store]"
#
# catch_discover_tests also sets SKIP_RETURN_CODE 4 on every entry, which is
# what Phase 1 Task 6 will use to skip the SIFT check when data/ is absent.
catch_discover_tests(lodestone_tests)
