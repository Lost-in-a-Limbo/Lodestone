add_executable(sift_check sift_check.cpp)
target_link_libraries(sift_check PRIVATE lodestone_core lodestone_options)

if(LODESTONE_BUILD_TESTS)
  # SIFT10K only. SIFT1M is a manual release-preset run: scalar squared L2 over
  # 10,000 queries x 1,000,000 vectors x 128 dims is roughly 1.3e12
  # multiply-adds on a deliberately un-vectorised baseline, which is minutes,
  # not something to put in a test suite.
  set(_siftsmall "${CMAKE_SOURCE_DIR}/data/siftsmall")

  add_test(
    NAME sift10k_recall_is_exactly_one
    COMMAND
      sift_check "${_siftsmall}/siftsmall_base.fvecs"
      "${_siftsmall}/siftsmall_query.fvecs"
      "${_siftsmall}/siftsmall_groundtruth.ivecs")

  # The tool returns 4 when the dataset is absent, which ctest then reports as
  # skipped rather than failed. CI has no data/ and must stay green; a laptop
  # that has run ./tools/download_sift.sh gets the real assertion. Same code
  # Catch2's own discovery uses, so "4 means skip" holds across the whole suite.
  set_tests_properties(sift10k_recall_is_exactly_one PROPERTIES SKIP_RETURN_CODE 4)
endif()
