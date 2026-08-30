add_executable(sift_check sift_check.cpp)
target_link_libraries(sift_check PRIVATE lodestone_core lodestone_options)

if(LODESTONE_BUILD_TESTS)
  # SIFT10K only. SIFT1M is a manual release-preset run: scalar squared L2 over
  # 10,000 queries x 1,000,000 vectors x 128 dims is roughly 1.3e12
  # multiply-adds on a deliberately un-vectorised baseline, which is minutes,
  # not something to put in a test suite.
  set(_siftsmall "${CMAKE_SOURCE_DIR}/data/siftsmall")

  # One entry per kernel. This is Phase 2's real correctness gate: a SIMD kernel
  # whose distances agree with scalar to 1e-5 can still return different
  # neighbours, because a reordering only needs the gap between ranks 10 and 11
  # to be smaller than the error. Requiring tie-aware recall@10 = 1.000000
  # through each kernel catches that; a tolerance test does not.
  foreach(_kernel auto scalar sse avx2)
    add_test(
      NAME sift10k_recall_is_exactly_one.${_kernel}
      COMMAND
        sift_check "--kernel=${_kernel}" "${_siftsmall}/siftsmall_base.fvecs"
        "${_siftsmall}/siftsmall_query.fvecs"
        "${_siftsmall}/siftsmall_groundtruth.ivecs")
    # 4 means skip: no dataset, or a kernel this CPU cannot run. CI has neither
    # a dataset nor a guaranteed AVX2 host and must stay green either way.
    set_tests_properties(sift10k_recall_is_exactly_one.${_kernel}
                         PROPERTIES SKIP_RETURN_CODE 4)
  endforeach()

endif()
