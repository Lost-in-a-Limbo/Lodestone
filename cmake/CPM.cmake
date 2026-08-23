# CPM.cmake — pinned bootstrap
#
# Task 2 of .claude/plans/phase0.md wanted this file *vendored*: the full ~30KB
# of CPM committed, so a fresh clone configures with no network at all. The
# Phase 0 session had no outbound network from the shell, so this is the
# official bootstrap shim instead. It fetches exactly one pinned version, once,
# into a shared source cache.
#
# The pinning guarantee — the reason vendoring was wanted — is preserved: a
# dependency manager that silently changes version under you is worse than no
# dependency manager. What is lost is only the offline-cold-clone property.
#
# TODO(phase1): after the first successful configure, copy the downloaded file
# from ${CPM_DOWNLOAD_LOCATION} over this one to complete the vendoring, and
# fill in CPM_HASH_SUM below from the release page.

set(CPM_DOWNLOAD_VERSION 0.40.2)

# Leave empty until the hash is read off the upstream release page and
# verified. An unverified hash is worse than none — it looks like a guarantee.
set(CPM_HASH_SUM "")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

get_filename_component(CPM_DOWNLOAD_DIRECTORY "${CPM_DOWNLOAD_LOCATION}" DIRECTORY)
file(MAKE_DIRECTORY "${CPM_DOWNLOAD_DIRECTORY}")

function(lodestone_fetch_cpm)
  if(EXISTS "${CPM_DOWNLOAD_LOCATION}")
    return()
  endif()

  message(STATUS "lodestone: downloading CPM v${CPM_DOWNLOAD_VERSION}")

  set(download_args
      "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
      "${CPM_DOWNLOAD_LOCATION}"
      STATUS download_status
      TLS_VERIFY ON)
  if(NOT CPM_HASH_SUM STREQUAL "")
    list(APPEND download_args EXPECTED_HASH "SHA256=${CPM_HASH_SUM}")
  endif()

  file(DOWNLOAD ${download_args})

  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    list(GET download_status 1 status_string)
    # Remove the partial file, or the EXISTS check above will happily reuse a
    # truncated CPM on the next configure and fail somewhere far less obvious.
    file(REMOVE "${CPM_DOWNLOAD_LOCATION}")
    message(FATAL_ERROR
            "lodestone: failed to download CPM v${CPM_DOWNLOAD_VERSION}: ${status_string}")
  endif()
endfunction()

lodestone_fetch_cpm()

include("${CPM_DOWNLOAD_LOCATION}")
