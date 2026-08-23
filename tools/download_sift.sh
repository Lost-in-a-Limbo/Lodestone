#!/usr/bin/env bash
#
# Fetch the TEXMEX SIFT corpora into data/.
#
#   ./tools/download_sift.sh            # SIFT10K only (~5 MB) — enough for ctest
#   ./tools/download_sift.sh --full     # adds SIFT1M (~170 MB) — needed to close Phase 1
#
# Two independent integrity checks, and both are load-bearing.
#
# 1. The upstream MD5, published alongside the archives at
#    ftp://ftp.irisa.fr/local/texmex/corpus/MD5SUM. Verified before extracting.
#
# 2. Exact extracted file sizes, which are *arithmetic* rather than trusted
#    from a third party. A .fvecs record is a 4-byte dimension plus dim*4 bytes
#    of payload, so a 128-dimensional file is exactly 516 bytes per vector and
#    nothing else. Both checks would have to fail in the same direction for bad
#    data to get through.
#
# Why bother, when the parser already validates? Because there is one
# corruption the format cannot detect: a file truncated by a *whole* number of
# records is indistinguishable from a shorter valid file, since the record count
# is derived from the length. See include/lodestone/io.hpp. A silently short
# sift_base.fvecs is exactly the input that produces a plausible recall of 0.98
# and costs an evening.

set -euo pipefail

readonly MIRROR="ftp://ftp.irisa.fr/local/texmex/corpus"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DATA_DIR="${REPO_ROOT}/data"

WANT_FULL=0
KEEP_ARCHIVE=0

log() { printf '  %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Fetch the TEXMEX SIFT corpora into data/.

Usage: tools/download_sift.sh [--full] [--keep-archive]

  (no args)        SIFT10K only, ~5 MB. Enough for the test suite.
  --full           Also fetch SIFT1M, ~170 MB. Phase 1 needs it to close.
  --keep-archive   Leave the .tar.gz in place after extracting.
  -h, --help       This message.

Downloads are verified twice: against the upstream MD5, and against exact
extracted file sizes derived from the record format. Re-running is safe and
skips anything already present and verified.
EOF
  exit 0
}

# ---------------------------------------------------------------------------
# Dataset facts
# ---------------------------------------------------------------------------

# Published upstream in the corpus MD5SUM file.
dataset_md5() {
  case "$1" in
    siftsmall) echo "0b8324a7a82d7f2663d7dcbd57642df7" ;;
    sift)      echo "b23d1b3b2ee8469d819b61ca900ef0ed" ;;
    *)         die "unknown dataset: $1" ;;
  esac
}

# "<path relative to data/> <exact size in bytes>", one per line.
#
# Every size below is derived, not observed:
#   .fvecs at dim 128       -> 4 + 128*4 = 516 bytes per vector
#   .ivecs ground truth     -> 4 + 100*4 = 404 bytes per row (top-100 per query)
#
#   SIFT10K   10,000 base   x 516 =    5,160,000     100 queries x 516 =    51,600
#             25,000 learn  x 516 =   12,900,000     100 rows    x 404 =    40,400
#   SIFT1M 1,000,000 base   x 516 =  516,000,000  10,000 queries x 516 = 5,160,000
#           100,000 learn   x 516 =   51,600,000  10,000 rows    x 404 = 4,040,000
dataset_files() {
  case "$1" in
    siftsmall)
      cat <<'EOF'
siftsmall/siftsmall_base.fvecs 5160000
siftsmall/siftsmall_query.fvecs 51600
siftsmall/siftsmall_learn.fvecs 12900000
siftsmall/siftsmall_groundtruth.ivecs 40400
EOF
      ;;
    sift)
      cat <<'EOF'
sift/sift_base.fvecs 516000000
sift/sift_query.fvecs 5160000
sift/sift_learn.fvecs 51600000
sift/sift_groundtruth.ivecs 4040000
EOF
      ;;
    *) die "unknown dataset: $1" ;;
  esac
}

# ---------------------------------------------------------------------------
# Portability shims
# ---------------------------------------------------------------------------

file_size() {
  # GNU stat, then BSD/macOS stat.
  stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    die "neither md5sum nor md5 is available; cannot verify downloads"
  fi
}

download() {
  local url="$1" dest="$2"
  # A progress bar when a human is watching, silence in CI.
  local progress="-sS"
  [[ -t 2 ]] && progress="--progress-bar"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${progress}" --retry 5 --retry-delay 2 -o "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${dest}" "${url}"
  else
    die "neither curl nor wget is available"
  fi
}

# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------

# Success means every expected file exists at exactly its expected size.
verify_extracted() {
  local dataset="$1" rel expected actual path
  while read -r rel expected; do
    path="${DATA_DIR}/${rel}"
    [[ -f "${path}" ]] || return 1
    actual="$(file_size "${path}")"
    [[ "${actual}" == "${expected}" ]] || return 1
  done < <(dataset_files "${dataset}")
  return 0
}

# Same walk, but reports precisely what is wrong. Split from the silent version
# so the idempotency check stays quiet while a real failure stays loud.
report_extracted() {
  local dataset="$1" rel expected actual path bad=0
  while read -r rel expected; do
    path="${DATA_DIR}/${rel}"
    if [[ ! -f "${path}" ]]; then
      log "MISSING  ${rel}"
      bad=1
      continue
    fi
    actual="$(file_size "${path}")"
    if [[ "${actual}" != "${expected}" ]]; then
      log "WRONG SIZE ${rel}: got ${actual} bytes, expected ${expected}"
      bad=1
    else
      log "ok  ${rel}  (${actual} bytes)"
    fi
  done < <(dataset_files "${dataset}")
  return "${bad}"
}

fetch_dataset() {
  local dataset="$1"
  local archive="${DATA_DIR}/${dataset}.tar.gz"
  local want got

  if verify_extracted "${dataset}"; then
    log "${dataset}: already present and verified, skipping"
    return 0
  fi

  want="$(dataset_md5 "${dataset}")"

  if [[ -f "${archive}" ]] && [[ "$(md5_of "${archive}")" == "${want}" ]]; then
    log "${dataset}: archive already downloaded and verified"
  else
    # A partial file from an interrupted run would otherwise be reused and fail
    # the checksum on every subsequent attempt.
    rm -f "${archive}"
    log "${dataset}: downloading from ${MIRROR}"
    download "${MIRROR}/${dataset}.tar.gz" "${archive}"

    got="$(md5_of "${archive}")"
    if [[ "${got}" != "${want}" ]]; then
      rm -f "${archive}"
      die "${dataset}.tar.gz md5 mismatch: got ${got}, expected ${want}"
    fi
    log "${dataset}: md5 ok (${want})"
  fi

  log "${dataset}: extracting"
  tar xzf "${archive}" -C "${DATA_DIR}"

  if ! report_extracted "${dataset}"; then
    die "${dataset}: extracted files failed the size check — refusing to leave bad data in place"
  fi

  if [[ "${KEEP_ARCHIVE}" -eq 0 ]]; then
    rm -f "${archive}"
  fi
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --full)         WANT_FULL=1 ;;
      --keep-archive) KEEP_ARCHIVE=1 ;;
      -h|--help)      usage ;;
      *)              die "unknown option: $1 (try --help)" ;;
    esac
    shift
  done

  mkdir -p "${DATA_DIR}"

  fetch_dataset siftsmall
  if [[ "${WANT_FULL}" -eq 1 ]]; then
    fetch_dataset sift
  else
    log "SIFT1M skipped — pass --full to fetch it (Phase 1 needs it to close)"
  fi

  log "data is in ${DATA_DIR}"
}

main "$@"
