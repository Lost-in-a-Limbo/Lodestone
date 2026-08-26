# Benchmarks

Every number in this file is produced by a tool in this repo and copied here.
**Nothing is typed by hand** — if a number appears here without a command that
regenerates it, that is a bug in the process, not a result.

From Phase 4 onward the generating tool is `bench/`, writing
`bench/results/results.json`. Phase 1's numbers come from
`tools/sift_check`, which is a correctness check that happens to time itself —
see `DECISIONS.md` D18 for why that is not the benchmark harness.

## Methodology — fixed for the whole project

- Single-threaded queries unless a row explicitly says otherwise
- 10-second warmup, discarded
- Minimum 10,000 queries per measurement
- Three runs, median reported
- Machine spec captured in every result
- **Losses are published.** Where Lodestone is slower than `hnswlib`, the row
  says so and the note says why.

CI never asserts on timing. GitHub's runners are not the machine publishing
these numbers — see `DECISIONS.md` D11.

### Stated deviations

Any departure from the rules above is listed here rather than left implicit.

| Phase | Rule | Deviation and why |
|---|---|---|
| 1 | 10-second warmup discarded | **Skipped.** A brute-force query streams the entire 488 MiB corpus, so there is no working set for a warmup to warm. Three runs still taken, median reported. |

---

## Machines

| Id | CPU | Cores | Cache | RAM | Compiler | Flags |
|---|---|---|---|---|---|---|
| M1 | AMD Ryzen 5 7530U (Zen 3) | 6C / 12T, 4546 MHz max | L1d 192 KiB, L2 3 MiB, L3 16 MiB | 15 GiB | GCC 11.4.0 | `-O3 -march=native`, C++20 |

M1 has AVX2 and FMA and **no AVX-512**, which is why Phase 2 ships SSE + AVX2
only (`IDEAS.md`). Built with CMake 4.3.1 / Ninja 1.13.2, `release` preset.

From Phase 4 this table is filled by the harness from `/proc/cpuinfo` and the
compiler version rather than by hand.

---

## Dataset facts

Properties of the corpora themselves, established while building Phase 1 and
relied on by every recall number in this file.

| Dataset | Vectors | Dim | Queries | Ground truth | Distinct vectors |
|---|---|---|---|---|---|
| SIFT10K | 10,000 | 128 | 100 | top-100 | — |
| SIFT1M | 1,000,000 | 128 | 10,000 | top-100 | **985,462** (14,538 duplicates, 1.4538%) |

The duplicate count is not trivia. It is why recall against SIFT1M ground truth
is reported twice — see `DECISIONS.md` D17.

---

## Phase 1 — load and brute force

Machine M1, `release` preset. Regenerate with:

```bash
./tools/download_sift.sh --full
./build/release/tools/sift_check data/sift/sift_base.fvecs \
    data/sift/sift_query.fvecs data/sift/sift_groundtruth.ivecs
```

### Load and memory, SIFT1M

| Metric | Value | Note |
|---|---|---|
| Load, 1,000,000 × 128 | **0.35 s** | Warm page cache. Observed 0.35–0.84 s depending on cache state. |
| — of which `reserve()` | 0.245 s | `aligned_alloc` + zero-fill of 488 MiB |
| — marginal cost of the zero-fill | **57 ms** | Removing the `memset` drops a full load only from 0.358 s to 0.301 s; the rest is first-touch page-fault cost paid either way |
| Peak RSS (`VmHWM`) | **501.6 MiB** | Honest, not lazily-mapped: the zero-fill touches every page |
| Store allocation | 488.3 MiB | 512,000,000 bytes |
| Overhead above the store | ~13.3 MiB | Queries, ground truth, binary |
| Bytes per vector, in memory | 512 | stride 128 == dim, so no padding at dim 128 |
| Bytes per vector, on disk | 516 | 4-byte dimension prefix per record |

### Brute-force exact k-NN, SIFT1M

Single-threaded, k = 10, all 10,000 queries, no warmup (see deviations above).

| Metric | Value |
|---|---|
| Throughput | **11.58 QPS** (median of 3) |
| Individual runs | 11.38, 11.58, 11.58 QPS — **1.7% spread** |
| Wall clock per pass | 863.8 s (median) |
| Distance computations | 1.0 × 10¹⁰ per pass |
| Effective read bandwidth | 5.5 GiB/s |
| **recall@10, strict id-set match** | **0.999440** |
| **recall@10, tie-aware** | **1.000000** |

The 1.7% spread across three runs is inside the project's 5% reproducibility
target. Both recall figures were **bit-identical** across all three runs, which
is the `(distance, id)` total order earning its keep — sorting on distance alone
would let equal-distance neighbours land in a different arrangement per run
(`DECISIONS.md` D16).

The PRD expected single-digit QPS; 11.58 is the right order of magnitude. The
scan sustains 1.48 × 10⁹ dimension-updates/s, which against a 4.5 GHz peak clock
is roughly 3 cycles per dimension — about what an un-vectorised
load-subtract-multiply-accumulate costs. That confirms `-fno-tree-vectorize` is
genuinely applied to `distance_scalar.cpp` (`DECISIONS.md` D5), and so that this
is an honest scalar baseline for Phase 2 to beat rather than a quietly
auto-vectorised one.

**On the two recall figures.** 56 of 10,000 queries return a different id set
from the reference, and every one of them differs only at the k-th position, on
a byte-identical duplicate vector. The strict figure reports how often the
tie-break convention differed; the tie-aware figure reports whether the search
was correct. Full reasoning and evidence in `DECISIONS.md` D17.

### Brute-force exact k-NN, SIFT10K

| Metric | Value |
|---|---|
| Load, 10,000 × 128 | 0.01 s |
| Peak RSS | 9.6 MiB |
| Throughput | ~1,270 QPS (100 queries — **indicative only**, below the 10,000-query minimum) |
| recall@10, strict | **1.000000** |
| recall@10, tie-aware | **1.000000** |

SIFT10K has no duplicates on any k-th boundary, so both measures agree. This is
the configuration `ctest` runs; it takes 0.9 s under the sanitised debug preset.

---

## Phase 2 — distance kernels

Machine M1, `release` preset. Regenerate with:

```bash
./build/release/bench/bench_distance --benchmark_min_time=1s \
    --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

**"ns per distance" is two numbers, not one.** Reporting only the first is how a
SIMD speedup gets overstated, so both fixtures are always published:

- **`l1`** — store ~16 KiB, inside L1d, so the same few vectors are re-read for
  the whole run. Compute-bound. This is what the kernel can do.
- **`stream`** — store 256 MiB, 16× the 16 MiB L3, walked linearly so every
  distance touches a cold line. Memory-bound. This is what predicts
  brute-force QPS.

Bandwidth counts payload bytes only (`dim` floats, not `stride`), so it is
comparable across dimensions and against the Phase 1 brute-force figure.

### Scalar baseline, squared L2

Median of 3. Stddev was 0.14–3.1 ns, i.e. **0.2–0.5%** — far inside the 5%
reproducibility target.

| Fixture | dim | ns/distance | GiB/s | stddev |
|---|---|---|---|---|
| `l1` | 128 | **76.52** | 6.23 | 0.18% |
| `stream` | 128 | **79.71** | 5.98 | 0.31% |
| `l1` | 960 | **648.10** | 5.52 | 0.45% |
| `stream` | 960 | **667.89** | 5.35 | 0.47% |

Inner product runs 4–8% faster than L2 at the same shape, which is the one
subtract it does not do.

**The baseline finding: `l1` and `stream` differ by only 4.2% at dim 128.** The
scalar kernel is slow enough to be compute-bound even while walking 256 MiB —
the memory system keeps up trivially at 6 GiB/s. So the two fixtures currently
say almost the same thing. They will diverge sharply once a SIMD kernel lands,
and that divergence is the measurement this phase exists to make.

**dim 960 costs 0.675 ns per dimension against dim 128's 0.598** — 12.9% more,
where per-distance loop overhead amortising over 960 elements instead of 128
should have made it *cheaper*. Most likely L1 pressure: at dim 960 each distance
streams a 3.84 KiB stored vector against a 3.84 KiB query, 7.7 KiB touched per
distance versus 1 KiB at dim 128. Recorded as an observation, not a conclusion —
worth confirming when the SIMD kernels give a second data point.

### The prediction, on record before the SIMD kernels exist

Now quantitative, from the numbers above rather than from Phase 1's estimate:

- **`l1` should approach the full SIMD width**, so roughly 8× for AVX2 — down
  from 76.5 ns toward ~10 ns at dim 128. Phase 2's ≥3× exit criterion is
  measured here and should pass comfortably.
- **`stream` cannot.** It already moves 5.98 GiB/s. Single-core streaming
  bandwidth on a mobile Zen 3 part is roughly 15–20 GiB/s — well under DRAM peak,
  because one core cannot keep enough misses outstanding. That floors `stream`
  at 79.7 × 5.98/20 ≈ **24 ns** to 79.7 × 5.98/15 ≈ **32 ns**, a speedup of only
  **2.5× to 3.3×**.

If `l1` shows 6–8× and `stream` shows ~3×, that is not a contradiction. It is the
memory wall, and it is the answer to "explain why the speedup isn't exactly 8×".
**Falsify this rather than confirm it** — if `stream` beats 3.3×, the
single-core bandwidth estimate was wrong and that is the more interesting result.

### Methodology note: CPU frequency scaling

Google Benchmark warns that CPU scaling is enabled. It is: the governor is
`powersave` under `amd-pstate-epp`, and pinning it needs root this session does
not have. Reported rather than quietly ignored — but the measured stddev of
0.2–0.5% across three runs says it is not materially affecting these numbers.

### Kernel comparison

Filled by tasks 3–5.

| Kernel | `l1` dim 128 | `stream` dim 128 | `l1` dim 960 | `stream` dim 960 | vs scalar |
|---|---|---|---|---|---|
| scalar | 76.52 ns | 79.71 ns | 648.10 ns | 667.89 ns | 1.00× |
| sse | — | — | — | — | — |
| avx2 | — | — | — | — | — |

## Phase 3 — HNSW

| ef | recall@10 | QPS | p50 | p95 | p99 | Machine |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

Report the tie-aware recall figure, or inherit a 0.06% penalty on SIFT1M that
has nothing to do with the index (`DECISIONS.md` D17).

## Phase 4 — versus hnswlib

| Index | recall@10 | QPS | Ratio | Note |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 5 — product quantisation

| m | bytes/vector | recall@10 | Machine |
|---|---|---|---|
| — | — | — | — |

## Phase 6 — the selectivity sweep

The collapse curve. See `FINDINGS.md` for the analysis.

| Selectivity | pre-filter | post-filter | in-filter | Correlation |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 7 — the attempt

| Selectivity | best baseline | Lodestone | Delta |
|---|---|---|---|
| — | — | — | — |
