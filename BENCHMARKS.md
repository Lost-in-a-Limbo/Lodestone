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
| 2 | Three runs, median reported | **Kept, plus random interleaving.** The machine thermally throttles (86 °C idle, 92 °C loaded), so a fixed benchmark order penalises whatever runs last. Google Benchmark's `--benchmark_enable_random_interleaving` with 9 repetitions. Absolute ns figures still drift ~20% with machine state; **ratios reproduce within ~2%**, so ratios are what the headline numbers report. |

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

The first measurement taken this phase, on a cold machine, before any SIMD
existed. Kept as the historical record; the comparison table below supersedes it
and was taken under interleaving on a hot one. Do not mix rows between the two.

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
should have made it *cheaper*. Recorded at the time as "most likely L1
pressure", explicitly as an observation rather than a conclusion.

> **Superseded.** It was **dependency-chain length**, not L1 pressure — see the
> accumulator experiment below. With four accumulators dim 960 becomes *cheaper*
> per dimension than dim 128, which L1 pressure cannot explain.

### The prediction that was on record before any SIMD existed, and how it went

Written here before the kernels were built, to be judged rather than quietly
revised afterwards.

| Predicted | Actual | Verdict |
|---|---|---|
| `l1` approaches full SIMD width, ~8×, toward ~10 ns at dim 128 | 11.1×, 8.49 ns | **Better than predicted** |
| `stream` floors at 24–32 ns, a speedup of only 2.5–3.3× | 28.26 ns, 3.51× | **ns right, speedup slightly beaten** |
| The `l1`/`stream` gap is the memory wall and answers "why not 8×" | 11.1× vs 3.51× | **Held** |

The shape of the argument survived; two of its numbers did not, and a later
claim about the bandwidth ceiling was wrong outright — see below.

### Kernel comparison, squared L2

**Ratios, not absolutes, are the trustworthy quantity on this machine.** See the
thermal note below. Measured with Google Benchmark's random interleaving,
9 repetitions, so thermal drift cannot map onto benchmark order.

| ns/distance | `l1` dim 128 | `l1` dim 960 | `stream` dim 128 | `stream` dim 960 |
|---|---|---|---|---|
| scalar | 94.33 | 777.40 | 99.30 | 836.07 |
| sse | 20.24 | 193.26 | 35.23 | 275.02 |
| avx2 (4 acc) | **8.49** | **46.91** | **28.26** | **209.31** |
| stddev, worst | 6.6% | 5.0% | 6.8% | 5.5% |

| Speedup vs scalar | `l1` dim 128 | `l1` dim 960 | `stream` dim 128 | `stream` dim 960 |
|---|---|---|---|---|
| sse | 4.66× | 4.02× | 2.82× | 3.04× |
| **avx2** | **11.11×** | **16.57×** | **3.51×** | **3.99×** |

**Phase 2's exit criterion is met with room to spare: AVX2 L2 is 11.1× scalar at
dim 128 in the compute-bound fixture, against a requirement of ≥3×.** It clears
3× in every fixture, including the memory-bound one.

### Why the speedup exceeds the 8-wide register width

11× and 16× from an 8-wide instruction set needs explaining as much as a
shortfall would, and the answer is that **the baseline is latency-bound, not
throughput-bound.**

Naive scalar `sum += diff * diff` is a dependent float-add chain, one link per
element, and Zen 3 float-add latency is 3 cycles. So the scalar kernel floors
near 3 cycles per dimension no matter how many execution units sit idle. The
AVX2 kernel advances 8 elements per link *and* runs 4 independent chains, so the
comparison is 32 elements in flight against 1. Bounded by load ports and memory,
that lands at 11–17×.

Which means the headline number combines two distinct wins — vectorisation and
breaking a dependency chain — and only the first is really "SIMD". A
multi-accumulator *scalar* kernel would narrow the gap; we have not measured by
how much, and that is logged in `IDEAS.md` as the honest caveat rather than
quietly left out. Under any reading the ≥3× criterion is met.

The real shortfall is `stream` at 3.5×, and its cause is memory bandwidth. The
`l1`/`stream` gap — 11.1× against 3.5× — is precisely what the two fixtures were
built to expose.

### Task 5: the accumulator experiment

ns/distance by accumulator count, AVX2, squared L2. Measured on a cooler thermal
state than the table above, so compare *within* this table only.

| Accumulators | `l1` dim 128 | `l1` dim 960 | `stream` dim 128 | `stream` dim 960 |
|---|---|---|---|---|
| 1 | 9.61 | 91.08 | 25.71 | 192.47 |
| 2 | 7.17 | 57.46 | 24.62 | 167.81 |
| **4** | **6.93** | 37.57 | **23.33** | 165.54 |
| 8 | 7.24 | **36.65** | 23.46 | 165.44 |
| gain, 1→best | **1.39×** | **2.48×** | **1.10×** | **1.16×** |

**The textbook argument is wrong here, and the reason is the interesting part.**
Theory says one accumulator serialises every FMA behind the previous one — 4
cycles of latency against 2/cycle of throughput — so saturation needs 8 chains
and the gain should be large. Measured, the gain is 1.39× at dim 128.

The cause: `distances_to()` computes independent distances back to back, so the
tail of one overlaps the head of the next. **Batching already supplies the
instruction-level parallelism that accumulators would otherwise provide.**

But only while the chain fits the out-of-order window. At dim 128 a distance is
16 dependent FMAs and the next distance can cover them; at dim 960 it is 120,
which cannot be hidden, and accumulators then buy **2.48×**. That dimension
dependence is the evidence for the explanation, not just a restatement of it.

Four accumulators ship: best at dim 128, within 2.5% of best at dim 960, and
eight costs register pressure for nothing.

**This corrects an earlier claim in this file.** Tasks 2 and 3 recorded a ~11–16%
per-dimension penalty at dim 960 versus dim 128 across scalar and SSE, and
attributed it to L1 pressure. It was **dependency-chain length**: with four
accumulators, dim 960 becomes *cheaper* per dimension than dim 128 (0.049 ns
versus 0.054 ns), which L1 pressure cannot explain and chain length does.

### Answered: does the 64-byte-aligned query buffer buy anything?

The open question carried from Phase 1 (`IDEAS.md`). Measured by swapping
`_mm256_load_ps` for `_mm256_loadu_ps` on both operands:

| | `l1` dim 128 | `l1` dim 960 | `stream` dim 128 | `stream` dim 960 |
|---|---|---|---|---|
| aligned | 9.35 ns | 86.86 ns | 26.60 ns | 169.8 / 173.9 ns |
| unaligned | 9.46 ns | 86.91 ns | 27.05 ns | 172.1 / 176.0 ns |
| difference | 1.2% | 0.05% | 1.7% | 1.3% |

**Answer: essentially nothing — 0–2%, at or barely above noise.** So the
alignment stays for **correctness, not speed**: a 32-byte `_mm256_load_ps` from
the 16-byte-aligned base that `std::vector<float>` guarantees is undefined
behaviour on half its offsets. That is why `detail::PreparedQuery` exists.

Worth recording how that number was reached. The first comparison showed 18% at
`stream` dim 960, wildly out of line with the other three. Repeating the pair
twice gave 1.3% both times; the 18% was an outlier. Publishing the first run
would have produced a confident and completely wrong finding.

### Two claims from task 3 that AVX2 falsified

Recorded because they were written down as predictions, and the point of writing
a prediction down is to be able to say plainly when it was wrong.

**Wrong: "the single-core bandwidth ceiling is 15.5 GiB/s, measured rather than
estimated."** AVX2 streams above 20 GiB/s. 15.5 was SSE's achievable rate.

**Wrong, and worse: "two shapes landing within 0.06% is not a coincidence, it is
a wall."** It was a coincidence — the next invocation had the same two numbers
6% apart. One run's agreement was read as structure on a machine already
documented here as drifting between runs.

**Wrong: "AVX2 `stream` dim 128 stays within ~10% of SSE."** It is 20–25% faster.

**Right: "AVX2 `l1` dim 128 lands near 10 ns"**, and **right: "extra
accumulators will gain little at dim 128"** — 1.39×, against a textbook
expectation of several times that.

The corrected picture: SSE's streaming was *partly compute-limited*, not
saturated. A wider kernel issues loads faster and keeps more misses outstanding,
so it extracts more bandwidth. The true single-core ceiling is **above 20 GiB/s
and still unmeasured** — nothing built here has reached it.

### End to end: SIFT1M brute force through the dispatched kernel

The number that says what the kernel work is actually worth. Same tool, same
data, same methodology as Phase 1 — only the kernel changed, and it changed by
`detected_kernel()` picking AVX2 with no caller edited.

| | scalar (Phase 1) | avx2 (Phase 2) | change |
|---|---|---|---|
| Throughput | 11.58 QPS | **32.32 QPS** | **2.79×** |
| Runs | 11.38 / 11.58 / 11.58 | 31.84 / 32.32 / 33.31 | 4.6% spread |
| Wall clock per pass | 863.8 s | **300–314 s** | |
| Time per query | 86.4 ms | **30.9 ms** | |
| Effective read bandwidth | 5.5 GiB/s | **15.4 GiB/s** | 2.8× |
| recall@10 strict / tie-aware | 0.999440 / 1.000000 | **0.999440 / 1.000000** | **identical** |

**The recall figures are bit-identical to the scalar run**, across 10,000 real
queries against 1,000,000 vectors. That is the strongest correctness evidence in
this phase — far stronger than any tolerance test, because it says the AVX2
kernel selects the *same neighbours*, not merely similar distances.

**2.79× end to end, against 11.1× in the compute-bound microbenchmark.** Not a
disappointment — an explanation. A brute-force query reads all 488.3 MiB of the
corpus, and 488.3 MiB at 15.4 GiB/s *is* 30.9 ms. The query time is now almost
exactly the time it takes to stream the corpus out of DRAM once; the arithmetic
has stopped mattering.

It is also below the `stream` fixture's 3.51×, and the gap is Amdahl's law: the
scan also maintains a bounded heap and generates ids, and neither of those got
faster. Roughly 8% of the query is now non-kernel work.

**This is the argument for Phase 3 stated in numbers.** No kernel can beat
30.9 ms while every query touches all 1M vectors — the only way past it is to
stop visiting all of them, which is what a navigable graph is for.

### Methodology: this laptop thermally throttles, and it matters

The single most important caveat on every absolute number in this section.

`thermal_zone0` reads **86 °C after five minutes idle** and **92 °C** at the end
of a benchmark run. Under sustained load the same measurement drifts a long way:
`scalar/l2/l1/dim128` was 74.8 ns on a cold machine and 108.3 ns on a hot one, a
45% spread on identical code.

Three consequences, all applied above:

1. **Ratios are stable even when absolutes are not.** AVX2-versus-scalar at
   `l1` dim 128 measured 11.17× on a cold machine and 11.11× on a hot one — 0.5%
   apart, across a thermal state that moved the absolutes by 22%. Every headline
   figure in this section is therefore a ratio.
2. **Random interleaving is required**, not optional. Without it, benchmarks run
   in a fixed order and the machine heats monotonically, so later kernels are
   systematically penalised — which would have made AVX2 look worse purely
   because it is registered last.
3. **Absolute ns/distance from this machine is not a portable number.** Phase 4's
   harness must either pin the governor (needs root, unavailable in this
   session), interleave as here, or report ratios only.

## Phase 3 — HNSW

Machine M1, `release` preset, AVX2 kernel selected automatically. Regenerate:

```bash
./build/release/tools/hnsw_bench data/sift/sift_base.fvecs \
    data/sift/sift_query.fvecs data/sift/sift_groundtruth.ivecs
```

Recall is reported **tie-aware** (D17). Strict id-set recall is printed
alongside and differs by <0.001 here — at 96% recall the genuine misses swamp
the duplicate-vector ties that dominated the exact-search figure.

### Build, SIFT1M

`M = 16`, `M_max0 = 32`, `ef_construction = 200`, seed 100, single-threaded.

| Metric | Value | Exit criterion |
|---|---|---|
| Build time | **403.4 s** (6.7 min), 2,479 vectors/s | under 20 min ✅ |
| Graph memory | **135.9 MiB** | — |
| Peak RSS | 660.5 MiB (store 488.3 MiB, so ~172 MiB above it) | — |
| Max level | 5 | — |
| Graph bytes per vector | ~142 | — |

The graph costs 28% of what the vectors cost. At `M_max0 = 32` a layer-0
neighbour list is 33 × 4 = 132 bytes; everything else — upper layers, the level
array, the visited set — is the remaining ~10.

### The recall/QPS curve, SIFT1M, k = 10

| ef | recall@10 | QPS | nodes visited | % of corpus |
|---|---|---|---|---|
| 8 | *skipped* | — | — | ef < k cannot return k results |
| 16 | 0.8023 | 17,831 | 484 | 0.048% |
| 32 | 0.9045 | 10,822 | 743 | 0.074% |
| **64** | **0.9644** | **6,046** | 1,230 | 0.123% |
| 128 | 0.9900 | 3,444 | 2,128 | 0.213% |
| 256 | 0.9980 | 1,855 | 3,728 | 0.373% |

**Exit criterion met: recall@10 ≥ 0.95 at ef = 64.**

`ef = 8` is skipped rather than clamped. PRD section 6 suggests sweeping from 8,
but ef bounds the layer-0 candidate list, so ef < k cannot return k results. A
row labelled `ef=8` that had quietly run at ef=10 would be a lie in a table.

### Against brute force — what the index is actually worth

Same machine, same kernel, same data, same tie-aware metric:

| | recall@10 | QPS | nodes visited/query |
|---|---|---|---|
| Brute force (Phase 2) | 1.000000 | 32.3 | 1,000,000 |
| HNSW, ef = 64 | 0.9644 | **6,046** | **1,230** |
| HNSW, ef = 128 | 0.9900 | 3,444 | 2,128 |

**187× the throughput at 96.4% recall; 107× at 99.0%.**

The mechanism is visible in the last column: the graph answers a query by
looking at **1,230 of a million vectors, 0.12% of the corpus.** Phase 2 ended
with the observation that brute force was pinned at 30.9 ms because it streams
488 MiB per query and no kernel work could change that. This is the other way
out — not moving the data faster, but not touching it at all.

### What the neighbour heuristic is worth

The single most-asked question about an HNSW implementation, measured rather
than asserted. Both selection rules stay in the code so this can be rerun
(`--selection=both`).

**SIFT1M, 2,000 queries for the simple run, 10,000 for the heuristic:**

| ef | heuristic (Alg 4) | simple (Alg 3) | gap |
|---|---|---|---|
| 16 | 0.8023 | 0.7171 | +0.085 |
| 64 | **0.9644** | 0.9135 | +0.051 |
| 256 | 0.9980 | 0.9825 | +0.016 |

The gap narrows as ef grows, which is the honest way to read it: **the heuristic
does not find answers simple selection cannot — it finds them with less search.**
Stated at matched recall, the convention that actually matters: heuristic hits
0.964 at ef = 64 and 6,046 QPS; simple needs roughly ef ≈ 120 for the same
recall, around 3,500 QPS. So the heuristic is worth about **1.7× throughput at
matched recall**, and it costs nothing at build time (369 s versus 403 s, the
simple rule being marginally cheaper to evaluate).

Build time and graph size are identical to three significant figures — 135.9 MiB
either way — so this is purely a question of *which* edges get kept.

### A claim I made from SIFT10K that SIFT1M did not support

Worth recording, because the mistake is the interesting part.

On SIFT10K the same comparison looked far more dramatic. Simple selection
appeared to **plateau**:

| ef | heuristic | simple |
|---|---|---|
| 16 | 0.9710 | 0.7030 |
| 64 | 1.0000 | 0.7470 |
| 256 | 1.0000 | **0.7500** |

Sixteen times the search effort buying 0.007 recall reads unmistakably as
*unreachable* regions rather than merely unexplored ones — and I wrote that
interpretation down.

**SIFT1M does not reproduce it.** Simple selection keeps improving all the way
to 0.9825 at ef = 256; there is no plateau, only a lag.

The likely cause is the measurement, not the algorithm: **SIFT10K ships only 100
queries.** That is two orders of magnitude below this project's own stated
minimum of 10,000 queries per measurement, and a handful of queries landing in a
badly-connected pocket is enough to pin the mean near 0.75 and make it look like
a ceiling. The 1M figure, over 2,000–10,000 queries, is the one to trust.

The rule the project already had would have caught this, and it is now applied
to SIFT10K rows too: **anything measured on 100 queries is indicative, never a
finding.**

### An unexpected result: the hierarchy does almost nothing at 20k

Recorded because it was measured and it contradicts the folk explanation of why
HNSW is fast.

Removing the greedy descent entirely — searching layer 0 directly from the entry
point — changed nothing measurable on a 20,000-vector corpus. Mean nodes visited
went from 478 to **445**, i.e. slightly *fewer*, and every recall test still
passed.

At that size layer 0 is well enough connected that greedy search converges from
almost anywhere, and the descent is pure overhead. The hierarchy earns its keep
at 1M — `max_level` reaches 5, and the descent is what avoids a long walk across
the graph — but it is not the reason a small index is fast, and this is why no
test asserts a visit-count bound tight enough to catch a missing descent. Stated
rather than papered over, because a test that cannot fail is worse than no test.

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
