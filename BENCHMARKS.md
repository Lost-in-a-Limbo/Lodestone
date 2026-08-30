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
| 4 | Numbers reproduce within 5% | **NOT MET.** 16 of 24 measurements exceed it, median spread 7.3%, worst 17.8%. Two causes: documented thermal throttling, and a 10-second warmup that is demonstrably too short — run 1 was slowest in 13 of 24 measurements against a chance expectation of 8. Recall reproduces exactly; only throughput carries this. |
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

Machine M1, `release` preset, AVX2 selected automatically. Regenerate with:

```bash
./build/release/bench/bench --all      # writes bench/results/results.json
```

Identical corpus, identical queries, identical ground truth, identical
`M = 16` / `ef_construction = 200`, single-threaded on both sides, 10 s warmup
discarded, median of 3 runs. hnswlib 0.8.0, header-only, compiled into the
benchmark with the same `-O3 -march=native` this project uses — so its own SIMD
path is active and the comparison is not handicapping it.

### The curve, SIFT1M, k = 10

| ef | recall (Lodestone) | recall (hnswlib) | QPS (Lodestone) | QPS (hnswlib) | ratio |
|---|---|---|---|---|---|
| 16 | 0.8023 | 0.8022 | 21,134 | 17,198 | 1.23× |
| 32 | 0.9045 | 0.9040 | 12,623 | 10,751 | 1.17× |
| 64 | 0.9644 | 0.9642 | 6,446 | 6,456 | 1.00× |
| 128 | 0.9900 | 0.9901 | 4,129 | 3,305 | 1.25× |
| 256 | 0.9980 | 0.9980 | 2,274 | 1,877 | 1.21× |

Full sweep including k = 1 and k = 100, latency percentiles and per-run figures
is in `bench/results/results.json`.

### The recall curves are the same curve

**Maximum absolute recall difference across all twelve operating points:
0.0018.** At k = 10, ef = 256 both report 0.9980; at ef = 128, 0.9900 versus
0.9901.

This is the most valuable thing in the comparison and it is not the speed. Two
independently written implementations of Algorithms 1–5, with different
languages of expression, different pruning code and different heap handling,
land on the same recall at every operating point. That is mutual validation:
either both are correct, or both are wrong in exactly the same way, and the
second is far less likely than the first.

It also means that **for this pair, matching on ef happens to be matching on
recall** — so the QPS column can be read directly. That was not safe to assume
in advance, and the harness does not assume it.

### On throughput: the direction is real, the magnitude is not well resolved

Lodestone is faster at **11 of 12 operating points**, tied at the twelfth, with
a median ratio of **1.17×**.

Taken point by point that would be over-reading the data: per-point run-to-run
spread reaches 17.8% (see below), which is larger than most of the individual
gaps. But the *sign* is a separate question from the magnitude, and 11 wins out
of 12 independent operating points has a one-sided sign-test **p = 0.0032**.
Noise does not produce that.

So: **the exit criterion "QPS within 5× of hnswlib at matched recall" is met by
a wide margin — Lodestone is at parity or slightly ahead.** The honest phrasing
is "comparable, consistently a little faster", not "1.2× faster".

### Why we might be ahead, and why some of it is not a fair win

Published because PRD §4 asks for the gap explained, and that obligation does
not lapse when the gap points the other way.

- **hnswlib's `searchKnn` returns a `std::priority_queue` by value.** That is a
  heap allocation per query. Lodestone writes into a caller-provided span. This
  is an API difference, not an algorithmic one, and it plausibly accounts for a
  good part of the gap — it should be roughly constant per query, and the
  advantage is indeed largest at low ef (1.23× at ef=16) where per-query time is
  smallest.
- **hnswlib does more.** It maintains a label→internal-id map, supports element
  deletion with tombstones, and carries locking hooks for concurrent insert.
  Lodestone has none of that. Comparing a feature-complete library against a
  narrower one and reporting only the speed is not a like-for-like result.
- **Lodestone batches through `distances_to()`.** Phase 2 established that
  independent distances overlap in the pipeline; hnswlib computes them one at a
  time with software prefetch. This one *is* an algorithmic difference and is
  the part of the gap that would survive an API fix.

Build time: 359.7 s versus 438.5 s, so Lodestone builds 1.22× faster — with the
same caveat, since hnswlib's `addPoint` maintains structures ours does not.

### Exit criterion NOT met: numbers do not reproduce within 5%

Stated plainly rather than buried. **16 of 24 measurements exceed the 5%
run-to-run target; the median spread is 7.3% and the worst is 17.8%.**

The cause is partly the documented thermal throttling (Phase 2: 86 °C idle,
92 °C loaded, absolute timings drifting 20–45%), but the data shows a second,
fixable cause:

> **Run 1 was the slowest of three in 13 of 24 measurements**, where chance
> alone would give about 8. The worst point's raw runs were 3,542 / 3,766 /
> 4,210 QPS — monotonically increasing.

A 10-second warmup is not enough to reach steady state on this workload. The
fix is to discard the first *measured* run as well as the warmup, or to warm up
until the throughput stops rising rather than for a fixed wall-clock time.
Recorded here rather than quietly re-run, because the harness's job is to make
this visible.

**What does reproduce exactly: recall.** It is computed from the graph and the
data, not from the clock, and it is identical across runs to every digit
reported. Every correctness claim in this file rests on that; only the
throughput figures carry this caveat.

## Phase 5 — product quantisation

Machine M1, `release` preset. Codebooks trained on `sift_learn.fvecs` — 100,000
**held-out** vectors, not the corpus being encoded. Regenerate with:

```bash
./build/release/bench/bench --all --pq --pq-m=8,16,32 --k=10 --ef=64,128
```

Asymmetric distance (ADC): the query stays full precision, only the stored
vectors are quantized. **No re-ranking** — it would recover most of the loss
this section exists to measure (D35).

### The recall/memory frontier, SIFT1M, k = 10

| | bytes/vector | reduction | corpus | reconstruction error | **recall@10** | QPS |
|---|---|---|---|---|---|---|
| exact float32 | 512 | 1× | 488.3 MiB | 0 | **1.000000** | 32.3 |
| PQ m = 32 | 32 | 16× | 30.5 MiB | 3,613 | **0.7234** | 42.9 |
| **PQ m = 16** | **16** | **32×** | **15.3 MiB** | 10,592 | **0.5449** | 96.9 |
| PQ m = 8 | 8 | 64× | 7.6 MiB | 23,688 | **0.3135** | 163.8 |

Codebook is 128 KiB at every m — `m × 256 × (128/m)` floats is 32,768 floats
regardless of m, which is why it is reported separately from the codes and why
it stops mattering above a few thousand vectors.

**Exit criterion met: 512 bytes → 16 bytes, a 32× reduction, at recall 0.5449.**

Training is 21–35 s and encoding 10–17 s for the full million, both
single-threaded and both one-time.

### Reading the frontier honestly

Recall falls steeply — 0.72 at 16×, 0.54 at 32×, 0.31 at 64×. That is what PQ
costs *without re-ranking*, and it is the number PRD §6 asks to quantify. A
production system fetches the top few thousand ADC candidates and re-scores them
with exact distances, which recovers most of this; it is deliberately not
implemented, because it recovers precisely the loss being measured (D35).

Reconstruction error and recall move together in the right direction at every
step — 23,688 → 10,592 → 3,613 against 0.31 → 0.54 → 0.72 — which is the
internal consistency check that says the measurement is sound. **It did not,
at first**, and that is recorded below.

### PQ is faster than exact brute force, but not by the compression ratio

| | vs exact brute force |
|---|---|
| m = 8 | 5.07× |
| m = 16 | 3.00× |
| m = 32 | 1.33× |

32× less memory buys 3× more throughput, not 32×. Phase 2 established that
brute force is memory-bandwidth-bound at 30.9 ms per query, so shrinking the
corpus 32× ought to have been worth far more than this.

It is not, because **ADC is not bandwidth-bound — it is bound by dependent
loads.** Each distance is `m` lookups into a 16 KiB table, and each lookup is a
load whose address depends on a byte just read from the code. The table sits in
L1, so the loads are cheap individually, but they do not vectorise the way the
exact kernel's 128 contiguous floats do. At m = 32 the lookups cost more than
the 512 bytes of streaming they replaced, which is why the win shrinks as m
grows and why m = 32 is barely faster than exact.

The compression is a *memory* result, not a speed result. Reported as both,
because reporting only the memory would imply the speed followed.

### The metric was wrong, and the frontier is how it was caught

The first PQ measurements had recall *rising* as the codebook got coarser —
0.9690 at m = 8 against 0.9240 at m = 16 — while reconstruction error moved the
other way. Two measurements of the same thing disagreeing about its direction is
not a result.

The cause was in `recall_at_k_tied`, not in the quantizer: it read the distance
out of the `Neighbor` the search returned, and compared it against a threshold
computed *exactly*. Fine for four phases, because every computer had been exact.
PQ fills that field with a quantized estimate, and the comparison then measures
nothing — reporting **0.9690 where the truth was 0.5600**, in the flattering
direction.

Fixed by recomputing the distance from the same computer that produced the
threshold. For an exact computer the value is bit-identical, so **no Phase 1–4
number moved** — re-verified against the SIFT10K exact check, still 1.000000.
Full account in `DECISIONS.md` D34.

### The warmup fix from Phase 4, applied and validated

Phase 4 reported that its 10-second warmup was too short and logged the fix
rather than applying it retroactively. Phase 5 applies it: one extra measured
run is taken and the first is discarded.

At the two `lodestone` operating points measured in both phases:

| k=10 | Phase 4 spread | Phase 5 spread |
|---|---|---|
| ef = 64 | 12.3% | **1.9%** |
| ef = 128 | 4.3% | **2.4%** |

Both now inside the 5% reproducibility target that Phase 4 failed. The
diagnosis was right and the fix works.

## Phase 6 — the selectivity sweep

The collapse curve. See `FINDINGS.md` for the analysis.

| Selectivity | pre-filter | post-filter | in-filter | Correlation |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 7 — the attempt

| Selectivity | best baseline | Lodestone | Delta |
|---|---|---|---|
| — | — | — | — |
