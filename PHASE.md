# Current phase

> Update this the moment a phase closes. It is the first thing read at the
> start of every session — it is how both you and Claude Code know where you are.

## → PHASE 4: Benchmark harness

**Goal:** the measurement rig everything downstream depends on. Build it
properly now — Phases 5, 6 and 7 all report through it.

**Exit criteria (PRD §6 Phase 4)**
- [ ] `./bench --all` produces `bench/results/results.json` with no manual steps
- [ ] recall@1, @10, @100; QPS; p50/p95/p99 latency; index memory; build time
- [ ] Machine spec captured automatically — CPU, cores, RAM, compiler, flags
- [ ] Warmup discarded, minimum 3 runs, median reported
- [ ] A comparison harness running `hnswlib` on identical data
- [ ] Numbers reproduce within 5% across three runs
- [ ] `BENCHMARKS.md` has our curve next to hnswlib's

**Resume bullet unlocks here.** This is the phase that makes the project
citable.

**Blocked on:** nothing.

**What Phase 3 leaves you:**

1. `tools/hnsw_bench` already sweeps ef and prints recall/QPS/visited. Phase 4's
   job is to make it emit JSON, add latency percentiles and recall@1/@100, and
   put `hnswlib` beside it — not to rewrite the sweep.
2. **Report tie-aware recall** (D17). Every Phase 3 number does.
3. **Three measurement traps this project has already fallen into**, all
   recorded: absolute timings drift ~20% with thermals so ratios are the stable
   quantity (D-note in `BENCHMARKS.md`); a single run's agreement is not
   evidence; and **anything measured on SIFT10K's 100 queries is indicative, not
   a finding** — that one produced a wrong conclusion in Phase 3 that SIFT1M
   overturned.
4. The `hnswlib` comparison is the one number this phase cannot fudge. Match on
   recall, not on ef — the two libraries will not agree on what a given ef buys.

---

## Progress

| Phase | Name | Status | Closed | Key number |
|---|---|---|---|---|
| 0 | Bootstrap | **closed** | 2026-08-23 | 8/8 tests, 3 presets green |
| 1 | Data + brute force | **closed** | 2026-08-24 | recall@10 = 1.000000 tie-aware |
| 2 | SIMD kernels | **closed** | 2026-08-27 | AVX2 11.1× scalar; 32.3 QPS brute force |
| 3 | HNSW | **closed** | 2026-08-27 | recall@10 0.9644 @ 6,046 QPS |
| 4 | Benchmark harness | **in progress** | — | full curve vs hnswlib |
| 5 | Product quantization | not started | — | bytes/vector, recall loss |
| 6 | **Filtered search** | not started | — | **collapse curve** |
| 7 | The attempt | not started | — | delta vs baselines |
| 8 | Showcase | not started | — | deployed URL |

**Resume-ready after Phase 4. Differentiating after Phase 6.**

---

## Phase 3 exit criteria — all met

- [x] Layer assignment with exponentially decaying probability (`mL = 1/ln M`)
- [x] Greedy descent per layer down to layer 0
- [x] `SEARCH-LAYER` with a candidate heap and an `ef`-bounded result list
- [x] The neighbour selection **heuristic** (Algorithm 4), *and* Algorithm 3
      kept alongside so the gap could be measured: **1.7× throughput at matched
      recall**
- [x] Bidirectional insertion with re-selection past `M_max`
- [x] Serialisation round-trips **bit-identically** — same edges, same answers
- [x] **Recall@10 ≥ 0.95 on SIFT1M: 0.9644 at ef = 64**
- [x] Build under 20 minutes single-threaded: **403 s (6.7 min)**
- [ ] QPS within 5× of `hnswlib` at matched recall — **deferred to Phase 4**,
      which is where the PRD puts the comparison harness. Our own curve is
      recorded; the comparison is not yet run, and it is the number this project
      cannot fudge.

**187× brute force at 96.4% recall** — 6,046 QPS against 32.3, by visiting 1,230
of a million vectors instead of all of them.

87 tests green on `debug`, `asan` and `release`.

---

## Phase 2 exit criteria — all met

- [x] Scalar inner product alongside scalar L2
- [x] SSE and AVX2 variants of both, hand-written intrinsics
- [x] Runtime dispatch on CPU feature detection, inside `detected_kernel()` —
      the body of one function, no caller changed
- [x] Google Benchmark microbenchmarks at dim 128 and dim 960
- [x] Every SIMD variant agrees with scalar. **Read as relative (1e-5), not
      absolute:** SIFT distances run to 5×10⁴ and reordering a float32 sum of
      128 terms perturbs it by ~10⁻³ absolute on entirely correct code, so
      absolute 1e-4 would fail correct kernels. And the *stronger* gate is
      end-to-end — SIFT10K through each kernel must give tie-aware
      recall@10 = 1.000000, which a tolerance test cannot catch.
- [x] **AVX2 L2 is 11.1× scalar**, against a requirement of ≥3×
- [x] Can explain why it isn't exactly 8× — it is *more* than 8× in the
      compute-bound fixture, because the scalar baseline is latency-bound rather
      than throughput-bound; and it is 3.5× in the memory-bound one, because of
      bandwidth. Both written up in `BENCHMARKS.md`.

71 tests green on `debug`, `asan` and `release`, including one end-to-end SIFT10K
recall check per kernel.

---

## Phase 1 exit criteria — all met

- [x] `.fvecs` / `.ivecs` parsers, with `tools/download_sift.sh` fetching SIFT1M
- [x] `VectorStore::reserve` / `add` — 64-byte aligned, contiguous
- [x] Brute-force exact k-NN
- [x] Recall@k calculator
- [x] Loads 1M vectors, reports load time and RSS
- [x] **Brute-force recall@10 against the provided ground truth = 1.000000** —
      by the tie-aware measure. The strict id-set measure is 0.999440, and the
      0.06% gap is fully explained: SIFT1M contains 14,538 byte-identical
      duplicate vectors, so when one lands on the k-th boundary several id sets
      are equally correct answers. Evidence and reasoning in `DECISIONS.md` D17.
      **This was investigated before the metric was changed, not after.**

56 tests green on `debug`, `asan` and `release`. Every task's claims were
verified by mutation — deleting a check and confirming a test goes red — because
most of Phase 1's code was new files, where a "failing test first" is just a
compile error and proves nothing.

---

## Session log

```
2026-08-23  Phase 0  scaffolding created; debug/release/asan all green;
                     8 tests passing; std::expected probe resolved (absent);
                     Phase 0 closed
2026-08-24  Phase 1  store, parsers, scalar L2 kernel, exact k-NN, recall,
                     download script, sift_check; SIFT1M recall investigated
                     to duplicate vectors; 56 tests; Phase 1 closed
2026-08-27  Phase 2  pushed Phase 0+1 to origin/main (13 commits, b20c81f..e73fcf4);
                     Phase 0's last exit criterion ticked; scalar IP, SSE and
                     AVX2 kernels, runtime dispatch, benchmark rig, accumulator
                     experiment; brute force 11.58 -> 32.32 QPS with recall
                     bit-identical; 71 tests; Phase 2 closed
2026-08-27  Phase 3  HNSW Algorithms 1-5, serialisation, ef sweep; SIFT1M
                     recall@10 0.9644 @ ef=64, 6046 QPS, build 403 s;
                     found+fixed a duplicate-in-results bug; corrected a
                     plateau claim made from 100 queries; 87 tests;
                     Phase 3 closed
```

---

## Numbers recorded so far

Full detail, methodology and regeneration commands in `BENCHMARKS.md`.

| Metric | Value | Machine | Date |
|---|---|---|---|
| **HNSW recall@10 / QPS** | **0.9644 @ 6,046 QPS** (ef=64) | M1 | 2026-08-27 |
| — at ef=128 | 0.9900 @ 3,444 QPS | M1 | 2026-08-27 |
| — at ef=256 | 0.9980 @ 1,855 QPS | M1 | 2026-08-27 |
| HNSW build, 1M | 403 s, 2,479 vectors/s | M1 | 2026-08-27 |
| HNSW graph memory | 135.9 MiB (~142 B/vector) | M1 | 2026-08-27 |
| Nodes visited/query, ef=64 | 1,230 — **0.12% of the corpus** | M1 | 2026-08-27 |
| Neighbour heuristic worth | ~1.7× throughput at matched recall | M1 | 2026-08-27 |
| Brute force, AVX2 | 32.32 QPS, recall 1.000000 | M1 | 2026-08-27 |
| Brute force, scalar | 11.58 QPS | M1 | 2026-08-24 |
| AVX2 vs scalar, compute-bound | 11.11× (dim 128) | M1 | 2026-08-27 |
| Load, 1M × 128 | 0.33–1.01 s | M1 | 2026-08-27 |
| SIFT1M distinct vectors | 985,462 of 1,000,000 | — | 2026-08-24 |
| Tests passing | 87/87 on 3 presets | M1 | 2026-08-27 |

**Machine M1:** AMD Ryzen 5 7530U (Zen 3, 6C/12T, 4546 MHz max), 15 GiB RAM,
GCC 11.4.0, CMake 4.3.1, Ninja 1.13.2. AVX2 + FMA, no AVX-512.

---

## Open questions

- [ ] **How does the curve compare to `hnswlib` at matched recall?** The one
      number this project cannot fudge, and the reason Phase 4 exists. Match on
      recall, never on ef.
- [ ] Build is 2,479 vectors/s single-threaded. Where does that go — the
      `ef_construction = 200` search, or the heuristic's O(M²) distance checks
      during selection? Never profiled.
- [ ] Does `distances_to()` batching still pay at graph-sized batches (M ≈ 16–32
      unvisited neighbours) as it did at 256? Carried from Phase 2, still open,
      and now there is a real consumer to measure.

### Resolved during Phase 3

- [x] **What does ef buy, and where does the curve bend?** 0.80 at ef=16 to
      0.998 at ef=256; the knee is around ef=64–128, where recall passes 0.96
      while still visiting only 0.12–0.21% of the corpus.
- [x] **How much is the neighbour heuristic worth?** ~1.7× throughput at matched
      recall on SIFT1M. Not the dramatic collapse SIFT10K suggested — see the
      correction in `BENCHMARKS.md`.
- [x] **Does the hierarchy matter?** Almost not at all at 20k: deleting the
      greedy descent changed visited counts from 478 to 445 and broke no test.
      It earns its keep at 1M (`max_level` 5). Recorded in D27 along with the
      consequence that no test can catch a missing descent at testable sizes.

### Resolved during Phase 2

- [x] **Does the streaming speedup cap near 3×?** Yes — 3.51× at dim 128, 3.99×
      at dim 960, and 2.79× end to end on real SIFT1M. But the *reason* given
      was wrong twice: "the ceiling is 15.5 GiB/s" and "the wall is hit at SSE"
      were both over-read from single runs. AVX2 exceeds 20 GiB/s and the true
      single-core ceiling is still unmeasured.
- [x] **Is the AVX2 speedup larger at dim 960 than dim 128?** Yes, 16.6× versus
      11.1× — the horizontal reduction is paid once per distance and amortises
      over 120 iterations instead of 16.
- [x] **Does the 64-byte-aligned query buffer buy anything?** No — 0–2%, at
      noise. It stays because `_mm256_load_ps` from a 16-byte-aligned base is
      UB, not because it is faster.
- [x] **Does the tail-free stride-wide loop help?** It is a *permission*, not a
      win: a `dim`-bounded loop computes the identical value because it still
      covers `ceil(dim/lanes)*lanes` elements over zeroed padding. What it buys
      is the absence of a masked-load tail, which is the buggiest part of a
      hand-written SIMD kernel.
- [x] **How much of the gap is load-port pressure?** With 4 accumulators at dim
      960 the loop runs ~1.42 cycles per 8-lane iteration against a floor of 1
      (2 loads per FMA, 2 load ports). Close enough that loads, not FLOPs, are
      the binding constraint at the top end.
- [x] **This laptop thermally throttles**, badly enough to move absolute
      timings 20–45%. Ratios under random interleaving reproduce within ~2%.
      Every Phase 2 headline is a ratio for that reason.

