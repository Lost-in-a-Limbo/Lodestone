# Current phase

> Update this the moment a phase closes. It is the first thing read at the
> start of every session — it is how both you and Claude Code know where you are.

## → PHASE 5: Product quantization

**Goal:** compress the vectors, measure what accuracy it costs. This is the ML
content of an ML-systems project — k-means, lossy compression, and the
accuracy/memory frontier.

**Exit criteria (PRD §6 Phase 5)**
- [ ] Per-subspace k-means, 256 centroids → one byte per subspace
- [ ] Codebook training
- [ ] Asymmetric distance with precomputed lookup tables
- [ ] 128-dim float32 (512 bytes) compressed to 16 bytes — 32× reduction
- [ ] Recall loss quantified at each compression ratio
- [ ] You can explain asymmetric versus symmetric distance computation
- [ ] **It must plug in through `DistanceComputer` without touching `hnsw.cpp`**

**Record:** memory per vector and recall@10 at m ∈ {8, 16, 32} subspaces.

**Blocked on:** nothing.

**This phase is the test of D1.** The distance seam was designed in Phase 0
specifically so that quantised distances could drop in here. `prepare_query()`
exists because PQ needs per-query state: the query is projected once into a
256×m lookup table, and every subsequent distance is m table lookups and adds.
If Phase 5 finds itself editing `hnsw.cpp`, the seam was wrong and *that* is the
thing to fix.

**What Phase 4 leaves you:**

1. `./bench --all` writes `bench/results/results.json` with machine specs.
   Phase 5 adds rows; it does not build a new harness.
2. **Apply the warmup fix before measuring, not after** (`IDEAS.md`). Phase 4's
   10-second warmup is demonstrably too short, and changing methodology after
   seeing numbers is how a benchmark stops being one.
3. Report tie-aware recall (D17), and the PQ recall loss is measured against
   *our own* exact figures, not against hnswlib.

---

## Progress

| Phase | Name | Status | Closed | Key number |
|---|---|---|---|---|
| 0 | Bootstrap | **closed** | 2026-08-23 | 8/8 tests, 3 presets green |
| 1 | Data + brute force | **closed** | 2026-08-24 | recall@10 = 1.000000 tie-aware |
| 2 | SIMD kernels | **closed** | 2026-08-27 | AVX2 11.1× scalar; 32.3 QPS brute force |
| 3 | HNSW | **closed** | 2026-08-27 | recall@10 0.9644 @ 6,046 QPS |
| 4 | Benchmark harness | **closed** | 2026-08-30 | parity with hnswlib, recall identical to 0.0018 |
| 5 | Product quantization | **in progress** | — | bytes/vector, recall loss |
| 6 | **Filtered search** | not started | — | **collapse curve** |
| 7 | The attempt | not started | — | delta vs baselines |
| 8 | Showcase | not started | — | deployed URL |

**Resume-ready after Phase 4. Differentiating after Phase 6.**

---

## Phase 4 exit criteria

- [x] `./bench --all` produces `bench/results/results.json` with no manual steps
- [x] recall@1, @10, @100; QPS; p50/p95/p99; index memory; build time
- [x] Machine spec captured automatically — CPU, cores, cache, RAM, compiler,
      the actual build flags, the selected kernel, **and the CPU governor and
      die temperature**, because on this machine those change the answer
- [x] Warmup discarded, 3 runs, median reported
- [x] `hnswlib` 0.8.0 on identical data, same M and ef_construction,
      single-threaded, same `-march=native`
- [x] `BENCHMARKS.md` has both curves side by side, with the reasons our win
      may be partly unearned recorded next to it
- [ ] **Numbers reproduce within 5% — NOT MET.** 16 of 24 measurements exceed
      it; median spread 7.3%, worst 17.8%. Two causes: documented thermal
      throttling, and a 10-second warmup shown by the data to be too short (run
      1 was slowest in 13 of 24 measurements against a chance expectation of 8).
      Recall reproduces exactly. See D31 — the fix is logged and deliberately
      not applied retroactively.

**The headline: recall curves agree with hnswlib to within 0.0018 at every one
of twelve operating points**, and Lodestone is faster at 11 of 12 (median 1.17×,
sign-test p = 0.0032). The exit criterion asked for QPS within 5× of hnswlib;
we are at parity or slightly ahead.

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
- [x] QPS within 5× of `hnswlib` at matched recall — **done in Phase 4**, where
      the PRD puts the comparison harness. Result: parity, Lodestone faster at
      11 of 12 operating points.

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
2026-08-30  Phase 4  bench harness -> results.json with machine specs;
                     hnswlib 0.8.0 comparison on identical data; recall curves
                     agree to 0.0018, QPS parity (11/12 points ahead,
                     p=0.0032); 5% reproducibility criterion NOT met and
                     reported; fixed CI format job red since Phase 0;
                     Phase 4 closed
```

---

## Numbers recorded so far

Full detail, methodology and regeneration commands in `BENCHMARKS.md`; raw data
in `bench/results/results.json`.

| Metric | Value | Machine | Date |
|---|---|---|---|
| **vs hnswlib, recall** | **identical to within 0.0018** at 12 operating points | M1 | 2026-08-30 |
| **vs hnswlib, QPS** | **parity** — faster at 11/12 points, median 1.17×, p=0.0032 | M1 | 2026-08-30 |
| HNSW recall@10 / QPS | 0.9644 @ 6,446 QPS (ef=64) | M1 | 2026-08-30 |
| — at ef=128 | 0.9900 @ 4,129 QPS | M1 | 2026-08-30 |
| — at ef=256 | 0.9980 @ 2,274 QPS | M1 | 2026-08-30 |
| HNSW recall@1 | 0.9998 @ 2,149 QPS (ef=256) | M1 | 2026-08-30 |
| HNSW recall@100 | 0.9863 @ 2,213 QPS (ef=256) | M1 | 2026-08-30 |
| p50 / p99 latency, ef=64 k=10 | 157 µs / 339 µs | M1 | 2026-08-30 |
| HNSW build, 1M | 360 s (hnswlib 439 s) | M1 | 2026-08-30 |
| HNSW graph memory | 135.9 MiB (~142 B/vector) | M1 | 2026-08-27 |
| Nodes visited/query, ef=64 | 1,230 — **0.12% of the corpus** | M1 | 2026-08-27 |
| Brute force, AVX2 | 32.32 QPS, recall 1.000000 | M1 | 2026-08-27 |
| AVX2 vs scalar, compute-bound | 11.11× (dim 128) | M1 | 2026-08-27 |
| SIFT1M distinct vectors | 985,462 of 1,000,000 | — | 2026-08-24 |
| Tests passing | 87/87 on 3 presets | M1 | 2026-08-30 |

**Machine M1:** AMD Ryzen 5 7530U (Zen 3, 6C/12T, 4546 MHz max), 15 GiB RAM,
GCC 11.4.0, CMake 4.3.1, Ninja 1.13.2. AVX2 + FMA, no AVX-512.

---

## Open questions

- [ ] Does the PQ lookup-table path actually beat the AVX2 exact kernel? m
      table lookups and adds against 16 FMAs is not obviously a win at dim 128 —
      the compression is the point, but the speed claim needs measuring.
- [ ] How much of the hnswlib throughput gap is the `priority_queue` allocation
      versus the `distances_to()` batching? Separable by patching a local copy
      of hnswlib to write into a span, which would say which part of the win is
      algorithmic.
- [ ] Build is 2,479 vectors/s single-threaded and has never been profiled —
      `ef_construction` search, or the heuristic's O(M²) checks?

### Resolved during Phase 4

- [x] **How does the curve compare to hnswlib at matched recall?** Parity.
      Recall identical to 0.0018 at all twelve operating points; Lodestone
      faster at 11 of 12, median 1.17×, sign test p = 0.0032. Reasons the win
      may be partly unearned are recorded in D32.
- [x] **Do the numbers reproduce within 5%?** No — and the harness found why.
      Beyond the known thermal drift, run 1 was slowest in 13 of 24
      measurements, so the 10-second warmup is too short. D31, fix in
      `IDEAS.md`.

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

