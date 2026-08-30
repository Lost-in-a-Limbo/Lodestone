# Current phase

> Update this the moment a phase closes. It is the first thing read at the
> start of every session — it is how both you and Claude Code know where you are.

## → PHASE 3: HNSW

**Goal:** the index itself. The largest single phase.

**Exit criteria (PRD §6 Phase 3)**
- [ ] Layer assignment with exponentially decaying probability
- [ ] Greedy search per layer, descending to layer 0
- [ ] `SEARCH-LAYER` with a candidate heap and an `ef`-bounded result list
- [ ] The neighbour selection *heuristic* (Algorithm 4), not just nearest-M —
      it matters a lot for recall
- [ ] Bidirectional insertion with pruning past `M_max`
- [ ] Serialisation that round-trips identically
- [ ] Recall@10 ≥ 0.95 on SIFT1M at some `ef`
- [ ] QPS within 5× of `hnswlib` at matched recall
- [ ] Build under 20 minutes single-threaded on 1M vectors

**Reference:** Malkov & Yashunin, arXiv 1603.09320, Algorithms 1–5. One
algorithm per task, each with tests.

**Blocked on:** nothing.

**What Phase 2 leaves you:**

1. `make_distance_computer(metric, store)` returns AVX2 automatically and the
   graph never names a kernel. Hold a `DistanceComputer&` and nothing else — if
   `l2_distance(` appears in `hnsw.cpp`, the rule has broken.
2. **Use `distances_to()`, not `distance_to()` in a loop.** A node's neighbour
   list is naturally a batch, and the batch is where the speed is: independent
   distances overlap in the pipeline, which is worth more than extra
   accumulators (`DECISIONS.md` D22).
3. **Report the tie-aware recall figure.** SIFT1M has 14,538 duplicate vectors;
   the strict id-set measure costs a free 0.06% that has nothing to do with your
   index (`DECISIONS.md` D17).
4. **The number to beat is 30.9 ms per query.** That is brute force with the
   fastest kernel this machine can run, and it is pure memory bandwidth — 488
   MiB streamed per query. No kernel improves it. HNSW wins by not visiting a
   million vectors, and that is the whole argument for the phase.

---

## Progress

| Phase | Name | Status | Closed | Key number |
|---|---|---|---|---|
| 0 | Bootstrap | **closed** | 2026-08-23 | 8/8 tests, 3 presets green |
| 1 | Data + brute force | **closed** | 2026-08-24 | recall@10 = 1.000000 tie-aware |
| 2 | SIMD kernels | **closed** | 2026-08-27 | AVX2 11.1× scalar; 32.3 QPS brute force |
| 3 | HNSW | **in progress** | — | recall@10, QPS |
| 4 | Benchmark harness | not started | — | full curve vs hnswlib |
| 5 | Product quantization | not started | — | bytes/vector, recall loss |
| 6 | **Filtered search** | not started | — | **collapse curve** |
| 7 | The attempt | not started | — | delta vs baselines |
| 8 | Showcase | not started | — | deployed URL |

**Resume-ready after Phase 4. Differentiating after Phase 6.**

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
```

---

## Numbers recorded so far

Full detail, methodology and regeneration commands in `BENCHMARKS.md`.

| Metric | Value | Machine | Date |
|---|---|---|---|
| Load, 1M × 128 | 0.33–1.01 s (page-cache dependent) | M1 | 2026-08-27 |
| Peak RSS, SIFT1M | 501.6 MiB | M1 | 2026-08-27 |
| Brute force, scalar | 11.58 QPS, k=10 | M1 | 2026-08-24 |
| **Brute force, AVX2** | **32.32 QPS**, k=10 (median of 3, 4.6% spread) | M1 | 2026-08-27 |
| — per query | 30.9 ms, 15.4 GiB/s — bandwidth-bound | M1 | 2026-08-27 |
| recall@10, tie-aware | **1.000000** (identical under every kernel) | M1 | 2026-08-27 |
| recall@10, strict id-set | 0.999440 (identical under every kernel) | M1 | 2026-08-27 |
| AVX2 vs scalar, compute-bound | **11.11×** (dim 128), 16.57× (dim 960) | M1 | 2026-08-27 |
| AVX2 vs scalar, memory-bound | 3.51× (dim 128), 3.99× (dim 960) | M1 | 2026-08-27 |
| SIFT1M distinct vectors | 985,462 of 1,000,000 | — | 2026-08-24 |
| Tests passing | 71/71 on 3 presets | M1 | 2026-08-27 |

**Machine M1:** AMD Ryzen 5 7530U (Zen 3, 6C/12T, 4546 MHz max), 15 GiB RAM,
GCC 11.4.0, CMake 4.3.1, Ninja 1.13.2. AVX2 + FMA, no AVX-512.

---

## Open questions

Things you don't understand yet and must resolve before the phase closes.
An empty list at phase close means you either understood everything or weren't
paying attention.

- [ ] What `ef` actually buys on SIFT1M, and where the recall/QPS curve bends.
- [ ] How much the neighbour-selection *heuristic* (Algorithm 4) is worth over
      plain nearest-M. Implement both, measure the gap, keep the number — it is
      the single most-asked question about an HNSW implementation.
- [ ] Does batching neighbour distances through `distances_to()` actually help
      inside `SEARCH-LAYER`, where the batch is M ≈ 16–32 rather than 256?
      Phase 2 showed the batch is where the parallelism comes from, but never
      measured it at graph-sized batches.

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

