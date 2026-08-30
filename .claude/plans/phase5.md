# Phase 5 — Product quantization

**Written:** 2026-08-30
**Goal:** compress the vectors, measure what accuracy it costs. This is the ML
content of an ML-systems project: k-means, lossy compression, and the
accuracy/memory frontier.
**Exit criteria (PRD §6):** per-subspace k-means with 256 centroids → one byte
per subspace; codebook training; asymmetric distance with precomputed lookup
tables; 512 bytes → 16 bytes (32×); recall loss quantified at each ratio; you
can explain asymmetric versus symmetric; **and it plugs into the distance
interface without touching `hnsw.cpp`.**
**Record:** memory per vector and recall@10 at m ∈ {8, 16, 32}.

---

## This phase is the test of a decision made in Phase 0

D1 chose an abstract `DistanceComputer` with `prepare_query()` over the faster
alternative — a concept plus templating the index on the kernel type — and the
justification given at the time was, verbatim, that PQ needs per-query state:

> the query is projected once into a 256×m lookup table, and every distance
> after that is m table lookups and adds.

That claim has been sitting unexercised for four phases. Either it pays off here
or the seam was over-designed. **If Phase 5 finds itself editing the graph
algorithms, D1 was wrong and that is the finding to report.**

---

## How PQ works, stated once so the code can be short

Split each 128-dim vector into `m` contiguous subspaces of `128/m` dims. Run
k-means with 256 centroids **independently per subspace**. Each vector becomes
`m` bytes: one centroid index per subspace.

| m | sub-dim | bytes/vector | reduction |
|---|---|---|---|
| 8 | 16 | 8 | 64× |
| **16** | **8** | **16** | **32×** ← the exit criterion |
| 32 | 4 | 32 | 16× |

**Asymmetric distance (ADC), which is what we implement.** The query stays
full-precision float. On `prepare_query()`, compute for every subspace `s` and
every centroid `c` the squared distance from the query's `s`-th slice to
centroid `c` — an `m × 256` table. Then any distance is `m` lookups and `m`
adds:

```
d(q, x) ≈ Σ_s  table[s][code_x[s]]
```

**Symmetric (SDC), which we do not.** Quantize the query too, and precompute a
256×256 inter-centroid table per subspace once at training time. The table is
then query-independent, so `prepare_query()` is free — but the query's own
quantization error is added to the stored vector's, and accuracy is
meaningfully worse for the same code size. ADC's table costs `m × 256` distance
computations per query, which is 4,096 at m=16 — real, but amortised over a
million distances.

The asymmetry is the whole trick: **only one side of the comparison is
approximated.**

---

## Assumptions (override any at review)

| # | Assumption |
|---|---|
| A1 | Trained on `sift_learn.fvecs` (100,000 vectors), not on the base set. Training a codebook on the corpus it will encode is a mild form of testing on the training set, and the dataset ships a learn split precisely to avoid it. |
| A2 | 256 centroids, fixed. That is what makes a code exactly one byte; 512 would need 9 bits and the packing would cost more than it saves. |
| A3 | k-means++ initialisation, fixed seed, Lloyd's iterations to a fixed count. Reproducibility over convergence guarantees — a codebook you cannot rebuild is a benchmark you cannot repeat. |
| A4 | Empty clusters are re-seeded from the farthest point, not left empty. An empty cluster wastes a code and silently reduces the effective codebook size. |
| A5 | The primary measurement is **brute-force ADC over the whole corpus**, which isolates quantization loss from graph loss. HNSW+PQ is reported second, as the practical configuration. |
| A6 | No re-ranking. Real systems re-score PQ candidates with exact distances; that recovers most of the recall and would hide exactly the loss this phase exists to measure. Logged in `IDEAS.md`. |

---

## Task list

### Task 1 — `ProductQuantizer`: training and encoding
`include/lodestone/quantizer.hpp`, `src/quantizer.cpp`, `tests/test_quantizer.cpp`

k-means++ init, Lloyd's iterations, empty-cluster repair, per-subspace
codebooks, encode a whole `VectorStore` to `m` bytes each. Tests on synthetic
data with known cluster structure, where the right codebook is knowable.

### Task 2 — The ADC `DistanceComputer`
The table build in `prepare_query()`, the sum in `distance_to()`. Tests that it
approximates exact L2, that error falls as m rises, and that it goes through the
factory like every other kernel.

### Task 3 — Wire PQ into HNSW without touching the algorithms
`make_hnsw_index` builds its own computers from `(store, metric)`. PQ needs a
different construction path, so an overload taking a computer *factory* is
added and the existing entry point delegates to it.

**That is a change to `hnsw.cpp`, and the honest claim is narrower than the exit
criterion's wording:** zero lines of `SEARCH-LAYER`, `INSERT`,
`SELECT-NEIGHBORS` or the search path change. Report the actual diff, not a
claim of "no changes".

### Task 4 — Measure
Brute-force ADC at m ∈ {8, 16, 32}: recall@10 against exact ground truth,
bytes/vector, and QPS. Then HNSW+PQ at the same m. Both into `results.json`.

### Task 5 — Close
`BENCHMARKS.md` recall-versus-memory curve, `DECISIONS.md`, `PHASE.md` →
Phase 6, `IDEAS.md`.

**Apply the Phase 4 warmup fix before measuring, not after** — `IDEAS.md` says
discard the first measured run as well as the warmup, and Phase 4's numbers
showed why.

---

## NOT in this phase

No filtering, no re-ranking, no OPQ or rotation, no SDC, no GPU, no
multi-threading, no frontend.
