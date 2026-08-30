# Phase 3 — HNSW

**Written:** 2026-08-27
**Goal:** the index itself. The largest single phase.
**Reference:** Malkov & Yashunin, arXiv 1603.09320. Algorithms 1–5 *are* the
spec; the code should be readable side by side with the paper.
**Exit criteria (PRD §6):** layer assignment; greedy descent; `SEARCH-LAYER`
with a candidate heap and an `ef`-bounded result list; the neighbour selection
**heuristic**, not just nearest-M; bidirectional insertion with pruning past
`M_max`; serialisation; recall@10 ≥ 0.95 on SIFT1M at some `ef`; QPS within 5×
of `hnswlib` at matched recall; build under 20 min single-threaded on 1M.
**Record:** build time, index memory, the recall/QPS curve at
`ef ∈ {8,16,32,64,128,256}`.

---

## The number this phase has to beat

Brute force with the fastest kernel this machine can run: **30.9 ms per query,
32.3 QPS, recall 1.000000.** That time is pure memory bandwidth — 488 MiB
streamed per query — so no kernel work can improve it. HNSW wins only by not
visiting a million vectors.

At `ef = 64` the graph should visit a few thousand nodes instead of 1,000,000.
If it does not beat 30.9 ms by two orders of magnitude, something is wrong with
the traversal, not with the constant factors.

---

## Assumptions (override any of these at review)

| # | Assumption |
|---|---|
| A1 | **Recall is reported tie-aware.** SIFT1M has 14,538 duplicate vectors; the strict id-set measure costs a free 0.06% that has nothing to do with the index (D17). Both printed, the tie-aware one asserted. |
| A2 | Single-threaded build and search, per the project methodology. Concurrent insert is a different design and is out of scope. |
| A3 | The index **references** a `VectorStore` and does not own it. Serialisation writes the graph only; the store is reloaded from `.fvecs` separately. Duplicating 488 MiB into an index file to make one API prettier is not a trade worth making. |
| A4 | Per-query scratch (visited set, heaps) lives in the index as mutable members, so `search()` is `const` but **not thread-safe**. Documented; a scratch parameter is the Phase 4 fix if threading arrives. |
| A5 | Distances go through `DistanceComputer` only. If `l2_distance(` appears in `hnsw.cpp`, the phase has failed. |
| A6 | Layer-0 neighbour lists live in one flat arena. Upper layers get a per-node allocation — they hold ~1/M of nodes per level and are touched ~log N times per query, so the pointer chase there is not on the hot path. |

---

## Design decisions to make before any code

### Graph storage

Layer 0 holds every node and carries the accuracy-critical hops. One flat
arena, stride `m_max0 + 1` ids, slot 0 holding the count:

```
level0_[i * (m_max0+1)]      = degree
level0_[i * (m_max0+1) + 1 .. ] = neighbours
```

At `M = 16`, `m_max0 = 32`, that is 33 × 4 = 132 bytes per node — a little over
two cache lines. D2's "one cache line at M=16" claim is about the *upper*
layers, where the budget is `M` not `2M`; layer 0 was always going to be wider.
Worth stating plainly rather than letting the earlier claim stand unqualified.

### The visited set is a performance decision, not a detail

`SEARCH-LAYER` must ask "have I seen this node?" thousands of times per query.
A `std::vector<bool>` cleared per query is O(N) per query — at 1M nodes that
alone would dwarf the search. Use **epoch stamping**: a `vector<uint32_t>` of
marks plus a per-query counter, so clearing is `++epoch` and the check is one
compare. 4 MB at 1M nodes, and it makes the reset O(1).

### Determinism

Same total order as Phase 1: `(distance, id)`. Ties broken by id so a rebuild
with the same seed produces the same graph and the same answers. A benchmark
you cannot reproduce is not a measurement.

---

## Task list

Eight tasks. 1→2→3→4→5 are strictly ordered; 6 and 7 need all of them.

### Task 1 — Graph storage, level assignment, visited set
`include/lodestone/hnsw.hpp`, `src/hnsw.cpp`, `tests/test_hnsw.cpp`

Level assignment is `floor(-ln(U(0,1)) · mL)` with `mL = 1/ln(M)`, seeded from
`HnswConfig::seed`. Tests: the level distribution is geometric with the right
ratio (~1/M of nodes at each level up); the same seed gives the same levels; the
arena addresses the right slots; the visited set survives an epoch wraparound.

### Task 2 — `SEARCH-LAYER` (Algorithm 2)
The heart of the index. A min-heap of candidates, a max-heap of results bounded
by `ef`, and the early termination when the nearest candidate is further than
the furthest result.

**Uses `distances_to()` on the whole neighbour list**, not `distance_to()` in a
loop — Phase 2 established that batching independent distances is where the
instruction-level parallelism comes from (D22). This is the first consumer at
graph-sized batches (M ≈ 16–32), which `IDEAS.md` flags as unmeasured.

Tests on a hand-built graph where the right answer is known by construction,
including: `ef = 1` degenerates to greedy; a larger `ef` finds a node greedy
misses; termination actually terminates on a cyclic graph.

### Task 3 — `SELECT-NEIGHBORS`, simple and heuristic (Algorithms 3 and 4)
Both, because the *comparison* is the point. Simple keeps the M nearest.
The heuristic keeps `e` only when `e` is closer to the query than to any already
selected neighbour — which produces a diverse, long-range-connected graph
instead of a cluster of mutually-redundant near neighbours.

Test that constructs a case where they differ and asserts the heuristic picks
the diverse set. Both stay in the code so task 7 can measure the recall gap;
this is the most-asked question about an HNSW implementation.

### Task 4 — `INSERT` (Algorithm 1)
Greedy descent from the entry point down to the new node's level, then
`SEARCH-LAYER` at `ef_construction` for each layer below, neighbour selection,
**bidirectional** linking, and pruning any neighbour that exceeded `m_max`.

Tests: every edge is reciprocal; no node exceeds its degree budget; the entry
point tracks the highest level; a fully-built small graph is connected.

### Task 5 — `K-NN-SEARCH` (Algorithm 5) and SIFT10K recall
Descend with `ef = 1`, then one `SEARCH-LAYER` at layer 0 with the caller's
`ef`. Registered as a ctest entry requiring recall@10 ≥ 0.95 on SIFT10K, which
is small enough to run under the sanitisers.

### Task 6 — Serialisation
Magic, version, config, entry point, levels, links. Round-trip test asserting
the reloaded index returns **bit-identical** results, not merely similar recall.

### Task 7 — SIFT1M: build, curve, and the honest comparison
Build 1M vectors, record build time and index memory. Sweep
`ef ∈ {8,16,32,64,128,256}`, recording recall@10 (tie-aware and strict), QPS,
and nodes visited per query. Then the heuristic-versus-simple gap at one `ef`.

`hnswlib` comparison is **Phase 4's** job — this phase records our own curve and
notes where it stands relative to the published expectation.

### Task 8 — Close
`BENCHMARKS.md` (curve, build time, memory), `DECISIONS.md`, `PHASE.md` →
Phase 4, `IDEAS.md`.

---

## What is explicitly NOT in this phase

No filtering, no quantisation, no `results.json`, no `hnswlib` linkage, no
threading, no CLI11, no deletion or update of inserted elements.
