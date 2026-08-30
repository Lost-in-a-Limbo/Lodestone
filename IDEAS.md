# Ideas

Scope-creep containment. Anything not in the current phase's deliverables lands
here instead of in the code. Scope creep is the primary failure mode of this
project — `PRD.md` says so twice.

An entry here is not a promise. Most of these should never be built.

---

## Deferred with a reason

### AVX-512 distance kernels
**Raised:** Phase 0, assumption C13. **Blocked on:** hardware.

`PRD.md` Phase 2 lists SSE / AVX2 / AVX-512 with runtime dispatch, and
`README.md`'s roadmap repeats it. The development machine is Zen 3, which has
no AVX-512, so the kernel could be written but never tested or measured —
and an untested SIMD kernel in a project whose entire value is measurement
discipline is worse than an absent one.

Phase 2 ships SSE + AVX2 with runtime dispatch. The dispatch table is the part
that matters; adding a fourth entry later is mechanical. If a Zen 4 / Sapphire
Rapids machine ever becomes available, this is a half-day.

### Warnings as errors
**Raised:** Phase 0.

`-Werror` in CI is cheap and valuable, but `-Wconversion` fires inside Catch2's
macros when they expand in our test translation units, so turning it on needs
Catch2's headers marked `SYSTEM` first. Not worth a Phase 0 detour. Revisit
when the test suite is large enough that a warning can hide in it.

### Brute force over an explicit candidate subset
**Raised:** Phase 1 task 4. **Needed by:** Phase 6, strategy A.

`brute_force_knn` scans the contiguous range `[0, count)`. Phase 6's pre-filter
strategy needs the same selection logic over an arbitrary id subset — the set
that passed the predicate. That is a second entry point sharing one internal
scan, not a rewrite, and the bounded-heap and blocking logic transfer unchanged.

Deliberately not built now. A filter strategy written before there is a graph
to filter is a guess, and the signature is the only part that had to be got
right early — which it was, by taking `count` rather than reaching into a store.

### A multi-accumulator *scalar* baseline
**Raised:** Phase 2 task 5. **Why it matters:** honesty about the headline.

The published "AVX2 is 11.1× scalar" compares tuned AVX2 against *naive* scalar,
and the naive scalar kernel is latency-bound on a dependent float-add chain at
~3 cycles per element. So the 11.1× combines two wins — vectorisation and
breaking a dependency chain — and only the first is really SIMD.

A scalar kernel with 4 independent accumulators would be meaningfully faster and
would narrow the gap. We have not measured by how much, so the honest statement
is that the SIMD-only contribution is somewhere below 11.1× and above the ≥3×
the phase required.

Not built because it is a benchmark artefact, not a kernel anyone would ship,
and Phase 2's exit criterion is met under any reading. Worth an hour if the
number is ever quoted outside this repo.

### Measure `distances_to()` at graph-sized batches
**Raised:** Phase 2 task 5. **Needed by:** Phase 3.

Phase 2 established that batching independent distances is where the
instruction-level parallelism comes from — it beats extra accumulators at
dim 128. But every measurement used a batch of 256, matching brute force.
`SEARCH-LAYER` will call with M ≈ 16–32. Whether the overlap still materialises
at that size is unmeasured and directly affects Phase 3's inner loop.

### DONE — 64-byte-align the prepared query buffer
**Raised:** Phase 1 task 3. **Answered:** Phase 2 task 4.

Measured at 0–2%, at or barely above noise. Kept anyway, because a 32-byte
`_mm256_load_ps` from the 16-byte-aligned base `std::vector<float>` guarantees
is undefined behaviour — correctness, not speed. See `DECISIONS.md` D23.

<details>
<summary>Original entry</summary>

`ScalarL2Computer::query_` is a plain `std::vector<float>`, so it is 16-byte
aligned at best. The stored vectors are 64-byte aligned and the query is not,
which means Phase 2's AVX2 kernel will want `_mm256_loadu_ps` on the query side
even while it can use aligned loads on the store side.

The reason not to fix it blind: the query stays resident in L1 for an entire
search — it is the same 512 bytes read over and over — whereas the store side
streams fresh cache lines on every distance. Unaligned loads on L1-resident
data have been close to free on x86 since Nehalem. So this is plausibly worth
nothing, and Phase 2 should *measure* it rather than pay for an aligned
allocator on the strength of an argument.
</details>

### Finish vendoring CPM
**Raised:** Phase 0, see DECISIONS.md D6.

Copy the downloaded `CPM.cmake` over `cmake/CPM.cmake` and fill in
`CPM_HASH_SUM` from the upstream release page, restoring the
offline-cold-clone property. Small, mechanical, do it during Phase 1.

---

### Measured and decided against

Kept here so the same idea does not get re-raised as an optimisation.

- **Avoiding the double copy in `load_fvecs`** (file → scratch → store). Would
  need `VectorStore` to expose writable uninitialised slots, which hands the
  zero-padding contract to the caller and lets a vector be left half-written.
  Measured: the read-and-copy portion of a 1M load is ~60–120 ms warm-cache, so
  there is nothing here worth that risk.
- **Chunked reads instead of the per-record loop.** The plan flagged this as a
  possible follow-up if load time was embarrassing. It is 0.35 s warm for
  516 MB, of which two thirds is page-fault cost that a different read strategy
  would not touch. Not worth doing.

## Untriaged

Things worth thinking about, in no order, with no commitment.

- Prefetching the next neighbour's vector while computing the current
  distance, inside `distances_to()`. This is exactly what the batch method in
  D1 was shaped to allow, so it should be measured in Phase 2 rather than
  guessed at.
- Storing neighbour lists in a separate arena from vectors, so a graph
  traversal touches only id-sized data until it needs a distance.
- `results.json` schema versioning, before Phase 8's frontend starts depending
  on the shape. Cheap now, annoying later.
