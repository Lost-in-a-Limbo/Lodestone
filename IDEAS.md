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

### Finish vendoring CPM
**Raised:** Phase 0, see DECISIONS.md D6.

Copy the downloaded `CPM.cmake` over `cmake/CPM.cmake` and fill in
`CPM_HASH_SUM` from the upstream release page, restoring the
offline-cold-clone property. Small, mechanical, do it during Phase 1.

---

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
