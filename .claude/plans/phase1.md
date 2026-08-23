# Phase 1 — Data and brute force

**Written:** 2026-08-23
**Goal:** ground truth. You cannot measure recall without knowing the right answer.
**Exit criteria (PRD §6 Phase 1):** `.fvecs`/`.ivecs` parsers; aligned vector
store; brute-force exact k-NN; recall@k calculator; loads 1M vectors reporting
load time and RSS; **brute-force recall@10 against the provided ground truth
= 1.000 exactly.** Any deviation means the parser or the metric is wrong —
stop and fix, do not proceed.
**Record:** load time, memory footprint, brute-force QPS (expect single digits).

---

## The one thing that makes this phase different

Phase 1 is where a wrong answer is *invisible*. A parser that reads the file
with a one-record offset still produces 1M plausible 128-dim vectors. A recall
calculator that averages wrongly still produces a number between 0 and 1.
Nothing crashes. The whole phase is therefore built around a single assertion
that cannot be satisfied by accident:

> recall@10 against the provided ground truth is **exactly 1.000**

Exact brute force over the same data the ground truth was computed from must
agree perfectly. 0.999 is not "close" — it means something is wrong, and the
phase does not close until it is 1.000 or the deviation is *explained* (see
Task 4 on ties).

---

## Assumptions (override any of these at review)

| # | Assumption |
|---|---|
| A1 | Distances are **squared** L2. Never take the sqrt — it is monotone, so it cannot change a ranking, and it costs a sqrt per distance. Ground-truth *ids* are unaffected. |
| A2 | Little-endian host. The format is LE; `static_assert` on `std::endian::native` so a hypothetical BE port fails loudly instead of silently producing garbage. |
| A3 | Padding floats in the store are **zeroed**, not garbage. This is load-bearing — see Task 1. |
| A4 | `ctest` runs SIFT10K (siftsmall) only. SIFT1M is a manual release-preset run. Follows Phase 0 assumption C15. |
| A5 | Ground-truth ids are 0-based indices into the base file, and each row is sorted ascending by distance. Both are **verified at load**, not assumed. |
| A6 | No CLI11 yet. The Phase 1 tool takes positional paths. CLI11 arrives in Phase 4 when there are real flags to parse — adding a dependency for `argv[1]` is not justified. |
| A7 | No mmap. Plain buffered reads. `mmap` would be faster but adds platform code for a one-off load, and persistence is an explicit non-goal. |
| A8 | RSS is read from `/proc/self/status` `VmHWM` (peak). Linux-only, guarded, reported as unavailable elsewhere. |

### Two files not in PRD §7's list

PRD §7 is the *bootstrap* list, not the final tree. Phase 1 adds:

- **`io.hpp` / `io.cpp`** — the parsers. A file-format reader is not storage and
  does not belong in `vector_store.cpp`.
- **`brute_force.hpp` / `brute_force.cpp`** — exact k-NN and the recall metric.

Both are flagged here rather than slipped in, because "the plan said six source
files and the diff has ten" is exactly how scope creep gets normalised.

---

## Task list

Seven tasks. Tasks 1→2→3→4 are strictly ordered by dependency; Task 5 is
independent and can run any time; Tasks 6–7 need everything.

---

### Task 1 — `VectorStore::reserve` and `add`

**Test first:** `tests/test_vector_store.cpp`
**Touches:** `src/vector_store.cpp` only (the header shape is already fixed)

Phase 0 left `reserve()` returning `not_implemented` and `add()` returning
`nullopt`. This task is the stride and alignment maths, and it is the one place
in the phase where the interesting decision lives.

**The stride rule.** Every vector must *start* on a 64-byte boundary, not just
the first one. 64 bytes is 16 floats, so:

```
stride_floats = ceil(dim / 16) * 16
```

| dim | stride | padding waste |
|---|---|---|
| 128 (SIFT) | 128 | 0% |
| 960 (GIST) | 960 | 0% |
| 100 | 112 | 10.7% |
| 129 | 144 | 10.4% |

**Note what that table says: both dimensions this project actually uses pad to
nothing.** SIFT's 128 and GIST's 960 are already multiples of 16. So a stride
bug is invisible on every dataset in the PRD. **Every store test therefore runs
at dim 100 as well as dim 128** — this is the open question `PHASE.md` recorded
at the end of Phase 0, and it is answered by testing the case the data hides.

**Why padding is zeroed, and why it is not merely tidiness.** With zeroed
padding, a distance kernel may process all `stride` floats instead of exactly
`dim`, because for squared L2 each padding term is `(0 - 0)² = 0` and for inner
product `0 * 0 = 0`. Both contribute nothing.

That buys Phase 2 something concrete: an AVX2 kernel can run whole 8-wide
iterations across the full stride with **no scalar tail loop** and no masked
final load, because the stride is always a multiple of 16. The tail is the
fiddliest and buggiest part of a hand-written SIMD kernel, and this deletes it.
The cost is one `memset` of the allocation at reserve time — ~0.1 s for 512 MB,
paid once.

It also makes the RSS number honest: zeroing touches every page, so `VmHWM`
reflects memory actually resident rather than lazily-unfaulted address space.

**Allocation.** `std::aligned_alloc(64, bytes)`. C11 requires `bytes` to be a
multiple of the alignment — glibc tolerates violations but it is UB, so this
must be guaranteed rather than hoped for. It falls out for free: `stride * 4` is
a multiple of 64 by construction, so `capacity * stride * 4` is too. Paired with
plain `free()`, which the Phase 0 destructor already does.

**Tests** (dim 128 *and* dim 100 for each, where applicable):
- `stride()` == 128 for dim 128, 112 for dim 100, 16 for dim 1
- base pointer is 64-aligned; **`get(i)` is 64-aligned for every i**, not just 0
- padding floats read back as exactly 0.0f
- `add()` returns 0, 1, 2, … ; `get()` round-trips the payload bit-exactly
- `add()` past capacity → `nullopt`, and `size()` unchanged
- `add()` with `src.size() != dim` → `nullopt`
- `reserve(0, n)` and `reserve(d, 0)` → `Status::invalid_argument`
- `reserve()` twice frees the first block (ASan proves it; the debug preset has
  it on)
- a moved-from store is empty and destructible; `bytes()` tracks capacity

**Proof command:** `./build/debug/tests/lodestone_tests "[vector_store]"`

---

### Task 2 — `.fvecs` / `.ivecs` parsers

**Test first:** `tests/test_io.cpp`
**Creates:** `include/lodestone/io.hpp`, `src/io.cpp`

**Format.** One record is a 4-byte little-endian `int32` dimension followed by
that many `float32` (`.fvecs`) or `int32` (`.ivecs`). Records repeat to EOF.
Record size is `4 + 4*d`, uniform across the file.

**The integrity checks are the point of this task.** The dimension prefix is
repeated on every single record, which is redundant — and redundancy is free
validation. A parser that ignores it will happily misread a truncated or
misaligned file:

1. Read `d` from record 0. Reject `d == 0` or `d > 65536` (a sane ceiling; GIST
   is 960).
2. `file_size % (4 + 4*d) != 0` → `io_error`. This catches truncation, which is
   the single most likely real-world failure after an interrupted download.
3. `count = file_size / (4 + 4*d)`.
4. **Verify the dimension prefix on every record while reading**, not just the
   first. Costs one comparison per vector against a load that is
   memory-bandwidth-bound anyway, and it is what turns "silently off by one
   record" into a loud error.

**Reading strategy.** The 4-byte prefix must be stripped per record, so a single
bulk read into the destination is not possible. Start with the simple loop —
`read(&d, 4)`, `read(dest, 4*d)` — with an enlarged stream buffer via
`pubsetbuf`, which makes each call a memcpy out of the stream buffer rather than
a syscall. That is 2M calls for SIFT1M; expected to be ~1–2 s and
bandwidth-bound.

Load time is a *recorded number*, not a gated one. So: measure it, write it
down, and only optimise (chunked read + in-buffer header stripping) if the
number is embarrassing. Optimising first would be guessing.

**API:**
```cpp
struct FvecsInfo { std::size_t dim; std::size_t count; };
std::optional<FvecsInfo> probe_fvecs(const std::filesystem::path&);
Status load_fvecs(const std::filesystem::path&, VectorStore& out);

struct IvecsData {                       // flat, with a row() accessor
  std::vector<std::int32_t> data;        // never vector<vector<int32_t>>
  std::size_t dim = 0, count = 0;
  std::span<const std::int32_t> row(std::size_t i) const;
};
std::optional<IvecsData> load_ivecs(const std::filesystem::path&);
```

`probe_fvecs` exists so a caller can size the store before allocating —
`load_fvecs` calls `reserve()` itself, which is why the store must not grow.

`IvecsData` is flat even though ground truth is read once and never touched on
the hot path. Consistency is cheaper than an exception to the rule that someone
later cites as precedent.

**Tests** — synthetic files written to a temp dir inside the test, so the suite
needs no downloaded data:
- round-trip: write 3 vectors at dim 4, read back bit-exactly
- round-trip at dim 100, so the store's padding path is exercised end to end
- `probe_fvecs` reports the right dim and count without reading the payload
- truncated file (drop the last 3 bytes) → `io_error`
- inconsistent dimension in record 2 → `io_error`
- `d == 0`, absurd `d`, zero-byte file, nonexistent path → the right `Status`
- `.ivecs` round-trip, including negative values (they are `int32`, not indices,
  as far as the parser is concerned)

**Proof command:** `./build/debug/tests/lodestone_tests "[io]"`

---

### Task 3 — the scalar L2 `DistanceComputer`

**Test first:** `tests/test_distance.cpp`
**Touches:** `src/distance_scalar.cpp`, `include/lodestone/distance.hpp`

**This task deliberately reaches into Phase 2's file, and here is why.** Brute
force needs a distance. Architecture rule 1 forbids `hnsw.cpp` — or anything
else — from calling a concrete kernel, so the distance must arrive through
`DistanceComputer`. Therefore Phase 1 cannot compute a single distance without
one concrete implementation existing. The alternatives are worse: computing L2
inline inside `brute_force.cpp` breaks rule 1 in the exact way `CLAUDE.md` says
turns a later phase into a rewrite.

So Phase 1 implements **scalar squared L2 only**. Phase 2 still owns everything
that phase is actually about: inner product, the SSE and AVX2 kernels, runtime
CPU dispatch, and the microbenchmarks. Nothing is taken from it.

There is a real benefit to this ordering. Phase 1 becomes the **first consumer
of the D1 interface**, which means the seam gets validated by working code now
rather than in Phase 3 when the graph depends on it. If `prepare_query` /
`distances_to` is the wrong shape, this is the cheapest possible moment to find
out.

**The factory is the seam.** Add to `distance.hpp`:
```cpp
std::unique_ptr<DistanceComputer> make_distance_computer(Metric, const VectorStore&);
```
Callers name a `Metric`, never a class. In Phase 1 the body returns the scalar
L2 computer; in Phase 2 the *same* body grows CPU feature detection and returns
an AVX2 computer instead. **Phase 2's dispatch work becomes one function body**,
and no caller changes. This is what D1 was for.

**Tests:**
- hand-computed squared L2 on 3-4-5 style vectors, exact
- no sqrt: distance of `[3,0,...]` to `[0,0,...]` is 9, not 3
- `distances_to()` batch agrees elementwise with `distance_to()`
- **at dim 100: the distance equals the sum over 100 terms, not 112.** This is
  the test that proves the zeroed-padding contract from Task 1 holds, and it
  fails loudly if `reserve()` ever stops zeroing.
- `dim()` and `metric()` report correctly
- `prepare_query` twice in a row, second query's answers are correct (no stale
  state — the failure mode that D1's no-copy rule exists to prevent)

**Proof command:** `./build/debug/tests/lodestone_tests "[distance]"`

---

### Task 4 — brute-force k-NN and recall@k

**Test first:** `tests/test_brute_force.cpp`
**Creates:** `include/lodestone/brute_force.hpp`, `src/brute_force.cpp`

```cpp
struct Neighbor { VectorId id; float distance; };
Status brute_force_knn(DistanceComputer&, const float* query,
                       std::size_t k, std::span<Neighbor> out);
double recall_at_k(std::span<const Neighbor> got,
                   std::span<const std::int32_t> truth, std::size_t k);
```

**Bounded max-heap of size k**, not sort-the-whole-thing. For k=10 over N=1M,
sorting 1M distances per query is 1M log(1M) comparisons where a size-10 heap is
N comparisons against the heap top plus a push only on the rare improvement.
After the first few hundred vectors almost every candidate loses to the top
immediately, so the common case is one float compare.

Distances are fetched in **blocks through `distances_to()`** into a small
scratch buffer, not one at a time. Batching is the whole reason D1 has a batch
method; the first real consumer should use it as intended, and a block sized to
fit L1 is what Phase 2's kernel will want.

**Ties — the honest handling.** Recall is computed by **id intersection**. If a
distance ties exactly at the k-th position, a different-but-equally-correct set
of ids scores below 1.000 through no fault of the implementation.

The response is *not* to loosen the metric into an epsilon comparison, which
would also hide real bugs. It is a **diagnostic**: when recall < 1.000, report
the offending query, the distances of the missed and the extra id, and whether
they are equal within float tolerance. Then the number is either explained (a
genuine tie, which for SIFT at rank 10 should be vanishingly rare) or it is a
bug. Either way it is *known*, which is what the exit criterion actually asks.

**Ground-truth validation** (A5, verified not assumed): every id `< count`, and
each row's distances non-decreasing under our own computer. If the provided
ground truth is not sorted by our metric, that is a signal our metric disagrees
with theirs — caught here rather than as a mysterious 0.97.

Ground truth rows are top-100; recall@10 takes the **first 10** entries.

**Tests** — all synthetic, no downloaded data:
- vectors placed on a line so the true top-k is known by construction; assert
  exact ids and distances
- k=1, k=k_max, k > store size (→ `invalid_argument`)
- duplicate distances present: assert exactly k results, no crash, no duplicates
- `recall_at_k` on hand-made sets: perfect → 1.0, half → 0.5, disjoint → 0.0
- `recall_at_k` where `got` is a permutation of `truth` → 1.0 (recall is a set
  measure, order within the k does not matter)

**Proof command:** `./build/debug/tests/lodestone_tests "[brute_force]"`

---

### Task 5 — `tools/download_sift.sh`

**Creates:** `tools/download_sift.sh`

Fetches `siftsmall.tar.gz` (SIFT10K, ~10 MB) and optionally `sift.tar.gz`
(SIFT1M, ~170 MB) into `data/`.

**Checksums are real, not invented.** The upstream corpus publishes an
`MD5SUM` file, confirmed reachable during planning:

```
0b8324a7a82d7f2663d7dcbd57642df7  siftsmall.tar.gz
b23d1b3b2ee8469d819b61ca900ef0ed  sift.tar.gz
```

These go in the script and are verified after download. This matters more than
usual here: a silently truncated `sift_base.fvecs` is precisely the input that
produces a plausible-looking recall of 0.98 and burns an evening.

**Second, independent check.** After extraction, assert exact file sizes, which
are derivable from the format rather than trusted from a third party —
`10000 × (4 + 128×4) = 5,160,000` bytes for `siftsmall_base.fvecs`,
`1000000 × 516 = 516,000,000` for `sift_base.fvecs`. Two independent integrity
checks, one from upstream and one from arithmetic.

`set -euo pipefail`; idempotent (skip what is already present and verified);
`curl --retry`; siftsmall by default, `--full` for SIFT1M.

**Proof command:** `./tools/download_sift.sh && ls -l data/siftsmall/`

---

### Task 6 — `tools/sift_check.cpp`, the command that proves the phase

**Creates:** `tools/sift_check.cpp`, `tools/CMakeLists.txt`

Two exit criteria — "loads 1M vectors, reports load time and RSS" and
"recall@10 = 1.000" — require something that prints numbers. This is that
thing, and it is **deliberately not the benchmark harness.** No JSON, no
parameter sweep, no latency percentiles, no `hnswlib`. Those are Phase 4, and
building them here would be the scope creep `PRD.md` §3 names as the primary
failure mode.

Takes three positional paths (base, query, ground truth), prints:

```
loaded 1000000 x 128 from sift_base.fvecs in 1.83 s
peak RSS 542 MiB   (store 504 MiB, 516 B/vector on disk)
brute force: 10000 queries, k=10, 7.4 QPS
mean recall@10 = 1.000000
```

Exits non-zero if mean recall@10 != 1.0, with the Task 4 tie diagnostic printed
for every offending query.

Registered as a ctest entry for **siftsmall only**, using
`SKIP_RETURN_CODE` so a checkout without `data/` still shows green rather than
red. CI has no dataset and must stay green; the laptop with data gets the real
assertion.

SIFT1M is run by hand under the release preset. Scalar non-vectorised L2 over
10,000 × 1,000,000 × 128 is ~1.3e12 multiply-adds — order of ten minutes, which
is exactly why A4 keeps it out of `ctest`. Note the irony worth writing down:
`-fno-tree-vectorize` on `distance_scalar.cpp` (D5) is what makes this slow, and
it is the right trade — a valid Phase 2 baseline is worth a slow Phase 1 script.

**Proof commands:**
```bash
ctest --test-dir build/debug --output-on-failure          # green, siftsmall
./build/release/tools/sift_check data/sift/sift_base.fvecs \
    data/sift/sift_query.fvecs data/sift/sift_groundtruth.ivecs   # must print 1.000000
```

---

### Task 7 — close the phase

- `BENCHMARKS.md`: load time, RSS, bytes/vector, brute-force QPS, recall@10,
  machine M1
- `DECISIONS.md`: zeroed padding and the no-tail-loop consequence; squared L2;
  the factory as the dispatch seam; scalar L2 landing in Phase 1 and why;
  id-intersection recall with a tie diagnostic rather than an epsilon
- `PHASE.md`: → Phase 2, tick Phase 1, log the session, record open questions
- `IDEAS.md`: anything that surfaced and was not built

---

## Verification for the whole phase

```bash
cmake --build build/debug && ctest --test-dir build/debug --output-on-failure
cmake --build build/asan  && ctest --test-dir build/asan  --output-on-failure
./tools/download_sift.sh --full
./build/release/tools/sift_check data/sift/sift_base.fvecs \
    data/sift/sift_query.fvecs data/sift/sift_groundtruth.ivecs
```

The `asan` run is not redundant with `debug`: it is the optimised sanitised
preset (D4), and it is the one that can afford to run over real data.

---

## Open questions to resolve before the phase closes

- [ ] What is the actual load time for 516 MB, and is the simple record loop
      good enough or does it need chunked reads?
- [ ] Does the 512 MB `memset` show up as a meaningful fraction of load time?
- [ ] Are there any exact distance ties at rank 10 in SIFT1M? If recall is
      1.000 the question is moot; if it is not, this is the first thing to check.
- [ ] Brute-force QPS: PRD expects single digits. If it is much faster than
      that, suspect that `-fno-tree-vectorize` is not being applied — the
      Phase 0 guard should make that impossible, so this doubles as a check on
      D5.

---

## Explicitly NOT in this phase

No HNSW, no graph, no SIMD kernels, no inner product, no runtime CPU dispatch,
no `results.json`, no `hnswlib`, no CLI11, no filtering, no quantisation, no
serialisation. If any of those appear in a Phase 1 diff, the phase has failed.
