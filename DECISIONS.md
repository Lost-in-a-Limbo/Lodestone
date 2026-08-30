# Decisions

One entry per non-obvious choice, with the alternative that lost and why. The
point is that in three months there is a written answer to "why is it like
that", and that an interviewer asking it gets a reason rather than a shrug.

Newest last.

---

## D1 — The distance seam is an abstract class with a batch method

**Phase 0.** `include/lodestone/distance.hpp`

`DistanceComputer` is an abstract base class with `prepare_query()`,
`distance_to()`, and `distances_to()`.

Three constraints had to hold simultaneously:

1. **Per-query state.** Phase 5's product quantisation uses asymmetric
   distance: the query stays float32, the stored vectors are one-byte
   subspace codes. The query is projected once into a 256×m lookup table, and
   every distance after that is m lookups and adds. A free function
   `f(a, b, dim)` has nowhere to keep that table. Hence `prepare_query()`.
2. **No indirect call per distance.** A virtual call on the hottest operation
   in the system would be self-defeating — which is what `distances_to()` is
   for. HNSW expands one node's neighbour list at a time, naturally a batch of
   M ids: one virtual call, M distances inside it. Dispatch amortises to
   nothing and the kernel can run AVX2 across the whole batch.
3. **Runtime kernel selection**, because Phase 2 dispatches on CPU feature
   detection.

**Rejected: a C++20 concept plus templating the index on the computer type.**
Strictly faster — full inlining, zero dispatch. But it binds the kernel at
compile time, and constraint 3 needs it bound at construction. Keeping it would
mean instantiating the entire index once per kernel and selecting between those
instantiations, which is a lot of object code and build time to avoid a call
that is already amortised over M distances. Revisit in Phase 3 **only if
profiling shows the batch call itself is hot** — not on principle.

---

## D2 — `VectorId` is `uint32_t`, not `size_t`

**Phase 0.** `include/lodestone/types.hpp`

Neighbour lists dominate index memory and are the hottest read in search. At
M = 16, a 32-bit list is 16 × 4B = 64B — exactly one cache line, one miss. With
64-bit ids the same list is 128B: two lines, two potential misses, on every
single node expansion. 1M vectors needs 20 bits; 32 leaves three orders of
magnitude of headroom, and the project's stated non-goals rule out
billion-scale, which is the only place this would bind.

`tests/test_types.cpp` asserts `sizeof(VectorId) == 4` so that widening it is a
deliberate act with a failing test attached, not a tidy-up.

---

## D3 — No `std::expected`. The error channel is `Status` + `std::optional`

**Phase 0.** Resolves assumption B4 in `.claude/plans/phase0.md`.

The plan assumed `std::expected` would be available because the compiler is
GCC 12, with a probe to confirm. **The plan's reasoning was wrong, and the
compiler version is not the reason.** `std::expected` is C++23, and libstdc++
gates `<expected>` behind `__cplusplus > 202002L`. Under `-std=c++20` with
`CXX_EXTENSIONS OFF` it is unavailable on *every* GCC, not merely old ones.

So `CLAUDE.md`'s "return `std::expected` or an error enum" resolves to the enum
branch for the whole project: `Status` for operations that only succeed or
fail, `std::optional<T>` alongside it where a value is also carried.

The probe still ships, as `check_cxx_source_compiles` in `CMakeLists.txt`, and
prints its verdict at configure time. A probe whose result is recorded is worth
keeping even when the answer is known — it is what will notice the day the
standard flag moves to C++23.

**Not chosen: bumping the project to C++20 + `-fconcepts` style partial C++23,
or vendoring `tl::expected`.** The first is a portability trap; the second adds
a dependency to the core, which architecture rule 3 forbids outright.

---

## D4 — Three presets, and `asan` is optimised rather than a second debug

**Phase 0.** `CMakePresets.json`

| Preset | Type | Flags |
|---|---|---|
| `debug` | Debug | `-O0`, ASan + UBSan, no `-march=native` |
| `release` | Release | `-O3 -march=native` |
| `asan` | RelWithDebInfo | `-O2 -g -march=native`, ASan + UBSan |

`-O0` plus ASan is roughly 20–50× slower than release, which is unusable
against 1M vectors — so the sanitised preset that can actually run a realistic
workload has to be optimised. It also catches the class of bug that only
appears after the optimiser has been at the code, which `debug` structurally
cannot see.

`debug` deliberately omits `-march=native`: its job is to find bugs portably,
and a sanitiser finding is not more valuable for being AVX2-flavoured.

All three use Ninja, and all use `-fno-sanitize-recover=all` so UBSan aborts
rather than printing and continuing — a warning nobody reads is not a check.

---

## D5 — `-fno-tree-vectorize` pinned to `distance_scalar.cpp`, with a build guard

**Phase 0.** `CMakeLists.txt`, `src/distance_scalar.cpp`

Under the release preset's `-O3 -march=native`, GCC auto-vectorises a plain L2
loop into AVX2. Phase 2's exit criterion is "AVX2 L2 is ≥3× scalar" — measured
against an auto-vectorised baseline, that comparison is AVX2-vs-AVX2 and the
number is meaningless. Building the isolation in during Phase 0 means Phase 2's
measurement is valid the first time, rather than being re-run after someone
notices the speedup is suspiciously close to 1.0.

The flag alone is fragile: it is one line in a build file, far away from the
code whose correctness depends on it. So the same `COMPILE_OPTIONS` property
also passes `-DLODESTONE_SCALAR_NOVEC=1`, and `distance_scalar.cpp` `#error`s
if that define is absent. Both live in one property and cannot drift apart, so
deleting the flag breaks the build loudly instead of quietly inflating a
published number.

`-mavx2 -mfma` are scoped the same way, to `distance_avx2.cpp` only.
Enabling them globally would let the compiler emit AVX2 in the scalar baseline
*and* in the dispatch code that is supposed to run on machines without it —
which would defeat Phase 2's runtime feature detection by crashing before it
could dispatch. That file `#error`s on missing `__AVX2__` / `__FMA__` for the
same reason.

---

## D6 — CPM is bootstrapped from a pinned version, not vendored

**Phase 0.** `cmake/CPM.cmake`

Plan task 2 wanted the full ~30KB of CPM committed, so a cold clone configures
with no network. The Phase 0 session had no outbound network from the shell, so
what shipped is the official bootstrap shim: one pinned version
(`0.40.2`), downloaded once into a shared source cache.

The pinning guarantee — the actual reason vendoring was wanted, since a
dependency manager that silently changes version under you is worse than none —
is preserved. What is lost is only the offline-cold-clone property.

`CPM_HASH_SUM` is deliberately left empty rather than filled with a
remembered value: an unverified hash looks like a guarantee and isn't one.
Open item, tracked in `IDEAS.md`: complete the vendoring and fill the hash
after the first successful configure.

---

## D7 — Sources are listed explicitly; no globs

**Phase 0.** `CMakeLists.txt`

`file(GLOB ...)` is evaluated at configure time, so a newly added source
silently does not build until something unrelated triggers a reconfigure. The
failure mode is a test that passes because the code under test was never
compiled, which is the worst kind of green.

---

## D8 — Types are `PascalCase`, contradicting `.claude/plans/phase0.md`

**Phase 0.**

`CLAUDE.md` states: snake_case for functions and variables, PascalCase for
types. `.claude/plans/phase0.md` sketched `vector_id` and `distance_computer`
in lower_case. Both files are authoritative-looking; they cannot both be
followed.

`CLAUDE.md` won, because it is the standing convention document and the plan's
mention was incidental — it was making a point about *width* (`uint32_t` vs
`size_t`), not about case. So: `VectorId`, `Status`, `HnswConfig`,
`SearchParams`, `Metric`, `DistanceComputer`, `VectorStore`.

Enumerators stay lower_case (`Status::ok`, `Metric::l2`) — they are values, not
types. Encoded in `.clang-tidy` so the decision is enforced rather than
remembered.

---

## D9 — One test binary, not one per source file

**Phase 0.** `tests/CMakeLists.txt`

`CLAUDE.md` says one test *file* per source file, which is kept. But Catch2 v3
is a compiled library, so one *binary* per file means one link against Catch2
per file, and Phase 3 alone will add several. `catch_discover_tests()` still
registers every `TEST_CASE` as its own `ctest` entry, so no granularity is lost
in the place where granularity matters — the test report.

---

## D10 — `.claude/plans/` is tracked; `.claude/settings.local.json` is not

**Phase 0.** `.gitignore`

The plans are the record of why the code looks the way it does, and belong in
history next to the diffs they produced. Local permission settings are machine
state.

---

## D11 — CI builds both GCC 11 and GCC 12

**Phase 0.** `.github/workflows/ci.yml`

The plan specified gcc-12 only. The development laptop has the Ubuntu 22.04
default, gcc-11. Pinning CI to a compiler the laptop does not have lets the two
diverge unnoticed, so the matrix covers both — four jobs, `{debug, release} ×
{g++-11, g++-12}`, which is free on public runners.

CI contains **no timing assertions and never will**. GitHub's runners are
shared Intel hosts with different cache behaviour and AVX-512; a performance
gate there would be measuring a machine that isn't the one publishing numbers.
CI proves correctness; the laptop produces every published number.

---

## D12 — Every distance is squared. The sqrt is never taken

**Phase 1.** `src/distance_scalar.cpp`

sqrt is monotone, so it cannot change a ranking, and it would cost a
transcendental on the hottest path in the system. Nothing downstream un-squares
it — not the recall metric, not the benchmark output, not `Neighbor::distance`.
`tests/test_distance.cpp` asserts that the distance from `[3,0,0,0]` to the
origin is 9 rather than 3, specifically so that a later "fix" adding the sqrt
back fails a test instead of silently halving the throughput.

---

## D13 — Padding is zeroed, and `prepare_query` copies

**Phase 1.** `src/vector_store.cpp`, `src/distance_scalar.cpp`

`VectorStore::reserve` rounds each vector's stride up to a multiple of 16 floats
(64 bytes) so that *every* vector starts on a cache line, not only the first,
and zero-fills the whole allocation.

With padding at zero, a squared-L2 term is `(0-0)² = 0` and an inner-product
term is `0*0 = 0`. A kernel may therefore process all `stride` floats instead of
exactly `dim`, which lets Phase 2's AVX2 kernel run whole 8-wide iterations with
**no scalar tail loop and no masked final load** — the fiddliest part of writing
a SIMD kernel by hand.

That only works if the *query* is padded too. The store zeroes its own padding,
but a caller's query array is only `dim` floats long, so a stride-wide kernel
would read off the end of it. Hence `prepare_query()` copies into an internal
buffer padded to the stride and zero-filled beyond `dim` — reversing the
header's original "does not copy" promise. This also deletes a lifetime
requirement from the interface, and it gives `prepare_query()` real work to do,
which is the justification D1 offered for its existence.

**Measured cost, since the estimate was wrong.** Zero-filling 488 MiB looks like
it costs 0.245 s of a 0.358 s load — 67%. It does not. Removing the `memset`
entirely drops a full load only to 0.301 s, so the *marginal* cost is **57 ms**,
about 16%. The rest is first-touch page-fault cost on fresh anonymous pages,
which is paid either way — lazily during the read instead of eagerly in
`reserve`. 57 ms once, to delete the tail loop from every future SIMD kernel and
to make the reported RSS reflect resident memory rather than lazily-mapped
address space, is cheap.

**Note for every dimension-sensitive test:** SIFT is 128 and GIST is 960, both
already multiples of 16, so both pad to nothing. A stride bug is invisible on
every dataset in the PRD. Tests run at dim 100 (stride 112) as well.

---

## D14 — Parsers return `Status`, not `std::optional`

**Phase 1.** `include/lodestone/io.hpp`

The Phase 1 plan specified `std::optional` returns. Changed, because the tests
that make the parsers worth having need to assert *which* error occurred —
`io_error` for a malformed file versus `dimension_mismatch` for a bad record
prefix — and `std::optional` discards precisely that. `Status` was already the
project's error channel (D3), so this needed no new abstraction.

The plan's `FvecsInfo` became `VecsInfo`: `float` and `int32` elements are both
4 bytes, so one probe serves both formats.

---

## D15 — Scalar L2 lands in Phase 1, and the factory is the dispatch seam

**Phase 1.** `src/distance.cpp`, `src/distance_scalar.cpp`

Brute force cannot compute a single distance without one concrete kernel
existing, and architecture rule 1 forbids reaching around the interface to get
one. So Phase 1 implements scalar squared L2 — and nothing else. Inner product,
the SSE and AVX2 kernels, runtime dispatch and the microbenchmarks all remain
Phase 2's work.

There is a benefit to that ordering: Phase 1 becomes the **first real consumer
of the D1 interface**, so the seam is validated by working code now rather than
in Phase 3 when the graph depends on it.

`make_distance_computer(Metric, const VectorStore&)` is the single place a
concrete kernel is chosen. It lives in its own translation unit compiled with
ordinary flags — no `-mavx2`, no `-fno-tree-vectorize` — because dispatch code
has to run on a machine lacking the instruction set it is about to select. In
Phase 2 the body of that one function grows the CPU feature check and no caller
changes.

---

## D16 — Exact k-NN orders by `(distance, id)`, not distance alone

**Phase 1.** `src/brute_force.cpp`

Distance alone leaves equal-distance neighbours arranged however the bounded
heap happened to leave them, so a rerun can return a different-but-equally-
correct answer. This function is the ground truth every later phase is measured
against; it has to be reproducible down to the id.

A bounded max-heap, not a full sort: at k=10 over a million vectors, sorting
does 1M·log(1M) comparisons to answer what 1M comparisons answer, because after
the first few hundred candidates almost everything loses to the heap top
immediately.

---

## D17 — SIFT1M contains duplicate vectors, so recall needs a tie-aware form

**Phase 1.** `src/brute_force.cpp`. **This is the phase's real finding.**

Exact brute force over SIFT1M scored **0.999440**, not 1.000. The PRD says any
deviation means the parser or the metric is wrong. It is neither.

The evidence, in the order it arrived:

1. `diagnose_recall` reported all 56 shortfalls with `worst_kept` *exactly* equal
   to `best_missed` — not close, equal.
2. The missed/extra id pairs had only **two distinct offsets**, 78816 and 64278.
   Two values across 56 queries is structure, not float noise.
3. Reading both records straight out of `sift_base.fvecs` settled it: they are
   **byte-identical**.
4. Counting the whole corpus: **14,538 of 1,000,000 vectors are duplicates**
   (1.4538%; 985,462 distinct).

When a duplicate lands on the k-th boundary, "the 10 nearest neighbours" is not
a well-defined *set* — several answers are equally correct. Strict id comparison
then measures whose tie-break convention won, not whether the search was right.
TEXMEX's generator keeps the higher id; ours keeps the lower, by D16.

So `recall_at_k_tied` thresholds on the k-th true *distance* — the
ANN-Benchmarks convention. It reports **1.000000 exactly**. Both numbers are
published; the assertion is on the tie-aware one, which reaches 1.0 exactly when
no returned neighbour is farther than the k-th true one, so it cannot forgive a
real miss.

**Why this is not the epsilon fudge `RecallDiagnosis` exists to prevent.** The
order of events is the entire distinction. The diagnostic proved the shortfall
sat exactly on the boundary, and the duplicates were confirmed bit-identical,
*before* the definition changed. Reaching for tie-awareness first would have
been tuning the metric until the number looked good. With the evidence in hand,
strict set recall is simply the wrong measure for a non-unique answer.

**This will matter again.** Phase 3's HNSW recall, Phase 5's quantisation loss
and Phase 6's filtered recall are all measured against this same ground truth on
this same corpus. Every one of them should report the tie-aware figure, or
inherit a 0.06% penalty that has nothing to do with the algorithm under test.

---

## D18 — Phase 1's QPS is a median of three with no warmup, and that is a stated deviation

**Phase 1.** `tools/sift_check.cpp`

The project's fixed methodology requires a discarded 10-second warmup. It is
skipped here, deliberately and visibly, because it is meaningless for this
workload: a brute-force query streams the entire 488 MiB corpus, so there is no
working set for a warmup to warm. Three runs are still taken and the median
reported.

`sift_check` is explicitly **not** the benchmark harness — no JSON, no sweep, no
latency percentiles, no `hnswlib`. Phase 4 builds that. This tool exists to
prove recall is 1.000 and to report three numbers while it is there. It prints a
warning whenever it is run with fewer than 10,000 queries, so a SIFT10K QPS
figure cannot be mistaken for a methodology-compliant measurement.

---

## D19 — Inner product returns the negated dot product

**Phase 2.** `src/distance_scalar.cpp` and every kernel since.

A larger inner product means *more* similar, which is backwards from L2. Every
consumer — the bounded heap in `brute_force.cpp`, Phase 3's candidate queues —
is written around "smaller is closer".

**Rejected: a per-metric comparator in each consumer.** That hands all future
graph code a chance to get the sign wrong, and the failure mode is a search that
returns the *farthest* neighbours while looking perfectly healthy. One negation
in one kernel cannot be got wrong twice.

The visible consequence is that a vector's inner-product distance to itself is
`-‖x‖²` rather than 0. A test asserts exactly that, so nobody "fixes" it and
inverts the ordering on the way past.

Adding the second metric also exposed a live bug in Phase 1's
`diagnose_recall`: `worst_kept` was seeded at `0.0F` and taken as a running max,
correct only while distances are non-negative. Confirmed live rather than
theoretical — reverting the fix turns a test red.

---

## D20 — `KernelKind` is a request, not a type

**Phase 2.** `include/lodestone/distance.hpp`, `src/distance.cpp`

`make_distance_computer(metric, store, kind = automatic)`. The default argument
means no existing caller changed when SSE and AVX2 landed — brute force and
every Phase 1 test compile untouched, which is the property D15's seam was
shaped to have. All of Phase 2's dispatch is the body of `detected_kernel()`.

Explicit selection exists because the microbenchmark and the correctness tests
must instantiate *each* kernel rather than whichever one the host prefers, and
`DistanceComputer::kernel()` is how a test verifies the request was honoured.
Graph code still names no class.

**A request this CPU cannot serve returns nullptr, never a silent downgrade.** A
benchmark that asked for AVX2 and quietly got scalar would report a speedup of
1.0 and read as a slow kernel rather than a missing one. And the AVX2 path is
gated on `__builtin_cpu_supports` for **both** `avx2` and `fma` — separate CPUID
bits, and the kernel uses `_mm256_fmadd_ps`. Handing back a kernel the machine
cannot execute is an illegal-instruction crash, not a wrong number.

`__builtin_cpu_supports` rather than hand-rolled CPUID: its AVX path already
accounts for the OS having enabled `XSAVE` state for the YMM registers, which a
raw feature bit does not.

---

## D21 — Each SIMD file's ISA is pinned, in both directions

**Phase 2.** `CMakeLists.txt`

| File | Flags | Guard fires when |
|---|---|---|
| `distance_scalar.cpp` | `-fno-tree-vectorize -DLODESTONE_SCALAR_NOVEC` | define absent |
| `distance_sse.cpp` | `-mno-avx -mno-avx2 -mno-fma` | `__AVX2__` **present** |
| `distance_avx2.cpp` | `-mavx2 -mfma` | `__AVX2__` **absent** |
| `distance.cpp` | none — plain `-march=native` | — |

D5 established the scalar guard. Phase 2 adds the mirror: the SSE file `#error`s
when AVX2 *is* enabled, because under `-march=native` the compiler would
vectorise any stray scalar code in it to AVX2 and the "SSE" row would be partly
an AVX2 row. Between the two guards, a mislabelled measurement cannot be
produced quietly. Both verified by removing the flag and confirming the build
fails.

`distance.cpp` deliberately gets no SIMD flags at all: dispatch code has to run
on a machine lacking the instruction set it is about to select.

---

## D22 — Four accumulators in the AVX2 kernel, chosen by measurement

**Phase 2.** `src/distance_avx2.cpp`. Curve in `BENCHMARKS.md`.

The textbook argument says a single accumulator makes every FMA wait on the
previous one, so with 4 cycles of latency and 2/cycle of throughput you need 8
independent chains to saturate.

**Measured, the curve is nowhere near that shape, and the reason is
instructive.** `distances_to()` computes independent distances back to back, so
the tail of one overlaps the head of the next — the batch already supplies
instruction-level parallelism that accumulators would otherwise have to.

But only when the chain is short enough to fit the out-of-order window. At dim
128 a distance is 16 dependent FMAs and extra accumulators buy ~1.4×; at dim 960
it is 120 and they buy **~2.5×**, because 120 dependent FMAs cannot be hidden by
the next distance.

Four is the choice: best at dim 128, within 2.5% of best at dim 960, and eight
costs register pressure for nothing.

**This corrects an earlier hypothesis in this file.** The ~11–16% per-dimension
penalty at dim 960 versus dim 128, which Phase 2 tasks 2 and 3 attributed to L1
pressure, was **dependency-chain length**. With four accumulators dim 960
becomes *cheaper* per dimension than dim 128, which the L1-pressure explanation
cannot account for.

---

## D23 — The query buffer is 64-byte aligned for correctness, not speed

**Phase 2.** `include/lodestone/detail/prepared_query.hpp`

Extracted from the kernels so all three share one prepared-query buffer, but the
substantive reason is alignment. `std::vector<float>` gets its storage from
`operator new`, which guarantees 16 bytes on x86-64 — enough for SSE, and **not
enough for AVX2**: a 32-byte `_mm256_load_ps` from a 16-byte-aligned base is
undefined behaviour on half its offsets.

**Measured: alignment is worth 0–2%, at or barely above noise.** So it stays for
legality, not performance. This closes the open question Phase 1 logged in
`IDEAS.md`.

Worth recording *how* that number was reached. The first comparison showed 18%
at `stream` dim 960, wildly out of line with the other three shapes. Repeating
the pair twice gave 1.3% both times. Publishing the first run would have
produced a confident and completely wrong finding — the same mistake made one
task earlier with the "0.06% is a wall" claim (see `BENCHMARKS.md`).
