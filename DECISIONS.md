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
