# Phase 0 — Bootstrap plan

**Written:** 2026-08-15
**Goal:** a repo Claude Code can work in productively, with a build that runs.
**Exit criteria (PHASE.md):** debug preset configures + builds; ctest runs one
passing test; CLAUDE.md under 150 lines; section 7 bootstrap files exist; first
commit pushed.

**No real logic ships in this phase.** Every `.cpp` is a stub that compiles.

---

## Assumptions made (override any of these at review)

Answered by you:

- **A1** — `cmake`, `ninja-build`, `g++-12`, `gcc-12` installed via apt. *Still
  pending at time of writing; every build step below is blocked until it lands.*
- **B4** — gcc-12 is the compiler, so `std::expected` is assumed available.
  **Will be verified by probe before any code depends on it.** If gcc-12's
  libstdc++ turns out to lack `<expected>`, fallback is `status` enum +
  `std::optional`, which `CLAUDE.md` already sanctions.
- **A3** — repo `Lost-in-a-Limbo/lodestone` exists, `gh` authenticated, local
  `main` tracks `origin/main`. Done, nothing further needed.

Assumed by me, unanswered:

| Q | Assumption |
|---|---|
| A2 | `.clang-format` + `.clang-tidy` written now; binaries not installed, so enforcement is CI-only for now |
| B5 | Distance seam = **abstract base class with a batch method** (see Task 7) |
| B6 | Per-file `-fno-tree-vectorize` on `distance_scalar.cpp` built in **now** |
| B7 | Three presets: `debug` (O0+ASan/UBSan), `release` (O3+native), `asan` (RelWithDebInfo+ASan/UBSan) |
| B8 | Catch2 **v3** via CPM |
| B9 | `CPM_SOURCE_CACHE` set to `~/.cache/CPM` — one dep copy shared across 3 build dirs |
| B10 | Your 10 items + `DECISIONS.md`, `IDEAS.md`, `BENCHMARKS.md` stubs. Deferred: `pages.yml`, `web/`, `download_sift.sh`, `gen_filtered_gt.cpp` |
| B11 | `filter.hpp` / `quantizer.hpp` = declaration-only placeholders, `// Phase N` comment, zero logic |
| B12 | CI = `ubuntu-22.04` + gcc-12 only. Correctness only — **no timing assertions in CI, ever** |
| C13 | **No AVX-512.** Zen 3 lacks it; untestable code is not written. → `IDEAS.md` |
| C14 | hnswlib is the Phase 4 baseline; FAISS optional, decided later |
| C15 | `ctest` always runs SIFT10K-scale or synthetic data. SIFT1M only under release/asan |
| C16 | `LODESTONE_PRD.md` → `git mv` to `PRD.md` so `CLAUDE.md`'s references resolve |
| C18 | Left open. No telemetry hook designed in yet |

---

## Task list

### Task 1 — `.gitignore`

**Creates:** `.gitignore`

**Key decision:** ignore `build/`, `data/`, `*.fvecs`, `*.ivecs`, and
`.claude/settings.local.json` — but **track `.claude/plans/`**. The plans are
the record of why the code looks the way it does; they belong in history next
to the diffs they produced.

---

### Task 2 — `cmake/CPM.cmake`

**Creates:** `cmake/CPM.cmake` (vendored, ~30KB, downloaded once)

**Key decision:** vendor the file rather than bootstrap-download it at configure
time, so a fresh clone configures without network for CPM itself. Pinned
version, not `latest` — a dependency manager that silently changes under you is
worse than no dependency manager.

---

### Task 3 — `CMakeLists.txt`

**Creates:** root `CMakeLists.txt`

**Structure:**
- `project(lodestone LANGUAGES CXX)`, C++20, `CXX_EXTENSIONS OFF` (we want
  `-std=c++20`, not `-std=gnu++20` — portable is the default)
- `lodestone_core` static library from `src/*.cpp`
- `lodestone_options` INTERFACE target carrying warnings
  (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`) so every target gets the
  same treatment from one place
- Options: `LODESTONE_BUILD_TESTS` (ON), `LODESTONE_BUILD_BENCH` (OFF in Phase 0)

**Key decision — the one that protects Phase 2's headline number:**
`distance_scalar.cpp` gets `-fno-tree-vectorize` as a *per-file* property. With
release-wide `-O3 -march=native`, gcc auto-vectorises the scalar L2 loop into
AVX2, and Phase 2's "AVX2 is 3× scalar" becomes AVX2-vs-AVX2 — a meaningless
number. Building the isolation in now means Phase 2's measurements are valid the
first time instead of being re-run.

---

### Task 4 — `CMakePresets.json`

**Creates:** `CMakePresets.json` (schema v3 — requires cmake ≥3.21; apt ships 3.22)

| Preset | Build type | Flags | Binary dir |
|---|---|---|---|
| `debug` | Debug | `-O0 -g`, ASan + UBSan | `build/debug` |
| `release` | Release | `-O3 -march=native` | `build/release` |
| `asan` | RelWithDebInfo | `-O2 -g`, ASan + UBSan | `build/asan` |

**Key decision:** the third preset is *optimised* + sanitised, not a duplicate of
debug. `-O0` + ASan is roughly 20–50× slower than release — unusable against 1M
vectors. `asan` is the preset that can actually sanitise a realistic workload,
and it catches the class of bug that only appears once the optimiser has been at
your code. All three use Ninja.

---

### Task 5 — `.clang-format`

**Creates:** `.clang-format`

**Key decision:** `BasedOnStyle: LLVM`, `ColumnLimit: 100` per section 7. Pinning
a shared format now means no diff in any later phase is ever polluted by
reformatting noise.

---

### Task 6 — `.clang-tidy`

**Creates:** `.clang-tidy`

**Key decision:** enable `bugprone-*`, `performance-*`, `readability-*`,
`modernize-*`; explicitly **disable** `readability-magic-numbers` (HNSW is full
of legitimately-named constants like `M`, `ef`) and `modernize-use-trailing-return-type`
(pure style churn). Config lands now; binaries aren't installed, so this is
CI-enforced until you install `clang-tidy-15`.

---

### Task 7 — Header stubs in `include/lodestone/`

**Creates:** `types.hpp`, `distance.hpp`, `vector_store.hpp`, `hnsw.hpp`,
`filter.hpp`, `quantizer.hpp`

**`types.hpp`** — `using vector_id = std::uint32_t;`

*Key decision:* 32-bit, not `size_t`. Neighbour lists dominate index memory: at
M=16, a node's list is 16×4B = 64B = exactly one cache line. With 64-bit ids it's
128B — two cache lines, two misses, on the hottest read in the entire search.
1M vectors needs 20 bits; 32 is ample.

Also: `enum class status`, `hnsw_config`, `search_params`.

**`distance.hpp`** — THE INJECTABLE INTERFACE. This is the load-bearing file.

```
class distance_computer {
public:
  virtual ~distance_computer() = default;
  virtual void  prepare_query(const float* q) = 0;
  virtual float distance_to(vector_id id) const = 0;
  virtual void  distances_to(std::span<const vector_id> ids,
                             std::span<float> out) const = 0;
};
```

*Key decision — why this shape, and it is the most consequential call in Phase 0:*

Three things had to be true at once.

1. **It needs per-query state.** Phase 5's product quantization uses *asymmetric*
   distance: the query stays full-precision float, the stored vectors are 1-byte
   codes. You project the query into each subspace once, build a 256×m lookup
   table, and then every subsequent "distance" is m table lookups and adds. That
   table is per-query state. A free function `f(a, b, dim)` has nowhere to put
   it. Hence `prepare_query()` — plain L2 simply stashes the pointer and does
   nothing else.

2. **It needs to not cost an indirect call per distance.** A virtual call on the
   single hottest operation in the system is exactly the wrong instinct — which
   is why `distances_to()` exists. HNSW's inner loop expands one node's
   neighbour list, which is naturally a batch of M ids. One virtual call, M
   distance computations inside it, so the dispatch overhead amortises to
   nothing while the kernel stays free to use AVX2 across the whole batch.

3. **It must never leak a concrete kernel into the graph.** `hnsw.cpp` will hold
   a `distance_computer&` and nothing else. If a later diff shows
   `l2_distance(` inside `hnsw.cpp`, that is the rule breaking and I stop.

*Alternative rejected:* a C++20 concept plus templating `hnsw` on the computer
type. Faster (full inlining, zero dispatch) but it bakes the kernel choice at
compile time, and Phase 2 requires *runtime* CPU-feature dispatch — which would
mean instantiating the entire index for each kernel and selecting at
construction. Revisit in Phase 3 if profiling shows the batch call is still hot.
Logged in `DECISIONS.md`.

**`vector_store.hpp`** — declares 64-byte aligned contiguous storage. 64B = one
cache line on this Zen 3 CPU; aligning to it means a 128-dim float vector (512B)
occupies exactly 8 lines with no straddling, and AVX2 can use aligned loads.

**`hnsw.hpp`, `filter.hpp`, `quantizer.hpp`** — declaration-only placeholders
marked with their phase. No logic.

---

### Task 8 — Skeleton sources in `src/`

**Creates:** `distance_scalar.cpp`, `distance_avx2.cpp`, `vector_store.cpp`,
`hnsw.cpp`, `filter.cpp`, `quantizer.cpp`

**Key decision:** each is a compiling stub — enough to prove the library links
and that the per-file compile flags actually apply, nothing more.
`distance_avx2.cpp` gets `-mavx2 -mfma` as a per-file property (verified working
on this CPU by probe) but stays empty until Phase 2.

---

### Task 9 — `tests/` + one trivial passing test

**Creates:** `tests/CMakeLists.txt`, `tests/test_types.cpp`

**Key decision:** the test asserts something real but trivial — that
`vector_id` is 4 bytes and the config defaults are sane. A test asserting
`1 == 1` proves the harness runs; a test asserting a design invariant also
proves it *keeps* running. Registered via `catch_discover_tests()` so each
Catch2 case appears as a separate ctest entry.

---

### Task 10 — `.github/workflows/ci.yml`

**Creates:** `.github/workflows/ci.yml`

**Key decision:** `ubuntu-22.04`, gcc-12, builds `debug` and `release`, runs
ctest on both. **No timing assertions, ever** — GitHub's runners are Intel with
AVX-512 and different cache behaviour; a performance gate there would be
measuring a machine that isn't yours. CI proves correctness; your laptop
produces every number that gets published.

---

### Task 11 — Doc stubs + PRD rename

**Creates:** `DECISIONS.md`, `IDEAS.md`, `BENCHMARKS.md`
**Renames:** `LODESTONE_PRD.md` → `PRD.md` (via `git mv`)
**Reconciles:** remote `README.md` already exists and is good — left as-is,
except its roadmap/AVX-512 line may need a note once C13 is confirmed.

**Key decision:** `DECISIONS.md` gets the B5 distance-seam entry and its rejected
alternative on day one. `IDEAS.md` gets AVX-512 immediately. Both files exist
from the first commit so there's never a moment where a decision has nowhere to
go — that's how scope creep and undocumented choices actually get in.

---

### Task 12 — Verify

```bash
cmake --preset debug && cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
cmake --preset release && cmake --build build/release
```

All three must be green. Plus a `std::expected` probe against gcc-12 to confirm
or kill assumption B4 before anything depends on it.

---

### Task 13 — Commit and close

Commit per task where it makes sense to, then update `PHASE.md` to Phase 1, tick
the Phase 0 boxes, log the session date, and push to `origin/main`.

---

## What is explicitly NOT in this phase

Deferred to their own phases, despite appearing in section 7's tree:
`tools/download_sift.sh` (Phase 1), `tools/gen_filtered_gt.cpp` (Phase 6),
`bench/bench_main.cpp` (Phase 4), `.github/workflows/pages.yml` + `web/`
(Phase 8), `FINDINGS.md` (Phase 6).

No distance kernel, no parser, no graph. If any of those appear in a Phase 0
diff, the phase has failed.
