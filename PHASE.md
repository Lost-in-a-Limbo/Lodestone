# Current phase

> Update this the moment a phase closes. It is the first thing read at the
> start of every session — it is how both you and Claude Code know where you are.

## → PHASE 1: Data and brute force

**Goal:** ground truth. You cannot measure recall without knowing the right
answer.

**Exit criteria**
- [ ] `.fvecs` / `.ivecs` parsers, with `tools/download_sift.sh` fetching SIFT1M
- [ ] `VectorStore::reserve` / `add` implemented — 64-byte aligned, contiguous
- [ ] Brute-force exact k-NN
- [ ] Recall@k calculator
- [ ] Loads 1M vectors, reports load time and RSS
- [ ] **Brute-force recall@10 against the provided ground truth = 1.000 exactly.**
      Any deviation means the parser or the metric is wrong — stop and fix it,
      do not proceed.

**Record:** load time, memory footprint, brute-force QPS (expect single digits).

**Blocked on:** nothing. `VectorStore`'s shape and special members already
exist; `reserve()` and `add()` return `not_implemented` and are the first thing
to write.

---

## Progress

| Phase | Name | Status | Closed | Key number |
|---|---|---|---|---|
| 0 | Bootstrap | **closed** | 2026-08-23 | 8/8 tests, 3 presets green |
| 1 | Data + brute force | **in progress** | — | recall@10 = 1.000 |
| 2 | SIMD kernels | not started | — | ns/distance |
| 3 | HNSW | not started | — | recall@10, QPS |
| 4 | Benchmark harness | not started | — | full curve vs hnswlib |
| 5 | Product quantization | not started | — | bytes/vector, recall loss |
| 6 | **Filtered search** | not started | — | **collapse curve** |
| 7 | The attempt | not started | — | delta vs baselines |
| 8 | Showcase | not started | — | deployed URL |

**Resume-ready after Phase 4. Differentiating after Phase 6.**

---

## Phase 0 exit criteria — all met

- [x] `cmake --preset debug && cmake --build build/debug` succeeds
- [x] `ctest --test-dir build/debug` runs passing tests — 8/8, under ASan + UBSan
- [x] `CLAUDE.md` in place, under 150 lines
- [x] Bootstrap files from PRD section 7 created
- [ ] First commit pushed to GitHub — **committed locally, push pending**

`release` and `asan` presets also configure, build and test green. Both Phase 0
build guards were verified by removing their flags and confirming the build
fails (see `DECISIONS.md` D5).

---

## Session log

```
2026-08-23  Phase 0  scaffolding created; debug/release/asan all green;
                     8 tests passing; std::expected probe resolved (absent);
                     Phase 0 closed
```

---

## Numbers recorded so far

No performance numbers yet — Phase 0 produces none by design, and Phase 1's
brute-force QPS is the first real measurement.

| Metric | Value | Machine | Date |
|---|---|---|---|
| Tests passing | 8/8 | M1 | 2026-08-23 |
| Presets green | debug, release, asan | M1 | 2026-08-23 |

**Machine M1:** AMD Ryzen 5 7530U (Zen 3, 6C/12T), 15 GiB RAM,
GCC 11.4.0, CMake 4.3.1, Ninja 1.13.2. Zen 3 has AVX2 + FMA and **no
AVX-512** — this confirms plan assumption C13 and is why Phase 2 ships SSE +
AVX2 only (`IDEAS.md`).

---

## Open questions

Things you don't understand yet and must resolve before the phase closes.
An empty list at phase close means you either understood everything or weren't
paying attention.

- [ ] **`cmake` and `ninja` are not installed system-wide.** Both presets were
      driven with CLion's bundled copies, at
      `/snap/clion/current/bin/{cmake/linux/x64/bin,ninja/linux/x64}`. That
      works, but it means the build commands in `README.md` and `CLAUDE.md` do
      not run as written on a bare shell. Fix with
      `sudo apt-get install -y cmake ninja-build`, or add those two paths to
      `PATH` in the shell profile and say so in the README.
- [ ] CPM is bootstrapped rather than vendored, and `CPM_HASH_SUM` is empty.
      Complete both during Phase 1 — see `IDEAS.md`.
- [ ] Phase 1: what does the `stride()` padding actually need to be for a
      non-multiple-of-16 dimension, and does SIFT's dim 128 hide the bug that a
      dim like 100 would expose? Write the test at dim 100, not only 128.
