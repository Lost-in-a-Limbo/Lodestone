# Phase 4 — Benchmark harness

**Written:** 2026-08-27
**Goal:** the measurement rig everything downstream depends on. Phases 5, 6 and
7 all report through it, so it gets built properly once.
**Exit criteria (PRD §6):** `./bench --all` emits `results.json` with no manual
steps; recall@1/@10/@100, QPS, p50/p95/p99, index memory, build time; machine
spec captured automatically; warmup discarded, ≥3 runs, median; a comparison
harness running `hnswlib` on identical data; numbers reproduce within 5%;
`BENCHMARKS.md` carries our curve next to hnswlib's.

**This is the phase that makes the project citable.** The resume bullet unlocks
here, and the number in it is the hnswlib ratio.

---

## The one rule that decides whether this phase is honest

**Match on recall, never on ef.** Two implementations will not agree on what a
given `ef` buys — different neighbour selection, different pruning, different
entry points. Comparing `ef=64` to `ef=64` compares two different operating
points and flatters whichever library happens to explore less.

The ANN-Benchmarks convention is a curve: recall on x, QPS on y. The comparison
is read *vertically* — at the same recall, who is faster. Everything below is
built to produce that curve for both libraries on identical data.

---

## Assumptions (override any at review)

| # | Assumption |
|---|---|
| A1 | **hnswlib only; FAISS deferred.** Phase 0 assumption C14 already said this. hnswlib is header-only and vendored read-only through CPM into `bench/` alone. FAISS pulls BLAS and OpenMP and is a substantially larger integration for a second data point — `IDEAS.md`, with the reason. |
| A2 | hnswlib is included **DOWNLOAD_ONLY** and added as a SYSTEM include, bypassing its CMakeLists entirely. It is header-only, so this avoids inheriting its flags, its examples and its test targets. Architecture rule 3: never linked into `src/`. |
| A3 | Build time is measured **once**, not three times. A median of three 1M builds is 20 minutes of wall clock to sharpen a number nothing depends on. Query measurements get the full three runs. Stated in the deviations table. |
| A4 | `nlohmann/json` for output. Hand-rolling JSON escaping to avoid a header-only dependency in a *benchmark* binary is false economy. |
| A5 | **No CLI11**, despite PRD §7 listing it. `bench --all` plus five flags does not justify a dependency; the existing tools parse argv directly and are readable. Revisit if the flag set grows. |
| A6 | Recall is tie-aware (D17) and `recall@k` is only measured where `ef >= k`, because ef bounds the candidate list. Combinations that cannot serve k are absent from the output rather than silently clamped (D28). |

---

## Task list

### Task 1 — `bench/bench_main.cpp`, the `bench` target
PRD §7 names the binary `bench`. Argument surface:

```
bench --all [--small] [--runs=3] [--warmup=10] [--ef=16,32,64,128,256]
            [--k=1,10,100] [--m=16] [--ef-construction=200] [--out=PATH]
            [--skip-hnswlib]
```

`--all` is the no-arguments-needed path the exit criterion demands.

### Task 2 — Machine spec capture
CPU model, physical cores, threads, cache sizes, RAM, compiler version, the
actual compile flags, and which distance kernel was selected. Read from
`/proc/cpuinfo`, `/proc/meminfo`, `/sys/devices/system/cpu/`, `__VERSION__`, and
a `-D` carrying `CMAKE_CXX_FLAGS`. A number without the machine it came from is
not a result.

### Task 3 — The measurement loop
Per (index, k, ef): discard a 10-second warmup, then three timed runs of the
full query set, recording per-query latency. Report median QPS across runs, and
p50/p95/p99 from the pooled latencies. Recall computed once — it does not vary
between runs, and asserting that it does not is itself a check.

### Task 4 — The hnswlib comparison
Same store, same queries, same ground truth, same k, same M and
ef_construction, single-threaded on both sides. Build time and memory recorded
for both.

### Task 5 — `results.json`
Schema-versioned from the start, because Phase 8's frontend will depend on its
shape and renaming a field later breaks a deployed page.

### Task 6 — Close
`BENCHMARKS.md` with both curves side by side and the losses stated,
`DECISIONS.md`, `PHASE.md` → Phase 5, `IDEAS.md`.

---

## What is NOT in this phase

No filtering, no quantisation, no FAISS, no threading, no frontend. If the
hnswlib comparison is unflattering, it is published unflattering — PRD §4
requires the gap explained, not closed.
