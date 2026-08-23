# Benchmarks

Every number in this file is produced by `bench/` into
`bench/results/results.json` and copied here. **Nothing is typed by hand** — if
a number appears here without a corresponding entry in `results.json`, it is a
bug in the process, not a result.

## Methodology — fixed for the whole project

- Single-threaded queries unless a row explicitly says otherwise
- 10-second warmup, discarded
- Minimum 10,000 queries per measurement
- Three runs, median reported
- Machine spec captured in every result
- **Losses are published.** Where Lodestone is slower than `hnswlib`, the row
  says so and the note says why.

CI never asserts on timing. GitHub's runners are not the machine publishing
these numbers — see `DECISIONS.md` D11.

---

## Machines

| Id | CPU | Cores | RAM | Compiler | Flags |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

Filled by the harness in Phase 4 from `/proc/cpuinfo` and the compiler version,
not by hand.

---

## Phase 1 — load and brute force

| Metric | Value | Machine | Date |
|---|---|---|---|
| — | — | — | — |

Expected: recall@10 exactly 1.000 against the provided ground truth. Anything
else means the parser or the metric is wrong, and Phase 1 does not close.

## Phase 2 — distance kernels

ns per distance computation, dim 128 and dim 960.

| Kernel | dim 128 | dim 960 | vs scalar | Machine |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 3 — HNSW

| ef | recall@10 | QPS | p50 | p95 | p99 | Machine |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## Phase 4 — versus hnswlib

| Index | recall@10 | QPS | Ratio | Note |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 5 — product quantisation

| m | bytes/vector | recall@10 | Machine |
|---|---|---|---|
| — | — | — | — |

## Phase 6 — the selectivity sweep

The collapse curve. See `FINDINGS.md` for the analysis.

| Selectivity | pre-filter | post-filter | in-filter | Correlation |
|---|---|---|---|---|
| — | — | — | — | — |

## Phase 7 — the attempt

| Selectivity | best baseline | Lodestone | Delta |
|---|---|---|---|
| — | — | — | — |
