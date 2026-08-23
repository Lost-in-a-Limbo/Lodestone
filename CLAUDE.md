# Lodestone

A filtered approximate nearest neighbor index in C++20. The research question:
how do you keep an HNSW graph navigable when a metadata predicate removes 99%
of its nodes?

Full spec is in `PRD.md`. Current phase is in `PHASE.md`. **Read both before
planning anything.**

## Build

```bash
cmake --preset debug && cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
cmake --preset release && cmake --build build/release
./build/release/bench --all          # writes bench/results/results.json
```

Debug preset has AddressSanitizer and UndefinedBehaviorSanitizer on. Keep it
that way.

## Architecture rules

These are not stylistic preferences. Breaking them turns a later phase into a
rewrite.

1. **`distance.hpp` is an injectable interface.** Graph code never calls a
   concrete kernel. Phase 2 swaps in SIMD and Phase 5 swaps in quantized
   distances through this seam. If you find yourself writing `l2_distance(...)`
   inside `hnsw.cpp`, stop.
2. **Vectors are stored contiguous and 64-byte aligned.** Never
   `vector<vector<float>>`.
3. **No dependency enters the core.** `hnswlib` and `faiss` are comparison
   baselines in `bench/` only. Never linked into `src/`.
4. **Everything that produces a number writes to `results.json`.** No number
   is ever hand-typed into docs or the frontend.

## Conventions

- C++20. `snake_case` for functions and variables, `PascalCase` for types.
- Headers in `include/lodestone/`, implementation in `src/`.
- `#pragma once`, not include guards.
- No exceptions in the search hot path. Return `std::expected` or an error enum.
- Tests use Catch2, live in `tests/`, one file per source file.
- Microbenchmarks use Google Benchmark, live in `bench/`.

## Benchmark methodology — never deviate

- Single-threaded queries unless explicitly labelled otherwise
- 10-second warmup, discarded
- Minimum 10,000 queries per measurement
- Three runs, report the median
- Always capture machine specs in the output
- **Publish the losses.** Where we are slower than hnswlib, say so and say why.

## Workflow

- Plan mode before any multi-file change. Write the plan to
  `.claude/plans/phaseN.md` before implementing.
- One task per prompt. Write the test first.
- Every task ends with a command that proves it worked.
- Commit per task, message describes what and why.
- When a phase's exit criteria are met: update `PHASE.md`, append numbers to
  `BENCHMARKS.md`, append design choices to `DECISIONS.md`.

## Scope control

Anything not in the current phase's deliverables goes in `IDEAS.md`. Do not
implement it. Scope creep is the primary failure mode of this project.

Explicit non-goals: SQL, persistence, distribution, GPU, billion-scale.

## Explain as you go

I am learning this domain, not just shipping it. When you implement something
non-obvious — the neighbor selection heuristic, asymmetric distance
computation, two-hop expansion — add a short comment explaining *why*, not
what. I need to be able to whiteboard this from memory.
