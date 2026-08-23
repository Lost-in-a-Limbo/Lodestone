# Lodestone — A Filtered Approximate Nearest Neighbor Index

**Product Requirements Document**

| Field | Value |
|---|---|
| Owner | Utkarsh Tuteja |
| Language | C++20 core, TypeScript/React showcase |
| Repo | `github.com/lost-in-a-limbo/lodestone` |
| Duration | 12 weeks (resume-ready at week 6) |
| Build tool | Claude Code |
| Status | Phase 0 — not started |

> A lodestone is a naturally magnetized rock that always finds north. This index finds nearest neighbors — even when most of the map has been closed off.

---

## 1. The problem

Vector search finds "things similar to this" by treating items as points in high-dimensional space and hopping along a graph of nearest-neighbor shortcuts (HNSW). It works beautifully — until you add a metadata filter.

Real queries are not "find similar." They are "find similar **where tenant_id = 47 and status = published**." The predicate deletes most of the graph's nodes, and the surviving subgraph is not a smaller navigable graph — it is disconnected islands. Greedy traversal walks into a region where every neighbor fails the filter and terminates. The correct answer exists, satisfies the predicate, and is unreachable.

**Selectivity** is the fraction of the corpus passing the predicate. This is where the strategies break:

| Strategy | Mechanism | Fails when |
|---|---|---|
| Pre-filter | Evaluate predicate, brute-force survivors | High selectivity (scanning 300k vectors) |
| Post-filter | Search normally, discard failures | Low selectivity (zero survivors in top-k) |
| In-filter (ACORN) | Check predicate during traversal | Below ~1% selectivity — graph fragments |

The literature confirms the third: ACORN-1 checks the predicate before distance computation and expands to two-hop neighbors when one-hop candidates run out, but below 5% selectivity traversal paths fragment and below 1% recall collapses. Qdrant's own July 2026 measurements show ACORN stalling at 67.7% recall on a 1% filter at 2.1–2.9× plain-graph latency.

**Why it stays unsolved:** you could pre-build extra edges guaranteeing connectivity per filter — that is roughly filtered-DiskANN — but it requires knowing the filters at index time. Price ranges and date windows make the predicate space effectively infinite. Approaches that depend on predicate label distributions also pay much higher build costs.

**The open question:** how do you keep the graph navigable under arbitrary, unknown predicates without paying for it at build time?

---

## 2. Why this project

### 2.1 Resume gaps it closes

| Gap | How Lodestone closes it |
|---|---|
| C++ listed with zero artifacts | 5–7K lines of C++20, the primary artifact |
| No low-level work | Hand-written SIMD kernels, cache-aware layout |
| No measurement discipline | Recall@k / QPS curves against published baselines |
| Quantification by surface area ("17 endpoints") | Quantification by recall, latency, memory |
| ML projects are library wrappers | Implements the retrieval layer libraries provide |
| Projects don't compound | Upgrades NexaCred from "used a vector store" to "built one" |

### 2.2 Why it beats a compiler or a toy database

Compilers and LSM stores are guided tours with step-by-step tutorials; finishing one proves you can follow a book. Filtered ANN is a live frontier with 2026 papers and competing vendor blog posts. You are not reimplementing a solved thing — you are measuring an unsolved one against published baselines.

### 2.3 Career targeting

Qdrant, Weaviate, LanceDB, Chroma, Turbopuffer, Milvus/Zilliz. Small, open-source, distributed by default, hiring from their contributor pool. The one channel where being in Bengaluru costs nothing.

---

## 3. Non-goals

- Not a database. No SQL, no persistence layer, no transactions.
- Not distributed. Single process, single node.
- Not GPU. Everything runs on a laptop CPU.
- Not beating FAISS. Being within a stated factor and explaining the gap honestly is the goal.
- Not billion-scale. 1M vectors is the working size.

Ideas outside a phase go in `IDEAS.md`. Scope creep is the primary failure mode of this project.

---

## 4. Success criteria

**Hard gates — the project has failed without these:**

1. Recall@10 vs QPS curve on SIFT1M, plotted against `hnswlib`, with the gap explained.
2. A selectivity sweep from 100% down to 0.1% showing where each filtering strategy breaks.
3. Reproduction of the recall collapse, with numbers.
4. One documented attempt at a fix, with honest results — including a negative result.
5. A showcase frontend that makes all of the above legible in 30 seconds.

**Soft goals:** a blog post, a Show HN, one merged PR into a real vector database.

---

## 5. Architecture

```
                  ┌───────────────────────────────┐
                  │   Showcase frontend (Phase 8) │
                  │   React + Vite + D3           │
                  │   GitHub Pages                │
                  └───────────────┬───────────────┘
                                  │ reads results.json
                  ┌───────────────▼───────────────┐
                  │   Benchmark harness (Phase 4) │
                  │   sweeps, recall, QPS, JSON   │
                  └───────────────┬───────────────┘
                                  │
   ═══════════════════════ engine boundary ═══════════════════════
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │              Filter strategies (Phase 6)          │
        │        pre-filter │ post-filter │ in-filter       │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │                HNSW index (Phase 3)               │
        │   layered graph, greedy search, neighbor heuristic│
        └───────────┬─────────────────────────┬─────────────┘
                    │                         │
     ┌──────────────▼─────────┐   ┌───────────▼─────────────┐
     │ Distance kernels (P2)  │   │  Vector store (Phase 1) │
     │ SIMD L2 / inner product│   │  aligned, contiguous    │
     └────────────────────────┘   └─────────────────────────┘
```

**Load-bearing design rule:** the distance function is an injectable interface from day one. Phase 2 swaps a scalar implementation for SIMD without touching the graph code, and Phase 5 swaps in quantized distances the same way. Hard-code `l2_distance()` into the search loop in Phase 3 and Phases 5 and 7 become rewrites.

---

## 6. Phases

Each phase states: goal, what Claude Code builds, exit criteria you can verify, and the numbers to record. **You are always in exactly one phase. `PHASE.md` tracks which.**

---

### Phase 0 — Bootstrap (2 days)

**Goal:** a repo Claude Code can work in productively, with a build that runs.

**Files created:** see Section 7 for the full bootstrap list.

**Exit criteria**
- [ ] `cmake -B build && cmake --build build` succeeds
- [ ] `ctest --test-dir build` runs one trivial passing test
- [ ] `CLAUDE.md` exists and is under 150 lines
- [ ] `PHASE.md` exists, says Phase 1
- [ ] First commit pushed

---

### Phase 1 — Data and brute force (week 1)

**Goal:** ground truth. You cannot measure recall without knowing the right answer.

**Build**
- `.fvecs` / `.ivecs` parser (SIFT1M format: 4-byte little-endian dimension, then that many float32s, repeated)
- Aligned vector store — 64-byte alignment, contiguous, no `vector<vector<float>>`
- Brute-force exact k-NN
- Recall@k calculator against provided ground truth

**Dataset:** SIFT1M — 1,000,000 base vectors, 128 dimensions, 10,000 queries, ground truth included. `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (also mirrored on Kaggle and HuggingFace). Start with SIFT10K, the small sibling, so iteration is fast.

**Exit criteria**
- [ ] Loads 1M vectors, reports load time and RSS
- [ ] Brute-force recall@10 against provided ground truth = 1.000 exactly
- [ ] Any deviation from 1.000 means the parser or the metric is wrong — stop and fix

**Record:** load time, memory footprint, brute-force QPS (expect single digits).

---

### Phase 2 — SIMD distance kernels (week 2)

**Goal:** your first real performance number, and the first place core CS pays off.

**Build**
- Scalar L2 and inner-product baseline
- SSE, AVX2, and (if your CPU supports it) AVX-512 variants using intrinsics
- Runtime dispatch on CPU feature detection
- A microbenchmark comparing all variants
- Correctness tests asserting SIMD output matches scalar within float tolerance

**Concepts you will actually learn:** memory alignment, why `-O3 -march=native` sometimes auto-vectorizes and sometimes doesn't, loop unrolling, horizontal sum reduction, cache line effects at 128 vs 960 dimensions.

**Exit criteria**
- [ ] AVX2 L2 is ≥3× faster than scalar
- [ ] All variants agree with scalar within 1e-4
- [ ] You can explain why the speedup isn't exactly 8× for 8-wide float SIMD

**Record:** ns per distance computation for each variant, at dim 128 and dim 960.

---

### Phase 3 — HNSW (weeks 3–4)

**Goal:** the index itself. The largest single phase.

**Build**
- Layer assignment with exponentially decaying probability
- Greedy search at each layer, descending to layer 0
- `SEARCH-LAYER` with a candidate heap and a dynamic result list bounded by `ef`
- The neighbor selection heuristic (not just "keep the M closest" — the diversity heuristic from the paper matters a lot for recall)
- Bidirectional edge insertion with pruning when degree exceeds `M_max`
- Serialization to disk

**Reference:** Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (arXiv 1603.09320). Algorithms 1–5 in that paper are the spec. Have Claude Code implement them one at a time, each with tests.

**Exit criteria**
- [ ] Recall@10 ≥ 0.95 on SIFT1M at some `ef` setting
- [ ] QPS within 5× of `hnswlib` at matched recall
- [ ] Index round-trips through serialization identically
- [ ] Build time under 20 minutes single-threaded on 1M vectors

**Record:** build time, index memory, the full recall/QPS curve at `ef` ∈ {8, 16, 32, 64, 128, 256}.

---

### Phase 4 — Benchmark harness (week 5)

**Goal:** the measurement rig everything downstream depends on. Build it properly now.

**Build**
- Runner that sweeps parameters and emits `results.json`
- Recall@1, @10, @100; QPS; p50/p95/p99 latency; index memory; build time
- Machine spec capture — CPU model, core count, RAM, compiler version, flags
- Warmup phase discarded, minimum 3 runs, median reported
- Direct comparison harness running `hnswlib` and `faiss` on identical data

**Methodology rules — fixed for the whole project:**
- Single-threaded queries unless explicitly labelled otherwise
- 10-second warmup discarded
- Minimum 10,000 queries per measurement
- Always report the machine spec alongside the number
- Publish the losses

**Exit criteria**
- [ ] `./bench --all` produces `results.json` with no manual steps
- [ ] Numbers reproduce within 5% across three runs
- [ ] `BENCHMARKS.md` has your curve next to hnswlib's

**Resume bullet unlocked — the project is now on your CV:**

> Implemented an HNSW approximate-nearest-neighbor index in C++20 with hand-written SIMD distance kernels, reaching [X]% recall@10 at [Y] QPS on SIFT1M — within [Z]× of hnswlib — with a reproducible benchmark harness reporting recall, latency percentiles, and memory.

---

### Phase 5 — Product quantization (week 6)

**Goal:** compress the vectors, measure what accuracy it costs.

**Build**
- k-means clustering per subspace (typically 8 or 16 subspaces of 16 dimensions each)
- Codebook training, 256 centroids per subspace → 1 byte per subspace
- Asymmetric distance computation with precomputed lookup tables
- Recall-vs-memory curve

**Why it belongs:** this is the ML content of an ML systems project. k-means, lossy compression, and the accuracy/memory frontier are all genuinely machine learning, applied to a systems problem.

**Exit criteria**
- [ ] 128-dim float32 (512 bytes) compressed to 16 bytes — 32× reduction
- [ ] Recall loss quantified at each compression ratio
- [ ] You can explain asymmetric vs symmetric distance computation

**Record:** memory per vector and recall@10 at m ∈ {8, 16, 32} subspaces.

---

### Phase 6 — Filtered search and the collapse (weeks 7–9)

**This is the phase the project exists for.**

**Build**
- Attribute store: each vector gets metadata (categorical tags, a numeric field)
- Predicate evaluator with a roaring-bitmap-style membership structure
- **Strategy A — pre-filter:** materialize the passing set, brute force it
- **Strategy B — post-filter:** search with over-fetch factor `f`, discard failures
- **Strategy C — in-filter:** predicate check before distance during traversal, with two-hop expansion when one-hop candidates are exhausted (this is ACORN-1)
- **Filtered ground truth generator** — exact k-NN restricted to the passing set, per predicate. Without this you cannot measure filtered recall at all.
- Selectivity sweep: 100%, 50%, 30%, 10%, 5%, 1%, 0.5%, 0.1%

**Two correlation regimes, both required:**
- *Uncorrelated* — the predicate is independent of vector position (random tags)
- *Correlated / negatively correlated* — passing vectors cluster away from queries. This is the adversarial case and where collapse is worst.

**Exit criteria**
- [ ] All three strategies implemented and measured across the full sweep
- [ ] **The collapse is reproduced** — recall visibly falls off below ~1% selectivity
- [ ] Crossover points identified: at what selectivity does pre-filter beat in-filter?
- [ ] `FINDINGS.md` documents the collapse with numbers and a plot

If you reproduce a clean collapse curve and nothing else, the project has already succeeded. Everything after this is upside.

---

### Phase 7 — Your attempt (week 10)

**Goal:** propose something, implement it, measure honestly.

You now have a measurement rig, ground truth, and a known failure. Pick one direction:

- **Adaptive fallback** — detect fragmentation mid-search (candidate list starving) and switch to exact scan of the passing set
- **Bridge nodes** — let filter-failing nodes act as transient routing hops without entering results
- **Stride-sampled expansion** — when expanding two-hop, sample for spatial diversity rather than taking the nearest
- **Selectivity-aware planning** — estimate selectivity from the bitmap cardinality, choose strategy per query

**Exit criteria**
- [ ] One approach implemented behind a flag
- [ ] Measured against all three baselines on the same sweep
- [ ] Result written up — **including if it did not work**

A rigorous negative result with a clear explanation of *why* is a strong outcome. Do not fabricate a win.

---

### Phase 8 — The showcase (weeks 11–12)

**Presentation is the multiplier.** A recruiter or engineer gives you 30 seconds. The frontend is what converts a repo into an interview.

**Stack:** Vite + React + TypeScript, D3 for the custom visualizations, Tailwind for layout, deployed free on GitHub Pages via GitHub Actions. Static — reads the `results.json` your benchmark harness emits. No backend.

**Five sections, in order:**

**1. Hero — the one-sentence claim**
Full-bleed. An animated force-directed graph in the background where nodes progressively grey out as a filter tightens, and the search path visibly fails to reach the target. Headline: the problem in one line. Below it, three stat cards: peak recall, peak QPS, vectors indexed.

**2. The problem, animated**
Side-by-side graph panels — unfiltered versus filtered. A slider controls selectivity from 100% to 0.1%. As the user drags, nodes disappear, edges break, and the traversal path animates and fails. This is the section people will screenshot.

**3. Recall vs QPS**
The standard ANN-Benchmarks plot: recall on x, QPS on y, log scale. Your curve, hnswlib's, FAISS's. Hoverable points showing the parameters. A toggle for dataset.

**4. The collapse**
Selectivity on x (log, reversed), recall on y. Four lines: pre-filter, post-filter, in-filter, and yours. The collapse below 1% should be visually unmissable. Annotated with the crossover points.

**5. Architecture and honesty**
A clean architecture diagram. Then a table of what you lose at and why — "FAISS is 2.3× faster on unfiltered scan because of X, which I did not implement." The honesty section is what makes senior engineers trust the rest of the page.

**Design direction:** dark, technical, restrained. One accent colour. Monospace for numbers, a clean sans for prose. No gradients, no stock illustrations, no marketing language. It should look like documentation from a serious infrastructure company, not a portfolio template.

**Exit criteria**
- [ ] Deployed and reachable at a public URL
- [ ] Loads in under 2 seconds
- [ ] Works on mobile
- [ ] Every number on the page traces back to `results.json` — nothing hand-typed
- [ ] Linked from the README, your resume, and your portfolio

**Also in this phase:** a blog post explaining one non-obvious thing you learned, and a Show HN.

---

## 7. Bootstrap file list

Everything Claude Code needs at `git init`. Section 8 tells you how to generate these.

```
lodestone/
├── CLAUDE.md                  ← highest-leverage file; keep under 150 lines
├── PHASE.md                   ← which phase you are on; update every phase
├── README.md                  ← pitch, benchmarks, quickstart
├── PRD.md                     ← this document
├── FINDINGS.md                ← the collapse, with numbers (Phase 6)
├── BENCHMARKS.md              ← all results with machine specs
├── DECISIONS.md               ← design choices + rejected alternatives
├── IDEAS.md                   ← scope-creep containment
├── CMakeLists.txt
├── CMakePresets.json          ← debug/release/asan presets
├── .clang-format              ← LLVM base, 100 col
├── .clang-tidy
├── .gitignore                 ← build/, data/, *.fvecs
├── .github/workflows/ci.yml   ← build + test on push
├── .github/workflows/pages.yml← deploy frontend (Phase 8)
├── cmake/CPM.cmake            ← dependency management
├── include/lodestone/
│   ├── types.hpp              ← vector_id, config structs
│   ├── distance.hpp           ← THE INJECTABLE INTERFACE
│   ├── vector_store.hpp
│   ├── hnsw.hpp
│   ├── filter.hpp
│   └── quantizer.hpp
├── src/
│   ├── distance_scalar.cpp
│   ├── distance_avx2.cpp
│   ├── vector_store.cpp
│   ├── hnsw.cpp
│   ├── filter.cpp
│   └── quantizer.cpp
├── tests/                     ← Catch2 or doctest
├── bench/
│   ├── bench_main.cpp
│   └── results/results.json   ← consumed by the frontend
├── tools/
│   ├── download_sift.sh
│   └── gen_filtered_gt.cpp    ← filtered ground truth
├── data/                      ← gitignored
└── web/                       ← Phase 8 frontend
```

**Dependencies, kept minimal:** Catch2 (tests), Google Benchmark (microbenchmarks), nlohmann/json (results), CLI11 (arg parsing), hnswlib and FAISS (comparison baselines only, never linked into the core). Fetch via CPM.cmake — header-only where possible, no vcpkg or Conan setup burden.

**Reference repos worth reading, not copying:** `nmslib/hnswlib` (the canonical implementation, small enough to read end to end), `facebookresearch/faiss` (for PQ), `erikbern/ann-benchmarks` (for methodology), `qdrant/qdrant` (for how filtering is done in production).

---

## 8. Claude Code manual

### 8.1 One-time setup

```bash
npm install -g @anthropic-ai/claude-code
mkdir lodestone && cd lodestone
git init
claude
```

Then, in the session:

```
/init
```

This scans the directory and drafts a `CLAUDE.md`. **Do not accept it as-is.** Auto-generated memory files are mediocre, and this is the single highest-leverage file in your setup. Replace it with the hand-written one provided alongside this PRD.

### 8.2 The five rules that matter

**Rule 1 — CLAUDE.md is precious, and short.** It loads into context on every single turn. Keep it under 150 lines. Frontier models reliably follow roughly 150–200 instructions and Claude Code's own system prompt already consumes about 50 of them. Past a couple hundred lines you get context rot: the rules that matter get diluted and adherence quietly drops. Progressive disclosure — point at `PRD.md`, don't inline it.

**Rule 2 — Plan mode before every non-trivial task.** `Shift+Tab` twice enters a read-only research mode: Claude can read files, search the codebase, and design a solution, but cannot create, modify, or delete anything until you approve. For anything bigger than a one-file change, use Plan → spec file → execute rather than Plan → code.

**Rule 3 — Scope the context, don't dump the repo.** "Go investigate" with no scope makes it read hundreds of files and flood the context. Name the files.

**Rule 4 — Verify, don't trust.** Every task ends with a command that proves it worked. Read the diff yourself. A plausible-looking change you never ran is the most common way this goes wrong.

**Rule 5 — `/clear` between phases.** Context from Phase 2 actively degrades Phase 5. Clear, then re-orient from `PHASE.md`.

### 8.3 The per-phase loop

Run this same cycle for every phase. It is the whole workflow.

**Step 1 — orient (start of session)**
```
Read PHASE.md and CLAUDE.md. Tell me which phase we are on
and what the exit criteria are. Do not write any code yet.
```

**Step 2 — plan** (`Shift+Tab` twice)
```
We are starting Phase 3, HNSW. Read PRD.md section 6 Phase 3
and include/lodestone/hnsw.hpp. Produce an implementation plan
broken into tasks small enough to test individually. Write it
to .claude/plans/phase3.md. Do not implement anything yet.
```

**Step 3 — review the plan yourself.** This is where you learn the most. If you can't follow the plan, you won't understand the code.

**Step 4 — execute one task at a time**
```
Implement task 1 from .claude/plans/phase3.md only.
Write the test first, then the implementation.
Stop after task 1 and run ctest.
```

**Step 5 — verify**
```bash
cmake --build build && ctest --test-dir build --output-on-failure
```
Read the diff. `git diff`. Every time.

**Step 6 — commit per task**
```
Commit this with a message describing what changed and why.
```

**Step 7 — close the phase**
```
All Phase 3 exit criteria in PRD.md are met. Update PHASE.md
to Phase 4, append the recorded numbers to BENCHMARKS.md, and
add any design decisions to DECISIONS.md.
```
Then `/clear`.

### 8.4 Phase-opening prompts

Copy these verbatim. Each assumes a fresh session after `/clear`.

**Phase 0**
```
Read PRD.md sections 5 and 7. Create the bootstrap scaffolding:
CMakeLists.txt with C++20, CMakePresets.json with debug/release/asan
presets, .clang-format (LLVM base, 100 columns), .gitignore, a GitHub
Actions CI workflow that builds and tests, cmake/CPM.cmake, and the
header stubs listed in section 7. Add one trivial passing test so
ctest has something to run. Do not implement any real logic.
```

**Phase 1**
```
Phase 1. Implement the .fvecs and .ivecs parsers and an aligned
vector store, then brute-force exact k-NN and a recall@k calculator.
Write tools/download_sift.sh to fetch and extract SIFT1M into data/.
Test against SIFT10K first. Brute-force recall@10 against the
provided ground truth must be exactly 1.000 — if it is not, the
parser or the metric is wrong and we stop and fix it.
```

**Phase 2**
```
Phase 2. distance.hpp must stay an injectable interface — the graph
code must never call a concrete kernel directly. Implement scalar L2
and inner product, then AVX2 versions using intrinsics, with runtime
CPU feature dispatch. Add Google Benchmark microbenchmarks comparing
them at dim 128 and 960, and tests asserting SIMD agrees with scalar
within 1e-4.
```

**Phase 3**
```
Phase 3. Implement HNSW following Malkov & Yashunin arXiv 1603.09320,
Algorithms 1 through 5. One algorithm per task, each with tests.
Include the neighbor selection heuristic, not just nearest-M — it
matters for recall. Then serialization. Do not start the benchmark
harness; that is Phase 4.
```

**Phase 4**
```
Phase 4. Build the benchmark harness per PRD.md section 6 Phase 4.
It must emit bench/results/results.json including machine specs,
and sweep ef across 8 to 256. Add a comparison harness that runs
hnswlib on identical data. Follow the fixed methodology rules
exactly — warmup discarded, median of three, single-threaded.
```

**Phase 5**
```
Phase 5. Implement product quantization: per-subspace k-means,
256 centroids, asymmetric distance with precomputed lookup tables.
It must plug into the distance interface without touching hnsw.cpp.
Produce a recall-versus-memory curve for m in 8, 16, 32.
```

**Phase 6**
```
Phase 6, the core of the project. Read PRD.md section 6 Phase 6
fully before planning. Build the attribute store, predicate
evaluator, all three filtering strategies, and the filtered ground
truth generator. Then the selectivity sweep from 100% to 0.1%,
under both uncorrelated and negatively-correlated predicates.
The goal is to REPRODUCE the recall collapse, not avoid it.
Write findings to FINDINGS.md with numbers.
```

**Phase 7**
```
Phase 7. Read FINDINGS.md. I want to implement [YOUR CHOSEN
APPROACH] behind a runtime flag and measure it against all three
baselines on the same sweep. Do not tune the baselines down to
make it look better. If it does not beat them, we report that.
```

**Phase 8**
```
Phase 8. Build the showcase frontend per PRD.md section 6 Phase 8.
Vite + React + TypeScript + D3 + Tailwind, static, reads
bench/results/results.json — no hand-typed numbers anywhere.
Five sections as specified. Dark, restrained, one accent colour,
monospace for numerals. Add a GitHub Actions workflow deploying
to Pages. Start with the interactive graph-fragmentation
visualization in section 2; it is the centrepiece.
```

### 8.5 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Claude ignores your conventions | CLAUDE.md too long | Prune to under 150 lines |
| It refactors things you didn't ask about | Task scope too broad | One task per prompt, name the files |
| Code compiles, results are wrong | You didn't verify | Every task ends with a command that proves it |
| It reinvents Phase 2 in Phase 5 | Distance interface got bypassed | Put the rule in CLAUDE.md; check the diff |
| Sessions get slow and confused | Context bloat | `/clear` between phases |
| You can't explain your own code | You skipped plan review | Read every plan before approving |

**The last one is the real risk.** An interviewer will ask "why did you choose that neighbor heuristic?" and Claude Code will not be in the room. Read the plans. Read the diffs. If a phase finishes and you couldn't whiteboard it, redo it slower.

---

## 9. Timeline

| Week | Phase | Milestone |
|---|---|---|
| 0 | 0 | Bootstrap, build runs |
| 1 | 1 | Ground truth, brute force |
| 2 | 2 | SIMD kernels, first real number |
| 3–4 | 3 | HNSW working |
| 5 | 4 | Benchmark harness — **resume-ready** |
| 6 | 5 | Product quantization |
| 7–9 | 6 | **The collapse reproduced** |
| 10 | 7 | Your attempt |
| 11–12 | 8 | Showcase, blog post, Show HN |

**Under time pressure, cut in this order: Phase 5, then Phase 7, then Phase 8's polish.** Never cut Phase 6 — it is the entire differentiator. A project that stops after Phase 6 with a clean collapse curve and a plain README still beats every other project on your resume.

---

## 10. Running alongside

- **LeetCode 150 → 400+**, targeting a Knight rating. Still outranks this project for on-campus screening.
- **One merged PR** into Qdrant, LanceDB, or hnswlib. You will be reading their internals during Phases 3 and 6 anyway — the PR costs marginal effort and carries disproportionate weight with exactly the companies you are targeting.

---

## Appendix A — Reading, in phase order

| Phase | Source |
|---|---|
| 1 | TEXMEX SIFT1M dataset documentation |
| 2 | Intel Intrinsics Guide; Agner Fog's optimization manuals |
| 3 | Malkov & Yashunin, arXiv 1603.09320 (the HNSW paper) |
| 3 | `nmslib/hnswlib` source — small enough to read fully |
| 4 | Aumüller et al., "ANN-Benchmarks", arXiv 1807.05614 |
| 5 | Jégou et al., "Product Quantization for Nearest Neighbor Search" |
| 6 | Patel et al., ACORN (2024); Qdrant's filtered vector search blog post |
| 6 | Gollapudi et al., "Filtered-DiskANN" |
| 8 | ANN-Benchmarks results site, for plot conventions |

## Appendix B — Glossary

- **ANN** — approximate nearest neighbor. Trades exactness for speed.
- **HNSW** — hierarchical navigable small world. Layered proximity graph, the dominant in-memory index.
- **Recall@k** — fraction of the true k nearest neighbors your search actually returned.
- **QPS** — queries per second. Plotted against recall; the tradeoff curve is the real result.
- **Selectivity** — fraction of the corpus passing a metadata predicate. Low selectivity is the hard case.
- **PQ** — product quantization. Splits vectors into subspaces, replaces each with a centroid id.
- **ef** — search-time candidate list size. Bigger means higher recall and lower QPS.
- **M** — max edges per node in the graph. Bigger means better connectivity and more memory.
