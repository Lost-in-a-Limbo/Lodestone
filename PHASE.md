# Current phase

> Update this the moment a phase closes. It is the first thing read at the
> start of every session — it is how both you and Claude Code know where you are.

## → PHASE 2: SIMD distance kernels

**Goal:** your first real performance number, and the first place core CS pays
off.

**Exit criteria**
- [ ] Scalar inner product to sit alongside the existing scalar L2
- [ ] SSE and AVX2 variants of both, using intrinsics
- [ ] Runtime dispatch on CPU feature detection, inside
      `make_distance_computer()` — the body of that one function, with no caller
      changes (`DECISIONS.md` D15)
- [ ] Google Benchmark microbenchmarks comparing every variant at dim 128 and
      dim 960
- [ ] Tests asserting every SIMD variant agrees with scalar within 1e-4
- [ ] AVX2 L2 is ≥3× scalar
- [ ] You can explain why the speedup isn't exactly 8×

**No AVX-512.** M1 is Zen 3 and has none, and an untested SIMD kernel in a
project whose value is measurement discipline is worse than an absent one. See
`IDEAS.md`.

**Blocked on:** nothing. The seam is in place and already has a working
consumer: `make_distance_computer()` returns the scalar kernel today, brute
force drives it, and 56 tests hold the contract.

**Two things Phase 1 leaves you:**

1. The store zeroes its padding and `prepare_query()` zero-pads the query, so an
   AVX2 kernel may run whole 8-wide iterations across the full stride with **no
   scalar tail loop and no masked final load** (`DECISIONS.md` D13). This is the
   single biggest simplification available to Phase 2 — use it.
2. `-fno-tree-vectorize` is pinned to `distance_scalar.cpp` with a build guard,
   so the baseline is honest. The measured ~3 cycles/dimension confirms it is
   really applied, so the speedup you report will be real.

---

## Progress

| Phase | Name | Status | Closed | Key number |
|---|---|---|---|---|
| 0 | Bootstrap | **closed** | 2026-08-23 | 8/8 tests, 3 presets green |
| 1 | Data + brute force | **closed** | 2026-08-24 | recall@10 = 1.000000 tie-aware |
| 2 | SIMD kernels | **in progress** | — | ns/distance |
| 3 | HNSW | not started | — | recall@10, QPS |
| 4 | Benchmark harness | not started | — | full curve vs hnswlib |
| 5 | Product quantization | not started | — | bytes/vector, recall loss |
| 6 | **Filtered search** | not started | — | **collapse curve** |
| 7 | The attempt | not started | — | delta vs baselines |
| 8 | Showcase | not started | — | deployed URL |

**Resume-ready after Phase 4. Differentiating after Phase 6.**

---

## Phase 1 exit criteria — all met

- [x] `.fvecs` / `.ivecs` parsers, with `tools/download_sift.sh` fetching SIFT1M
- [x] `VectorStore::reserve` / `add` — 64-byte aligned, contiguous
- [x] Brute-force exact k-NN
- [x] Recall@k calculator
- [x] Loads 1M vectors, reports load time and RSS
- [x] **Brute-force recall@10 against the provided ground truth = 1.000000** —
      by the tie-aware measure. The strict id-set measure is 0.999440, and the
      0.06% gap is fully explained: SIFT1M contains 14,538 byte-identical
      duplicate vectors, so when one lands on the k-th boundary several id sets
      are equally correct answers. Evidence and reasoning in `DECISIONS.md` D17.
      **This was investigated before the metric was changed, not after.**

56 tests green on `debug`, `asan` and `release`. Every task's claims were
verified by mutation — deleting a check and confirming a test goes red — because
most of Phase 1's code was new files, where a "failing test first" is just a
compile error and proves nothing.

---

## Session log

```
2026-08-23  Phase 0  scaffolding created; debug/release/asan all green;
                     8 tests passing; std::expected probe resolved (absent);
                     Phase 0 closed
2026-08-24  Phase 1  store, parsers, scalar L2 kernel, exact k-NN, recall,
                     download script, sift_check; SIFT1M recall investigated
                     to duplicate vectors; 56 tests; Phase 1 closed
2026-08-27  Phase 2  pushed Phase 0+1 to origin/main (13 commits, b20c81f..e73fcf4);
                     Phase 0's last exit criterion ticked; Phase 2 planning
```

---

## Numbers recorded so far

Full detail, methodology and regeneration commands in `BENCHMARKS.md`.

| Metric | Value | Machine | Date |
|---|---|---|---|
| Load, 1M × 128 | 0.33–0.84 s (page-cache dependent) | M1 | 2026-08-24 |
| Peak RSS, SIFT1M | 501.6 MiB | M1 | 2026-08-24 |
| Brute force | 11.58 QPS, k=10 (median of 3, 1.7% spread) | M1 | 2026-08-24 |
| recall@10, tie-aware | **1.000000** | M1 | 2026-08-24 |
| recall@10, strict id-set | 0.999440 | M1 | 2026-08-24 |
| SIFT1M distinct vectors | 985,462 of 1,000,000 | — | 2026-08-24 |
| Tests passing | 56/56 on 3 presets | M1 | 2026-08-24 |

**Machine M1:** AMD Ryzen 5 7530U (Zen 3, 6C/12T, 4546 MHz max), 15 GiB RAM,
GCC 11.4.0, CMake 4.3.1, Ninja 1.13.2. AVX2 + FMA, no AVX-512.

---

## Open questions

Things you don't understand yet and must resolve before the phase closes.
An empty list at phase close means you either understood everything or weren't
paying attention.

- [ ] **The prediction on record for Phase 2.** Brute force already moves
      5.5 GiB/s streaming the corpus. Single-core streaming bandwidth on a
      mobile Zen 3 part is roughly 15–20 GiB/s — far below DRAM peak, because a
      single core cannot keep enough misses outstanding. So the *scan* speedup
      should cap around **3×** however fast the kernel gets, while an L1-resident
      *microbenchmark* should show close to full SIMD width. Measure both and
      report them separately; the gap between them is the answer to Phase 2's
      "explain why it isn't 8×". **Falsify this rather than assume it.**
- [ ] Is the query buffer's lack of 64-byte alignment worth fixing? It is
      L1-resident for a whole search while the store side streams, so it is
      plausibly worth nothing. Measure before paying for an aligned allocator
      (`IDEAS.md`).
- [ ] Does the tail-free stride-wide loop actually help AVX2, or does the
      compiler handle a `dim`-bounded loop just as well at dim 128 where
      stride == dim? No test forces stride-wide looping — it is a permission the
      store grants, not an obligation — so this is genuinely open.

### Resolved during Phase 1

- [x] `cmake` / `ninja` not installed system-wide. Still true; the build is
      driven with CLion's bundled copies at
      `/snap/clion/current/bin/{cmake/linux/x64/bin,ninja/linux/x64}`. Works, but
      the commands in `README.md` and `CLAUDE.md` do not run as written on a bare
      shell. Fix with `sudo apt-get install -y cmake ninja-build`, or put those
      two paths on `PATH`.
- [x] Load time for 516 MB, and whether the simple record loop is good enough.
      0.35 s warm. Two thirds of that is `reserve()`'s page-fault cost, which no
      read strategy would change. Chunked reads not worth doing — `IDEAS.md`.
- [x] Does the 488 MiB zero-fill cost meaningful time? It looks like 0.245 s of
      0.358 s, but the *marginal* cost is 57 ms: without it the pages fault in
      during the read anyway.
- [x] Are there exact distance ties at rank 10 in SIFT1M? Yes — 56 of 10,000
      queries, every one caused by a byte-identical duplicate vector.
- [x] Would brute-force QPS be suspiciously fast, implying
      `-fno-tree-vectorize` was not applied? No. 11.58 QPS is 1.48 × 10⁹
      dimension-updates/s, roughly 3 cycles per dimension against a 4.5 GHz peak
      clock — what un-vectorised scalar costs.
- [x] Does stride padding matter at the dimensions in use? No — SIFT's 128 and
      GIST's 960 are both multiples of 16 and pad to nothing, which is exactly
      why tests run at dim 100 as well.
