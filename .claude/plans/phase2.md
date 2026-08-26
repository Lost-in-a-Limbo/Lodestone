# Phase 2 — SIMD distance kernels

**Written:** 2026-08-27
**Goal:** the first real performance number, and the first place core CS pays off.
**Exit criteria (PRD §6 Phase 2):** scalar L2 and inner product; SSE and AVX2
variants using intrinsics; runtime dispatch on CPU feature detection; a
microbenchmark comparing all variants; tests asserting SIMD agrees with scalar;
**AVX2 L2 ≥3× scalar**; and you can explain why the speedup isn't exactly 8×.
**Record:** ns per distance computation for each variant, at dim 128 and dim 960.

---

## Two corrections to the exit criteria, before anything is built

**1. "All variants agree with scalar within 1e-4" cannot mean absolute error.**

SIFT squared-L2 distances are around 5 × 10⁴. Changing the summation order —
which is exactly what vectorising does — perturbs a float32 sum of n terms by
roughly `n · ε · magnitude`. At n = 128, ε ≈ 1.2 × 10⁻⁷ and magnitude 5 × 10⁴,
that is an expected absolute difference of order **10⁻³ to 10⁻²**, an order of
magnitude past the stated 1e-4 and not a bug in anything.

So the criterion is read as **relative** agreement, and tests assert 1e-5
relative. Absolute 1e-4 on a value of 5 × 10⁴ is 2 × 10⁻⁹ relative, which
float32 simply cannot deliver — asserting it would guarantee a red test on
correct code.

**2. Distance agreement is the weak test. Recall is the strong one.**

Two kernels can agree to 1e-5 and still return different neighbours, because a
reordering only needs the gap between ranks 10 and 11 to be smaller than the
error. That is the same float32 effect Phase 1 met head-on.

So the real correctness gate reuses what Phase 1 already built: run SIFT10K
end-to-end through each kernel and require tie-aware recall@10 = 1.000000. A
kernel that computes distances "close enough" but ranks them differently fails
that, and passes a tolerance test. Task 7.

---

## The measurement design, which is the heart of this phase

**"ns per distance computation" is two different numbers, and reporting one is
how a SIMD speedup gets overstated.**

`PHASE.md` carries a prediction made at the close of Phase 1, before any of this
was written, precisely so it can be falsified:

- A **microbenchmark with the vectors resident in L1** is compute-bound. It
  should show close to the full SIMD-width speedup.
- A **full 1M scan** streams 488 MiB per query and already moves 5.5 GiB/s.
  Single-core streaming bandwidth on a mobile Zen 3 part is roughly 15–20 GiB/s
  — far below DRAM peak, because one core cannot keep enough misses outstanding.
  So the scan speedup should cap near **3×** however fast the kernel gets.

Both get measured and both get published. If the microbenchmark shows 6× and
brute-force QPS improves 3×, that is not a contradiction, it is the memory wall,
and it is the answer to "explain why the speedup isn't exactly 8×".

There are in fact **four** reasons the speedup will not be 8×, and the phase
should be able to attribute the shortfall to each:

| Cause | Where it shows |
|---|---|
| FMA latency vs a single accumulator chain | Task 5 — the biggest effect, and the most instructive |
| Horizontal reduction, paid once per distance | Larger at dim 128 (16 loop iterations) than dim 960 (120) |
| Memory bandwidth | Streaming regime only; invisible in L1 |
| Loads, not arithmetic, being the bottleneck | 2 loads per FMA — the loop is load-port bound before it is FLOP bound |

---

## Assumptions (override any of these at review)

| # | Assumption |
|---|---|
| A1 | **No AVX-512.** M1 is Zen 3 and has none. Already in `IDEAS.md`; the dispatch table makes adding a fourth entry mechanical if a machine ever appears. |
| A2 | SSE means SSE2 + SSE3 (`_mm_hadd_ps`), which every x86-64 CPU has. Its role is the guaranteed fallback, and the 4-wide midpoint between scalar and AVX2. |
| A3 | CPU detection via `__builtin_cpu_supports`, not hand-rolled CPUID. It is a GCC/Clang builtin, needs no inline asm, and its AVX path already accounts for OS `XSAVE` state — a raw CPUID feature bit does not, and would happily select AVX2 on a kernel that does not preserve YMM registers. |
| A4 | dim 960 is measured on **synthetic random data**. We have no GIST corpus and do not need one: this is ns/distance, not recall. Flagged in the output so it is never mistaken for a GIST result. |
| A5 | Google Benchmark via CPM, `BENCHMARK_ENABLE_TESTING OFF`. Built by the `release` preset only, and **never registered as a ctest** — CI must not time anything (D11). |
| A6 | All computers keep the convention **smaller means closer**. See Task 1 for what that forces on inner product. |
| A7 | `-ffast-math` is never used, anywhere. It would let the compiler reassociate the scalar baseline's sum and vectorise it despite `-fno-tree-vectorize`, and it changes results in ways that are invisible until a recall number moves. |

---

## Task list

Eight tasks. 1→2 must precede the kernels; 3, 4, 5 are the kernel work in order;
6, 7, 8 close it.

---

### Task 1 — Scalar inner product, and explicit kernel selection

**Test first:** extend `tests/test_distance.cpp`
**Touches:** `src/distance_scalar.cpp`, `src/distance.cpp`, `include/lodestone/distance.hpp`

**The inner-product convention, which is a real decision.** Larger inner product
means *more* similar, which is backwards from L2. Every consumer — the bounded
heap in `brute_force.cpp`, and Phase 3's candidate priority queues — is written
around "smaller is closer". Two options: teach every consumer to flip its
comparator per metric, or have the inner-product computer return **negated** dot
product.

Negation wins, and not narrowly. A per-metric comparator means every future
piece of graph code has a chance to get the sign wrong, and the failure mode is
a search that returns the *farthest* neighbours while looking perfectly healthy.
One negation in one kernel cannot be got wrong twice. The cost is that
`distance_to(id)` for a vector against itself is `-‖x‖²` rather than 0, which is
merely surprising, not dangerous — and it gets a test asserting it so nobody
"fixes" it.

**Kernel selection.** Add to `distance.hpp`:

```cpp
enum class KernelKind : std::uint8_t { automatic, scalar, sse, avx2 };

std::unique_ptr<DistanceComputer> make_distance_computer(
    Metric, const VectorStore&, KernelKind = KernelKind::automatic);

KernelKind detected_kernel();          // what `automatic` would choose here
std::string_view kernel_name(KernelKind);
```

The default argument means no existing caller changes — brute force and the
tests keep compiling untouched, which is the property D15 was set up to have.
Explicit selection exists because the microbenchmark and the correctness tests
must instantiate *each* kernel, not whichever one this laptop happens to pick;
and `detected_kernel()` exists so Phase 4's `results.json` can record which
kernel produced a number.

Graph code still names no class. `KernelKind` is a request, not a type.

**Tests:** hand-computed dot products; the negation convention including the
self-distance case; `automatic` resolves to something non-`automatic`;
requesting a kernel this CPU lacks returns nullptr rather than crashing.

**Proof:** `./build/debug/tests/lodestone_tests "[distance]"`

---

### Task 2 — The benchmark rig, before any kernel is optimised

**Creates:** `bench/CMakeLists.txt`, `bench/bench_distance.cpp`

Built first, deliberately. Optimising before the measurement exists is how you
end up believing an improvement that was noise.

Two fixtures, and the difference between them is the whole point:

- **`l1_resident`** — a store small enough to sit in L1 (a few KiB), one query,
  distances computed over the same handful of vectors repeatedly. Compute-bound.
  This is the number that shows what the kernel can do.
- **`streaming`** — a store far larger than L3's 16 MiB, walked linearly so
  every distance touches cold lines. Memory-bound. This is the number that
  predicts brute-force QPS.

Both at **dim 128** and **dim 960**, both through `distances_to()` in batches,
because that is how the graph will call it.

Reported as **ns per distance** and, for the streaming fixture, **effective
GiB/s** — because a bandwidth figure is what makes the ceiling legible instead
of mysterious.

Google Benchmark handles warmup and repetition itself; the project's
median-of-three rule applies to the numbers copied into `BENCHMARKS.md`.

**Proof:** `./build/release/bench/bench_distance --benchmark_min_time=0.5s`

---

### Task 3 — SSE kernels

**Creates:** `src/distance_sse.cpp`
**Test first:** extend `tests/test_distance.cpp`

4-wide L2 and inner product. Straightforward, and its real value is being the
control: if SSE lands near 4× and AVX2 near 8× in the L1 fixture, the scaling
is real. If SSE lands at 4× and AVX2 also at 4×, something is wrong with the
AVX2 kernel — and without SSE there is nothing to notice that against.

**The compile-flag trap, and it is the same shape as D5.** Under
`-march=native`, GCC compiles SSE intrinsics to VEX-encoded 128-bit
instructions. That alone does not widen them, so the measurement stays 4-wide —
but any scalar code left in the file *can* be auto-vectorised to AVX2, and the
result would be an "SSE" number that is partly AVX2.

So this file is compiled `-msse4.2 -mno-avx -mno-avx2 -mno-fma`, appended after
`-march=native` so the negations win, and it carries an **inverted** guard:

```cpp
#ifdef __AVX2__
#error "distance_sse.cpp must be compiled with -mno-avx2 ..."
#endif
```

The AVX2 file `#error`s when `__AVX2__` is *absent*; this one `#error`s when it
is *present*. Both flags then fail the build loudly rather than quietly
producing a number that means something other than its label.

**Proof:** tests green, plus the SSE row appearing in the bench output.

---

### Task 4 — AVX2 kernels, single accumulator

**Touches:** `src/distance_avx2.cpp` (currently a stub with its guards already in place)

8-wide, `_mm256_fmadd_ps`, one accumulator. Written this way *first* on purpose,
because Task 5 is the measurement that explains the phase.

**The horizontal reduction.** The loop leaves eight partial sums in a `__m256`
that must collapse to one float. The idiom:

```cpp
__m128 lo = _mm256_castps256_ps128(acc);        // free — just a register view
__m128 hi = _mm256_extractf128_ps(acc, 1);
lo = _mm_add_ps(lo, hi);                        // 8 -> 4
lo = _mm_hadd_ps(lo, lo);                       // 4 -> 2
lo = _mm_hadd_ps(lo, lo);                       // 2 -> 1
return _mm_cvtss_f32(lo);
```

Worth understanding rather than copying: it is ~5 dependent instructions with
real latency, paid **once per distance**, not per element. At dim 128 the main
loop runs 16 iterations, so the reduction is a visible fraction of the work; at
dim 960 it is 120 iterations and the reduction nearly vanishes. **That alone
predicts the AVX2 speedup will be larger at dim 960 than at dim 128**, and the
bench should confirm it. This is the cheapest available piece of the "why not 8×"
answer.

**Use the tail-free property.** `stride` is always a multiple of 16 floats, the
store zeroes its padding, and `prepare_query` zero-pads the query (D13). So the
loop is `for (i = 0; i < stride; i += 8)` with **no tail and no masked load**.
That is the single biggest simplification Phase 1 handed this phase.

**Proof:** tests green; `[distance]` agreement at 1e-5 relative; first AVX2
numbers recorded.

---

### Task 5 — The accumulator experiment

**Touches:** `src/distance_avx2.cpp`

This is the task that earns the phase's exit criterion, and it is an experiment
rather than an implementation.

`vfmadd231ps` on Zen 3 has ~4 cycles of latency and 2/cycle of throughput. A
single accumulator makes every FMA depend on the previous one, so the loop runs
at **one FMA per 4 cycles** instead of two per cycle — a factor of 8 left on the
floor, which is very nearly the entire theoretical SIMD win. Breaking the chain
into k independent accumulators summed at the end lets k FMAs be in flight at
once, and saturation needs `latency × throughput = 4 × 2 = 8` of them.

So: measure 1, 2, 4 and 8 accumulators at both dimensions, **record the whole
curve in `BENCHMARKS.md`**, and keep the winner. The prediction is a steep climb
from 1 to 4 and a flattening by 8, where the loop becomes load-port bound
instead — 2 loads per FMA, against 2 load ports.

Recording the losing configurations is the point. "AVX2 is 5.8× scalar" is a
number; "a naive AVX2 kernel is 1.9× scalar and four accumulators make it 5.8×,
because FMA latency is 4 cycles" is the thing worth knowing, and it is what
someone will ask about in an interview.

**Proof:** the accumulator table, and AVX2 L2 ≥3× scalar in the L1 fixture.

---

### Task 6 — Runtime dispatch

**Touches:** `src/distance.cpp`

`make_distance_computer(..., KernelKind::automatic)` picks AVX2 if
`__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")`, else SSE,
else scalar. Both checks, because AVX2 and FMA are separate feature bits and the
kernel uses `_mm256_fmadd_ps`.

Still the body of one function, still no caller changes — the shape D15 was
chosen for, now actually exercised.

**Tests:** `detected_kernel()` never returns `automatic`; on a CPU reporting
AVX2 it returns `avx2`; every kernel that `detected_kernel()` claims is
available can actually be constructed and produces correct distances. That last
one is the check that matters — a dispatch table that selects a kernel the
machine cannot run is an illegal-instruction crash, not a wrong number.

---

### Task 7 — End-to-end correctness through every kernel

**Touches:** `tools/sift_check.cpp`, `tools/CMakeLists.txt`

Add `--kernel scalar|sse|avx2|auto`. Register three ctest entries running
SIFT10K through each available kernel, each asserting tie-aware
recall@10 = 1.000000, each skipping with code 4 when `data/` is absent.

This is the strong correctness gate promised at the top. A kernel whose
distances agree to 1e-5 but whose *ranking* differs passes a tolerance test and
fails this one. It costs ~1 s per kernel under the sanitised preset.

Then re-run SIFT1M through the dispatched kernel and record brute-force QPS,
which is the streaming number the Phase 1 prediction is about.

**Proof:**
```bash
ctest --test-dir build/debug --output-on-failure
./build/release/tools/sift_check data/sift/sift_base.fvecs \
    data/sift/sift_query.fvecs data/sift/sift_groundtruth.ivecs
```

---

### Task 8 — Close the phase

- `BENCHMARKS.md`: ns/distance for scalar, SSE, AVX2 at dim 128 and 960, in
  **both** the L1 and streaming fixtures; the accumulator curve; the new
  brute-force QPS; effective GiB/s and how close it is to the bandwidth ceiling
- `DECISIONS.md`: the negated inner product; `KernelKind` as a request rather
  than a type; `-mno-avx2` on the SSE file; the accumulator count and its
  measured justification; relative-not-absolute tolerance
- `PHASE.md` → Phase 3, and **settle the Phase 1 prediction explicitly** — it
  was written down to be falsified, so say plainly whether it held
- `IDEAS.md`: whatever surfaced and was not built

---

## Open questions to resolve before the phase closes

- [ ] Does the streaming speedup really cap near 3×, or is the single-core
      bandwidth estimate wrong? **This is a prediction to falsify, not confirm.**
- [ ] Is the AVX2 speedup larger at dim 960 than dim 128, as the
      reduction-cost argument predicts?
- [ ] Does the 64-byte-aligned store actually buy anything measurable over
      unaligned loads, given that Zen 3 handles unaligned loads cheaply when
      they do not cross a line? Carried over from Phase 1 (`IDEAS.md`).
- [ ] Does the tail-free stride-wide loop beat a `dim`-bounded loop with a tail,
      at dim 128 where `stride == dim` and the tail would be empty anyway?
- [ ] How much of the remaining gap is load-port pressure rather than FLOPs?
      2 loads per FMA against 2 load ports says the loop may be load-bound at
      the top end, which would cap it near 4× regardless of accumulator count.

---

## Explicitly NOT in this phase

No HNSW, no graph, no `results.json`, no `hnswlib`, no CLI11, no filtering, no
quantisation, no AVX-512, no multithreading, no prefetching (logged in
`IDEAS.md` — it is a Phase 2-shaped idea but it is not on the exit criteria, and
the accumulator experiment is the one that answers the exit question).
