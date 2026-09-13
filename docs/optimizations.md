# Optimization ladder

Each rung gets a branch, a benchmark, and a PR into `main`. Tags (`rung-01`,
`rung-02`, …) preserve the code; compare adjacent tags for the diff:

```sh
git diff rung-02 rung-03 -- src/
```

| Stage | Main change to inspect | What it targets | Decode tokens/s |
| --- | --- | --- | ---: |
| [01: FP32](https://github.com/ykumards/unremarkable/tree/rung-01) | One sum per matrix row | Baseline | 1.08 |
| [02: prefetch](https://github.com/ykumards/unremarkable/tree/rung-02) | Request weights 256 bytes ahead | Memory waits | 1.44 |
| [03: SIMD](https://github.com/ykumards/unremarkable/tree/rung-03) | Four NEON vectors, 16 partial sums | Arithmetic | 1.75 |
| [04: Q8](https://github.com/ykumards/unremarkable/tree/rung-04) | 32 int8 weights per scale | Memory traffic and size | 3.80 |
| [05: two threads](https://github.com/ykumards/unremarkable/tree/milestone/05-q8-threads) | Split projection rows across cores | Parallel execution | 5.76 |
| [06: six-op dot](https://github.com/ykumards/unremarkable/tree/milestone/06-q8-six-op) | Pair products before widening | Instructions per Q8 block | 6.32 |
| [07: batched prefill](https://github.com/ykumards/unremarkable/tree/milestone/07-batched-prefill) | Eight prompt tokens per projection; final logits only | Time to first token: 3.94 → 2.64 s | 6.05¹ |
| [08: serial steps](https://github.com/ykumards/unremarkable/tree/milestone/08-q8-serial-steps) | RoPE angles once per token; NEON quantization | Serial work between projections | 6.35 |
| [09: attention](https://github.com/ykumards/unremarkable/tree/milestone/09-attention) | NEON weighted sum; heads split across cores | Attention over cached tokens | 6.95 |

¹ Rung 07 changes prefill. Its matched baseline decoded at 5.92 tok/s; it does
not establish a decode improvement or regression against rung 06's older run.

The runs below measure each change and explain why prefetch comes before SIMD.

## Measured: 01 on the tablet (2026-09-12)

[Raw runs](../runs/naive-fp32-20260912b/). Rung 01 reorganizes the original scalar
engine (`898d904`). Both executables ran on the same boot: one warmup each, then
rung 01, original, original, rung 01.

| Executable | Decode tokens/s | Mean | Mean TTFT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Rung 01 (`c34027a7`) | 1.079, 1.083 | 1.081 | 23.62 s | 539.5 MiB |
| Original scalar, from `898d904` (`374f424e`) | 1.083, 1.085 | 1.084 | 23.68 s | 539.4 MiB |

The 0.3% gap is within the observed run variation. All four outputs match
(`860192f8`), with 64 generated tokens and a limit stop.

Workload: reMarkable 2, `ondemand` governor, SmolLM2-135M-Instruct FP32
(`82335f56`), 26 prompt tokens, no system prompt, greedy, seed 1, context 512.
Build with `make tablet`. Hashes in the tables are SHA-256 prefixes; full hashes
are saved with the runs. TTFT means time to first token, excluding model loading.

```sh
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
./unremarkable models/smollm2-135m.bin -z models/smollm2-135m.tok \
  -c 512 -y '' -t 0 -s 1 -i "$prompt" -n 64
```

Decode speed excludes prefill and the first generated token. The same prompt,
model, and generation settings are used below.

## Measured: 02 on the tablet (2026-09-12)

[Raw runs](../runs/fp32-neon-20260912b/). Four builds of the same experimental
`matvec` ran on the boot used for rung 01: one warmup each, then scalar, NEON,
prefetch, NEON plus prefetch, and back in reverse. The prefetch build needed a
correction and a later run, detailed below.

| `matvec` | Decode tokens/s | Mean | vs 01 | Mean TTFT | Output |
| --- | --- | ---: | ---: | ---: | --- |
| 01: scalar (`c34027a7`) | 1.080, 1.080 | 1.080 | 1.00x | 23.82 s | `860192f8` |
| **02: scalar plus prefetch** (`cf1de842`) | 1.436 | 1.436 | **1.33x** | 17.61 s | `860192f8` |
| NEON, four accumulators (`89800510`) | 1.027, 1.028 | 1.028 | 0.95x | 24.87 s | `860192f8` |
| NEON plus prefetch (`85a8d5ee`) | 1.749, 1.751 | 1.750 | 1.62x | 14.42 s | `860192f8` |

Rung 02 keeps the scalar loop and adds prefetch (`cf1de842`). It requests weights
256 bytes ahead, once per 16 floats, without changing addition order. Host logits
and tablet text match rung 01 exactly.

Prefetch improved decode by 33% in this comparison. NEON alone was 5% slower;
combined with prefetch it was 62% faster than scalar. These timings support the
rung order, but do not isolate the cause of every stall.

**Prefetch has only one valid run.** The first build skipped every other
16-float chunk. Runs 3 and 6 produced invalid text (`331a5336`) and are excluded
from the table but retained on disk. The corrected build ran about 25 minutes
later on the same boot. The other variants' paired runs agreed within 0.2%.

The tablet UI (`xochitl`) was stopped for this experiment. With it running,
a NEON attempt was OOM-killed: little memory remained outside the CMA reserve,
which could not satisfy the failed allocation. Stopping the UI freed about
150 MiB. Scalar measured 1.080 tok/s with it stopped versus 1.081 in rung 01
with it running.

```sh
systemctl stop xochitl   # restart with: systemctl start xochitl
```

## Measured: 03 on the tablet (2026-09-12)

Rung 03 reproduces the NEON-plus-prefetch binary (`85a8d5ee`) from
[02's experiment](../runs/fp32-neon-20260912b/), so it uses those measurements.
`-DUNREMARKABLE_SCALAR` reproduces rung 02's binary (`cf1de842`).

| `matvec` | Decode tokens/s | Mean | vs 02 | vs 01 | Mean TTFT | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 02: scalar plus prefetch (`cf1de842`) | 1.436 | 1.436 | 1.00x | 1.33x | 17.61 s | `860192f8` |
| **03: NEON plus prefetch** (`85a8d5ee`) | 1.749, 1.751 | 1.750 | **1.22x** | 1.62x | 14.42 s | `860192f8` |

Four NEON vectors hold 16 partial sums. Each multiply handles four values;
separate sums reduce the dependency on one running total. The measured gain over
prefetch alone is 22%, with only one prefetch-only run for comparison.

Addition order changes. Across 18 tested prompt positions, host logits differed
by at most 2e-4, with the same top-1 token at every position. The 64-token tablet
outputs also matched.

Rung 04 reduces weight storage from about 538 MB to 151 MB.

## Measured: 04 on the tablet (2026-09-12)

The [raw run](../runs/q8-neon-20260912/) is one 64-token run on the boot that
measured 01 to 03, with the UI running.

| Engine | Checkpoint | Decode tokens/s | vs 03 | vs 01 | TTFT | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 03: FP32 (`85a8d5ee`) | FP32, 538 MB (`82335f56`) | 1.750 | 1.00x | 1.62x | 14.42 s | 539.6 MiB |
| **04: Q8** (`14b247ca`) | Q8_0, 151 MB (`f91ef591`) | **3.804** | **2.17x** | **3.52x** | 6.37 s | 170.8 MiB |

Q8_0 stores every matrix as groups of 32 int8 weights that share one FP32 scale:
36 bytes where FP32 needs 128. Each projection input is quantized the same way,
so every group is an exact integer dot product (`vmull_s8`, widened with
`vpadalq_s16`) scaled by both groups' scales, and the groups add in FP32. Norm
vectors and the KV cache stay FP32.

Weight traffic falls by about 3.6x; measured decode speed rises by 2.17x.
The comparison uses one Q8 run with the UI on and two FP32 runs with it off.
The next steps add a second core and reduce instructions per block.

Peak memory falls from 539.6 to 170.8 MiB. This Q8 run completed with the tablet
UI running.

Quantization changes results: perplexity rises 0.6% (see the accuracy section
below), and the 64-token text differs from FP32's (`4ea6b63a`).

```sh
make chat-model-q8   # tools/export_hf.py -q q8_0; needs numpy
./unremarkable models/smollm2-135m-q8.bin -z models/smollm2-135m-q8.tok \
  -c 512 -y '' -t 0 -s 1 -i "$prompt" -n 64
```

UNRK version 2 adds the matrix format after the RoPE base; version 1 files still
load as FP32.

### Accuracy (2026-09-13)

Q8_0 versus FP32 on the WikiText-2 test text: the first 40 chunks of 512
tokens, selected before seeing the results. Only the second half of each chunk is scored, so every prediction has at
least 256 tokens of context: 10,200 scored tokens. `make accuracy` reproduces it
on the host; the [raw rows](../runs/accuracy-q8-20260913/) hold every token.

| Checkpoint | Perplexity | vs FP32 | Mean KL | 99th percentile KL | Same top token |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 18.79 | | | | |
| **Q8_0** | 18.90 | **+0.60% ± 0.09%** | 0.0038 | 0.018 | 95.7% |
| Control: 15 levels per group | 27.83 | +48% | 0.40 | 1.82 | 64.3% |

The ± is one standard error over chunks, accounting for correlation between
neighbouring tokens. Q8 loss is higher in 35 of 40 chunks (sign test p = 1.4e-6)
and 52.9% of scored tokens. A one-chunk trial had shown a small improvement;
the larger evaluation reversed it.

Validation: FP32 against itself gives zero KL and loss difference. An independent
Python calculation reproduces loss and KL at four positions. For the control,
the exporter uses limits of 7 instead of 127 in `quantize_q8_0`: 15 levels
(−7…7) instead of 255. Its perplexity rises 48%.

Scores come from the host build. This comparison measures the quantization loss
on WikiText-2, not general instruction-following quality.

## Measured: 05 on the tablet (2026-09-13)

[Raw runs and command](../runs/q8-threads-20260913b/) · [PR #5](https://github.com/ykumards/unremarkable/pull/5).
SmolLM2-135M Q8, context 512, 26 prompt tokens, greedy seed 1, 64 generated tokens
(63 timed decode steps). The UI stayed running, both cores were online, and the
CPU governor was `ondemand`. No CPU affinity was set.

One warmup per configuration, then rung 04 as a reference and an interleaved
one-thread, two-thread, two-thread, one-thread comparison:

| Configuration | Runs (tok/s) | Mean ± sample std | Mean TTFT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Rung 04 | 3.801 | 3.801 (one run) | 6.39 s | 170.89 MiB |
| Rung 05, `-j 1` | 3.818, 3.820 | 3.819 ± 0.001 | 6.39 s | 170.90 MiB |
| **Rung 05, `-j 2`** | 5.727, 5.792 | **5.760 ± 0.046** | **3.98 s** | 170.90 MiB |

Two threads improve decode by **1.51×** and reduce TTFT by **38%** in this short
comparison. All five measured outputs are byte-identical (`4ea6b63a`); full
binary/model hashes and per-run timings are saved with the runs.

`Engine::project()` quantizes the input once, then assigns whole rows to the
caller and one persistent worker. Each row keeps the same dot-product order.
The worker finishes before the input buffer is reused. Jobs under 64 rows stay
on the caller; `-j 1` creates no worker. Quantization and the other operators
remain serial. SmolLM2 has 211 projections per token, each with a completion wait.

Validation: 17 engine tests and the worker test pass normally and with
ASan/UBSan; the worker test also passes ThreadSanitizer. FP32 and Q8 logits match
between thread counts, including reset, odd row counts, and padded Q8 tails.
ARMv7 build and formatting pass. The kernel arithmetic is unchanged from rung 04.

## Measured: 06 on the tablet (2026-09-13)

[Raw runs](../runs/q8-six-op-20260913/) · [PR #6](https://github.com/ykumards/unremarkable/pull/6).

The Q8 dot product now pairs products in int16 before widening to int32:

| Per 32-value block | Rung 05 | Rung 06 |
| --- | --- | --- |
| Multiply into int16 | 4 `vmull_s8` | 2 `vmull_s8` + 2 `vmlal_s8` |
| Sum into int32 | 4 `vpadalq_s16` | 1 `vpaddlq_s16` + 1 `vpadalq_s16` |
| Arithmetic instructions | 8 | 6 |

Loads, final lane reduction, scale application, and the worker are unchanged.
Paired products fit int16: `2 × 128 × 127 = 32512`. Input values must stay in
`[-127, 127]`, as the quantizer guarantees; weights may include `-128`.

| Kernel, two threads | Runs (tok/s) | Mean ± sample std | Mean TTFT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Rung 05: eight-op | 5.770, 5.787 | 5.779 ± 0.012 | 3.972 s | 170.86 MiB |
| **Rung 06: six-op** | 6.324, 6.311 | **6.318 ± 0.009** | **3.578 s** | 170.90 MiB |

Decode improved **9.3%** and TTFT fell **9.9%** in this short comparison.
All four outputs are byte-identical (`4ea6b63a`). Each variant has two measured
runs; compare against this run's baseline, rather than pooling older timings.

The benchmark used SmolLM2-135M Q8, two threads, context 512, the 26-token
lighthouse prompt, greedy seed 1, and 64 generated tokens (63 timed decode
steps). Both binaries ran on the same boot with the UI active and the governor
set to `ondemand`: one 16-token warmup each, then eight-op, six-op, six-op,
eight-op. Warmups are excluded from the table.

Validation: all 65,280 supported input/weight pairs, mixed signs, scales, zero
inputs, and tails pass on host NEON, host scalar, and tablet ARMv7. The 17 engine
tests and worker tests pass, including ASan/UBSan; formatting passes. Host
SmolLM2 logits match rung 05 exactly at four tested positions. The saved ARMv7
assembly excerpt confirms six arithmetic instructions.

## Measured: 07 on the tablet (2026-09-13)

[Raw runs](../runs/prefill-20260913/) · [PR #7](https://github.com/ykumards/unremarkable/pull/7).

`Engine::prefill()` processes up to eight prompt tokens per chunk. Each layer's
projections become matrix-matrix products: `X[batch, input] × Wᵀ → Y[batch, output]`.
The kernel reuses a weight row through cache across tokens, using the existing
NEON dot product. It has no packed weights or register tiling yet. The worker
splits weight rows as before, with one wake/wait per batched projection.

Only the last prompt token runs the final norm and vocabulary projection.
`-b 1` isolates that saving; `-b 8` adds batching. `-b 0` retains the original
forward loop. Decode still uses `forward()`.

| Prefill mode | TTFT runs (s) | Mean TTFT ± sample std | Prefill tok/s ± sample std | Decode tok/s ± sample std |
| --- | --- | ---: | ---: | ---: |
| Original loop, `-b 0` | 4.120, 3.753 | 3.937 ± 0.260 | 6.62 ± 0.44 | 5.918 ± 0.156 |
| Final logits only, `-b 1` | 3.348, 3.229 | 3.289 ± 0.084 | 7.91 ± 0.20 | 5.986 ± 0.006 |
| **Batch eight, `-b 8`** | 2.643, 2.633 | **2.638 ± 0.007** | **9.86 ± 0.02** | 6.053 ± 0.004 |

Mean TTFT falls **33%** against the original loop. Batching accounts for a
further **20%** reduction after skipping unused logits. Prefill throughput is
prompt tokens divided by prefill time, distinct from decode throughput. The
turtle remains at rung 06's measured 6.32 tok/s; this step targets prompt latency.

Same 26-token lighthouse prompt, SmolLM2-135M Q8, two threads, context 512,
greedy seed 1, and 64 generated tokens (63 timed decode steps). One warmup per
mode and parent binary, then parent, `0, 1, 8, 8, 1, 0`. Both cores were online,
the governor was `ondemand`, and the UI stayed active on the same boot. No CPU
affinity was set. The separately built parent (`a247971`) gave 3.796 s TTFT and
6.043 decode tok/s in one run.

All seven measured outputs match (`4ea6b63a`), including the parent. Peak RSS
was 171.16 MiB with batching versus 170.88 MiB for the parent; added Q8 scratch
storage is 174.6 KiB. The baseline pair varies by 0.37 s, and the first parent
warmup was slower still. Warmups are saved but excluded; this short comparison
does not establish a decode speedup or predict longer-prompt performance.

Validation: 19 engine tests pass normally and with ASan/UBSan, plus worker and
Q8 kernel tests. Batched FP32/Q8 logits and subsequent decode exactly match
sequential forward passes on host and ARMv7 tablet fixtures, covering prefixes,
partial chunks, reset, context boundaries, and invalid input. Formatting and
the ARMv7 build pass. [Data flow and shapes](inference.md#batched-prefill).

## Measured: 08 on the tablet (2026-09-13)

[Raw runs](../runs/rung08-exact-20260913/).

RoPE angles are computed once per token and reused across heads and layers.
Input quantization uses NEON for full 32-value blocks and rounds halves away
from zero without calling `lround`. SwiGLU keeps the original `std::exp` loop.

| Build, two threads | Decode runs (tok/s) | Mean ± sample std | Mean TTFT ± sample std |
| --- | --- | ---: | ---: |
| Baseline, rung 07 inference | 6.077, 6.089 | 6.083 ± 0.008 | 2.658 ± 0.035 s |
| **Rung 08** | 6.314, 6.379 | **6.347 ± 0.046** | **2.494 ± 0.011 s** |

Decode improves **4.3%** and TTFT falls **6.2%** in this comparison. All four
outputs are byte-identical (`4ea6b63a`). SmolLM2-135M Q8, batch 8, two threads,
context 512, 26-token lighthouse prompt, greedy seed 1, 64 generated tokens.
One warmup each, then baseline, candidate, candidate, baseline on the same boot
with the UI active and the `ondemand` governor.

Validation: 20 host tests pass; 19 run under ASan/UBSan (the profile-build test
is skipped). Kernel tests pass on host NEON, scalar, and ARMv7, including exact
quantization rounding and RoPE at positions 0..600. Formatting passes.

The earlier 6.51 tok/s result included approximate SwiGLU and is superseded.
Those [timing](../runs/rung08-20260913/), [profile](../runs/rung08-profile-pair-20260913/),
and [accuracy](../runs/accuracy-rung08-20260913/) runs remain as historical data.

## Measured: 09 on the tablet (2026-09-13)

[Raw runs and patches](../runs/attention-heads-20260913/).

NEON computes four components of the weighted value sum at once. Each keeps
its original token order. The square root moves outside the loops, and the
existing worker takes five of SmolLM2's nine heads. Query–key dots and softmax
stay scalar; no cache layout or Q8 kernel changes.

| Build | Decode tok/s ± sample std | TTFT ± sample std |
| --- | ---: | ---: |
| Baseline, rung 08 | 6.299 ± 0.032 | 2.487 ± 0.003 s |
| NEON weighted sum + hoisted square root | 6.519 ± 0.246 | 2.475 ± 0.038 s |
| **Plus heads split across cores** | **6.949 ± 0.045** | **2.453 ± 0.016 s** |

Decode improves **10.3%** against the matched baseline. Rung 08 previously
recorded 6.35 tok/s; its fresh baseline here is 6.30. Three runs per build,
after a warmup each. SmolLM2-135M Q8, batch 8, two threads, context 512,
26-token lighthouse prompt, greedy seed 1, 64 generated tokens. UI active,
`ondemand` governor. All nine texts match. The slow single-core attention run
is included; no samples were dropped.

With the same prompt and a [192-token generation](../runs/attention-long-20260913/),
decode rises from **5.491 ± 0.022 to 6.514 ± 0.008 tok/s (+18.6%)**. Two runs
per build, 191 timed decode steps each; all four texts match. This longer history
increases attention's share of the work. The turtle uses the 64-token benchmark.

An [earlier comparison](../runs/attention-20260913/) put the single-core variant
at 6.698 versus 6.365 tok/s baseline. SIMD query–key dots reached 6.722, but
changed logits and generated text. That patch and the earlier
[Q8 accumulation experiments](../runs/rung09-q8-20260913/) are parked.

Host tests, ASan/UBSan, formatting, and ARMv7 attention tests pass. The retained
build matches baseline host logits exactly at eight checked positions. Tests
cover grouped heads, split ranges, SIMD tails, and causal boundaries.

## Memory ceiling

Sequential reads measured on the tablet on 2026-09-10: a 144 MiB buffer,
eight passes, using NEON loads and prefetch.

| Threads | Read bandwidth (GB/s) |
| --- | ---: |
| 1 | 1.423 |
| 2 | 2.708 |
| 3 | 2.643 |
| 4 | 2.630 |

Q8 weights occupy about 0.151 GB including scales. Streaming them once per token
gives **2.708 ÷ 0.151 ≈ 17.9 tok/s**. This estimate assigns all measured bandwidth
to weight reads; it excludes arithmetic, KV-cache traffic, and synchronization.
