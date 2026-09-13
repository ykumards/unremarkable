# Optimization ladder

Each rung gets a branch, a benchmark, and a PR into `main`. Tags `rung-01`
through `rung-04` preserve the code; compare adjacent tags for the diff:

```sh
git diff rung-02 rung-03 -- src/
```

| Stage | Main change to inspect | What it targets | Decode tokens/s |
| --- | --- | --- | ---: |
| [01: FP32](https://github.com/ykumards/unremarkable/tree/rung-01) | One sum per matrix row | Baseline | 1.08 |
| [02: prefetch](https://github.com/ykumards/unremarkable/tree/rung-02) | Request weights 256 bytes ahead | Memory waits | 1.44 |
| [03: SIMD](https://github.com/ykumards/unremarkable/tree/rung-03) | Four NEON vectors, 16 partial sums | Arithmetic | 1.75 |
| [04: Q8](https://github.com/ykumards/unremarkable/tree/rung-04) | 32 int8 weights per scale | Memory traffic and size | 3.80 |
| Two threads | Split projection rows across cores | Parallel work and coordination costs | |
| Six-op dot product | Pair integer products before widening | Instructions inside each Q8 group | |

The remaining steps exist on [`optimized`](https://github.com/ykumards/unremarkable/tree/optimized).
We port and measure them separately. The runs below explain why prefetch comes
before SIMD in this ladder.

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
below), and the 64-token text differs from FP32's (`4ea6b63a`). The logits are
byte-identical to the Q8 engine on `optimized`.

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

Scores come from the host build, where Q8_0 logits match `optimized` byte for
byte. SmolLM2 is instruction-tuned and has likely seen Wikipedia, so its absolute
perplexity here says less than the gap between the two checkpoints.
