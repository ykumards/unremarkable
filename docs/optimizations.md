# Optimization ladder

`main` starts with the readable FP32 scalar engine. Each next optimization is
explained, benchmarked, and merged into `main`; milestone branches preserve the
completed checkpoints.

[`milestone/01-fp32-scalar`](https://github.com/ykumards/unremarkable/tree/milestone/01-fp32-scalar)
preserves the first rung. It branches from the original scalar engine
(`898d904`) and reorganizes ownership and names while preserving the mathematical
operation order. It is a refactored baseline, not an exact snapshot of an old
timed executable.
[`milestone/02-fp32-prefetch`](https://github.com/ykumards/unremarkable/tree/milestone/02-fp32-prefetch)
preserves the second,
[`milestone/03-fp32-neon`](https://github.com/ykumards/unremarkable/tree/milestone/03-fp32-neon)
the third, and
[`milestone/04-q8-neon`](https://github.com/ykumards/unremarkable/tree/milestone/04-q8-neon)
the fourth.

| Stage | Main change to inspect | What it targets | Decode tokens/s |
| --- | --- | --- | ---: |
| 01: naive FP32 | Ordinary row-by-row dot products | Establish the computation and data flow | 1.08 |
| 02: prefetch | Request each weight cache line 256 bytes early | Time spent waiting for memory | 1.44 |
| 03: explicit SIMD | Four NEON accumulators, 16 partial sums | Arithmetic, once memory waits shrink | 1.75 |
| 04: Q8 | Groups of 32 int8 weights sharing one scale | Bytes read per token and memory footprint | 3.80 |
| Two threads | Split projection rows across cores | Parallel work and coordination costs | |
| Six-op dot product | Pair integer products before widening | Instructions inside each Q8 group | |

Later rows describe experiments already explored on
[`optimized`](https://github.com/ykumards/unremarkable/tree/optimized), which
preserves the previous `main` history and six-op improvements. There, SIMD came
before prefetch; measured one change at a time, the order reverses (see 02).
Keep each milestone's source, command, model, output hash, and measured runs
together. Preserve regressions as part of the explanation.

The naive engine retains the KV cache: earlier tokens are not recomputed on each
step. Its simplification is FP32 scalar arithmetic on one thread, rather than
removing essential sequence state.

## Measured: 01 on the tablet (2026-09-12)

The [raw runs](../runs/naive-fp32-20260912b/) include timings, generated text,
warmups, environment details, and hashes copied from the tablet.

This branch and the original scalar executable ran on one boot, one warmup each,
then interleaved as this branch, original, original, this branch:

| Executable | Decode tokens/s | Mean | Mean TTFT | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| This branch (`c34027a7`) | 1.079, 1.083 | 1.081 | 23.62 s | 539.5 MiB |
| Original scalar, from `898d904` (`374f424e`) | 1.083, 1.085 | 1.084 | 23.68 s | 539.4 MiB |

The 0.3% gap is smaller than the spread between runs: the refactor costs nothing
measurable. All four outputs are byte-identical (`860192f8`), 64 tokens each,
stopping on the limit.

Workload: reMarkable 2, `ondemand` governor, SmolLM2-135M-Instruct FP32
(`82335f56`), 26 prompt tokens, no system prompt, greedy, seed 1, context 512.
The executable is `make tablet`.

```sh
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
./unremarkable models/smollm2-135m.bin -z models/smollm2-135m.tok \
  -c 512 -y '' -t 0 -s 1 -i "$prompt" -n 64
```

Decode speed excludes prefill and the first generated token. Each token reads all
538 MB of FP32 weights, so 1.08 tokens/s is about 0.58 GB/s of weight traffic,
roughly 40% of the 1.40 GB/s one core can stream (measured on `optimized`). The
later stages start from that gap. The optimized measurements remain on `optimized`.

## Measured: 02 on the tablet (2026-09-12)

The [raw runs](../runs/fp32-neon-20260912b/) hold every run below. Four builds of
one experimental `matvec`, differing only in compile-time switches, ran on the
boot that measured 01: one warmup each, then scalar, NEON, prefetch,
NEON plus prefetch, and back in reverse.

| `matvec` | Decode tokens/s | Mean | vs 01 | Mean TTFT | Output |
| --- | --- | ---: | ---: | ---: | --- |
| 01: scalar (`c34027a7`) | 1.080, 1.080 | 1.080 | 1.00x | 23.82 s | `860192f8` |
| **02: scalar plus prefetch** (`cf1de842`) | 1.436 | 1.436 | **1.33x** | 17.61 s | `860192f8` |
| NEON, four accumulators (`89800510`) | 1.027, 1.028 | 1.028 | 0.95x | 24.87 s | `860192f8` |
| NEON plus prefetch (`85a8d5ee`) | 1.749, 1.751 | 1.750 | 1.62x | 14.42 s | `860192f8` |

This branch keeps only the prefetch loop, which compiles to the measured
`cf1de842`. The hint asks for each 64-byte weight line 256 bytes before the loop
reaches it, so the next miss is already in flight while the current line is
summed. On `optimized`, distances from 192 to 1024 bytes performed the same. It
changes no arithmetic: host logits are byte-identical to 01, and so is the tablet
output. 1.44 tokens/s is about 0.77 GB/s of weight traffic, 55% of the single-core
stream rate.

NEON alone is slower, because while every row still waits on misses, wider
arithmetic has nothing to overlap with. Once prefetch hides the waits, NEON adds
1.22x on top; that is the next rung. It also reorders the additions: over 18
prompt positions, host logits move by at most 2e-4 against a mean range of 33,
with the top-1 token unchanged at every position and the same 64-token text.

The prefetch row is one run. The first prefetch build had a loop bug that skipped
every other 16-float chunk; its runs 3 and 6 (output `331a5336`, gibberish) remain
in the raw runs but are excluded. The fixed build ran once, about 25 minutes later
on the same boot; both runs of every other variant agreed within 0.2%.

The UI (`xochitl`) was stopped for these runs. An attempt with it running was
killed by the OOM killer during the NEON run: with FP32 SmolLM2 resident, only a
few MiB remained outside the CMA region the kernel reserves for movable pages, and
the failed allocation could not use CMA. Stopping the UI frees about 150 MiB and
does not change the comparison: scalar decodes at 1.080 with it stopped and 1.081
with it running.

```sh
systemctl stop xochitl   # restart with: systemctl start xochitl
```

## Measured: 03 on the tablet (2026-09-12)

No new runs were needed. This branch's default build is byte-identical to the
NEON-plus-prefetch executable in [02's raw runs](../runs/fp32-neon-20260912b/)
(`85a8d5ee`), and building with `-DUNREMARKABLE_SCALAR` reproduces 02's
executable (`cf1de842`).

| `matvec` | Decode tokens/s | Mean | vs 02 | vs 01 | Mean TTFT | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 02: scalar plus prefetch (`cf1de842`) | 1.436 | 1.436 | 1.00x | 1.33x | 17.61 s | `860192f8` |
| **03: NEON plus prefetch** (`85a8d5ee`) | 1.749, 1.751 | 1.750 | **1.22x** | 1.62x | 14.42 s | `860192f8` |

With prefetch hiding most memory waits, the scalar loop's limit is its own
dependency chain: all 16 additions per cache line go into one `sum`, and each
must wait for the one before it. NEON multiplies four adjacent values per
instruction into four accumulators, 16 independent partial sums in all, so the
additions overlap. 1.75 tokens/s is about 0.94 GB/s of weight traffic, 67% of the
single-core stream rate.

The cost is exactness. The partial sums add in a different order, so results are
no longer bit-identical to 01 and 02: over 18 prompt positions, host logits move
by at most 2e-4 against a mean range of 33. The top-1 token is unchanged at every
position, and the 64-token tablet text is the same. Without prefetch, the same
NEON loop measured 0.95x of scalar (see 02).

At the 1.40 GB/s one core can stream, 538 MB of FP32 weights per token cap this
design near 2.6 tokens/s. The next rung reads fewer bytes instead: Q8 stores the
same weights in 151 MB.

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

Decode reads 3.6x fewer bytes and runs 2.17x faster. It pulls only about 0.58 GB/s
of weights, 41% of the single-core stream rate, so the loop is no longer mostly
waiting on memory; its time goes to the arithmetic in each group. The next rungs
work on that with a second core and fewer instructions per group.

Peak memory drops from 539.6 to 170.8 MiB, so SmolLM2 now runs beside the tablet
UI without risk of the OOM killer.

Quantization changes results. Over 18 prompt positions, host logits move by 0.19
on average from FP32 (worst 1.20, against a mean range of 33). The top-1 token
agrees at 17 positions, and the top-5 sets share 86 of 90 tokens. The 64-token
text differs from FP32's (`4ea6b63a`). These logits are byte-identical to the
Q8 engine on `optimized`.

```sh
make chat-model-q8   # tools/export_hf.py -q q8_0; needs numpy
./unremarkable models/smollm2-135m-q8.bin -z models/smollm2-135m-q8.tok \
  -c 512 -y '' -t 0 -s 1 -i "$prompt" -n 64
```

UNRK version 2 adds the matrix format after the RoPE base; version 1 files still
load as FP32.
