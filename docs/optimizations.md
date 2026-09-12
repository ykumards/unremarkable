# Optimization ladder

`milestone/01-fp32-scalar` is the readable starting point. It branches from the
original scalar engine (`898d904`) and reorganizes ownership and names while
preserving the mathematical operation order. It is a refactored baseline, not
an exact snapshot of an old timed executable.

| Stage | Main change to inspect | What it targets |
| --- | --- | --- |
| 01: naive FP32 (this branch) | Ordinary row-by-row dot products | Establish the computation and data flow |
| Explicit SIMD | Process several values per instruction | Arithmetic throughput; it can still regress |
| Prefetch | Request upcoming weight cache lines earlier | Time spent waiting for memory |
| Q8 | Store groups of integer weights plus scales | Bytes read per token and memory footprint |
| Two threads | Split projection rows across cores | Parallel work and coordination costs |
| Six-op dot product | Pair integer products before widening | Instructions inside each Q8 group |

Only the first milestone branch is created here. Later rows describe experiments
already explored on `main`; they are not links to branches that exist yet.
Keep each future milestone's source, command, model, output hash, and measured
runs together. Preserve regressions as part of the explanation.

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
roughly 40% of the 1.40 GB/s one core can stream (measured on `main`). The
later stages start from that gap. The optimized measurements remain on `main`.
