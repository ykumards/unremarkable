# Tablet performance

[benchmarks.csv](benchmarks.csv) stores one row per completed measured run for
plotting progress. Append new variants and runs; retain the individual numbers,
output hashes, and executable hashes. Exclude warmups and failed attempts.
Raw output and JSON metrics remain under ignored `results/`.

## Six-op Q8 dot product (2026-09-12)

The retained kernel pairs int8 products in int16 before widening, reducing the
inner arithmetic from eight instructions to six per 32 values. A follow-up uses
`vpaddl` for the first reduction instead of adding into a zeroed accumulator.
Loads, quantization scales, and the FP32 accumulation order are unchanged.

Final comparison against the original kernel, on the same boot, in ABBA order:

| Kernel, two threads | Runs | Decode tok/s (mean ± sample std) | Mean TTFT |
| --- | ---: | ---: | ---: |
| Original eight-op | 2 | 5.321 ± 0.004 | 3.995 s |
| Six-op with direct reduction | 2 | **5.835 ± 0.004** | **3.627 s** |

Decode improved **9.7%**, and TTFT fell **9.2%** in this comparison. These are
short runs, not a stability guarantee. Compare within this test, rather than
against the older 5.77 tok/s result from a different device state and workload.

The first experiment averaged 5.403 ± 0.127 tok/s for eight-op versus
5.851 ± 0.299 for six-op (three runs each, 8.3% improvement). Its variability
motivated the follow-up: six-op averaged 5.629 ± 0.006 versus 5.844 ± 0.012
with direct reduction (two runs each, 3.8%). All 14 measured runs are retained
in [benchmarks.csv](benchmarks.csv); do not pool these separate comparisons.

Each run used SmolLM2-135M Q8, context 512, two threads, the lighthouse prompt
below, no system prompt, greedy sampling, and 128 new tokens (127 timed decode
steps). Each comparison included a separate 16-token warmup for each binary.
The UI stayed running and the governor remained `ondemand`. All 14 outputs
were byte-identical, SHA-256
`22c7678796cc03773b23487eb80b72fa4e9b0230722e76fd8f7ed49c20e2d232`.

`make check`, `make check-sanitize`, and `make check-format` passed for the final
source, as did the ARMv7 kernel test. The new test covers every activation in
[-127, 127] against every int8 weight, including -128. Pairing cannot overflow:
`2 × 128 × 127 = 32512 < 32767`. The range requirement is documented in
`kernels.h`; projection inputs already satisfy it through `quantize_q8`.

Raw outputs, metrics, commands, environment, and binary/model hashes are in
`results/six-op-20260912/`, `results/six-direct-20260912/`, and
`results/six-final-20260912/`. Experiment binaries are separate from the tablet's
installed engine. Build flags match the ARMv7 flags below; only the kernel changes.

## NEON matvec, corrected (2026-09-10)

An earlier entry here reported the NEON kernel as about 24% faster than scalar.
That comparison was confounded: the two builds ran on different boots, and this
device's throughput varies more between boots than the kernel change was worth.
Measured back to back on one boot, in both orderings, the four-partial-sum kernel
is slightly *slower* than scalar, and a prefetch hint is what actually pays.

| matvec, isolated, 576x8192 | GB/s | vs scalar |
| --- | ---: | ---: |
| scalar | 0.620 | 1.00x |
| NEON, four partial sums | 0.585 | 0.94x |
| NEON plus `__builtin_prefetch` | 1.022 | 1.65x |

Six runs per variant, interleaved in both directions, spread under 1%. A
streaming read of the same buffer reaches 1.40 GB/s, so the scalar kernel was
already at 44% of what the core can pull, and the loop waits on cache misses
rather than on arithmetic. Widening it alone therefore buys nothing; the extra
instructions have nothing to do while a miss is outstanding. Interleaving several
rows is worse still, at 0.35 GB/s, because this core sustains too few outstanding
misses to serve more than one stream. Prefetch distance is insensitive between
192 and 1024 bytes.

The prefetch hint is not applied to the FP32 kernel in this branch; the quantized
kernel carries it.

## Q8_0 quantization (2026-09-10)

| Engine | Runs | Decode tokens/s | Mean TTFT | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| FP32, scalar | 1 | 1.082 | 23.56 s | 538.41 MiB |
| FP32, NEON four partial sums | 2 | 1.028 | 25.06 s | 538.59 MiB |
| FP32, NEON plus prefetch | 2 | 1.742 | 14.55 s | 538.68 MiB |
| **Q8_0** | 3 | **3.826 ± 0.017** | **6.57 s** | **170.87 MiB** |

All runs produced byte-identical output within their variant. Q8_0 is 3.54x the
scalar baseline on decode and 3.6x smaller resident.

Memory is the more useful number. At 538 MiB the engine was killed by the OOM
reaper on a device with 1003 MiB usable and no swap; that, not an unexplained
reset, is what ended the interrupted baseline below. At 171 MiB the failure mode
is gone.

Quantization also made the tablet usable while generating. Q8_0 moves about
580 MB/s of weight traffic against the FP32 prefetch kernel's 940 MB/s, so it is
2.2x faster while taking 38% less of the memory bandwidth the UI shares. Under
FP32 with prefetch an ssh connection could not complete its banner exchange;
under Q8_0 a round trip takes 490-670 ms against a 348-415 ms idle baseline.

Accuracy against the FP32 checkpoint, over 11 positions of 49152 logits: mean
absolute error 0.29 against a mean logit range of 28.9, worst 1.73, top-1
agreement 10 of 11, top-5 overlap 53 of 55.

## Memory roofline (2026-09-10)

Sequential reads over a 144 MiB working set, eight repeats, using the same NEON
loop with a prefetch hint used for the kernel measurements:

| Threads | GB/s | vs one core |
| --- | ---: | ---: |
| 1 | 1.423 | 1.00x |
| 2 | 2.708 | 1.90x |
| 3 | 2.643 | 1.86x |
| 4 | 2.630 | 1.85x |

Two cores deliver 1.90x the single-core throughput. Three and four threads add
no cores, so their slightly lower throughput does not establish that the DRAM
controller is saturated. Use 2.708 GB/s as the measured two-core streaming rate
for this loop and device state, not an absolute hardware limit.

The earlier single-core 1.40 GB/s figure understates the available aggregate
bandwidth. NXP's LPDDR3-1066 specification gives a theoretical 4.264 GB/s at
32 bits (`1066 million transfers/s × 4 bytes/transfer`), not measured tablet
throughput or confirmation of its configured bus width and clock.
See the [NXP datasheet](https://www.nxp.com/docs/en/data-sheet/IMX7DCEC.pdf).

Dividing by approximately 151 MB of Q8 weights read per token, including scales,
gives a memory-only estimate: **2.708 / 0.151 = 17.9 tokens/s** on two cores,
or 9.4 on one. These use decimal GB and assume each token streams the weights
once, with all bandwidth available to them. Compute, activations, KV-cache
traffic, and synchronization are omitted.

The recorded 5.77 tokens/s is about 32% of that estimate. Its implied weight
traffic is 0.87 GB/s; this is calculated from token throughput, not measured
DRAM traffic. The gap suggests investigating arithmetic, memory access, and
coordination costs, but does not prove an achievable 3x speedup. Likewise,
inference scaling 1.51x versus streaming's 1.90x does not isolate synchronization
cost: the workloads differ, and streaming bandwidth alone cannot distinguish
compute limits from memory stalls or scheduling overhead.

## Fixed workload

- reMarkable 2, Cortex-A7, default `ondemand` governor. Runs are single-threaded
  unless the variant says otherwise; `benchmarks.csv` records the thread count.
- SmolLM2-135M-Instruct; 513.134 MiB of FP32 weights or 144.415 MiB of Q8_0,
  plus 22.5 MiB of KV cache. The cache stays FP32 in both.
- Prompt: `Tell me a short story about a lighthouse keeper who discovers a message in a bottle.`
- No system prompt; greedy decoding, seed 1, context 512, maximum 256 new tokens.
- 26 prompt tokens; the 2026-09-09 rows generate 256 tokens and the 2026-09-10
  rows 64, each stopping on the token limit.
- TTFT excludes loading. Decode speed excludes prefill, the first generated token,
  printing, and terminal tokens. See the metric definitions below.
- Codex Linux 5.8.203, firmware 3.28.0.172, kernel 5.4.70-v1.6.3-rm11x.
- GCC 13.3.0, `-O2 -march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4
  -mfloat-abi=hard -ffp-contract=off -static-libstdc++ -static-libgcc`.

`make tablet` builds the engine. Define `UNREMARKABLE_SCALAR` at compile time
for the scalar path. The comparison executable lives in the tablet's
`/home/root/unremarkable/benchmarks/`; the installed UI executable was not replaced.

The only inference change is four-lane accumulation in `matvec`; weights and
buffers are unchanged. Summation order differs, so FP32 outputs need not always
be identical. On the Mac, 196,608 SmolLM2 logits differed by at most 0.000069.
The independent forward-reference tests and sanitizers passed. The kernel test
also passed on ARMv7, covering small widths, column tails, and unaligned buffers.

## Metric definitions

- `load_ms`: checkpoint validation, full read into RAM, and inference-buffer setup.
- `prefill_ms`: forward passes for all prompt tokens, including BOS.
- `ttft_ms`: tokenization through first generated token, excluding model/tokenizer
  loading. Zero if generation stops before producing a token.
- `decode_tok_s`: forward plus sampling for generated tokens **after the first**.
  Excludes terminal BOS/EOS, prompt processing, and printing. Zero if none.
- `generation_ms`: tokenization, prefill, generation, and text output together.
- `weights_mib`: resident checkpoint payload, including unused legacy RoPE tables.
- `kv_cache_mib`: allocated FP32 key/value cache capacity.
- `peak_rss_mib`: process peak resident memory from `getrusage`.

For comparisons, keep the checkpoint, prompt, context, seed, compiler flags, and
hardware fixed. Record token counts and stop reason alongside speed; short runs
that stop early are different workloads. Use fresh processes and keep warmup
runs separate from measured runs.
