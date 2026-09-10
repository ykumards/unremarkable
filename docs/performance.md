# Tablet performance

[benchmarks.csv](benchmarks.csv) stores one row per completed measured run for
plotting progress. Append new variants and runs; retain the individual numbers,
output hashes, and executable hashes. Exclude warmups and failed attempts.
Raw output and JSON metrics remain under ignored `results/`.

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

Sequential reads over a 144 MiB working set, the same NEON loop with a prefetch
hint used for the kernel measurements:

| Threads | GB/s | vs one core |
| --- | ---: | ---: |
| 1 | 1.423 | 1.00x |
| 2 | 2.708 | 1.90x |
| 3 | 2.643 | 1.86x |
| 4 | 2.630 | 1.85x |

One core does not saturate the memory controller. Two scale almost perfectly and
then stop: three and four threads on two cores are slightly slower than two,
which is what a controller limit rather than a core limit looks like. An earlier
version of this file treated the single-core 1.40 GB/s as a device ceiling. It
is not one, and neither is NXP's LPDDR3-1066 figure of 4.26 GB/s on a 32-bit
bus, which is the bus rather than anything sustainable.

Dividing by the 151 MB of weights a token reads gives the useful roofline:
**17.9 tokens per second** on two cores, 9.4 on one. Decode currently reaches
5.88, which is 33% of the two-core roofline, so this engine is bound by
arithmetic with roughly threefold headroom before memory becomes the limit. That
also explains why threading returned 1.51x where a pure streaming read returns
1.90x: the shortfall is the per-projection handoff and the sequential remainder,
not bandwidth.

## Fixed workload

- reMarkable 2, Cortex-A7, single-threaded, default `ondemand` governor.
- SmolLM2-135M-Instruct; 513.134 MiB of FP32 weights or 144.415 MiB of Q8_0,
  plus 22.5 MiB of KV cache. The cache stays FP32 in both.
- Prompt: `Tell me a short story about a lighthouse keeper who discovers a message in a bottle.`
- No system prompt; greedy decoding, seed 1, context 512, maximum 256 new tokens.
- 26 prompt tokens, 256 generated tokens, 255 timed decode steps; stop: token limit.
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
