# Tablet performance

[benchmarks.csv](benchmarks.csv) stores one row per completed measured run for
plotting progress. Append new variants and runs; retain the individual numbers,
output hashes, and executable hashes. Exclude warmups and failed attempts.
Raw output and JSON metrics remain under ignored `results/`.

## First optimization: NEON matvec (2026-09-09)

| Implementation | Runs | Decode tokens/s (mean ± sample std) | Mean TTFT | Mean peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 2 | 0.991 ± 0.021 | 24.92 s | 538.65 MiB |
| NEON, four partial sums | 1 | 1.228 | 19.73 s | 539.58 MiB |

The NEON run was about 24% faster than the earlier scalar mean. All runs
produced the same 256-token output, byte-for-byte. This is an initial comparison:
NEON has only one measured run, after a reboot with the stock UI; the scalar runs
preceded the reboot. It is not a controlled repeatability or stability result.
The interrupted scalar attempt is excluded; its reset cause remains unknown.

## Fixed workload

- reMarkable 2, Cortex-A7, single-threaded, default `ondemand` governor.
- SmolLM2-135M-Instruct, FP32; 513.134 MiB weights and 22.5 MiB KV cache.
- Prompt: `Tell me a short story about a lighthouse keeper who discovers a message in a bottle.`
- No system prompt; greedy decoding, seed 1, context 512, maximum 256 new tokens.
- 26 prompt tokens, 256 generated tokens, 255 timed decode steps; stop: token limit.
- TTFT excludes loading. Decode speed excludes prefill, the first generated token,
  printing, and terminal tokens. See the metric definitions below.
- Codex Linux 5.8.203, firmware 3.28.0.172, kernel 5.4.70-v1.6.3-rm11x.
- GCC 13.3.0, `-O2 -march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4
  -mfloat-abi=hard -ffp-contract=off -static-libstdc++ -static-libgcc`.

`make tablet` builds the NEON engine. Define `UNREMARKABLE_SCALAR` at compile time
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
