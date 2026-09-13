# Batched prefill profile, 2026-09-13

Source: `8f8c26ef1fee6e8fdeb9c645e9305bf7fa0b4fa4` on `tools/profiler`, based on
rung 07 (`29c5b9c`). Built with `make tablet-profile tablet`.

SmolLM2-135M Q8, `-b 8 -j 2 -c 512`, 26-token lighthouse prompt, no system prompt,
greedy seed 1, 64 generated tokens (63 timed decode steps). UI active, ondemand
governor, no CPU affinity. `run.sh` contains the exact commands: one 16-token
warmup each, then normal, profile, profile, normal.

The normal files contain CLI timing JSON. Profile files add a second line with
phase counters. Text outputs, warmups, device state, and binary/model hashes
are retained. `summary.json` uses arithmetic means and sample standard deviations
across two runs. Prefill milliseconds per token are the whole prompt's time
divided by 26, including the final classifier once; decode divides by 63.

Quantization and caller waiting are already included in projection time.
Worker elapsed time overlaps caller time. In prefill, the first residual add
is grouped with the feed-forward norm. Compare timings within this session;
older profiles are historical references, not matched controls.

Regenerate the summary with:

```sh
python3 runs/profile-prefill-20260913/summarize.py
```
