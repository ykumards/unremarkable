# Attention with a longer generation

Same binaries as [the short comparison](../attention-heads-20260913/): baseline
`1e6abf4` versus NEON weighted sums, a hoisted square root, and split heads.
Local binary hashes match the tablet's recorded hashes.

Same 26-token prompt and settings, with the generation limit raised to 192.
All four runs reach that limit (191 timed decode steps) and produce identical
text. One warmup each, then baseline, candidate, candidate, baseline.
The UI stayed active on the same boot; see `run.sh` and environment files.

Baseline: **5.491 ± 0.022 tok/s**. Candidate: **6.514 ± 0.008 tok/s**, **18.6%**
faster. Mean and sample standard deviation over two runs each; per-run data and
TTFT in `summary.json`. Attention reads more cached positions as generation
continues. These results cover histories up to 217 tokens, not a full context.
