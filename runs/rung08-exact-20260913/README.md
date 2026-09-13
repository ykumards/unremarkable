# Rung 08 without approximate SwiGLU

Source: `a4637b9`. Baseline: profiler branch `8f8c26e`, with rung 07 inference.
Both use the original scalar SwiGLU. Only RoPE reuse and faster quantization
change inference in the candidate.

`run.sh` records the exact commands: SmolLM2-135M Q8, batch 8, two threads,
context 512, 26-token lighthouse prompt, greedy seed 1, 64 generated tokens.
One 16-token warmup each, then baseline, candidate, candidate, baseline.
UI active, ondemand governor, same boot. No CPU affinity was set.

`summary.json` reports means and sample standard deviations for two measured
runs per build. All four generated texts match the baseline SHA-256. Binary,
model, tokenizer hashes and device state are included.

Normal host checks pass (20 tests), as do ASan/UBSan (19 engine tests; the
profile-build test is skipped there), formatting, and the tablet kernel tests.
The RoPE test compares positions 0..600 at both frequency bases exactly. Its
sanitizer run initially differed by one float rounding step because macOS used
separate sine/cosine calls in the implementation and a combined call in the
reference. Computing both values before storing them fixes the mismatch; the
ARMv7 executable remains byte-identical before and after this adjustment.

Earlier `rung08-20260913`, `rung08-profile-pair-20260913`, and
`accuracy-rung08-20260913` runs include approximate SwiGLU and are superseded.
