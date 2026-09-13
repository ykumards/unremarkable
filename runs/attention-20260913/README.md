# Attention: weighted sums and dot products

Baseline: `1e6abf4`. Apply `exact.patch` or `simd.patch` to that commit and run
`make tablet`. Local binary hashes were checked against `hashes.txt`.

`exact` hoists the square root and uses NEON for four output components at a
time, keeping additions in token order. `simd` also uses the existing FP32 dot
kernel for attention scores, changing their addition order.

Three interleaved runs per variant after a warmup each; see `run.sh` for commands
and `summary.json` for means and sample standard deviations. All runs generated
64 tokens (63 timed decode steps), with 26 prompt tokens, batch 8, two threads,
context 512, UI active, and the `ondemand` governor.

The exact variant improves decode by 5.2%: 6.698 versus 6.365 tok/s. Its tablet
text and eight checked host logit positions match baseline exactly. The SIMD-dot
variant reaches 6.722 tok/s, changes tablet text, and changes host logits by up
to 1.30. It is parked; these checks do not establish its quality.

Attention tests cover a scalar reference, grouped heads, vector tails, and
causal boundaries. Host checks, ASan/UBSan, formatting, and ARMv7 attention tests
pass for the SIMD variant; host logit comparisons are in `logit-comparison.json`.
