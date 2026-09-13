# Q8 dot-product experiments

Baseline: `1e6abf4` (rung 08 inference). Apply `inline.patch` or `vector.patch`
to that commit, then `make tablet`. Both experiments are parked; attention
starts from the baseline.
Binary hashes match the local builds; model and tokenizer hashes are included.

`run.sh` records the commands. One 16-token warmup per build, then baseline,
inline, vector, vector, inline, baseline: 64 generated tokens, 63 timed decode
steps. Same boot, UI active, two cores, `ondemand`, no CPU affinity.
`summary.json` gives mean and sample standard deviation over two runs per build.
All six generated texts match (`4ea6b63a`).

Inlining gains 3.4% decode speed with unchanged arithmetic. Vector accumulation
gains 2.6% decode speed and cuts TTFT by 10.4% against baseline; it is archived
because it changes results without improving decode over the simpler candidate.
Two runs per build are a small timing sample.

`logit-comparison.json` records the host probe inputs and results against baseline
on the same Q8 checkpoint. Inlining is exact. Vector accumulation changes logits
by up to 1.09 (mean absolute difference 0.166); top tokens match at all eight
positions. This is not a quality evaluation.

The vector variant fails the kernel's exact-equality test, while the 20 engine
tests pass within their existing checks. Tests were not relaxed. The inline
variant passes host NEON/scalar kernel tests, 20 engine tests, ASan/UBSan
(19 engine tests; profile test skipped), formatting, and tablet kernel tests.
