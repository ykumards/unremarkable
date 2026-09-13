# Attention across both cores

Baseline: `1e6abf4`. `exact.patch` adds NEON weighted sums and hoists the square
root. `threaded.patch` adds those changes plus disjoint head ranges, the engine
wrapper, and tests. Apply either patch to baseline and `make tablet`.
Local binary hashes match the tablet's recorded hashes.

Three interleaved runs per build after one warmup each; exact commands in
`run.sh`. SmolLM2-135M Q8, batch 8, two threads, context 512, 26 prompt tokens,
64 generated tokens (63 timed decode steps). Same boot, UI active, `ondemand`,
no CPU affinity. All nine output texts match (`4ea6b63a`).

`summary.json` reports means and sample standard deviations. Threaded attention
reaches 6.949 ± 0.045 tok/s versus baseline's 6.299 ± 0.032: +10.3%.
The single-core variant has one slow run, 6.237 tok/s; it is included.

The retained variant matches eight baseline host logit positions exactly with
two threads (`logit-comparison.json`). Host tests, ASan/UBSan, formatting, and
ARMv7 attention tests pass. Worker tests exercise both projection and head
thresholds; attention tests check scalar arithmetic, grouped and split heads,
vector tails, and causal boundaries. Logs are included.
