# Grouped Q/K/V and gate/up projections

Baseline: `f24f9d6` (rung 09). Apply `candidate.patch` and `make tablet` for the
candidate. Quantize a shared input once, then split the combined matrix rows
into one worker job. Matrix storage, output strides, and kernel arithmetic stay
unchanged. Single projections and batches use the same helper.

Three measured runs per build after a 16-token warmup each, ordered baseline,
candidate, candidate, baseline, baseline, candidate. SmolLM2-135M Q8, batch 8,
two threads, context 512, 26-token lighthouse prompt, greedy seed 1, 64 generated
tokens (63 timed decode steps). UI active, same boot, `ondemand`, no CPU affinity.
`run.sh` records commands; binary hashes match local builds.

Decode: **6.961 ± 0.035 → 7.279 ± 0.012 tok/s (+4.6%)**.
TTFT: **2.456 ± 0.015 → 2.388 ± 0.003 s (−2.8%)**.
Mean ± sample standard deviation over three runs. All six texts match
(`4ea6b63a`); peak RSS stays about 171 MiB. Per-run numbers are in `summary.json`.

Host checks and ASan/UBSan pass. Prefill tests cover worker splits inside Q,
at its end, and inside K, plus shared/separate classifiers, odd hidden widths,
Q8 tails, prefixes, partial batches, reset, and subsequent decode. Eight host
logit positions match baseline exactly (`logit-comparison.json`).

The tablet also ran `tests/prefill.cpp` on two fixtures made with
`tests/test_engine.py::reference_model`: `shared=False`, `tagged=True`, `dim=64`,
`hidden=73`, `context=18`; FP32 uses two KV heads and Q8 uses four. Both pass.
Their hashes and the ARMv7 test-binary hash are in `local-hashes.json`.
