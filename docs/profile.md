# Profiling

```sh
make build/unremarkable-profile  # Mac/Linux
make tablet-profile             # ARMv7
```

Run with the usual model and CLI arguments. A second stderr JSON line reports
per-step milliseconds for prefill and decode. Use normal builds for speed tests.

## reMarkable 2: batched prefill and decode

SmolLM2-135M Q8, rung 07, `-b 8 -j 2 -c 512`, UI active. Mean of two runs on
2026-09-13: 26 prompt tokens, 63 decode steps. [Raw runs](../runs/profile-prefill-20260913/)
· [Older unbatched profile](../runs/profile-rung06-20260913/).

| Operation | Prefill ms/token | Decode ms/token |
| --- | ---: | ---: |
| Layer projections | 88.3 | 114.3 |
| Classifier | 0.8 | 22.5 |
| Attention | 4.4 | 20.9 |
| Other ops | 8.6 | 9.2 |
| **Total** | **102.1** | **166.8** |

Prefill divides the whole prompt's time by 26; its classifier runs only once.
Projection times include input quantization (7.0 / 7.4 ms) and caller waiting
(3.0 / 17.0 ms), prefill / decode. Worker time overlaps the caller.
From rung 09, worker and waiting counters also include attention-head jobs.

Normal builds: **2.648 ± 0.015 s TTFT**, **6.026 ± 0.013 decode tok/s**
(mean ± sample std, two runs). All four outputs match. Profile builds averaged
5.964 tok/s; use their counters for the breakdown, normal runs for speed.
