# Profiling

```sh
make build/unremarkable-profile  # Mac/Linux
make tablet-profile             # ARMv7
```

Run with the usual model and CLI arguments. A second stderr JSON line reports
per-step milliseconds for prefill and decode. Use normal builds for speed tests.

## reMarkable 2: decode time

SmolLM2-135M Q8, rung 06, two threads. One run on 2026-09-13: 26 prompt tokens,
context 512, 63 decode steps, UI active. [Raw timings](../runs/profile-rung06-20260913/).

| Operation | ms/token |
| --- | ---: |
| Projections | 132.1 |
| Attention | 20.3 |
| SwiGLU | 4.9 |
| RoPE | 2.9 |
| Norms, residual adds, embedding | 1.5 |
| **Total** | **161.7** |

Projection time includes 7.3 ms of input quantization and 16.9 ms waiting for the
worker. Worker time overlaps the caller; don't add it to the total.
