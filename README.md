<h1 align="center">unremarkable</h1>

<p align="center">
  <img src="assets/mascot.webp" alt="Unremarkable mascot: an unimpressed orange cat" width="300">
</p>

The readable starting point: FP32 weights, one CPU thread, ordinary scalar loops.
Follow one token through its calculations and memory. We improve `main` one
measured step at a time; [milestone branches](docs/optimizations.md) preserve each
checkpoint. The existing optimized engine lives on
[`optimized`](https://github.com/ykumards/unremarkable/tree/optimized).

On the reMarkable 2 it generates SmolLM2-135M at **1.08 tokens/s**, after a
23.6 s wait for the first token: the same speed as the engine it was refactored
from. [How that was measured](docs/optimizations.md#measured-01-on-the-tablet-2026-09-12).

## Ceiling

![A right-facing tortoise at 1.08 tok/s, heading toward the estimated single-core FP32 ceiling of 2.65 tok/s](assets/tps-progress.svg)

For this FP32, one-thread baseline: **1.423 GB/s** measured streaming bandwidth
divided by roughly **0.538 GB of weights per token** gives **2.65 tok/s**.
We're at **41%** of that memory-only estimate; arithmetic and other memory traffic
reduce achievable speed. Q8 on two cores has a different estimate:
[17.9 tok/s](https://github.com/ykumards/unremarkable/blob/optimized/docs/performance.md#memory-roofline-2026-09-10).

## Start reading

1. [One token, end to end](docs/inference.md): data flow, shapes, and memory lifetimes.
2. [The forward pass](src/engine.cpp): every operation in execution order.
3. [The scalar kernels](src/kernels.cpp): the actual arithmetic loops.
4. [The optimization ladder](docs/optimizations.md): how later stages change this baseline.

## Run locally

Requires a C++23 compiler, Make, Python 3.11+, and curl on Linux or macOS.

```sh
make
make models
./build/unremarkable models/stories15M.bin -z models/tokenizer.bin -i "Once upon a time"
```

For SmolLM2, run `make chat-model`, then use `models/smollm2-135m.bin` with
`-z models/smollm2-135m.tok -c 512`. This is the FP32 checkpoint; Q8 files belong
to later stages. Prefer TinyStories 15M for the first tablet run: FP32 SmolLM2
needs about 538 MiB of process memory and has previously exhausted tablet memory.

```sh
make test-models
make check
make check-sanitize
make check-format
```

[Build and deployment](docs/development.md) · [Model files](models/README.md) ·
[Attribution](THIRD_PARTY.md)
