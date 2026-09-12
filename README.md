<h1 align="center">unremarkable</h1>

<p align="center">
  <img src="assets/mascot.webp" alt="Unremarkable mascot: an unimpressed orange cat" width="300">
</p>

A tiny LLM running entirely on a reMarkable 2.

An inference engine written in C++23, built to learn how to make language models
run faster on limited hardware. It starts as plain FP32 loops and gets faster one
measured optimization at a time; each step is tagged (`rung-01`, `rung-02`, …)
and explained in [the optimization ladder](docs/optimizations.md). Runs
TinyStories 15M and SmolLM2-135M-Instruct, with optional tablet UI patches.

## Ceiling

![1.75 tok/s toward a 17.9 tok/s target](assets/tps-progress.svg?v=1.75)

SmolLM2-135M, Q8, two cores: **2.708 GB/s ÷ 0.151 GB/token ≈ 17.9 tok/s**
memory-only ceiling. [Calculation](https://github.com/ykumards/unremarkable/blob/optimized/docs/performance.md#memory-roofline-2026-09-10)
· [FP32 baseline runs](docs/optimizations.md#measured-01-on-the-tablet-2026-09-12)
· [Prefetch runs](docs/optimizations.md#measured-02-on-the-tablet-2026-09-12)
· [SIMD runs](docs/optimizations.md#measured-03-on-the-tablet-2026-09-12).

## Start reading

1. [One token, end to end](docs/inference.md): data flow, shapes, and memory lifetimes.
2. [The forward pass](src/engine.cpp): every operation in execution order.
3. [The kernels](src/kernels.cpp): the actual arithmetic loops.
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
