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

![7.28 tok/s toward a 17.9 tok/s target](assets/tps-progress.svg?v=7.28)

SmolLM2-135M, Q8, two cores: **2.708 GB/s ÷ 0.151 GB/token ≈ 17.9 tok/s**
memory-only ceiling. [Calculation](docs/optimizations.md#memory-ceiling)
· [FP32 baseline runs](docs/optimizations.md#measured-01-on-the-tablet-2026-09-12)
· [Prefetch runs](docs/optimizations.md#measured-02-on-the-tablet-2026-09-12)
· [SIMD runs](docs/optimizations.md#measured-03-on-the-tablet-2026-09-12)
· [Q8 run](docs/optimizations.md#measured-04-on-the-tablet-2026-09-12)
· [Two-thread runs](docs/optimizations.md#measured-05-on-the-tablet-2026-09-13)
· [Six-op runs](docs/optimizations.md#measured-06-on-the-tablet-2026-09-13)
· [Serial-step runs](docs/optimizations.md#measured-08-on-the-tablet-2026-09-13)
· [Attention runs](docs/optimizations.md#measured-09-on-the-tablet-2026-09-13)
· [Grouping runs](docs/optimizations.md#measured-10-on-the-tablet-2026-09-13).

Batched prefill cuts the wait for the first token from **3.94 s to 2.64 s** on
our 26-token prompt. [Runs](docs/optimizations.md#measured-07-on-the-tablet-2026-09-13)
— the turtle tracks decode speed.

## Code and docs

- [Inference](docs/inference.md): follow a token through memory and the model.
- [Forward pass](src/engine.cpp) · [Batched prefill](src/prefill.cpp) · [Kernels](src/kernels.cpp)
- [Optimization ladder](docs/optimizations.md): changes, measurements, and diffs.

## Run locally

Requires a C++23 compiler, Make, Python 3.11+, and curl on Linux or macOS.

```sh
make
make models
./build/unremarkable models/stories15M.bin -z models/tokenizer.bin -i "Once upon a time"
```

For SmolLM2, run `make chat-model-q8` (needs numpy), then use
`models/smollm2-135m-q8.bin` with `-z models/smollm2-135m-q8.tok -c 512 -j 2`; it needs
about 171 MiB. `make chat-model` writes the 538 MiB FP32 checkpoint the first three
rungs were measured on; on the tablet it can exhaust memory while the UI runs.

```sh
make test-models
make check
make check-sanitize
make check-format
```

[Build and deployment](docs/development.md) · [Model files](models/README.md) ·
[Attribution](THIRD_PARTY.md)
