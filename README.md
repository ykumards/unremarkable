<h1 align="center">unremarkable</h1>

<p align="center">
  <img src="assets/mascot.webp" alt="Unremarkable mascot: an unimpressed orange cat" width="400">
</p>

A tiny LLM running entirely on a reMarkable 2.

An inference engine written in C++23, built to learn how to make language models
run faster on limited hardware.
Loads weights into RAM and implements the forward pass, tokenizer, and sampling.
Runs TinyStories 15M and SmolLM2-135M-Instruct, with optional tablet UI patches.

## Try it locally

Requires Linux or macOS, a C++23 compiler, Make, Python 3.11+, and curl.

```sh
make
make models  # downloads TinyStories 15M (~62 MB with tokenizer)
./build/unremarkable models/stories15M.bin -z models/tokenizer.bin -i "Once upon a time"
```

## Docs

- [Usage](docs/usage.md): SmolLM2, sampling, context, and output.
- [Tablet setup](docs/tablet.md): cross-compilation, deployment, and UI patches.
- [Performance](docs/performance.md): measurements and the [raw numbers](docs/benchmarks.csv).
- [Development](docs/development.md): code layout, forward pass, and checks.

Adapted from [llama2.c](THIRD_PARTY.md).
