# Development

## Code layout

| File | Responsibility |
| --- | --- |
| [engine.cpp](../src/engine.cpp) | Model loading, inference buffers, forward pass, sampler |
| [kernels.cpp](../src/kernels.cpp) | Numerical operations: matvec, normalization, attention, and activations |
| [tokenizer.cpp](../src/tokenizer.cpp) | Tokenizer loading, BPE, byte fallback |
| [main.cpp](../src/main.cpp) | CLI, chat template, generation loop, metrics |
| [export_hf.py](../tools/export_hf.py) | Hugging Face checkpoint and tokenizer conversion |

## Forward pass

`Engine::forward(token, position)` returns a borrowed logits span. Positions
start at zero and must be consecutive. `reset()` starts a new sequence while
retaining loaded weights and allocated buffers.

The engine selects each layer's weights and cache slots, then calls kernels in
transformer order. Kernels write into preallocated buffers; their shapes and
in-place behavior are documented in [kernels.h](../src/kernels.h).
Owning buffers use `std::vector`; the CLI catches parsing and configuration exceptions.

## Model formats

The loader accepts legacy llama2.c FP32 checkpoints and the tagged `UNRK` format
produced by the exporter, rather than GGUF. Tagged checkpoints include the RoPE
base and omit the legacy format's unused RoPE tables.
[Published measurements](performance.md) specify the model and precision tested.

`UNRK` version 2 adds a quantization field. Version 1 stays FP32, so existing
checkpoints and the pinned fixtures keep working. Under `q8_0` the matrices hold
thirty-two int8 values per FP32 scale, the scale stored ahead of its values so a
row reads as one sequential stream; norm vectors stay FP32 in either format. A
row therefore occupies `ceil(columns / 32) * 36` bytes rather than `columns * 4`,
which is why the weight layout is walked in bytes rather than floats.

Products accumulate in int32 within a group and in FP32 across groups. Integer
accumulation is exact, so the quantized kernel produces bit-identical results on
every target, and unlike the FP32 path it depends on neither `-ffp-contract=off`
nor NEON's flush-to-zero behavior.

The exporter reorders query and key rows because Hugging Face rotates halves of
each attention head while this engine rotates adjacent pairs.

The tokenizer supports the legacy SentencePiece export and byte-level BPE.
Byte-level pretokenization approximates non-ASCII character categories, so it
can differ from the reference tokenizer for text such as emoji.

## Checks

```sh
make test-models     # ~1 MB TinyStories 260K fixture
make check          # byte-level tokenizer checks also need make chat-model
make check-sanitize # AddressSanitizer + UndefinedBehaviorSanitizer
make check-format   # requires clang-format
```

Checks cover reference logits and greedy output, attention, cache behavior,
sampling, tokenizer behavior, malformed files, and matvec edge cases. Desktop
checks do not establish tablet performance; keep those measurements separate.

Define `UNREMARKABLE_SCALAR` at compile time to disable NEON. FP32 summation
order differs between paths, so small rounding differences are expected.

The math and legacy tokenizer derive from a pinned llama2.c revision under MIT;
see [THIRD_PARTY.md](../THIRD_PARTY.md).
