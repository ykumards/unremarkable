# Development

| Module | Responsibility |
| --- | --- |
| [model.h](../src/model.h), [model.cpp](../src/model.cpp) | Checkpoint loading, immutable weight views, tensor dimensions |
| [engine.h](../src/engine.h), [engine.cpp](../src/engine.cpp) | Scratch and KV ownership, reset, one-token forward pass |
| [kernels.h](../src/kernels.h), [kernels.cpp](../src/kernels.cpp) | FP32 numerical loops and buffer contracts |
| [sampler.h](../src/sampler.h), [sampler.cpp](../src/sampler.cpp) | Greedy or temperature/top-p next-token selection |
| [tokenizer.cpp](../src/tokenizer.cpp) | Legacy and byte-level BPE tokenization |
| [main.cpp](../src/main.cpp) | Arguments, prompt formatting, prefill/decode loop, timing |

`make check` runs the independent forward oracle, published greedy-output
fixture, tokenizer checks, context/reset checks, sampling checks, and malformed
input checks. `make check-sanitize` adds address/undefined-behavior sanitizers.
`make check-format` enforces the repository's 100-column C++ style.

For TinyStories fixtures run `make test-models` first. Tokenizer parity tests also
need the SmolLM2 tokenizer downloaded by `make chat-model`.

## Tablet cross-build

```sh
make tablet-image  # requires Docker; creates the ARMv7 toolchain image
make tablet        # produces build/unremarkable-armv7
```

The executable produced by plain `make` is for the host. The ARMv7 build targets
Cortex-A7 and links the C++ runtime statically; the scalar source uses no NEON
intrinsics. Copy the executable and matching model/tokenizer to a directory under
`/home/root` on the tablet. Start with the small TinyStories model.

Output text goes to stdout and timing JSON to stderr. The full model is loaded
into RAM once per process. There is no GPU transfer, mmap-based weight access,
quantization, worker pool, or batched prefill in this milestone.
