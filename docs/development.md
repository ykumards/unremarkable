# Development

| Module | Responsibility |
| --- | --- |
| [model.h](../src/model.h), [model.cpp](../src/model.cpp) | Checkpoint loading, immutable weight views, tensor dimensions |
| [engine.h](../src/engine.h), [engine.cpp](../src/engine.cpp) | Scratch and KV ownership, reset, one-token forward pass |
| [prefill.cpp](../src/prefill.cpp) | Prompt chunks, batched projections, final-token logits |
| [kernels.h](../src/kernels.h), [kernels.cpp](../src/kernels.cpp) | FP32 and Q8 kernels and buffer contracts |
| [worker.h](../src/worker.h), [worker.cpp](../src/worker.cpp) | Split projection rows or attention heads across the caller and one worker |
| [profile.h](../src/profile.h) | Per-step timing, compiled in only with `-DUNREMARKABLE_PROFILE` |
| [sampler.h](../src/sampler.h), [sampler.cpp](../src/sampler.cpp) | Greedy or temperature/top-p next-token selection |
| [tokenizer.cpp](../src/tokenizer.cpp) | Legacy and byte-level BPE tokenization |
| [main.cpp](../src/main.cpp) | Arguments, prompt formatting, prefill/decode loop, timing |

`make check` runs the independent forward oracle, published greedy-output
fixture, tokenizer checks, context/reset checks, sampling checks, and malformed
input checks. `make check-sanitize` adds address/undefined-behavior sanitizers.
The worker tests cover row coverage, repeated jobs, shutdown, and thread limits;
the forward tests compare one-thread and two-thread logits and reset. Q8 kernel
tests cover all supported input/weight pairs, mixed lanes, scales, and tails
on both NEON and scalar paths.
Attention tests compare with scalar arithmetic across grouped heads, vector
tails, split head ranges, and causal boundaries.
Prefill tests compare exact logits and subsequent decode against sequential
forward passes: FP32/Q8, one/two threads, prefixes, partial chunks, reset,
context boundaries, and rejected inputs.
`make check-format` enforces the repository's 100-column C++ style.

For TinyStories fixtures run `make test-models` first. Tokenizer parity tests also
need the SmolLM2 tokenizer downloaded by `make chat-model`.

## Tablet cross-build

```sh
make tablet-image  # requires Docker; creates the ARMv7 toolchain image
make tablet        # produces build/unremarkable-armv7
```

The executable produced by plain `make` is for the host. The ARMv7 build targets
Cortex-A7, enables the NEON kernel, and links the C++ runtime statically. Copy the
executable and matching model/tokenizer to a directory under
`/home/root` on the tablet. Start with the small TinyStories model.

Output text goes to stdout and timing JSON to stderr. The full model is loaded
into RAM once per process. FP32 and Q8 checkpoints are supported. `-j 2` splits projection rows across
two threads; `-j 1` is the default. `-b 8` batches up to eight prompt tokens;
`-b 1` skips unused logits without batching, and `-b 0` selects the original
prefill loop for comparison. Decode still processes one token at a time.

## Profiling

```sh
make build/unremarkable-profile  # host
make tablet-profile              # produces build/unremarkable-armv7-profile
```

Profile builds add a second JSON line on stderr with per-step timings for
prefill and decode. [Tablet timings](profile.md).
