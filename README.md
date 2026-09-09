# unremarkable

A small C++23 LLM inference engine for experiments on the reMarkable 2.

The baseline loads a legacy llama2.c FP32 checkpoint fully into RAM and runs a
single-threaded dense Llama forward pass: RMSNorm, rotary attention, KV cache,
SwiGLU, and vocabulary projection. It includes a BPE tokenizer and greedy or
temperature/top-p sampling. Plain loops, with no hand-written SIMD or quantization.

Two model families work. TinyStories 15M continues a prompt into a story and is
not instruction-tuned; SmolLM2-135M-Instruct answers questions using a ChatML
template. The engine reads legacy llama2.c `.bin` checkpoints and the tagged
format written by `tools/export_hf.py`, not GGUF.

## Run

Requires a C++23 compiler and standard library, Make, and Linux or macOS.
Downloads and tests use Python 3.11+; downloads also require curl.
Inference has no third-party library dependencies.

```sh
make
make models  # downloads and verifies the 15M checkpoint and tokenizer (~62 MB)

./build/unremarkable models/stories15M.bin \
  -z models/tokenizer.bin \
  -i "Once upon a time" \
  -n 256
```

For an instruction-following model instead:

```sh
make chat-model  # downloads ~270 MB and converts it to a 538 MB FP32 checkpoint

./build/unremarkable models/smollm2-135m.bin \
  -z models/smollm2-135m.tok \
  -c 512 \
  -i "What is the capital of France?" \
  -n 200
```

A vocabulary carrying `<|im_start|>` and `<|im_end|>` selects the ChatML template
automatically: the prompt is wrapped as a user turn, the reply stops at the end of
the turn, and only the reply is printed. `-y` replaces the system prompt. Prompt
text can never encode to a role marker, so it cannot forge a turn boundary.

Text goes to stdout. One JSON metrics record goes to stderr:

```sh
./build/unremarkable models/stories15M.bin \
  -i "Once upon a time" -n 256 > story.txt 2> metrics.json
```

`-n` limits **new** tokens. Generation stops on EOS, the legacy TinyStories BOS
delimiter, or context exhaustion. The 15M checkpoint has a 256-position context,
so a nonempty prompt leaves room for fewer than 256 new tokens. The final sampled
token does not need a cache slot until it is fed back into the model.

Default sampling is greedy. For a repeatable sampled story:

```sh
./build/unremarkable models/stories15M.bin \
  -i "Once upon a time" -t 0.8 -p 0.9 -s 42 -n 256
```

`-c 128` reduces cache capacity; it cannot exceed the checkpoint's maximum.
`--help` lists the options. Model binaries and build outputs are ignored by Git;
see [model instructions](models/README.md).

## reMarkable 2

`make tablet` cross-compiles for the device inside a container pinned to Ubuntu
24.04, which matches Codex Linux exactly: glibc 2.39 and GCC 13. libstdc++ and
libgcc are linked statically because the toolchain ships 6.0.33 while the tablet
has 6.0.32, and `-ffp-contract=off` matters more here than on the host, since
VFPv4 has FMA and contraction would change results.

```sh
make tablet                       # build/unremarkable-armv7
scp build/unremarkable-armv7 remarkable:/home/root/unremarkable/unremarkable
scp -r models/ device/ remarkable:/home/root/unremarkable/
```

Everything belongs under `/home/root`. The root filesystem has around 11 MiB
free, and filling it can leave the device unbootable.

The 260K fixture is the cross-compile check: identical greedy output on ARMv7
and on the build host means the port did not drift numerically.

| File | Responsibility |
| --- | --- |
| `device/ask.sh` | Runs one preset or a free-text subject, streaming into `state/` |
| `device/ask-bg.sh` | Detaches a run so the UI thread never blocks |
| `device/unremarkable.qmd` | Quick Settings panel with the presets |
| `device/unremarkable-selection.qmd` | Poem about a handwriting selection |

The two `.qmd` files are [qmldiff](https://github.com/asivery/qmldiff) patches
applied by qt-resource-rebuilder under [xovi](https://github.com/asivery/xovi),
installed with `vellum add xovi qt-resource-rebuilder qt-command-executor`. Copy
them to `/home/root/xovi/exthome/qt-resource-rebuilder/` and run `xovi/start`;
`xovi/stock` reverts. The systemd override lives on a tmpfs, so a reboot always
returns the device to stock and the patches must be started again.

The selection patch reuses the device's own handwriting recognizer, so the
circled text becomes the prompt. Recognised text reaches `ask.sh` as an argv
entry rather than through a shell string, so it cannot be read as shell syntax.
Avoid regular-expression literals inside a `.qmd`: a quote inside one starts a
string as far as qmldiff is concerned, and the corrupted QML takes down every
component that depends on it.

## Measurements

- `load_ms`: checkpoint validation, full read into RAM, and inference-buffer setup.
- `prefill_ms`: forward passes for all prompt tokens, including BOS.
- `ttft_ms`: tokenization through first generated token, excluding model/tokenizer
  loading. Zero if generation stops before producing a token.
- `decode_tok_s`: forward plus sampling for generated tokens **after the first**.
  Excludes terminal BOS/EOS, prompt processing, and printing. Zero if none.
- `generation_ms`: tokenization, prefill, generation, and text output together.
- `weights_mib`: resident checkpoint payload, including unused legacy RoPE tables.
- `kv_cache_mib`: allocated FP32 key/value cache capacity.
- `peak_rss_mib`: process peak resident memory from `getrusage`.

For comparisons, keep the checkpoint, prompt, context, seed, compiler flags, and
hardware fixed. Record token counts and stop reason alongside speed; short runs
that stop early are different workloads. Use fresh processes and keep warmup
runs separate from measured runs.

## Code

| File | Responsibility |
| --- | --- |
| `src/engine.h`, `src/engine.cpp` | Owned model weights, cache and scratch, forward pass, sampler |
| `src/kernels.h`, `src/kernels.cpp` | Embedding lookup, matvec, RMSNorm, RoPE, causal attention, softmax, SwiGLU, residual addition |
| `src/tokenizer.h`, `src/tokenizer.cpp` | Tokenizer loading, BPE, byte fallback |
| `src/main.cpp` | CLI, chat template, generation loop, metrics |
| `tools/export_hf.py` | Hugging Face Llama checkpoint and tokenizer conversion |

`Engine::forward(token, position)` returns a borrowed logits span. Positions are
consecutive from zero; `reset()` starts a new sequence while retaining the loaded
weights and allocated buffers. Parsing and configuration errors use exceptions,
caught by the CLI. Owning buffers use `std::vector`.

`forward()` selects each layer's weights and cache slots, then calls the kernels
in transformer order. Numerical loops live in `kernels.cpp`; `kernels.h` documents
their buffer shapes and in-place behavior. Kernels write into buffers allocated
by the engine and do not allocate memory themselves.

The loader accepts both header layouts. A tagged checkpoint begins with the magic
`UNRK`, carries its RoPE base in the header, and omits the two unused RoPE tables
that legacy checkpoints store. `tools/export_hf.py` also reorders each head's
query and key rows, because Hugging Face rotates halves of a head where this
engine rotates adjacent pairs.

The tokenizer reads both the legacy SentencePiece export, with byte-fallback IDs
3..258, and a byte-level BPE export whose scores are negated merge ranks, which
lets one merge loop serve both. Byte-level encoding splits text on the reference's
word boundaries first; that split classifies ASCII exactly and treats bytes above
ASCII as letters, which matches the reference for Latin and CJK text but can
differ for symbols such as emoji.

The math and legacy tokenizer are adapted from a pinned llama2.c revision under MIT;
see [THIRD_PARTY.md](THIRD_PARTY.md). The structure follows SlowMoE's C++23,
Makefile, RAII, and separate kernel/tokenizer conventions.

## Check

```sh
make test-models     # downloads and verifies the ~1 MB 260K test fixture
make check           # byte-level tokenizer checks need `make chat-model` too
make check-sanitize  # AddressSanitizer + UndefinedBehaviorSanitizer
make check-format   # clang-format; optional for building/running
```

Checks cover published 200-token greedy output, logits against an independent
Python reference for both header layouts and both RoPE bases, multihead and
grouped-query attention, shared/separate classifier weights, cache reset and
bounds, BOS/EOS, sampling, UTF-8 input, malformed files, and byte-level
tokenization against identifiers from the reference tokenizer.

The executable built by `make` targets the build machine. The reMarkable needs an
ARMv7 Linux build with a compatible C++ toolchain/runtime. This baseline has not
yet been tested on the tablet; desktop timings are not tablet benchmarks.
