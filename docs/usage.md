# Usage

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

Or quantized, which is what the tablet runs:

```sh
make chat-model-q8  # the same weights as Q8_0: 144 MiB instead of 513 MiB

./build/unremarkable models/smollm2-135m-q8.bin \
  -z models/smollm2-135m-q8.tok \
  -c 512 \
  -i "What is the capital of France?" \
  -n 200
```

Quantizing requires numpy; every other path, inference included, needs only the
standard library. The tokenizer is identical either way. Both checkpoints run on
the same engine, which reads the format from the header.

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

`-j N` runs the projections on N threads; on the tablet's two cores `-j 2` is
about 1.5x faster to decode and 1.6x to first token. Rows are split between
threads and never summed apart, so the output does not depend on N. The default
is 1, which is what the published measurements use.

`-c 128` reduces cache capacity; it cannot exceed the checkpoint's maximum.
`--help` lists the options. Model binaries and build outputs are ignored by Git;
see [model instructions](../models/README.md).
