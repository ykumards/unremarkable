The transformer operators, BPE tokenizer, and sampler in `src/` are
adapted from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c),
revision `350e04fe35433e6d2941dce5a1f53308f87058eb` (MIT; see `LICENSE`).
The published 260K greedy-output fixture in `tests/test_engine.py` is from
`test_all.py` at that revision.

The C++ ownership, loader, CLI, generation accounting, build, and test harness
are local changes. Sampling ties are ordered by token ID for reproducibility.
The baseline retains the reference's scalar arithmetic order, without OpenMP
or fast-math. Weights are read into owned RAM rather than memory mapped.

SmolLM2-135M-Instruct is downloaded from
[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
(Apache-2.0) and converted locally by `tools/export_hf.py`; the byte-level
pre-tokenizer and GPT-2 byte map it reimplements are described by that
repository's `tokenizer.json`.

TinyStories model files are downloaded separately from
[karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas), revision
`0bd21da7698eaf29a0d7de3992de8a46ef624add`.
The 15M tokenizer comes from the pinned llama2.c revision above.
Model binaries are downloaded separately; URLs and SHA-256 checksums are in
`tools/download.py`. Files in `models/` are not committed.
