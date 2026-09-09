# Model files

Run `make models` for TinyStories 15M or `make test-models` for the 260K test model.
Downloads are pinned and SHA-256 verified.

Checkpoint and tokenizer binaries are downloaded separately and ignored by Git.
For tablet deployment, copy the executable plus the matching files together:

```text
unremarkable
stories15M.bin
tokenizer.bin
```

Then run `./unremarkable stories15M.bin -z tokenizer.bin -i "Once upon a time"`.

The loader reconstructs the tensor layout from the checkpoint's 28-byte header in
the legacy format's fixed storage order, and validates the header, total file
size, and that the derived layout fits. Any legacy llama2.c FP32 checkpoint works
without extra preparation; no Python is needed on the tablet.
