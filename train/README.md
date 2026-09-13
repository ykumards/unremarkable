# Bedtime-story fine-tune

Fine-tunes SmolLM2-135M-Instruct to tell a short bedtime story about whatever
you circle in your notebook: a word, a phrase, a list, a note or a diary entry.
Any Python with PyTorch (CUDA) and Transformers works; `setup.sh` creates one in
`train/.venv` and downloads TinyStories-Instruct to `train/data`.

## Data

`generate.py` has a larger model write the training pairs through any
OpenAI-compatible server; we used Qwen3.5-9B AWQ on vLLM (set
`VLLM_USE_FLASHINFER_SAMPLER=0` if FlashInfer's kernel build rejects your GCC).
Each pair takes two calls:

1. A selection: a random person, mood and day, and a random kind of text
   (a single word, a phrase, a list, a note, a diary entry, or a copied line),
   with its key, the concrete thing a story should be built around.
2. A story that sees only the selection: 70 to 120 words, simple words, built
   around the key and named plainly, with a small plot that works out. It avoids
   stock mood words (soft, warm, cozy, quiet…) and never mentions a writer or
   notebook; nobody has to fall asleep at the end.

`prepare.py --generated` drops pairs whose story loses the key, leans on mood
words, mentions the notebook, or runs outside 60 to 180 words; gives 15% of
prompts recognition-style typos; and mixes in a TinyStories-Instruct sample
(CDLA-Sharing-1.0, BadEnding stories dropped). The prompt is
`prepare.story_prompt`, which the tablet's `device/ask.sh` must match. Without
`--generated` it writes all of TinyStories-Instruct, as for the first fine-tune.

## Training and evaluation

`finetune.py` trains on the chat template exactly as the engine builds it with
`-y ""` (no system turn), with loss only on the story and its `<|im_end|>`.
Batches are sized in tokens (`--batch-tokens`).

`evaluate.py` runs 24 hand-written selections (never generated) through the
engine, two seeds each, and scores every story: did it end on its own, keep the
selection's key, lean on mood words, mention the notebook, or end with someone
asleep. It also writes a blind comparison sheet.

```sh
sh train/setup.sh
PY=train/.venv/bin/python                  # or an existing environment
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit --port 8766 --max-model-len 2048 \
  --served-model-name writer
$PY train/generate.py --count 60000        # train/data/stories.jsonl; rerun to resume
$PY train/prepare.py --generated train/data/stories.jsonl --stories 50000
$PY train/finetune.py --examples 1000000   # the whole mix, into train/out/final
python3 tools/export_hf.py train/out/final -o models/story-q8.bin -c 2048 -q q8_0
python3 train/evaluate.py --engine build/unremarkable \
  --model stock models/smollm2-135m-q8.bin models/smollm2-135m-q8.tok story \
  --model story models/story-q8.bin models/story-q8.tok story
```

## Results

Generation took about 95 minutes on an RTX 4090 for 56,910 pairs; the filter
kept 53,663 (94%). Each variant trained for one epoch on two copies of those
pairs, the first with 50k TinyStories mixed in. On the 24 hand-written
selections, two seeds each, at temperature 0.8 and a 240-token limit:

| model | ended | kept key | mood words /100 | notebook leaks | ended asleep | median tokens |
|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 9/48 | 32/48 | 1.2 | 2 | 8/48 | 240 |
| earlier diary fine-tune | 48/48 | 38/48 | 9.7 | 3 | 46/48 | 134 |
| generated + TinyStories | 46/48 | 46/48 | 0.2 | 0 | 0/48 | 132 |
| generated only | 47/48 | 48/48 | 0.2 | 0 | 3/48 | 134 |

The generated-only model ships; with TinyStories mixed in, one story repeated
"Tick." until the limit. At temperature 0.5 and a 300-token limit it ended
48/48 and kept the key in 45/48, and reading the stories it makes fewer
non-sequiturs than at 0.8, so the tablet uses 0.5. It is still a 135M model:
expect the odd slip, such as reading "Miss grandpa today" as a name.
