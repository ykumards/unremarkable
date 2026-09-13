# Bedtime-story fine-tune

Fine-tunes SmolLM2-135M-Instruct on
[TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
(CDLA-Sharing-1.0), so it tells short, complete stories that end on their own.
Prompts ask for a bedtime story, often with words the story must use; stories
tagged `BadEnding` are dropped. Prompts match the engine's chat template with
`-y ""` (no system turn), and the loss covers only the story and its closing
`<|im_end|>`.

Any Python with PyTorch (CUDA) and Transformers works; `setup.sh` creates one in
`train/.venv` and downloads the data to `train/data`.

```sh
sh train/setup.sh
PY=train/.venv/bin/python                 # or an existing environment
$PY train/prepare.py                      # data/train.jsonl, data/valid.jsonl
$PY train/finetune.py --throughput-steps 50
$PY train/finetune.py --examples 600000   # batches are sized in tokens: --batch-tokens
```

The result lands in `train/out/final` as BF16 safetensors, which the exporter
reads. `evaluate.py` then writes one story per word from each model, with a blind
comparison sheet:

```sh
python3 tools/export_hf.py train/out/final -o models/bedtime-q8.bin -c 2048 -q q8_0
python3 train/evaluate.py --engine build/unremarkable \
  --model stock models/smollm2-135m-q8.bin models/smollm2-135m-q8.tok chat \
  --model finetuned models/bedtime-q8.bin models/bedtime-q8.tok chat \
  --model tinystories models/stories15M.bin models/tokenizer.bin opening
```
