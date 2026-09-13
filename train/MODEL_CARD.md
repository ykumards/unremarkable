---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- bedtime-stories
- smollm2
- on-device
- remarkable
---

# SmolLM2-135M bedtime stories

Give it something from your notebook, whether a word, a to-do list or a few
lines about your day, and it tells a short, gentle bedtime story about it. A
fine-tune of
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
made for [unremarkable](https://github.com/ykumards/unremarkable), which runs
it on a reMarkable 2: circle some handwriting, tap a button, and the story is
written on the tablet itself.

> **Notebook:** Hmm, woke up on the wrong side of the bed again. How does this
> keep happening? Always on the right! Subconscious, do better!
>
> **Story:** Once upon a time, there was a little frog named Emil. He woke up on
> the wrong side of the bed again. His legs felt wobbly. He wanted to jump high
> to see the big green pond. But the water was too deep for him. Emil did not
> give up… He landed right on the edge of the pond… He is happy and settled.

## Prompt

It was tuned on exactly this user turn, with **no system turn**. Start the
reply with `Once upon a time, there was a little`: without that opening it
tends to ramble in the second person.

```
<|im_start|>user
Here is something from my notebook:

{text}

Tell me a short bedtime story about it.<|im_end|>
<|im_start|>assistant
Once upon a time, there was a little
```

The story ends with `<|im_end|>`. Sample at temperature 0.5, top-p 0.9; keep
the notebook text under about 120 words.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "ykumards/smollm2-135m-bedtime-stories"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

text = "Took the kids to the zoo. The penguins were the best part."
opening = "Once upon a time, there was a little"
request = f"Here is something from my notebook:\n\n{text}\n\nTell me a short bedtime story about it."
prompt = tokenizer.apply_chat_template([{"role": "user", "content": request}],
                                       tokenize=False, add_generation_prompt=True) + opening
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.5, top_p=0.9)
print(opening + tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The bundled chat template adds no system prompt (SmolLM2's original inserts a
default one, which this model never saw), so `apply_chat_template` builds the
training prompt; append the opening yourself.

## On a reMarkable 2

`unremarkable/story-q8.bin` and `unremarkable/story-q8.tok` are the same weights
in the unremarkable engine's Q8_0 format. On the tablet's two Cortex-A7 cores it
takes about 5 s to the first word and writes about 6.4 tokens/s: roughly 30 s
per story, in 171 MiB of memory.

## Training

- **Data:** 56,910 synthetic pairs from Qwen3.5-9B (AWQ, on vLLM). Each pair is
  a piece of notebook text written from a random person, mood and day (a single
  word, a phrase, a list, a note, a diary entry or a copied line) with its key
  thing, and a story of about 70 to 120 words written from the text alone:
  built around the key, a small plot that works out, simple words, no stock
  mood words, and nobody has to fall asleep at the end.
- **Filtering:** pairs whose story lost the key, leaned on words like soft,
  warm and cozy, mentioned the notebook, or ran outside 60 to 180 words were
  dropped, leaving 53,663 (94%). 15% of prompts got handwriting-style typos.
- **Recipe:** one epoch over two copies of the pairs (105k examples), loss on
  the story only, AdamW at 1e-4 with cosine decay, bf16 autocast, batches of
  8,192 tokens, on one RTX 4090.

Code: [`train/`](https://github.com/ykumards/unremarkable/tree/main/train).

## Evaluation

24 hand-written selections (never generated), two seeds each, at temperature
0.8 through the Q8 engine, without the opening:

| model | kept the key | ended asleep | mood words /100 | notebook leaks |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 32/48 | 8/48 | 1.2 | 2 |
| an earlier diary fine-tune | 38/48 | 46/48 | 9.7 | 3 |
| this model | 48/48 | 3/48 | 0.2 | 0 |

## Limitations

- It is a 135M model. Stories stay on topic but the middle can wander, and it
  slips: penguins with orange feathers, "Miss grandpa today" read as a name.
- It does not get jokes or sarcasm.
- For hard entries ("Mom is in hospital. Scared.") it stays gentle but cannot
  really comfort. It is not a substitute for talking to someone.
- English only. The training data is synthetic.

## License

Apache 2.0, like the base model. The training stories were generated with
Qwen3.5-9B.
