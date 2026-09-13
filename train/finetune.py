#!/usr/bin/env python3
"""Fine-tune SmolLM2-135M-Instruct on bedtime-story chat examples.

Prompts are tokenized exactly as src/main.cpp builds them with `-y ""`: no
system turn. Loss covers only the story and its closing <|im_end|>. The result
is saved in BF16 as one model.safetensors, which tools/export_hf.py reads.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tokenizer, prompt, story):
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    context = [im_start] + text(f"user\n{prompt}") + [im_end] + text("\n")
    context += [im_start] + text("assistant\n")
    target = text(story) + [im_end]
    return context + target, [-100] * len(context) + target


def load(path, tokenizer, limit, max_length, rng):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rng.shuffle(rows)
    examples = []
    for row in rows:
        ids, labels = encode(tokenizer, row["prompt"], row["story"])
        # Truncating would cut off the ending, which is what we want to teach.
        if len(ids) <= max_length:
            examples.append((ids, labels))
        if len(examples) == limit:
            break
    return examples


def group(examples, budget, rng=None):
    """Batches of similar length, each at most `budget` padded tokens."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    groups, current, width = [], [], 0
    for i in order:
        length = len(examples[i][0])
        if current and max(width, length) * (len(current) + 1) > budget:
            groups.append(current)
            current, width = [], 0
        current.append(i)
        width = max(width, length)
    if current:
        groups.append(current)
    if rng:
        rng.shuffle(groups)
    return groups


def batches(examples, groups, pad):
    for indices in groups:
        chunk = [examples[i] for i in indices]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad)
        labels = torch.full((len(chunk), width), -100)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (i, l) in enumerate(chunk):
            ids[row, :len(i)] = torch.tensor(i)
            labels[row, :len(l)] = torch.tensor(l)
            mask[row, :len(i)] = 1
        yield ids.cuda(), labels.cuda(), mask.cuda()


def story_loss(model, ids, labels, mask):
    """Summed loss over story tokens, and their count.

    The vocabulary projection runs only where there is a target, skipping
    prompt and padding positions.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        targets = labels[:, 1:]
        keep = targets != -100
        logits = model.lm_head(hidden[:, :-1][keep])
    loss = torch.nn.functional.cross_entropy(logits.float(), targets[keep], reduction="sum")
    return loss, keep.sum()


@torch.no_grad()
def evaluate(model, examples, budget, pad):
    model.eval()
    total = count = 0.0
    for ids, labels, mask in batches(examples, group(examples, budget), pad):
        loss, n = story_loss(model, ids, labels, mask)
        total += loss.item()
        count += n.item()
    model.train()
    return total / count


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--data", type=Path, default=here / "data")
    parser.add_argument("--out", type=Path, default=here / "out")
    parser.add_argument("--examples", type=int, default=400_000)
    parser.add_argument("--valid-examples", type=int, default=2_000)
    parser.add_argument("--batch-tokens", type=int, default=8_192,
                        help="padded tokens per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--throughput-steps", type=int, default=0,
                        help="time this many steps, report tokens/s, and exit")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda()
    model.config.use_cache = False

    train = load(args.data / "train.jsonl", tokenizer, args.examples, args.max_length, rng)
    valid = load(args.data / "valid.jsonl", tokenizer, args.valid_examples, args.max_length, rng)
    groups = group(train, args.batch_tokens, rng)
    steps = len(groups)
    print(f"{len(train)} training examples in {steps} batches, {len(valid)} validation; "
          f"attention: {model.config._attn_implementation}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.95), fused=True)
    warmup = max(1, steps // 50)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) *
        0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    if not args.throughput_steps:
        baseline = evaluate(model, valid, args.batch_tokens, pad)
        print(f"step 0: validation loss {baseline:.4f}", flush=True)
        log.write(json.dumps({"step": 0, "valid_loss": baseline}) + "\n")

    started, tokens = time.time(), 0
    for step, (ids, labels, mask) in enumerate(batches(train, groups, pad), start=1):
        total, count = story_loss(model, ids, labels, mask)
        loss = total / count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += int(mask.sum())
        if args.throughput_steps and step == args.throughput_steps:
            torch.cuda.synchronize()
            rate = tokens / (time.time() - started)
            memory = torch.cuda.max_memory_allocated() / 2**30
            print(f"{rate:,.0f} tokens/s, peak memory {memory:.1f} GiB, "
                  f"{tokens / step:,.0f} real tokens per batch", flush=True)
            return
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} "
                  f"{tokens / (time.time() - started):,.0f} tokens/s", flush=True)
        if step % args.eval_every == 0 or step == steps:
            value = evaluate(model, valid, args.batch_tokens, pad)
            print(f"step {step}: validation loss {value:.4f}", flush=True)
            log.write(json.dumps({"step": step, "loss": loss.item(), "valid_loss": value}) + "\n")
            log.flush()

    final = args.out / "final"
    model.to(torch.bfloat16).save_pretrained(final, max_shard_size="4GB")
    tokenizer.save_pretrained(final)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
