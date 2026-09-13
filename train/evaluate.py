#!/usr/bin/env python3
"""Tell stories about hand-written notebook selections with the engine, and score them.

Selections are written by hand, never generated: single words, phrases, lists,
notes and diary entries, each with a pattern for what the story should keep.
Writes results.jsonl, blind.md (seed-1 stories per selection in shuffled order)
and key.json, and prints per-model scores.
"""
import argparse
import json
from pathlib import Path
import random
import re
import statistics
import subprocess

from generate import MOOD_WORDS
from prepare import story_prompt

SELECTIONS = [
    ("hello", r"hello"),
    ("dragon", r"dragons?\b"),
    ("lighthouse", r"lighthouse"),
    ("my cat", r"cats?\b"),
    ("tired today", r"tired"),
    ("pizza with friends", r"pizza"),
    ("Buy milk, call dentist, finish slides", r"milk|dentist|slides?\b"),
    ("I got the job!!!", r"job"),
    ("Mom is in hospital. Scared.", r"hospital|mom"),
    ("teh meetign was lnog and boring, i want to sleep", r"meeting"),
    ("The mitochondria is the powerhouse of the cell.", r"mitochondri"),
    ("Can't stop thinking about what she said at dinner.", r"dinner"),
    ("Ran 5k for the first time without stopping!", r"5k|ran\b|run"),
    ("Rain all day. Stayed in, made soup, watched an old film with Anna.", r"soup|film|anna"),
    ("Deadline tomorrow and nothing works. Spent 4 hours debugging a null pointer.",
     r"deadline|debug|code|bug"),
    ("We went to the beach and built a huge sandcastle, then the tide took it.",
     r"sandcastle"),
    ("Long day. Deadline moved up again and I skipped lunch. Came home and the oven light "
     "finally died, so I cooked pasta in the half dark. Called mum, she sounded well. "
     "Tired.", r"oven|pasta"),
    ("Missed the bus by ten seconds and walked in the rain. Soaked socks all morning. But "
     "the new colleague brought cinnamon buns and we laughed about it.", r"bus\b|buns?\b"),
    ("Quiet Sunday. Repotted the basil, read two chapters, fell asleep on the sofa. Garden "
     "needs weeding but not today.", r"basil"),
    ("Interview tomorrow at 9. Ironed my shirt twice. Can't stop going over the answers in "
     "my head. What if I freeze?", r"interview"),
    ("Took Rufus to the vet. Nothing serious, just a sore paw, but he looked so sorry for "
     "himself. Bought him a new ball on the way home.", r"rufus|vet\b"),
    ("Had a stupid argument with Sam about the dishes. We both apologised later. Still "
     "feel a bit heavy.", r"dishes|argu"),
    ("First snow! Walked to work through the park, everything silent and white. Kids "
     "building a snowman by the pond.", r"snow"),
    ("Miss grandpa today. Found his old watch in a drawer. It still ticks.", r"watch"),
]
PROMPTS = {
    "story": story_prompt,
    # The first diary fine-tune's template, to compare it on its own terms.
    "diary": lambda text: (f"My diary entry for today:\n\n{text}\n\n"
                           "Tell me a short bedtime story about it."),
}
LEAK = re.compile(r"\b(writer|diary|notebook|journal)\b", re.I)
SLEEP = re.compile(r"\b(sleep|asleep|slept|dream|bed|goodnight)", re.I)


def generate(engine, model, tokenizer, prompt, seed, limit, temperature):
    args = [str(engine), str(model), "-z", str(tokenizer), "-i", prompt, "-c", "512",
            "-y", "", "-j", "2", "-b", "8", "-t", str(temperature), "-p", "0.9",
            "-s", str(seed), "-n", str(limit)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if proc.returncode:
        raise RuntimeError(f"{model} failed: {proc.stderr.strip()}")
    return proc.stdout.strip(), json.loads(proc.stderr.splitlines()[0])


def score(text, pattern):
    words = max(len(text.split()), 1)
    ending = " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[-2:])
    return {"kept": bool(re.search(rf"\b(?:{pattern})", text, re.I)),
            "mood": sum(len(re.findall(rf"\b{w}", text, re.I)) for w in MOOD_WORDS) * 100 / words,
            "leak": bool(LEAK.search(text)), "sleep_end": bool(SLEEP.search(ending))}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", nargs=4, action="append", required=True,
                        metavar=("LABEL", "CHECKPOINT", "TOKENIZER", "PROMPT"),
                        help="PROMPT is story (current template) or diary (the first "
                             "diary fine-tune's)")
    parser.add_argument("--out", type=Path, default=here / "out" / "eval")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)  # as device/ask.sh
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    results, key, sheet = [], {}, ["# Notebook stories, blind\n"]
    for n, (selection, pattern) in enumerate(SELECTIONS):
        stories = []
        for label, model, tokenizer, prompt in args.model:
            for seed in range(1, args.seeds + 1):
                text, metrics = generate(args.engine, model, tokenizer,
                                         PROMPTS[prompt](selection), seed, args.limit,
                                         args.temperature)
                row = {"selection": selection, "model": label, "seed": seed,
                       "ended": metrics["stop"] in ("turn", "eos", "bos"),
                       "tokens": metrics["generated_tokens"], "text": text,
                       **score(text, pattern)}
                results.append(row)
                if seed == 1:
                    stories.append(row)
        rng.shuffle(stories)
        key[n] = {chr(65 + i): row["model"] for i, row in enumerate(stories)}
        sheet.append(f"\n## {n}. {selection}\n")
        for i, row in enumerate(stories):
            sheet.append(f"\n**{chr(65 + i)}**\n\n{row['text']}\n")
        print(f"{n + 1}/{len(SELECTIONS)} done", flush=True)
    (args.out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (args.out / "blind.md").write_text("".join(sheet))
    (args.out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{'model':12} {'ended':>7} {'kept key':>9} {'mood/100w':>10} {'leaks':>6} "
          f"{'sleep end':>10} {'median tok':>11}")
    for label, *_ in args.model:
        rows = [r for r in results if r["model"] == label]
        total = len(rows)
        print(f"{label:12} {sum(r['ended'] for r in rows):>3}/{total:<3} "
              f"{sum(r['kept'] for r in rows):>5}/{total:<3} "
              f"{statistics.mean(r['mood'] for r in rows):>10.1f} "
              f"{sum(r['leak'] for r in rows):>6} "
              f"{sum(r['sleep_end'] for r in rows):>6}/{total:<3} "
              f"{statistics.median(r['tokens'] for r in rows):>11}")


if __name__ == "__main__":
    main()
