#!/usr/bin/env python3
"""Generate one bedtime story per word from each model with the engine.

Writes results.jsonl (every story with its stop reason, length, and whether the
word appears), blind.md (stories per word in shuffled order), and key.json.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess

from prepare import diary_prompt

WORDS = ["dragon", "lighthouse", "rain", "moon", "grandma", "robot", "sock", "cloud"]
# Hand-written, not generated; keyed by a detail the story should keep.
DIARIES = {
    "oven": "Long day. Deadline moved up again and I skipped lunch. Came home and the oven "
            "light finally died, so I cooked pasta in the half dark. Called mum, she sounded "
            "well. Tired.",
    "bus": "Missed the bus by ten seconds and walked in the rain. Soaked socks all morning. "
           "But the new colleague brought cinnamon buns and we laughed about it.",
    "basil": "Quiet Sunday. Repotted the basil, read two chapters, fell asleep on the sofa. "
             "Garden needs weeding but not today.",
    "interview": "Interview tomorrow at 9. Ironed my shirt twice. Can't stop going over the "
                 "answers in my head. What if I freeze?",
    "vet": "Took Rufus to the vet. Nothing serious, just a sore paw, but he looked so sorry "
           "for himself. Bought him a new ball on the way home.",
    "dishes": "Had a stupid argument with Sam about the dishes. We both apologised later. "
              "Still feel a bit heavy.",
    "snow": "First snow! Walked to work through the park, everything silent and white. Kids "
            "building a snowman by the pond.",
    "watch": "Miss grandpa today. Found his old watch in a drawer. It still ticks.",
}


def prompt(style, case):
    if style == "chat":
        return f"Tell me a bedtime story with the word {case}."
    if style == "diary":
        return diary_prompt(DIARIES[case])
    return f"Once upon a time, there was a little {case} who was getting ready for bed."


def generate(engine, model, tokenizer, style, word, seed, limit):
    args = [str(engine), str(model), "-z", str(tokenizer), "-i", prompt(style, word),
            "-t", "0.8", "-p", "0.9", "-s", str(seed), "-n", str(limit), "-j", "2"]
    if style != "opening":
        # The tablet's settings; TinyStories keeps its own 256-token context.
        args += ["-c", "512", "-y", ""]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if proc.returncode:
        raise RuntimeError(f"{model} failed on {word!r}: {proc.stderr.strip()}")
    metrics = json.loads(proc.stderr.splitlines()[0])
    return proc.stdout.strip(), metrics


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", nargs=4, action="append", required=True,
                        metavar=("LABEL", "CHECKPOINT", "TOKENIZER", "STYLE"),
                        help="STYLE is chat (word instruction), diary, or opening "
                             "(continuation)")
    parser.add_argument("--task", choices=("word", "diary"), default="word",
                        help="diary uses the entries above; every model needs STYLE diary")
    parser.add_argument("--out", type=Path, default=here / "out" / "eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results, key, sheet = [], {}, ["# Bedtime stories, blind\n"]
    for word in (WORDS if args.task == "word" else DIARIES):
        stories = []
        for label, model, tokenizer, style in args.model:
            text, metrics = generate(args.engine, model, tokenizer, style, word, args.seed,
                                     args.limit)
            ended = metrics["stop"] in ("turn", "eos", "bos")
            # The engine echoes an opening prompt, which already contains the word.
            written = text[len(prompt(style, word)):] if style == "opening" else text
            row = {"word": word, "model": label, "ended": ended, "stop": metrics["stop"],
                   "tokens": metrics["generated_tokens"], "uses_word": word in written.lower(),
                   "text": text}
            results.append(row)
            stories.append(row)
        rng.shuffle(stories)
        key[word] = {chr(65 + i): row["model"] for i, row in enumerate(stories)}
        sheet.append(f"\n## {word}\n")
        for i, row in enumerate(stories):
            sheet.append(f"\n**{chr(65 + i)}**\n\n{row['text']}\n")
        print(f"{word}: done", flush=True)
    (args.out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (args.out / "blind.md").write_text("".join(sheet))
    (args.out / "key.json").write_text(json.dumps(key, indent=1))
    for label, *_ in args.model:
        rows = [r for r in results if r["model"] == label]
        print(f"{label}: ended {sum(r['ended'] for r in rows)}/{len(rows)}, "
              f"used the word {sum(r['uses_word'] for r in rows)}/{len(rows)}, "
              f"median {sorted(r['tokens'] for r in rows)[len(rows) // 2]} tokens")


if __name__ == "__main__":
    main()
