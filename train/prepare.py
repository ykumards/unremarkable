#!/usr/bin/env python3
"""Build chat examples (JSON lines) from generated stories and TinyStories-Instruct.

With --generated, pairs from generate.py are filtered (the story must keep the
key, avoid stock mood words and never mention the notebook), some prompts get
recognition-style typos, and a TinyStories sample is mixed in. Without it, all
of TinyStories-Instruct is written, as for the first fine-tune.

TinyStories records are fields in any order; the story runs from "Story:" to
the next field. Stories marked BadEnding are skipped.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from generate import MOOD_WORDS

FIELDS = ("Features:", "Words:", "Summary:", "Random sentence:", "Story:")
LEAK = re.compile(r"\b(writer|diary|notebook|journal)\b", re.I)
STOPWORDS = {"the", "and", "one", "with", "for", "from", "into", "our", "your", "his", "her",
             "its", "their", "this", "that"}


def story_prompt(text):
    return f"Here is something from my notebook:\n\n{text}\n\nTell me a short bedtime story about it."


def records(path):
    lines = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "<|endoftext|>":
                if lines:
                    yield lines
                lines = []
            else:
                lines.append(line.rstrip("\n"))
    if lines:
        yield lines


def parse(lines):
    fields, story, in_story = {}, [], False
    for line in lines:
        field = next((f for f in FIELDS if line.startswith(f)), None)
        if field == "Story:":
            in_story = True
            rest = line[len(field):].strip()
            if rest:
                story.append(rest)
        elif field:
            in_story = False
            fields[field[:-1]] = line[len(field):].strip()
        elif in_story:
            story.append(line)
    return fields, "\n".join(story).strip()


def prompt_for(fields, rng):
    words = [w.strip() for w in fields.get("Words", "").split(",") if w.strip()]
    choice = rng.random()
    if words and choice < 0.45:
        listed = ", ".join(words[:-1]) + f" and {words[-1]}" if len(words) > 1 else words[0]
        return f"Tell me a bedtime story using the words {listed}."
    if words and choice < 0.80:
        return f"Tell me a bedtime story with the word {rng.choice(words)}."
    return "Tell me a bedtime story."


def tinystories(path, rng):
    kept = skipped = 0
    for lines in records(path):
        fields, story = parse(lines)
        if not story or "BadEnding" in fields.get("Features", ""):
            skipped += 1
            continue
        kept += 1
        yield {"prompt": prompt_for(fields, rng), "story": story}
    print(f"{path.name}: {kept} examples, {skipped} skipped")


def rejected(row):
    story = row["story"]
    words = len(story.split())
    if not 60 <= words <= 180:
        return "length"
    # Any content word of the key, by stem, so "football match" accepts "the match".
    parts = [w for w in re.findall(r"[\w']+", row["key"])
             if len(w) > 2 and w.lower() not in STOPWORDS] or [row["key"]]
    if not any(re.search(rf"\b{re.escape(w[:max(4, len(w) - 2)])}", story, re.I)
               for w in parts):
        return "key"
    if sum(len(re.findall(rf"\b{w}", story, re.I)) for w in MOOD_WORDS) * 100 / words > 3:
        return "mood"
    if LEAK.search(story):
        return "leak"
    return None


def typos(text, rng):
    # Swaps or drops a letter in one to three longer words, like misread handwriting.
    words = text.split(" ")
    eligible = [i for i, word in enumerate(words) if len(word) >= 4 and word.isalpha()]
    for i in rng.sample(eligible, min(len(eligible), rng.randint(1, 3))):
        word = words[i]
        j = rng.randrange(1, len(word) - 2)
        words[i] = (word[:j] + word[j + 1] + word[j] + word[j + 2:] if rng.random() < 0.5
                    else word[:j] + word[j + 1:])
    return " ".join(words)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generated", type=Path, help="generate.py output to train on")
    parser.add_argument("--stories", type=int, default=50_000,
                        help="TinyStories training examples mixed with --generated")
    parser.add_argument("--repeat", type=int, default=2, help="copies of each generated example")
    parser.add_argument("--valid", type=int, default=1_000,
                        help="validation examples from each source with --generated")
    parser.add_argument("--typos", type=float, default=0.15,
                        help="share of generated prompts given recognition-style typos")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    source = lambda split: tinystories(args.data / f"TinyStories-Instruct-{split}.txt", rng)
    if not args.generated:
        splits = {"valid": source("valid"), "train": source("train")}
    else:
        rows = sorted((json.loads(line) for line in args.generated.open()),
                      key=lambda r: r["id"])
        reasons = [rejected(r) for r in rows]
        dropped = Counter(reason for reason in reasons if reason)
        kept = [r for r, reason in zip(rows, reasons) if not reason]
        print(f"{args.generated.name}: kept {len(kept)} of {len(rows)}; dropped "
              + (", ".join(f"{n} {reason}" for reason, n in dropped.most_common()) or "none"))
        generated = [{"prompt": story_prompt(typos(r["text"], rng) if rng.random() < args.typos
                                             else r["text"]), "story": r["story"]}
                     for r in kept]
        sample = lambda split, n: rng.sample(list(source(split)), n) if n else []
        valid = sample("valid", min(args.valid, args.stories)) + generated[:args.valid]
        train = sample("train", args.stories) + generated[args.valid:] * args.repeat
        rng.shuffle(train)
        splits = {"valid": valid, "train": train}
    for split, rows in splits.items():
        count = 0
        with (args.data / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
                count += 1
        print(f"{split}.jsonl: {count} examples")


if __name__ == "__main__":
    main()
