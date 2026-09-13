#!/usr/bin/env python3
"""Turn TinyStories-Instruct into bedtime-story chat examples (JSON lines).

Each record is a set of fields in any order; the story runs from "Story:" to
the next field. Stories marked BadEnding are skipped. With --diary, a sample of
them is mixed with diary-entry stories from generate.py.
"""
import argparse
import json
from pathlib import Path
import random

FIELDS = ("Features:", "Words:", "Summary:", "Random sentence:", "Story:")


def diary_prompt(entry):
    return f"My diary entry for today:\n\n{entry}\n\nTell me a short bedtime story about it."


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


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--diary", type=Path, help="generate.py output to mix in")
    parser.add_argument("--stories", type=int, default=150_000,
                        help="TinyStories training examples kept with --diary")
    parser.add_argument("--repeat", type=int, default=2, help="copies of each diary example")
    parser.add_argument("--valid", type=int, default=1_000,
                        help="validation examples from each source with --diary")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    splits = {s: tinystories(args.data / f"TinyStories-Instruct-{s}.txt", rng)
              for s in ("valid", "train")}
    if args.diary:
        rows = sorted((json.loads(line) for line in args.diary.open()), key=lambda r: r["id"])
        diary = [{"prompt": diary_prompt(r["diary"]), "story": r["story"]} for r in rows]
        valid = rng.sample(list(splits["valid"]), args.valid) + diary[:args.valid]
        train = rng.sample(list(splits["train"]), args.stories) + diary[args.valid:] * args.repeat
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
