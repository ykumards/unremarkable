#!/usr/bin/env python3
"""Turn TinyStories-Instruct into bedtime-story chat examples (JSON lines).

Each record is a set of fields in any order; the story runs from "Story:" to
the next field. Stories marked BadEnding are skipped.
"""
import argparse
import json
from pathlib import Path
import random

FIELDS = ("Features:", "Words:", "Summary:", "Random sentence:", "Story:")


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


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "data")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for split in ("valid", "train"):
        source = args.data / f"TinyStories-Instruct-{split}.txt"
        kept = skipped = 0
        with (args.data / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for lines in records(source):
                fields, story = parse(lines)
                if not story or "BadEnding" in fields.get("Features", ""):
                    skipped += 1
                    continue
                out.write(json.dumps({"prompt": prompt_for(fields, rng), "story": story}) + "\n")
                kept += 1
        print(f"{split}: {kept} examples, {skipped} skipped")


if __name__ == "__main__":
    main()
