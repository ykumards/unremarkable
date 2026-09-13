#!/usr/bin/env python3
"""Generate notebook selections and bedtime stories built around them.

Talks to an OpenAI-compatible server (vLLM). Each example is two calls: text
someone might circle in their notebook (a word, a phrase, a list, a note, a
diary entry or a copied line) with its key thing, then a story built around
that key which sees only the text, as the tablet will. Appends to --out and
resumes from it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import urllib.request

PERSONAS = [
    "a nurse on night shifts", "a primary school teacher", "a software developer",
    "a retired carpenter", "a university student", "a new parent", "a baker",
    "a bus driver", "a graphic designer", "a grandmother who gardens",
    "a shop assistant", "a researcher", "a freelance musician", "a farmer",
    "a teenager in high school", "a cook in a busy restaurant", "an architect",
    "someone who just moved to a new city", "a librarian", "a delivery cyclist",
    "a veterinarian", "a stay-at-home dad", "a translator", "an electrician",
    "a ten-year-old", "a marathon runner", "a pilot", "a painter", "a firefighter",
]
MOODS = [
    "happy", "proud", "tired but content", "stressed", "anxious", "sad", "lonely",
    "frustrated", "bored", "grateful", "overwhelmed", "hopeful", "calm", "restless",
    "a little disappointed", "excited", "nervous about tomorrow", "quietly relieved",
]
TOPICS = [
    "a deadline at work", "a walk in the rain", "a small argument with a friend",
    "cooking something new", "a sick pet", "a long commute", "a phone call with family",
    "a job interview", "tidying the house", "a lost key", "a good book",
    "a doctor's appointment", "a child's first day at school", "a garden",
    "a noisy neighbour", "a trip to the sea", "a birthday", "a bad night's sleep",
    "a kind stranger", "moving boxes", "a broken bicycle", "a concert",
    "missing someone far away", "a snowy morning", "an exam", "a picnic",
    "a mistake at work", "learning to knit", "a visit to grandparents", "a thunderstorm",
    "a train that was late", "fixing something in the kitchen", "a museum",
    "a quiet Sunday", "a first run", "a new puppy", "a football match", "a power cut",
    "a wedding", "a hospital visit", "a school play", "a camping trip", "a lighthouse",
    "a dragon from a bedtime book", "the moon", "a robot kit", "a lost sock",
]
STYLES = [
    "short plain sentences", "fragments, like quick notes", "rambling and honest",
    "starting with 'Dear diary'", "reflective and a bit poetic", "matter-of-fact",
]
KINDS = [
    (0.15, "word", "a single word someone might circle on a page: an object, animal, "
                   "place, person or feeling"),
    (0.15, "phrase", "a short phrase of two to six words, like a note in a margin"),
    (0.15, "list", "a short list of three to six items: things to do or things that "
                   "happened, one per line"),
    (0.15, "note", "one or two quick sentences"),
    (0.35, "entry", "a diary entry of about {words} words, in {style}"),
    (0.05, "line", "a line copied into a notebook: a fact, a saying or a sentence "
                   "from a book"),
]
NAMES = [
    "Aiko", "Ben", "Chidi", "Dara", "Emil", "Farah", "Gus", "Hana", "Ines", "Jonas",
    "Kofi", "Leena", "Mateo", "Nora", "Oskar", "Priya", "Quinn", "Rosa", "Sami", "Tove",
    "Umar", "Vera", "Wren", "Xiu", "Yusuf", "Zoe", "Anouk", "Bram", "Cleo", "Dev",
    "Esme", "Femi", "Greta", "Hugo", "Ida", "Juno", "Kai", "Lotte", "Moss", "Nia",
    "Otto", "Paz", "Ravi", "Signe", "Teo", "Ulla", "Vik", "Wanda", "Yara", "Ziggy",
]
ANIMALS = [
    "hedgehog", "otter", "fox", "badger", "owl", "rabbit", "mouse", "tortoise", "seal",
    "sparrow", "bear cub", "squirrel", "duckling", "mole", "deer", "cat", "snail", "frog",
]
# Stock words the first diary model overused; the filter in prepare.py uses them too.
MOOD_WORDS = ["soft", "warm", "cozy", "quiet", "still", "safe", "gentle", "drift",
              "peaceful", "blanket", "whisper", "hum", "lullaby"]

SELECTION = ("You are {persona}. Today involved {topic}, and you feel {mood}. Write {kind} "
             "from your notebook, in everyday language. Also give its key: the main concrete "
             "thing in the text that a story about it should be built around (an object, "
             "animal, food, place, person or event, not a feeling or a verb), copied exactly "
             "as it appears in the text.")
SCHEMA = {"type": "object", "required": ["text", "key"],
          "properties": {"text": {"type": "string"}, "key": {"type": "string"}}}
STORY_SYSTEM = (
    "You write short stories that someone reads at bedtime, about some text they circled "
    "in their notebook. The story is {words} words, in simple words and short sentences. "
    "Build it around \"{key}\": name it plainly and keep it what it is (a dragon stays a "
    "dragon, a job stays a job). If the text is about their day, the story follows what "
    "happened. Give it a real little plot: {teller} wants something or meets a small "
    "problem, does a few concrete things, and it works out. Keep it kind and hopeful, "
    "with nothing scary and no lessons or advice. End on a happy, settled moment once "
    "the problem is solved; nobody needs to go to bed or fall asleep. Do not use these "
    "words: " + ", ".join(MOOD_WORDS) + ". Never mention a writer, a diary, a notebook or "
    "the text itself. Write only the story, with no title.")


def chat(url, model, system, user, temperature, max_tokens, schema=None):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "top_p": 0.95, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "selection", "schema": schema}}
    request = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        choice = json.load(response)["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("truncated")
    return choice["message"]["content"].strip()


def teller(rng):
    name, animal = rng.choice(NAMES), rng.choice(ANIMALS)
    return rng.choices([f"a {animal} named {name}", f"a person named {name}",
                        "the reader, called 'you'"], weights=[4, 4, 2])[0]


def example(url, model, index, seed):
    rng = random.Random(seed * 1_000_003 + index)
    kind = rng.choices(KINDS, weights=[k[0] for k in KINDS])[0]
    meta = {"kind": kind[1], "persona": rng.choice(PERSONAS), "mood": rng.choice(MOODS),
            "topic": rng.choice(TOPICS)}
    request = SELECTION.format(kind=kind[2].format(words=rng.choice([30, 50, 70, 100]),
                                                   style=rng.choice(STYLES)),
                               persona=meta["persona"], mood=meta["mood"], topic=meta["topic"])
    selection = json.loads(chat(url, model, None, request, 1.0, 400, SCHEMA))
    text, key = selection["text"].strip(), selection["key"].strip()
    if not text or not key or key.lower() not in text.lower():
        raise ValueError(f"key {key!r} not in text")
    meta["teller"] = teller(rng)
    system = STORY_SYSTEM.format(words=rng.choice(["70 to 100", "90 to 120"]), key=key,
                                 teller=meta["teller"])
    story = chat(url, model, system, f"The text they circled:\n\n{text}", 0.9, 450)
    return {"id": index, "text": text, "key": key, "story": story, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/v1")
    parser.add_argument("--model", default="writer")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "stories.jsonl")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    todo = [i for i in range(args.count) if i not in done]
    failed = 0
    with args.out.open("a") as out, ThreadPoolExecutor(args.concurrency) as pool:
        jobs = [pool.submit(example, args.url, args.model, i, args.seed) for i in todo]
        for n, job in enumerate(as_completed(jobs), 1):
            try:
                out.write(json.dumps(job.result()) + "\n")
                out.flush()
            except Exception as error:  # truncations, bad keys and timeouts; rerun retries
                failed += 1
                print(f"skipped: {error}", flush=True)
            if n % 100 == 0:
                print(f"{len(done) + n}/{args.count}, {failed} skipped", flush=True)
    print(f"done: {len(todo) - failed} written, {failed} skipped")


if __name__ == "__main__":
    main()
