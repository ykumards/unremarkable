#!/usr/bin/env python3
"""Generate diary entries and bedtime stories written from them.

Talks to an OpenAI-compatible server (vLLM). Each example is two calls: a diary
entry from a random persona, mood, topic and style, then a story that sees only
the entry, as the tablet will. Appends to --out and resumes from it.
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
    "a train that was late", "fixing something in the kitchen", "a museum", "a quiet Sunday",
]
STYLES = [
    "short plain sentences", "a few quick notes, fragments allowed",
    "rambling and honest", "starting with 'Dear diary'", "a list of what happened",
    "reflective and a bit poetic", "matter-of-fact",
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


def teller(rng):
    name, animal = rng.choice(NAMES), rng.choice(ANIMALS)
    return rng.choice([
        f"a small {animal} named {name} who lives through a day like the writer's",
        f"a character named {name}",
        "the writer, addressed as 'you'",
        "an object from the entry that comes to life",
        f"a gentle {animal} in a faraway, cosy place",
    ])

DIARY = ("Write one private diary entry of about {words} words by {persona}. "
         "The day involved {topic}. Overall mood: {mood}. Style: {style}. "
         "First person, everyday language. Write only the entry: no title, no date.")
STORY_SYSTEM = (
    "You write bedtime stories that people read on their tablet just before sleep, "
    "based on their diary entry. Rules: {words} words. Simple words and short "
    "sentences. Gentle, warm and positive; nothing scary; no lessons or advice. "
    "Tell a small story about {teller}, carrying over one or two concrete details "
    "from the entry. If the day was hard, let the story find comfort without "
    "denying it. End calm and sleepy. Write only the story, with no title.")


def chat(url, model, system, user, temperature, max_tokens):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "top_p": 0.95, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        choice = json.load(response)["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("truncated")
    return choice["message"]["content"].strip()


def example(url, model, index, seed):
    rng = random.Random(seed * 1_000_003 + index)
    meta = {"persona": rng.choice(PERSONAS), "mood": rng.choice(MOODS),
            "topic": rng.choice(TOPICS), "style": rng.choice(STYLES),
            "teller": teller(rng)}
    diary = chat(url, model, None, DIARY.format(words=rng.choice([30, 50, 70, 100]),
                                                **meta), 1.0, 300)
    story = chat(url, model, STORY_SYSTEM.format(words=rng.choice(["80 to 110", "100 to 140"]),
                                                 teller=meta["teller"]),
                 f"My diary entry for today:\n\n{diary}", 0.9, 450)
    return {"id": index, "diary": diary, "story": story, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/v1")
    parser.add_argument("--model", default="writer")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "diary.jsonl")
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
            except Exception as error:  # truncations and timeouts are skipped; rerun retries
                failed += 1
                print(f"skipped: {error}", flush=True)
            if n % 100 == 0:
                print(f"{len(done) + n}/{args.count}, {failed} skipped", flush=True)
    print(f"done: {len(todo) - failed} written, {failed} skipped")


if __name__ == "__main__":
    main()
