#!/bin/sh
# Turns a diary entry into a bedtime story, streamed into state files the UI polls.
# Text lands in state/out.txt as it is generated; state/status tracks progress.
set -u

DIR=/home/root/unremarkable
STATE="$DIR/state"
MODEL="$DIR/models/diary-q8.bin"
TOKENIZER="$DIR/models/diary-q8.tok"
LOCK="$STATE/lock"
LIMIT="${2:-240}"

# Recognized handwriting is arbitrary text; 1000 characters keeps the prompt
# near 250 tokens, leaving room for the story in the 512-token context.
ENTRY=$(printf '%s' "${1:-}" | tr -d '\000' | cut -c1-1000)
[ -z "$ENTRY" ] && { echo "usage: ask.sh \"<diary entry>\" [max_tokens]" >&2; exit 2; }

# Must match diary_prompt in train/prepare.py, the prompt the model was tuned on.
PROMPT=$(printf 'My diary entry for today:\n\n%s\n\nTell me a short bedtime story about it.' "$ENTRY")

mkdir -p "$STATE"

# A different seed every run, so the same entry gets a new story each time.
SEED=$(( ($(date +%s) + $$) % 2147483647 ))
[ "$SEED" -lt 1 ] && SEED=1

# A run killed before its trap fires (dropped ssh, reboot) leaves the lock
# behind and would disable every later run, so clear an ownerless one first.
if [ -d "$LOCK" ] && ! pidof unremarkable >/dev/null 2>&1; then
    rmdir "$LOCK" 2>/dev/null
fi

# One generation at a time: a second run would compete for both cores.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "A generation is already running." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

date +%s > "$STATE/started"
printf '%s' "$ENTRY" > "$STATE/prompt"
: > "$STATE/out.txt"
: > "$STATE/meta.json"   # cleared so the UI never shows the previous run's stats
printf 'running' > "$STATE/status"

# No system turn: the fine-tune was trained without one.
"$DIR/unremarkable" "$MODEL" -z "$TOKENIZER" -c 512 -y "" -j 2 -b 8 \
    -t 0.8 -p 0.9 -s "$SEED" -i "$PROMPT" -n "$LIMIT" \
    > "$STATE/out.txt" 2> "$STATE/meta.json"

printf 'done' > "$STATE/status"
