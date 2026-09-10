#!/bin/sh
# Runs one preset prompt and streams the reply into state files the UI polls.
# Text lands in state/out.txt as it is generated; state/status tracks progress.
set -u

DIR=/home/root/unremarkable
STATE="$DIR/state"
MODEL="$DIR/models/smollm2-135m-q8.bin"
TOKENIZER="$DIR/models/smollm2-135m.tok"
LOCK="$STATE/lock"
LIMIT="${2:-80}"

mkdir -p "$STATE"

# A different seed and non-zero temperature every run, otherwise greedy decoding
# returns a byte-identical poem for the same subject forever.
SEED=$(( ($(date +%s) + $$) % 2147483647 ))
[ "$SEED" -lt 1 ] && SEED=1

case "${1:-}" in
  # Subjects curated by measurement: naming a real product pulled the model into
  # marketing prose, and some subjects made it write about them instead of from them.
  poem|story)
    SUBJECTS="a tiny processor thinking
silicon and electricity
transistors switching in the dark
electrons racing through copper
an old chip in a quiet machine
the hum of a circuit board
a clock signal keeping time
a cache remembering what it saw"
    COUNT=$(printf '%s\n' "$SUBJECTS" | wc -l)
    PICK=$(( SEED % COUNT + 1 ))
    SUBJECT=$(printf '%s\n' "$SUBJECTS" | sed -n "${PICK}p")
    # Over 64 runs, stories came out well formed 98% of the time against 80% for
    # poems: this model is trained heavily on synthetic stories. A system prompt
    # makes both worse, since it is instruction tuned without one.
    if [ "$1" = "story" ]; then
      PROMPT="Tell me a short story about $SUBJECT."
    else
      PROMPT="Write a short poem about $SUBJECT."
    fi
    ;;
  about)
    # Subject comes from a selection on the page, so it is arbitrary text.
    SUBJECT=$(printf '%s' "${2:-}" | tr -d '\000' | cut -c1-200)
    [ -z "$SUBJECT" ] && { echo "ask.sh about: empty subject" >&2; exit 2; }
    PROMPT="Write a short poem about $SUBJECT."
    LIMIT="${3:-80}"
    ;;
  *) echo "usage: ask.sh {story|poem} [max_tokens]" >&2
     echo "       ask.sh about \"<subject>\" [max_tokens]" >&2; exit 2 ;;
esac

# A run killed before its trap fires (dropped ssh, reboot) leaves the lock
# behind and would disable every later run, so clear an ownerless one first.
if [ -d "$LOCK" ] && ! pidof unremarkable >/dev/null 2>&1; then
    rmdir "$LOCK" 2>/dev/null
fi

# One generation at a time. At 171 MiB memory is no longer the reason it was at
# 540 MiB; the two Cortex-A7 cores are.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "A generation is already running." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

date +%s > "$STATE/started"
printf '%s' "$PROMPT" > "$STATE/prompt"
: > "$STATE/out.txt"
: > "$STATE/meta.json"   # cleared so the UI never shows the previous run's stats
printf 'running' > "$STATE/status"

# No system prompt: it costs 21 of 37 prompt tokens and the model is instruction
# tuned without one, which cuts time to first token from ~36s to ~15s.
"$DIR/unremarkable" "$MODEL" -z "$TOKENIZER" -c 512 -y "" \
    -t 0.8 -p 0.9 -s "$SEED" -i "$PROMPT" -n "$LIMIT" \
    > "$STATE/out.txt" 2> "$STATE/meta.json"

printf 'done' > "$STATE/status"
