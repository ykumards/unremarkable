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
OPENING=""

mkdir -p "$STATE"

# A different seed and non-zero temperature every run, otherwise greedy decoding
# returns a byte-identical poem for the same subject forever.
SEED=$(( ($(date +%s) + $$) % 2147483647 ))
[ "$SEED" -lt 1 ] && SEED=1

case "${1:-}" in
  # Concrete storybook subjects, and the reply opened with "Once upon a time,"
  # so the model starts in narrative voice instead of writing *about* the
  # subject. Measured over 96 runs, every subject below came out clean 8 times
  # out of 8; unseeded prompts managed 80%, and seeding further into the action
  # dropped back to 93% because the model loses the thread mid-scene.
  poem|story)
    SUBJECTS="a lighthouse keeper
a little robot who was afraid of the dark
a cat who wanted to fly
a girl who found a strange machine in the attic
an old clock that could not stop ticking
a boy who built a boat out of newspaper
a dog who was scared of thunder
a snowman who did not want to melt
a little train that lost its way home
a mouse who lived in a library
a star that fell into someone's garden
a bear who could not sleep all winter"
    COUNT=$(printf '%s\n' "$SUBJECTS" | wc -l)
    PICK=$(( SEED % COUNT + 1 ))
    SUBJECT=$(printf '%s\n' "$SUBJECTS" | sed -n "${PICK}p")
    if [ "$1" = "story" ]; then
      PROMPT="Tell me a short story about $SUBJECT."
      OPENING="Once upon a time,"
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
# -a opens the reply for the model, so the words it continues from are ours.
if [ -n "$OPENING" ]; then
  "$DIR/unremarkable" "$MODEL" -z "$TOKENIZER" -c 512 -y "" \
      -t 0.8 -p 0.9 -s "$SEED" -i "$PROMPT" -a "$OPENING" -n "$LIMIT" \
      > "$STATE/out.txt" 2> "$STATE/meta.json"
else
  "$DIR/unremarkable" "$MODEL" -z "$TOKENIZER" -c 512 -y "" \
      -t 0.8 -p 0.9 -s "$SEED" -i "$PROMPT" -n "$LIMIT" \
      > "$STATE/out.txt" 2> "$STATE/meta.json"
fi

printf 'done' > "$STATE/status"
