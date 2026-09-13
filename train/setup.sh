#!/bin/sh
# Python environment and TinyStories-Instruct (CDLA-Sharing-1.0) next to this
# script. Delete .venv and data to undo.
set -u
cd "$(dirname "$0")"
mkdir -p data
(
  base=https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main
  curl -sfL -o data/TinyStories-Instruct-valid.txt "$base/TinyStories-Instruct-valid.txt" &&
    curl -sfL -o data/TinyStories-Instruct-train.txt "$base/TinyStories-Instruct-train.txt" &&
    echo ok > data/downloaded
) &
download=$!
if python3 -m venv .venv && .venv/bin/python -m pip install -q --upgrade pip &&
  .venv/bin/pip install -q -r requirements.txt; then
  .venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
else
  echo "ENV FAILED"
fi
wait "$download"
ls -la data
echo "SETUP DONE"
