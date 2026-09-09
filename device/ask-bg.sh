#!/bin/sh
# Detach ask.sh so QML's executeCommand returns immediately. Arguments are
# passed through as argv, never through a shell string, so recognised text
# cannot be interpreted as shell syntax.
nohup /home/root/unremarkable/ask.sh "$@" >/dev/null 2>&1 &
