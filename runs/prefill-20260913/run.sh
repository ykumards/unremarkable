#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/prefill-20260913
mkdir "$out"
cp benchmarks/run-prefill-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/rung06-host-armv7 benchmarks/rung07-prefill-armv7 benchmarks/test-prefill-armv7 benchmarks/prefill-fp32.bin benchmarks/prefill-q8.bin models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
./benchmarks/test-prefill-armv7 benchmarks/prefill-fp32.bin > "$out/test-fp32.txt" 2>&1
./benchmarks/test-prefill-armv7 benchmarks/prefill-q8.bin > "$out/test-q8.txt" 2>&1
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  batch=$2
  limit=$3
  binary=./benchmarks/rung07-prefill-armv7
  if [ "$batch" = old ]; then
    binary=./benchmarks/rung06-host-armv7
    set --
  else
    set -- -b "$batch"
  fi
  "$binary" models/smollm2-135m-q8.bin "$@" -j 2 -z models/smollm2-135m.tok -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-old old 16
run warmup-b0 0 16
run warmup-b1 1 16
run warmup-b8 8 16
run 0-old old 64
run 1-b0 0 64
run 2-b1 1 64
run 3-b8 8 64
run 4-b8 8 64
run 5-b1 1 64
run 6-b0 0 64
sha256sum "$out"/[0-6]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
