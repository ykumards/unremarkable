#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/q8-threads-20260913b
mkdir "$out"
cp benchmarks/run-q8-threads-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/rung04-armv7 benchmarks/rung05-armv7 models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  variant=$2
  limit=$3
  if [ "$variant" = old ]; then
    set -- ./benchmarks/rung04-armv7
  else
    set -- ./benchmarks/rung05-armv7 -j "$variant"
  fi
  binary=$1
  shift
  "$binary" models/smollm2-135m-q8.bin "$@" -z models/smollm2-135m.tok -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-rung04 old 16
run warmup-one 1 16
run warmup-two 2 16
run 1-rung04 old 64
run 2-one 1 64
run 3-two 2 64
run 4-two 2 64
run 5-one 1 64
sha256sum "$out"/[1-5]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
