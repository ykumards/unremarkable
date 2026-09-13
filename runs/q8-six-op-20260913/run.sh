#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/q8-six-op-20260913
mkdir "$out"
cp benchmarks/run-q8-six-op-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/rung05-eight-op-armv7 benchmarks/rung06-six-op-armv7 benchmarks/test-q8-rung06 models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
./benchmarks/test-q8-rung06 > "$out/kernel-test.txt" 2>&1
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  variant=$2
  limit=$3
  case "$variant" in
    eight) binary=./benchmarks/rung05-eight-op-armv7 ;;
    six) binary=./benchmarks/rung06-six-op-armv7 ;;
  esac
  "$binary" models/smollm2-135m-q8.bin -j 2 -z models/smollm2-135m.tok -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-eight eight 16
run warmup-six six 16
run 1-eight eight 64
run 2-six six 64
run 3-six six 64
run 4-eight eight 64
sha256sum "$out"/[1-4]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
