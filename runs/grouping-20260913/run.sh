#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/grouping-20260913
mkdir "$out"
cp benchmarks/run-grouping-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/grouping-baseline-armv7 benchmarks/grouping-candidate-armv7 models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
./benchmarks/grouping-test-prefill-armv7 benchmarks/grouping-test-fp32.bin > "$out/test-prefill-fp32.txt"
./benchmarks/grouping-test-prefill-armv7 benchmarks/grouping-test-q8.bin > "$out/test-prefill-q8.txt"
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  variant=$2
  limit=$3
  ./benchmarks/grouping-$variant-armv7 models/smollm2-135m-q8.bin -z models/smollm2-135m.tok -b 8 -j 2 -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-baseline baseline 16
run warmup-candidate candidate 16
run 1-baseline baseline 64
run 2-candidate candidate 64
run 3-candidate candidate 64
run 4-baseline baseline 64
run 5-baseline baseline 64
run 6-candidate candidate 64
sha256sum "$out"/[1-6]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
