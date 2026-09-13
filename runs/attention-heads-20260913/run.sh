#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/attention-heads-20260913
mkdir "$out"
cp benchmarks/run-attention-heads-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/attention-baseline-armv7 benchmarks/attention-exact-armv7 benchmarks/attention-threaded-armv7 models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
./benchmarks/attention-heads-test-armv7 > "$out/test-attention.txt"
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  variant=$2
  limit=$3
  ./benchmarks/attention-$variant-armv7 models/smollm2-135m-q8.bin -z models/smollm2-135m.tok -b 8 -j 2 -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-baseline baseline 16
run warmup-exact exact 16
run warmup-threaded threaded 16
run 1-baseline baseline 64
run 2-exact exact 64
run 3-threaded threaded 64
run 4-exact exact 64
run 5-threaded threaded 64
run 6-baseline baseline 64
run 7-threaded threaded 64
run 8-baseline baseline 64
run 9-exact exact 64
sha256sum "$out"/[1-9]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
