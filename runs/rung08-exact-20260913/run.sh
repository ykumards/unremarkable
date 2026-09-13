#!/bin/sh
set -eu
cd /home/root/unremarkable
out=benchmarks/rung08-exact-20260913
mkdir "$out"
cp benchmarks/run-rung08-exact-20260913.sh "$out/run.sh"
{ date -u; uname -a; uptime; free -m; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/online; systemctl is-active xochitl; } > "$out/environment.txt"
sha256sum benchmarks/prefill-normal-8f8c26e benchmarks/rung08-exact-armv7 benchmarks/test-rung08-exact-armv7 models/smollm2-135m-q8.bin models/smollm2-135m.tok > "$out/hashes.txt"
./benchmarks/test-rung08-exact-armv7 > "$out/test-kernels.txt" 2>&1
prompt='Tell me a short story about a lighthouse keeper who discovers a message in a bottle.'
run() {
  label=$1
  variant=$2
  limit=$3
  case "$variant" in
    main) binary=./benchmarks/prefill-normal-8f8c26e ;;
    rung08) binary=./benchmarks/rung08-exact-armv7 ;;
  esac
  "$binary" models/smollm2-135m-q8.bin -z models/smollm2-135m.tok -b 8 -j 2 -c 512 -y '' -t 0 -s 1 -i "$prompt" -n "$limit" > "$out/$label.txt" 2> "$out/$label.json"
  printf '%s: ' "$label"
  cat "$out/$label.json"
}
run warmup-main main 16
run warmup-rung08 rung08 16
run 1-main main 64
run 2-rung08 rung08 64
run 3-rung08 rung08 64
run 4-main main 64
sha256sum "$out"/[1-4]-*.txt > "$out/output-hashes.txt"
{ date -u; uptime; free -m; cat /proc/sys/kernel/random/boot_id; systemctl is-active xochitl; } > "$out/environment-after.txt"
printf 'complete\n' > "$out/status.txt"
