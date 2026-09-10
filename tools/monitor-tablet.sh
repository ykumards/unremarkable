#!/bin/sh
# Run one engine command in a new directory. Stream health samples over SSH so
# the Mac retains them even if the tablet resets. Engine output stays in files.
set -u
if [ "$#" -lt 2 ]; then
    echo "usage: monitor-tablet.sh NEW_RUN_DIRECTORY COMMAND [ARGUMENTS...]" >&2
    exit 2
fi
RUN=$1
shift
if pidof unremarkable benchmark-armv7 profile-armv7 >/dev/null 2>&1; then
    echo "An inference process is already running." >&2
    exit 1
fi
mkdir "$RUN" || exit 1
cd "$RUN" || exit 1
printf '%s\n' "$@" > command.txt
printf 'epoch,available_kib,rss_kib,temp_millic,cpu_khz,output_bytes\n' | tee health.csv
"$@" > output.txt 2> metrics.json &
ENGINE_PID=$!
trap 'kill "$ENGINE_PID" 2>/dev/null; wait "$ENGINE_PID" 2>/dev/null; exit 130' INT TERM HUP
while kill -0 "$ENGINE_PID" 2>/dev/null; do
    AVAILABLE=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    RSS=$(awk '/^VmRSS:/ {print $2}' "/proc/$ENGINE_PID/status" 2>/dev/null)
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    BYTES=$(wc -c < output.txt)
    printf '%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$AVAILABLE" "${RSS:-0}" \
        "$TEMP" "$FREQ" "$BYTES" | tee -a health.csv
    # This guard reduces risk; two-second sampling cannot prevent every OOM.
    if [ "$AVAILABLE" -lt 65536 ]; then
        echo "Stopped: less than 64 MiB system memory available." > stopped.txt
        kill "$ENGINE_PID" 2>/dev/null
        break
    fi
    sleep 2
done
wait "$ENGINE_PID"
STATUS=$?
printf '%s\n' "$STATUS" > exit-status.txt
exit "$STATUS"
