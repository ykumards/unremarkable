#!/usr/bin/env python3
"""Repeat an engine command; retain outputs/metrics and report mean + sample std.

Pass the command after --. It can invoke the engine directly or through ssh.
The engine must write text to stdout and one JSON metrics record to stderr.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.warmup < 0 or args.runs < 1:
        parser.error("provide a command, nonnegative warmups, and at least one run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measured = []
    hashes = set()
    # Refuse to overwrite a previous measurement.
    with args.output.open("x") as stream:
        for index in range(args.warmup + args.runs):
            phase = "warmup" if index < args.warmup else "measured"
            print(f"Starting {phase} {index + 1}/{args.warmup + args.runs}", flush=True)
            start = time.monotonic()
            result = subprocess.run(command, capture_output=True)
            record = dict(phase=phase, command=command, returncode=result.returncode,
                          wall_seconds=time.monotonic() - start,
                          stdout=result.stdout.decode("utf-8", errors="replace"),
                          stderr=result.stderr.decode("utf-8", errors="replace"),
                          output_sha256=hashlib.sha256(result.stdout).hexdigest())
            if result.returncode == 0:
                try:
                    record["metrics"] = json.loads(result.stderr)
                except ValueError:
                    pass
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if "metrics" not in record:
                raise SystemExit(f"Command failed or did not emit JSON metrics; see {args.output}")
            hashes.add(record["output_sha256"])
            metrics = record["metrics"]
            print(f"  {metrics['generated_tokens']} tokens, stop={metrics['stop']}, "
                  f"decode={metrics['decode_tok_s']:.3f} tok/s", flush=True)
            if phase == "measured":
                measured.append(metrics)
    if len(hashes) != 1:
        raise SystemExit("Outputs differed between runs; do not combine these measurements")
    for field in ("load_ms", "prefill_ms", "ttft_ms", "decode_tok_s", "peak_rss_mib"):
        values = [run[field] for run in measured]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{field}: {statistics.mean(values):.3f} +/- {deviation:.3f}")
    print(f"output_sha256: {hashes.pop()}")
    print(f"Raw runs: {args.output}")


if __name__ == "__main__":
    main()
