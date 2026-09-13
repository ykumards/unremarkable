import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

out = Path("runs/prefill-20260913")
assert (out / "status.txt").read_text().strip() == "complete"
local = json.loads((out / "local-hashes.json").read_text())
remote = {line.split()[1]: line.split()[0] for line in (out / "hashes.txt").read_text().splitlines()}
names = {"build/rung06-armv7": "benchmarks/rung06-host-armv7", "build/unremarkable-armv7": "benchmarks/rung07-prefill-armv7"}
for path, digest in local.items():
    remote_path = names.get(path, "benchmarks/" + Path(path).name)
    assert remote[remote_path] == digest, remote_path

hashes = {path.stem: hashlib.sha256(path.read_bytes()).hexdigest() for path in out.glob("[0-6]-*.txt")}
assert len(hashes) == 7 and len(set(hashes.values())) == 1
assert next(iter(hashes.values())) == "4ea6b63a2f2397457b1066150e6e49e40d3102a26513077bf2193e70dcd2026e"
summary = {"parent_commit": "a247971", "source_commit": "3fecb0b", "output_sha256": next(iter(hashes.values())), "comparisons": {}}
for name, runs in {"parent": ["0-old"], "legacy_b0": ["1-b0", "6-b0"], "last_logits_b1": ["2-b1", "5-b1"], "batched_b8": ["3-b8", "4-b8"]}.items():
    records = [json.loads((out / (r + ".json")).read_text()) for r in runs]
    assert all(r["prompt_tokens"] == 26 and r["generated_tokens"] == 64 and r["decode_tokens"] == 63 and r["stop"] == "limit" for r in records)
    values = {"runs": runs}
    for field in ("prefill_ms", "ttft_ms", "decode_tok_s"):
        numbers = [r[field] for r in records]
        values[field + "_mean"] = mean(numbers)
        values[field + "_sample_std"] = stdev(numbers) if len(numbers) > 1 else None
    rates = [r["prompt_tokens"] * 1000 / r["prefill_ms"] for r in records]
    values["prefill_tok_s_mean"] = mean(rates)
    values["prefill_tok_s_sample_std"] = stdev(rates) if len(rates) > 1 else None
    values["peak_rss_mib_max"] = max(r["peak_rss_mib"] for r in records)
    summary["comparisons"][name] = values
c = summary["comparisons"]
summary["ttft_reduction_fraction_vs_legacy"] = 1 - c["batched_b8"]["ttft_ms_mean"] / c["legacy_b0"]["ttft_ms_mean"]
summary["ttft_reduction_fraction_vs_last_logits"] = 1 - c["batched_b8"]["ttft_ms_mean"] / c["last_logits_b1"]["ttft_ms_mean"]
summary["prefill_speedup_vs_legacy"] = c["legacy_b0"]["prefill_ms_mean"] / c["batched_b8"]["prefill_ms_mean"]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "local-hashes.json").write_text(json.dumps(local, indent=2) + "\n")
print(json.dumps(summary, indent=2))
