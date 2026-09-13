import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

out = Path(__file__).resolve().parent
assert (out / "status.txt").read_text().strip() == "complete"
remote = {line.split()[1]: line.split()[0] for line in (out / "hashes.txt").read_text().splitlines()}
for name, digest in json.loads((out / "local-hashes.json").read_text()).items():
    assert remote[name] == digest
names = ["1-normal", "2-profile", "3-profile", "4-normal"]
records = {name: [json.loads(line) for line in (out / (name + ".json")).read_text().splitlines()] for name in names}
hashes = {hashlib.sha256((out / (name + ".txt")).read_bytes()).hexdigest() for name in names}
assert hashes == {"4ea6b63a2f2397457b1066150e6e49e40d3102a26513077bf2193e70dcd2026e"}
for lines in records.values():
    m = lines[0]
    assert (m["prompt_tokens"], m["decode_tokens"], m["generated_tokens"], m["prefill_batch"], m["threads"], m["stop"]) == (26, 63, 64, 8, 2, "limit")

def stats(values):
    return {"mean": mean(values), "sample_std": stdev(values)}

summary = {"source_commit": "8f8c26ef1fee6e8fdeb9c645e9305bf7fa0b4fa4", "output_sha256": next(iter(hashes)), "timings": {}, "profile_ms_per_token": {}}
for variant, runs in {"normal": ["1-normal", "4-normal"], "profile": ["2-profile", "3-profile"]}.items():
    summary["timings"][variant] = {"runs": runs, **{key: stats([records[r][0][key] for r in runs]) for key in ("prefill_ms", "ttft_ms", "decode_tok_s", "peak_rss_mib")}}
for phase, tokens in (("prefill", 26), ("decode", 63)):
    values = [records[name][1]["profile"][phase] for name in ("2-profile", "3-profile")]
    for v in values:
        assert v["tokens"] == tokens
        assert abs(sum(v["steps_ms"].values()) + v["unattributed_ms"] - v["total_ms"]) < 0.02
    steps = {key: stats([v["steps_ms"][key] / tokens for v in values]) for key in values[0]["steps_ms"]}
    totals = {key: stats([v[key] / tokens for v in values]) for key in ("total_ms", "quantize_ms", "waiting_ms", "worker_ms", "unattributed_ms")}
    summary["profile_ms_per_token"][phase] = {"tokens_per_run": tokens, "steps_ms": steps, **totals}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for phase, values in summary["profile_ms_per_token"].items():
    print(phase, {key: round(value["mean"], 3) for key, value in values["steps_ms"].items()})
    print({key: round(values[key]["mean"], 3) for key in ("total_ms", "quantize_ms", "waiting_ms", "worker_ms")})
print(json.dumps(summary["timings"], indent=2))
