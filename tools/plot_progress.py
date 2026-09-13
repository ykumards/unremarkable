# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<3.11"]
# ///
"""Render the README chart from archived runs: uv run tools/plot_progress.py."""
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Explicit selections exclude warmups and superseded/invalid experiments.
RUNGS = [
    ("FP32", "naive-fp32-20260912b", ["1-naive", "4-naive"]),
    ("Prefetch", "fp32-neon-20260912b", ["9-scalar-prefetch-fixed"]),
    ("NEON", "fp32-neon-20260912b", ["4-neon-prefetch", "5-neon-prefetch"]),
    ("Q8", "q8-neon-20260912", ["1-q8-neon"]),
    ("Two cores", "q8-threads-20260913b", ["3-two", "4-two"]),
    ("Six-op\nQ8 dot", "q8-six-op-20260913", ["2-six", "3-six"]),
    ("Batched\nprefill", "prefill-20260913", ["3-b8", "4-b8"]),
    ("RoPE +\nquantize", "rung08-exact-20260913", ["2-rung08", "3-rung08"]),
    ("Attention", "attention-heads-20260913", ["3-threaded", "5-threaded", "7-threaded"]),
    ("Grouped\nprojections", "grouping-20260913", ["2-candidate", "3-candidate", "6-candidate"]),
]


def measurements():
    points = []
    for label, directory, names in RUNGS:
        rows = [json.loads((ROOT / "runs" / directory / f"{name}.json").read_text())
                for name in names]
        if not all(row["generated_tokens"] == 64 and row["stop"] == "limit" for row in rows):
            raise ValueError(f"Unexpected workload in {directory}")
        points.append(mean(row["decode_tok_s"] for row in rows))
    return points


def render(theme, values):
    dark = theme == "dark"
    background, foreground = ("#111820", "#edf2f7") if dark else ("#faf9f6", "#222c36")
    muted, grid = ("#96a5b5", "#2b3540") if dark else ("#697582", "#e3e6e8")
    accent = "#f3ad58" if dark else "#bf6b18"
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "svg.fonttype": "none",
                         "svg.hashsalt": "unremarkable-progress", "font.size": 10})
    fig = plt.figure(figsize=(12, 6.6), facecolor=background)
    ax = fig.add_axes((0.075, 0.23, 0.87, 0.57), facecolor=background)
    x = list(range(1, len(values) + 1))
    ax.set_xlim(0.65, len(values) + 0.4)
    ax.set_ylim(0, 8.35)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=grid, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="both", length=0, colors=muted, pad=10)
    ax.set_xticks(x, [f"{i:02d}\n{rung[0]}" for i, rung in enumerate(RUNGS, 1)], fontsize=9)
    ax.fill_between(x, values, color=accent, alpha=0.09)
    ax.plot(x, values, color=accent, linewidth=2.8, solid_capstyle="round", zorder=3)
    ax.scatter(x, values, s=40, facecolor=background, edgecolor=accent, linewidth=1.8, zorder=4)
    ax.scatter([4, 5, 10], [values[i - 1] for i in [4, 5, 10]],
               s=55, facecolor=accent, edgecolor=background, linewidth=1.5, zorder=5)
    for i, value in enumerate(values, 1):
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, color=foreground, weight="bold" if i == 10 else "normal")
    ax.annotate("Q8 weights", (3.75, values[2] + 0.75 * (values[3] - values[2])), xytext=(3.2, 4.9),
                fontsize=11, weight="bold", color=foreground,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8,
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=6"})
    ax.annotate("Second core", (4.65, values[3] + 0.65 * (values[4] - values[3])), xytext=(4.4, 7.25),
                fontsize=11, weight="bold", color=foreground,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8,
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=6"})
    ax.text(7, 5.35, "Prefill: 33% less wait", ha="center", color=muted, fontsize=9)
    fig.text(0.075, 0.925, "Unremarkable performance", color=foreground, fontsize=21, weight="bold")
    fig.text(0.075, 0.88, "SmolLM2-135M · reMarkable 2 · measured on the tablet", color=muted, fontsize=11)
    fig.text(0.945, 0.925, f"{values[-1] / values[0]:.1f}×", ha="right", color=accent, fontsize=26, weight="bold")
    fig.text(0.945, 0.88, "initial throughput", ha="right", color=muted, fontsize=10)
    fig.text(0.075, 0.815, "DECODE · TOKENS / SECOND", color=muted, fontsize=8.5, weight="bold")
    fig.text(0.075, 0.07, "Historical means across different runs. Rung 07 optimizes prefill; decode varies between runs.",
             color=muted, fontsize=9)
    fig.savefig(ROOT / "assets" / f"performance-{theme}.svg", metadata={"Date": None})
    fig.savefig(ROOT / "build" / f"performance-{theme}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "build").mkdir(exist_ok=True)
    values = measurements()
    for theme in ("light", "dark"):
        render(theme, values)
    print("Rendered assets/performance-{light,dark}.svg from archived runs.")
