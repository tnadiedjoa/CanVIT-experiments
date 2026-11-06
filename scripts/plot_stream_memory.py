"""Figure of scripts/stream_memory.py: a slot memory trained on 4-chunk streams, tested on one informative chunk
followed by 256 noise chunks. Writes outputs/stream_memory.png.

    .venv/bin/python scripts/plot_stream_memory.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import GRID, INK, MUTED  # noqa: E402  (also sets the rcParams)
from plot_sink_memory import KINDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results/stream_memory", ROOT / "outputs"
# (model, label, Write kind for the hue, line style)
WRITES = [("softmax_s0", "softmax", "softmax", "-"), ("out_s0", "softmax + output projection", "out", "-"),
          ("null_s0", "zero-value null key", "null", "-"), ("null_open_s0", "null key open by default", "null", "--"),
          ("sigmoid_s0", "sigmoid", "sigmoid", "-"), ("gate_s0", "output gate", "gate", "-"),
          ("delta_s0", "delta", "delta", "-")]
# (model, evaluation suffix, hue, line style, label): the softmax variants share its hue, the oracle is in ink.
FIXES = [("softmax_s0", "", KINDS["softmax"][1], "-", "softmax"),
         ("softmax_pnoise1_s0", "", KINDS["softmax"][1], "--", "softmax + penalty in noise chunks"),
         ("softmax_pall1_s0", "", KINDS["softmax"][1], "-.", "softmax + penalty in every chunk"),
         ("null_open_s0", "", KINDS["null"][1], "--", "null key open by default"),
         ("softmax_s0", "_abstain", INK, ":", "softmax, no Write in noise chunks (oracle)")]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def axis(ax: plt.Axes, ylabel: str, title: str, log_y: bool = False) -> None:
    ax.axvline(4.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    ax.set_xticks([], minor=True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("chunks seen (log scale; training streams: 4)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)


def main() -> None:
    if load("softmax_s0_noise") is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for name, label, kind, style in WRITES:
        run = load(f"{name}_noise")
        if run is None:
            continue
        x = range(1, len(run["accuracy"]) + 1)
        axes[0].plot(x, run["accuracy"], color=KINDS[kind][1], linestyle=style, linewidth=1.8, label=label)
        axes[2].plot(x, run["memory_centered_norm"], color=KINDS[kind][1], linestyle=style, linewidth=1.8)
    for name, suffix, color, style, label in FIXES:
        run = load(f"{name}_noise{suffix}")
        if run is None:
            continue
        axes[1].plot(range(1, len(run["accuracy"]) + 1), run["accuracy"], color=color, linestyle=style,
                     linewidth=1.8, label=label)
    axis(axes[0], "recall accuracy on the first chunk's 8 keys", "Every Write kind, after one informative chunk")
    axis(axes[1], "recall accuracy", "Softmax: what stops the forgetting")
    axis(axes[2], "slot norm after removing its mean", "What LayerNorm divides by (Write kinds of the left panel)",
         log_y=True)
    for ax in axes[:2]:
        ax.set_ylim(0, 1.02)
        ax.axhline(1 / 64, color=MUTED, linewidth=1)
        ax.text(1.05, 1 / 64 + 0.015, "chance", fontsize=7, color=MUTED)
    for ax in axes[:2]:
        ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
    fig.suptitle("A slot memory written by cross-attention from a stream: one chunk of facts, then noise (seed 0)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "stream_memory.png", dpi=150)
    print("wrote", FIGURES / "stream_memory.png")


if __name__ == "__main__":
    main()
