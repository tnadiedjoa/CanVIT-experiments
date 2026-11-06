"""Figures of the write-restriction study across the released CanViT family (results/ablations), of the choice of the
outside cells a Write reaches (results/grouping, results/sink) and of ImageNet-1k by glimpse (results/imagenet), in
the style of scripts/plot_results.py.

    .venv/bin/python scripts/plot_family.py

Each figure is drawn only when its results exist.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import GRID, INK, MUTED  # noqa: E402  (also sets the rcParams)

ROOT = Path(__file__).resolve().parents[1]
ABL, FIGURES = ROOT / "results/ablations", ROOT / "outputs"
MODELS = {
    "in21k": "flagship (IN-21k)", "in1k": "IN-1k pretraining", "abl-baseline": "ablation baseline",
    "abl-qkvo-dcan256": "QKVO, D_can 256", "abl-qkvo-dcan384": "QKVO, D_can 384", "abl-dcan256": "D_can 256",
    "abl-rw-stride6": "1 Read / 1 Write", "abl-no-reads": "no Reads", "abl-no-vpe": "no VPE", "abl-vit-s": "ViT-S backbone",
    "abl-no-bptt": "no BPTT", "abl-no-dense": "no dense loss", "abl-no-fiid-1riid": "no F-IID (1 R-IID)",
    "abl-no-fiid-2riid": "no F-IID (2 R-IID)",
}
WRITES = {"released": ("released Writes", "#8a8985"), "local21": ("local Writes from level 3", "#2a78d6"),
          "object": ("object Writes", "#6250d6"), "objtop": ("object Writes, fixed reach", "#1baf7a")}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def curve(values: list) -> tuple[np.ndarray, np.ndarray]:
    """Glimpses and values where the mIoU was evaluated (runs with an evaluation stride leave nulls)."""
    t = np.array([i for i, v in enumerate(values) if v is not None])
    return t, np.array([values[i] for i in t])


def label_ends(ax: plt.Axes, ends: list[tuple[float, float, str]], gap: float) -> None:
    """End-of-line values, nudged apart vertically by at least `gap` (data units) so they do not overlap."""
    placed: list[float] = []
    for x, y, color in sorted(ends, key=lambda e: e[1]):
        label_y = max(y, placed[-1] + gap) if placed else y
        placed.append(label_y)
        ax.plot([x], [y], "o", color=color, markersize=4)
        ax.annotate(f"{y:.2f}", (x, y), xytext=(14, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK) if label_y == y else ax.annotate(
            f"{y:.2f}", (x, y), xytext=(x + 2.5, label_y), textcoords="data", va="center", fontsize=8, color=INK,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 1})


def family_changes() -> None:
    """Per pretrained model: mIoU change from glimpse 20 to glimpse 84 of C2F (32×32 canvas), for each Write rule."""
    rows = []
    for model, name in MODELS.items():
        runs = {w: load(ABL / f"{model}_c2f_c32_t85_{w}.json") for w in WRITES}
        if runs["released"] is None:
            continue
        t20 = runs["released"]["miou"][20]
        rows.append((name, t20, {w: r["miou"][84] - t20 for w, r in runs.items() if r is not None}))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
    for i, (name, t20, changes) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([min(changes.values()), max(changes.values())], [y, y], color=GRID, linewidth=2, zorder=1)
        for w, change in changes.items():
            ax.plot(change, y, "o", color=WRITES[w][1], markersize=7, zorder=2,
                    markeredgecolor="white", markeredgewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{name} ({t20:.1f})" for name, t20, _ in rows][::-1], fontsize=8)
    ax.set_xlabel("mIoU change from glimpse 20 to glimpse 84 (points)")
    ax.grid(axis="y", visible=False)
    for w, (label, color) in WRITES.items():
        ax.plot([], [], "o", color=color, label=label, markersize=7)
    ax.legend(fontsize=7.5, loc="lower center", ncol=4, bbox_to_anchor=(0.4, 1.0))
    fig.suptitle("Every released pretraining loses mIoU past C2F's level 2; local Writes and object Writes with a "
                 "fixed reach keep improving\n(in brackets: mIoU at glimpse 20)", fontsize=8.5, x=0.01, ha="left", color=MUTED, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_changes.png", bbox_inches="tight")
    plt.close(fig)


def family_leak() -> None:
    """Leak measured on 128 scenes (leaked sum over own sum, level 3) against the released Writes' relative mIoU
    change from glimpse 20 to 84 (the models' mIoU ranges from 18 to 44, so absolute changes are not comparable)."""
    points = []
    for model, name in MODELS.items():
        leak, run = load(ABL / f"leak_{model}_c32.json"), load(ABL / f"{model}_c2f_c32_t85_released.json")
        if leak is None or run is None:
            continue
        points.append((name, leak["level3_leaked_over_own"], 100 * (run["miou"][84] - run["miou"][20]) / run["miou"][20]))
    if len(points) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x, y = np.array([p[1] for p in points]), np.array([p[2] for p in points])
    ax.plot(x, y, "o", color=WRITES["released"][1], markersize=7, markeredgecolor="white", markeredgewidth=1)
    offsets = {"QKVO, D_can 384": (5, -10), "QKVO, D_can 256": (4, 6), "flagship (IN-21k)": (-5, 6), "IN-1k pretraining": (5, -10),
               "ablation baseline": (5, -10)}
    for name, xi, yi in points:
        dx, dy = offsets.get(name, (5, 2))
        ax.annotate(name, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK,
                    ha="right" if dx < 0 else "left")
    rank = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rank(x), rank(y))[0, 1]
    ax.set_xlabel("leaked residual over own residual, summed over level 3")
    ax.set_ylabel("relative mIoU change, glimpse 20 → 84 (%)")
    ax.set_title(f"The larger the leak, the larger the drop (Spearman ρ = {rho:.2f}, {len(points)} models)",
                 fontsize=8.5, loc="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_leak.png", bbox_inches="tight")
    plt.close(fig)


def which_cells() -> None:
    """C2F with a 64×64 canvas, flagship: which outside cells each glimpse writes. Left, rules without labels; right,
    the ground-truth oracle and its two halves (cells the probe gets right or wrong before the glimpse)."""
    sink, grouping = ROOT / "results/sink", ROOT / "results/grouping"
    released = ("released Writes (every cell)", grouping / "c2f_c64_t85_released.json", "#8a8985", "-")
    local = ("local Writes from level 3 (no outside cell)", grouping / "c2f_c64_t85_local21.json", "#2a78d6", "-")
    obj = ("object Writes (canvas similarity)", grouping / "c2f_c64_t85_object0.9_from5_late0.97.json", "#6250d6", "-")
    panels = {
        "Without labels": [released,
                           ("random cells, object Writes' share", sink / "c2f_c64_t85_random_like_object0.95_from5.json",
                            "#eda100", "-"),
                           local,
                           ("predicted class shared with the crop", grouping / "c2f_c64_t85_predicted_from5.json",
                            "#1baf7a", "-"),
                           ("probe probabilities similar (≥ 0.99)", grouping / "c2f_c64_t85_probs0.99_from5.json",
                            "#eb6834", "-"),
                           obj],
        "Ground-truth class shared with the crop (oracle)": [
            local, obj,
            ("all such cells", grouping / "c2f_c64_t85_oracle_from5.json", INK, "--"),
            ("only those the probe gets right", grouping / "c2f_c64_t85_oracle_correct_from5.json", "#e34948", "--"),
            ("only those the probe gets wrong", grouping / "c2f_c64_t85_oracle_wrong_from5.json", "#008300", "--")],
    }
    if not all(path.exists() for curves in panels.values() for _, path, _, _ in curves):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, curves) in zip(axes, panels.items()):
        ends = []
        for label, path, color, style in curves:
            t, values = curve(load(path)["miou"])
            ax.plot(t, values, style, color=color, label=label, linewidth=1.8)
            ends.append((t[-1], values[-1], color))
        label_ends(ax, ends, gap=0.3)
        for start in (5, 21):
            ax.axvline(start, color=GRID, linewidth=1, zorder=0)
        ax.set_xlim(-1, 97)
        ax.set_ylim(41, 49)
        ax.set_xlabel("glimpse t (C2F: level 2 from 5, level 3 from 21)")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("mIoU (%)")
    fig.suptitle("Writing outside the crop helps a cell whose representation matches the crop, and the oracle gains "
                 "only on the cells the model still gets wrong", fontsize=9, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "write_which_cells.png", bbox_inches="tight")
    plt.close(fig)


def policies() -> None:
    """Flagship, 64×64 canvas, 85 glimpses: released, local and object Writes under C2F, EG-C2F and R-IID (the C2F
    settings unchanged), and, on C2F, averaging the probe's probabilities over the glimpses instead."""
    folder, grouping = ROOT / "results/policies", ROOT / "results/grouping"
    rules = [("released Writes", "released", "#8a8985"), ("local Writes from glimpse 21", "local21", "#2a78d6"),
             ("object Writes", "object", "#6250d6")]
    panels = {"C2F": {label: grouping / f"c2f_c64_t85_{name}.json" for label, name, _ in rules[:2]}
              | {rules[2][0]: grouping / "c2f_c64_t85_object0.9_from5_late0.97.json"},
              "EG-C2F (tiles by probe entropy)": {label: folder / f"egc2f_c64_t85_{name}.json" for label, name, _ in rules},
              "R-IID (the pretraining policy)": {label: folder / f"riid_c64_t85_{name}.json" for label, name, _ in rules}}
    readout = {"released Writes, mean of the probabilities from glimpse 21":
               (ROOT / "results/readout/c2f_c64_t85_released_mean_from21.json", "#eb6834", "-"),
               "released Writes, mean from glimpse 0": (ROOT / "results/readout/c2f_c64_t85_released_mean_from0.json",
                                                        "#eda100", "-")}
    if not all(path.exists() for curves in panels.values() for path in curves.values()):
        return
    colors = {label: color for label, _, color in rules}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (title, curves) in zip(axes, panels.items()):
        ends = []
        for label, path in curves.items():
            t, values = curve(load(path)["miou"])
            ax.plot(t, values, color=colors[label], label=label, linewidth=1.8)
            ends.append((t[-1], values[-1], colors[label]))
        if title == "C2F":
            for label, (path, color, style) in readout.items():
                if path.exists():
                    t, values = curve(load(path)["miou"])
                    ax.plot(t, values, style, color=color, label=label, linewidth=1.4)
                    ends.append((t[-1], values[-1], color))
        label_ends(ax, ends, gap=0.3)
        ax.set_xlim(-1, 97)
        ax.set_ylim(40, 47.5)
        ax.set_xlabel("glimpse t")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("mIoU (%)")
    fig.suptitle("The Write rules carry over to other policies unchanged; averaging the predictions recovers part of "
                 "C2F's drop but never rises", fontsize=9, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "policies.png", bbox_inches="tight")
    plt.close(fig)


def scaling() -> None:
    """Flagship, 64×64 canvas, 257 glimpses of R-IID and F-IID (the pretraining policies): released and object
    Writes, against the glimpses on a log scale."""
    folder = ROOT / "results/scaling"
    rules = [("released Writes", "released", "#8a8985"), ("object Writes", "object", "#6250d6")]
    panels = {"R-IID": "riid", "F-IID (full view, then R-IID)": "fiid"}
    if not all((folder / f"{p}_c64_t257_{name}.json").exists() for p in panels.values() for _, name, _ in rules):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, (title, policy) in zip(axes, panels.items()):
        ends = []
        for label, name, color in rules:
            t, values = curve(load(folder / f"{policy}_c64_t257_{name}.json")["miou"])
            ax.plot(t + 1, values, color=color, label=f"{label}, seed 0", linewidth=1.8)
            ends.append((t[-1] + 1, values[-1], color))
            seed1 = folder / f"{policy}_c64_t257_{name}_s1.json"
            if seed1.exists():
                t, values = curve(load(seed1)["miou"])
                ax.plot(t + 1, values, color=color, label=f"{label}, seed 1", linewidth=1.2, linestyle="--")
        label_ends(ax, ends, gap=0.3)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 21, 85, 257], ["1", "5", "21", "85", "257"])
        ax.set_ylim(40, 47)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlabel("glimpses seen (log scale)")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("mIoU (%)")
    fig.suptitle("Past the training horizon, released Writes stall or decline; object Writes keep improving",
                 fontsize=9, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def imagenet() -> None:
    """ImageNet-1k top-1 by glimpse (fine-tuned classifier, C2F, 32×32 canvas, 10,000 images)."""
    folder = ROOT / "results/imagenet"
    released = load(folder / "c2f_c32_t85_released_evidence.json") or load(folder / "c2f_c32_t85_released.json")
    if released is None:
        return
    curves = {"recurrent CLS, released Writes": (released["top1"], "#8a8985", "-")}
    local = load(folder / "c2f_c32_t85_local21.json")
    if local is not None:
        curves["recurrent CLS, local Writes from level 3"] = (local["top1"], "#2a78d6", "-")
    if "top1_mean_probs" in released:
        curves["mean of its probabilities so far, released Writes"] = (released["top1_mean_probs"], "#eb6834", "-")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = []
    for label, (values, color, style) in curves.items():
        t = np.arange(len(values))
        ax.plot(t, values, style, color=color, label=label, linewidth=1.8)
        ends.append((t[-1], values[-1], color))
    label_ends(ax, ends, gap=0.12)
    ax.set_xlim(-1, 95)
    ax.set_xlabel("glimpse t")
    ax.set_ylabel("top-1 (%)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("ImageNet-1k: the recurrent CLS forgets at level 3, local Writes or not;\naveraging its "
                 "predictions over the glimpses recovers most of the drop",
                 fontsize=8.5, loc="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "imagenet_glimpses.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    family_changes()
    family_leak()
    which_cells()
    policies()
    scaling()
    imagenet()
