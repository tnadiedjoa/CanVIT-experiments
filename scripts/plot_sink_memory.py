"""Figure of the from-scratch study (scripts/sink_memory.py, results/sink_memory): the pixel error after every glimpse
for each kind of Write, under C2F and repeated full views, with the test-time rules, and the residual the Writes add
to the cells outside the glimpse's crop.

    .venv/bin/python scripts/plot_sink_memory.py            # outputs/sink_memory.png

Seed 0 is drawn; a kind or rule is skipped while its results are missing.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import GRID, INK, MUTED  # noqa: E402  (also sets the rcParams)

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results/sink_memory", ROOT / "outputs"
# Okabe–Ito hues, in a fixed order, one per kind of Write.
KINDS = {"softmax": ("softmax (CanViT's)", "#0072B2"), "out": ("softmax + output projection", "#56B4E9"),
         "null": ("zero-value null key", "#E69F00"),
         "sigmoid": ("sigmoid", "#009E73"), "gate": ("output gate", "#CC79A7"), "delta": ("delta", "#D55E00")}
RULES = {"": ("released", "o"), "_local0": ("kept in the crop", "s"), "_local21": ("kept in the crop from t = 21", "D"),
         "_object": ("object Writes from t = 5", "^"), "_visible0": ("outside: only what the readout sees", "P"),
         "_invisible0": ("outside: only what it does not see", "X")}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def marks(ax: plt.Axes, horizon: bool = True) -> None:
    if horizon:
        ax.axvline(3.5, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("training\nhorizon", (3.5, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                    textcoords="offset points", va="top", fontsize=7, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("glimpse t")


def by_glimpse(ax: plt.Axes, policy: str, key: str, ylabel: str, title: str) -> None:
    for kind, (label, color) in KINDS.items():
        run = load(f"{kind}_s0_{policy}")
        if run is not None:
            ax.plot(range(len(run[key])), run[key], color=color, linewidth=2, label=label)
    if policy == "c2f":
        ax.axvline(20.5, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("level 3", (20.5, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                    textcoords="offset points", va="top", fontsize=7, color=MUTED)
    marks(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)


def rules_at_end(ax: plt.Axes) -> None:
    """C2F pixel error at the last glimpse, per kind of Write (rows) and test-time rule (markers)."""
    kinds = [k for k in KINDS if load(f"{k}_s0_c2f") is not None]
    for row, kind in enumerate(kinds):
        _, color = KINDS[kind]
        ends = {r: load(f"{kind}_s0_c2f{r}") for r in RULES}
        values = [run["mse"][-1] for run in ends.values() if run is not None]
        ax.plot([min(values), max(values)], [row, row], color=GRID, linewidth=2, zorder=1)
        for rule, run in ends.items():
            if run is not None:
                ax.plot(run["mse"][-1], row, RULES[rule][1], color=color, markersize=8, zorder=2,
                        markeredgecolor="white", markeredgewidth=1)
    ax.set_yticks(range(len(kinds)), [KINDS[k][0] for k in kinds])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("pixel MSE at t = 84 (normalized pixels)")
    ax.set_title("C2F at t = 84: the test-time rules", loc="left", fontsize=9)


def main() -> None:
    if load("softmax_s0_c2f") is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    by_glimpse(axes[0, 0], "c2f", "mse", "pixel MSE (normalized pixels)", "C2F, 85 glimpses: every Write drifts")
    rules_at_end(axes[0, 1])
    by_glimpse(axes[1, 0], "rfs", "mse", "pixel MSE (normalized pixels)",
               "The full view repeated: no cell is outside the crop")
    by_glimpse(axes[1, 1], "c2f", "residual_outside", "residual norm per outside cell",
               "C2F: what each glimpse writes outside its crop (both Writes)")
    kinds = [plt.Line2D([], [], color=color, linewidth=2, label=label) for kind, (label, color) in KINDS.items()
             if load(f"{kind}_s0_c2f") is not None]
    rules = [plt.Line2D([], [], linestyle="none", marker=m, color=MUTED, markersize=7, label=label)
             for label, m in RULES.values()]
    fig.legend(handles=kinds, fontsize=8, frameon=False, loc="upper left", ncol=6, bbox_to_anchor=(0.01, 0.955))
    fig.legend(handles=rules, fontsize=8, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.925),
               title="markers, top right (outside the crop from t = 0 unless noted):", title_fontsize=8,
               alignment="left")
    fig.suptitle("A small canvas model trained from scratch on 4-glimpse rollouts (ADE20K, 64 px scenes, seed 0)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "sink_memory.png", dpi=150)
    print("wrote", FIGURES / "sink_memory.png")
    hidden_writes()


def toy_models() -> list[str]:
    """Every toy model with a C2F result, as <kind>[_seen]_s<seed> (the hidden-penalty and 16-glimpse runs aside)."""
    names = sorted(p.name.removesuffix("_c2f.json") for p in RESULTS.glob("*_c2f.json"))
    return [n for n in names if "_hp" not in n and "_g16" not in n]


def kind_of(name: str) -> str:
    return "null" if name.startswith("null_open") else name.split("_")[0]


def short_horizon_leak(run: dict) -> float:
    """Mean residual per outside cell over C2F's glimpses 1-3 (inside the training horizon), over the canvas norm."""
    return sum(run["residual_outside"][1:4]) / 3 / run["canvas_norm"][3]


def spearman_note(ax: plt.Axes, text: str, x: list[float], y: list[float], where=(0.97, 0.04)) -> None:
    rho, p = spearmanr(x, y)
    ax.text(*where, f"{text}Spearman ρ = {rho:.2f} (p = {p:.3f}, n = {len(x)})", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color=MUTED)


def toy_diagnostic(ax: plt.Axes) -> None:
    xs, ys = [], []
    for name in toy_models():
        run = load(f"{name}_c2f")
        _, color = KINDS[kind_of(name)]
        seen = "_seen" in name
        x, y = short_horizon_leak(run), run["mse"][84] / run["mse"][20]
        ax.plot(x, y, "o", markersize=7, zorder=2, color="white" if seen else color, markeredgecolor=color,
                markeredgewidth=1.6)
        if kind_of(name) != "delta":
            xs.append(x)
            ys.append(y)
    spearman_note(ax, "without delta: ", xs, ys)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.4], ["0.05", "0.1", "0.2", "0.4"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_yticks([], minor=True)
    ax.set_xlabel("leak / canvas norm, glimpses 1-3 of C2F")
    ax.set_ylabel("pixel MSE at t = 84 / at t = 20")
    ax.set_title("From scratch: the leak inside the training\nhorizon predicts the drift past it", loc="left",
                 fontsize=9)
    present = {kind_of(name) for name in toy_models()}
    handles = [plt.Line2D([], [], linestyle="none", marker="o", color=color, markersize=6, label=label.split(" (")[0])
               for kind, (label, color) in KINDS.items() if kind in present]
    handles += [plt.Line2D([], [], linestyle="none", marker="o", color=MUTED, markersize=6, label="scene loss"),
                plt.Line2D([], [], linestyle="none", marker="o", color="white", markeredgecolor=MUTED,
                           markeredgewidth=1.6, markersize=6, label="seen loss")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left", ncol=2)


def family_diagnostic(ax: plt.Axes) -> None:
    folder = ROOT / "results/ablations"
    xs, ys = [], []
    for path in sorted(folder.glob("leak_*_c32.json")):
        model = path.name.removeprefix("leak_").removesuffix("_c32.json")
        released = folder / f"{model}_c2f_c32_t85_released.json"
        if not released.exists():
            continue
        miou = json.loads(released.read_text())["miou"]
        x = json.loads(path.read_text())["levels"][1]["outside_over_inside"]
        flagship = model == "in21k"
        xs.append(x)
        ys.append(100 * (miou[84] - miou[20]) / miou[20])
        ax.plot(x, ys[-1], "o", markersize=7, color="#0072B2" if flagship else MUTED,
                markeredgecolor="white", markeredgewidth=1, zorder=3 if flagship else 2)
        if flagship:
            ax.annotate("flagship", (x, 100 * (miou[84] - miou[20]) / miou[20]), xytext=(6, 2),
                        textcoords="offset points", fontsize=7, color="#0072B2")
    spearman_note(ax, "", xs, ys, where=(0.97, 0.93))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_xlabel("outside / inside Write residual, C2F level 1 (glimpses 1-4)")
    ax.set_ylabel("mIoU change from t = 20 to t = 84 (%)")
    ax.set_title("The 14 released CanViT-B: the same prediction", loc="left", fontsize=9)


PROJECTIONS = {"_c2f": ("released", "o"), "_c2f_invisible0": ("outside: only what the readout does not see", "X"),
               "_c2f_visible0": ("outside: only what the readout sees", "P"), "_c2f_local0": ("kept in the crop", "s")}


def toy_projection(ax: plt.Axes) -> None:
    names = [n for n in toy_models() if all(load(f"{n}{s}") is not None for s in PROJECTIONS)]
    for row, name in enumerate(names):
        _, color = KINDS[kind_of(name)]
        values = [load(f"{name}{s}")["mse"][84] for s in PROJECTIONS]
        ax.plot([min(values), max(values)], [row, row], color=GRID, linewidth=2, zorder=1)
        for value, (_, marker) in zip(values, PROJECTIONS.values()):
            ax.plot(value, row, marker, color=color, markersize=7, zorder=2, markeredgecolor="white",
                    markeredgewidth=0.8)
    ax.set_yticks(range(len(names)), [n.replace("_seen", ", seen loss").replace("_s", " seed ") for n in names],
                  fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("pixel MSE at t = 84 of C2F")
    ax.set_title("From scratch: the hidden part of the leak makes the drift", loc="left", fontsize=9)
    ax.legend(handles=[plt.Line2D([], [], linestyle="none", marker=m, color=MUTED, markersize=6, label=label)
                       for label, m in PROJECTIONS.values()], fontsize=7, frameon=False, loc="lower right")


def canvit_projection(ax: plt.Axes) -> bool:
    folder = ROOT / "results/hidden_leak"
    runs = {"released": ("released Writes", "#8a8985", "-"),
            "local21": ("kept in the crop", "#0072B2", "-"),
            "probe_visible21": ("outside: only what the probe sees", "#009E73", "-"),
            "probe_invisible21": ("outside: only what the probe does not see", "#D55E00", "-"),
            "head_visible21": ("outside: only what the pretraining head sees", "#009E73", "--"),
            "head_invisible21": ("outside: only what the pretraining head does not see", "#D55E00", "--")}
    if not all((folder / f"c2f_c64_t85_{r}.json").exists() for r in list(runs)[:4]):
        return False
    for run, (label, color, style) in runs.items():
        path = folder / f"c2f_c64_t85_{run}.json"
        if not path.exists():
            continue
        miou = json.loads(path.read_text())["miou"]
        t = [i for i, v in enumerate(miou) if v is not None]
        ax.plot(t, [miou[i] for i in t], color=color, linewidth=1.8, linestyle=style, label=label)
    ax.axvline(20.5, color=MUTED, linewidth=1, linestyle=":")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("glimpse t")
    ax.set_ylabel("mIoU (%)")
    ax.set_xlim(10, 85)
    ax.set_title("CanViT, C2F, 64×64 canvas, the same test from level 3:\nneither part of the leak alone is harmless",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    return True


def hidden_writes() -> None:
    """outputs/hidden_writes.png: the short-horizon diagnostic (toy and CanViT family) and the causal test."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    toy_diagnostic(axes[0, 0])
    family_diagnostic(axes[0, 1])
    toy_projection(axes[1, 0])
    if not canvit_projection(axes[1, 1]):
        axes[1, 1].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "hidden_writes.png", dpi=150)
    print("wrote", FIGURES / "hidden_writes.png")


if __name__ == "__main__":
    main()
