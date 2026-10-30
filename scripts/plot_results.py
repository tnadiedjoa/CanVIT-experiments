"""Figures and a summary table from the ADE20K rollout results that scripts/ade20k_rollouts.py writes to results/.

    .venv/bin/python scripts/plot_results.py

Each figure is drawn only when its results exist. Paper curves come from the paper's exported data
(external/CanViT/paper/exports/ade20k_seg.json), never typed by hand.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "outputs"
PAPER_EXPORT = ROOT / "external/CanViT/paper/exports/ade20k_seg.json"

# Categorical slots in the fixed order of the reference palette; an entity keeps its color in every figure.
COLORS = {
    "C2F": "#2a78d6", "EG-C2F": "#eb6834", "F-IID": "#1baf7a", "R-IID": "#eda100",
    "F2C": "#e87ba4", "RFS": "#008300", "canvas contrast": "#6250d6", "oracle error": "#e34948",
    "low entropy": "#8a8985",
}
POLICY_LABELS = {
    "coarse_to_fine": "C2F", "entropy_coarse_to_fine": "EG-C2F", "full_then_random": "F-IID", "random": "R-IID",
    "fine_to_coarse": "F2C", "repeated_full_scene": "RFS",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150,
})


@dataclass(frozen=True)
class Runs:
    """Results of the same condition over seeds: [n_runs, T] arrays per recorded quantity."""

    runs: list[dict]

    def stack(self, key: str) -> np.ndarray:
        return np.array([run[key] for run in self.runs])

    def mean(self, key: str) -> np.ndarray:
        return self.stack(key).mean(axis=0)


def runs(*names: str) -> Runs | None:
    """The results named exactly, or by a prefix followed by _s<seed>; None when none exists."""
    found = []
    for name in names:
        seeded = sorted(p for p in RESULTS.glob(f"{name}_s*.json") if re.fullmatch(rf"{re.escape(name)}_s\d+", p.stem))
        paths = [RESULTS / f"{name}.json"] if (RESULTS / f"{name}.json").exists() else seeded
        found += [json.loads(p.read_text()) for p in paths]
    return Runs(found) if found else None


def paper_curve(policy: str, canvas_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI of the paper's mIoU after every glimpse, in %."""
    curves = json.loads(PAPER_EXPORT.read_text())["policy_curves"]
    (curve,) = [c for c in curves if c["policy"] == policy and c["canvas_grid"] == canvas_grid and c["scene_size"] == 512]
    pt = curve["per_timestep"]
    return tuple(100 * np.array([p[k] for p in pt]) for k in ("mean", "ci_lo", "ci_hi"))  # type: ignore[return-value]


def label_line_end(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.plot([x], [y], "o", color=color, markersize=4)
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def reproduction() -> None:
    policies = list(POLICY_LABELS)
    measured = {p: runs(f"{prefix}_c64_t21") for p, prefix in zip(policies, ["c2f", "egc2f", "fiid", "riid", "f2c", "rfs"])}
    if not all(measured.values()):
        return
    fig, (curves_ax, diff_ax) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for policy in policies:
        label, result = POLICY_LABELS[policy], measured[policy]
        assert result is not None
        miou = result.stack("miou")
        t = np.arange(miou.shape[1])
        color = COLORS[label]
        curves_ax.plot(t, miou.mean(0), color=color, label=f"{label} (n={len(miou)})")
        if len(miou) > 1:
            curves_ax.fill_between(t, miou.min(0), miou.max(0), color=color, alpha=0.15, linewidth=0)
        paper_mean, lo, hi = paper_curve(policy, 64)
        curves_ax.plot(t, paper_mean, "o", markersize=4, markerfacecolor="none", markeredgecolor=color, markeredgewidth=1)
        diff_ax.plot(t, miou.mean(0) - paper_mean, color=color, label=label)
        diff_ax.fill_between(t, lo - paper_mean, hi - paper_mean, color=color, alpha=0.10, linewidth=0)
    curves_ax.set(xlabel="glimpse t", ylabel="ADE20K val mIoU (%)", title="Canvas 64×64: measured (lines) vs paper (circles)")
    curves_ax.legend(loc="lower right", fontsize=8)
    diff_ax.axhline(0, color=MUTED, linewidth=1)
    diff_ax.set(xlabel="glimpse t", ylabel="measured − paper (mIoU points)",
                title="Difference to the paper (bands: paper's 95% CI)")
    diff_ax.legend(loc="lower right", fontsize=8, ncols=2)
    save(fig, "ade20k_reproduction.png")


def drift() -> None:
    series = [(label, runs(name), style) for label, name, style in [
        ("RFS", "rfs_c64_t85", "-"), ("C2F", "c2f_c64_t85", "-"), ("F-IID", "fiid_c64_t85", "-"),
        ("R-IID", "riid_c64_t85", "-"), ("RFS", "rfs_c32_t85", "--"), ("C2F", "c2f_c32_t85", "--"),
    ]]
    series = [(label, result, style) for label, result, style in series if result is not None]
    if not series:
        return
    panels = [("miou", "mIoU (%)", "linear"), ("canvas_patch_norm", "canvas patch norm (raw)", "linear"),
              ("delta_previous", "Δ to previous canvas", "log"), ("entropy", "probe entropy (nats)", "linear")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (key, ylabel, yscale) in zip(axes, panels, strict=True):
        for label, result, style in series:
            values = result.mean(key)
            ax.plot(np.arange(len(values)), values, style, color=COLORS[label], linewidth=1.6 if style == "--" else 2,
                    label=f"{label}, canvas {'64' if style == '-' else '32'}")
        for t in (1, 5, 21):
            ax.axvline(t - 0.5, color=GRID, linewidth=1, linestyle=":", zorder=0)
        ax.axvspan(19.5, ax.get_xlim()[1], color="#f3f2ee", zorder=-1, linewidth=0)
        ax.set(xlabel="glimpse t", ylabel=ylabel, yscale=yscale)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_title("Shaded: past 20 glimpses (<0.1% of pretraining rollouts)", fontsize=8, loc="left", color=MUTED)
    save(fig, "ade20k_long_rollouts.png")


def antialiasing() -> None:
    pairs = [("C2F, canvas 64", "c2f_c64_t21_s0"), ("C2F, canvas 32", "c2f_c32_t21_s0"),
             ("RFS, canvas 64", "rfs_c64_t21"), ("RFS, canvas 32", "rfs_c32_t21")]
    available = [(label, name) for label, name in pairs
                 if (RESULTS / f"{name}.json").exists() and (RESULTS / f"{name}_aa.json").exists()]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, name in available:
        bilinear = np.array(json.loads((RESULTS / f"{name}.json").read_text())["miou"])
        aa = np.array(json.loads((RESULTS / f"{name}_aa.json").read_text())["miou"])
        policy = label.split(",")[0]
        ax.plot(np.arange(len(aa)), aa - bilinear, "-" if "64" in label else "--", color=COLORS[policy], label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(xlabel="glimpse t", ylabel="mIoU(antialiased) − mIoU(bilinear), points",
           title="Anti-aliased glimpse extraction, paired runs")
    ax.legend(fontsize=8)
    save(fig, "ade20k_antialiasing.png")


def tile_orders() -> None:
    c2f = runs("c2f_c64_t21")
    others = [(label, runs(name)) for label, name in [
        ("EG-C2F", "egc2f_c64_t21"), ("canvas contrast", "c2f_canvas_contrast_c64_t21"),
        ("low entropy", "c2f_low_entropy_c64_t21"), ("oracle error", "c2f_oracle_error_c64_t21"),
    ]]
    others = [(label, result) for label, result in others if result is not None]
    if c2f is None or len(others) < 2:
        return
    fig, (full_ax, gain_ax) = plt.subplots(1, 2, figsize=(11, 4))
    reference = c2f.stack("miou")
    t = np.arange(reference.shape[1])
    full_ax.plot(t, reference.mean(0), color=COLORS["C2F"], label=f"C2F, random order (n={len(reference)})")
    full_ax.fill_between(t, reference.min(0), reference.max(0), color=COLORS["C2F"], alpha=0.15, linewidth=0)
    gain_ax.fill_between(t, reference.min(0) - reference.mean(0), reference.max(0) - reference.mean(0),
                         color=COLORS["C2F"], alpha=0.15, linewidth=0, label="C2F seed range")
    for label, result in others:
        miou = result.mean("miou")
        style = "--" if label == "low entropy" else "-"
        full_ax.plot(t, miou, style, color=COLORS[label], label=label)
        gain_ax.plot(t, miou - reference.mean(0), style, color=COLORS[label], label=label)
    for ax in (full_ax, gain_ax):
        for boundary in (1, 5):
            ax.axvline(boundary - 0.5, color=GRID, linewidth=1, linestyle=":", zorder=0)
        ax.set_xlabel("glimpse t")
    full_ax.set(ylabel="ADE20K val mIoU (%)", title="Tile order within C2F's levels, canvas 64×64")
    gain_ax.axhline(0, color=MUTED, linewidth=1)
    gain_ax.set(ylabel="mIoU − C2F mean, points", title="Gain over random tile order")
    full_ax.legend(fontsize=8, loc="lower right")
    save(fig, "ade20k_tile_orders.png")


def save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        if ax.get_xlabel() == "glimpse t":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGURES / name}")


def summary_table() -> None:
    """results/summary.md: mIoU at chosen glimpses and averaged over glimpses 1-20, per condition (mean over seeds)."""
    names = sorted({re.sub(r"_s\d+$", "", p.stem) for p in RESULTS.glob("*.json")})
    lines = ["| condition | runs | t=0 | t=1 | t=4 | t=12 | t=20 | mean t=1..20 | t=last | last t |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        result = runs(name)
        if result is None:
            continue
        miou = result.mean("miou")
        at = lambda t: f"{miou[t]:.2f}" if t < len(miou) else ""  # noqa: E731
        lines.append(f"| {name} | {len(result.runs)} | {at(0)} | {at(1)} | {at(4)} | {at(12)} | {at(20)} | "
                     f"{miou[1:21].mean():.2f} | {miou[-1]:.2f} | {len(miou) - 1} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    reproduction()
    drift()
    antialiasing()
    tile_orders()
    summary_table()
