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
        found += [result for p in paths if "miou" in (result := json.loads(p.read_text()))]
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), gridspec_kw={"wspace": 0.4})
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


def level3_controls() -> None:
    series = [(label, runs(name), color) for label, name, color in [
        ("C2F: level 3 (zoom past native resolution)", "c2f_c64_t85", COLORS["C2F"]),
        ("C2F to level 2, then level-2 tiles again", "c2f_then_level2_revisits_c64_t85", "#6250d6"),
        ("C2F to level 2, then the full scene again", "c2f_then_full_scene_c64_t85", "#e34948"),
        ("RFS", "rfs_c64_t85", COLORS["RFS"]),
    ]]
    if sum(result is not None for _, result, _ in series) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (key, ylabel) in zip(axes, [("miou", "mIoU (%)"), ("entropy", "probe entropy (nats)")], strict=True):
        for label, result, color in series:
            if result is not None:
                ax.plot(result.mean(key), color=color, label=label)
        ax.axvline(20.5, color=MUTED, linewidth=1, linestyle=":")
        ax.set(xlabel="glimpse t", ylabel=ylabel)
    axes[0].set_title("Canvas 64×64; glimpses 0-20 are C2F's levels 0-2 in every run", fontsize=9, loc="left")
    axes[0].legend(fontsize=7, loc="lower right")
    save(fig, "ade20k_level3_controls.png")


def scene_detail() -> None:
    # One blue ramp, darker for more detail.
    series = [(label, runs(name), color) for label, name, color in [
        ("native 512 px", "c2f_c64_t21_s0", "#104281"), ("detail of 256 px", "c2f_c64_t21_s0_detail256", "#3987e5"),
        ("detail of 128 px", "c2f_c64_t21_s0_detail128", "#9ec5f4"),
    ]]
    if any(result is None for _, result, _ in series):
        return
    native = series[0][1].mean("miou")  # type: ignore[union-attr]
    fig, (curves_ax, diff_ax) = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, result, color in series:
        miou = result.mean("miou")  # type: ignore[union-attr]
        curves_ax.plot(miou, color=color, label=label)
        diff_ax.plot(miou - native, color=color, label=label)
    for ax in (curves_ax, diff_ax):
        for boundary in (1, 5):
            ax.axvline(boundary - 0.5, color=GRID, linewidth=1, linestyle=":", zorder=0)
        ax.set_xlabel("glimpse t")
    curves_ax.set(ylabel="mIoU (%)", title="C2F on scenes with the fine detail removed (seed 0, canvas 64×64)")
    diff_ax.axhline(0, color=MUTED, linewidth=1)
    diff_ax.set(ylabel="mIoU − native, points", title="Loss from the missing detail")
    curves_ax.legend(fontsize=8, loc="lower right")
    save(fig, "ade20k_scene_detail.png")


def canvas_grids() -> None:
    policies = [("C2F, seed 0", ["c2f_c32_t21_s0", "c2f_c64_t21_s0", "c2f_c128_t21_s0"]),
                ("RFS", ["rfs_c32_t21", "rfs_c64_t21", "rfs_c128_t21"])]
    if not all(runs(name) for _, names in policies for name in names):
        return
    ramp = ["#9ec5f4", "#3987e5", "#104281"]  # one blue ramp, darker for a finer canvas
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (policy, names) in zip(axes, policies, strict=True):
        for name, color, grid in zip(names, ramp, (32, 64, 128), strict=True):
            result = runs(name)
            assert result is not None
            probe = " (64×64 probe)" if grid == 128 else ""
            ax.plot(result.mean("miou"), color=color, label=f"canvas {grid}×{grid}{probe}")
        ax.set(xlabel="glimpse t", title=policy)
    axes[0].set_ylabel("mIoU (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, "ade20k_canvas_grids.png")


def official() -> dict | None:
    """The official CLI's runs, collected by scripts/export_official.py."""
    path = RESULTS / "official.json"
    return json.loads(path.read_text()) if path.exists() else None


def cost_frontier() -> None:
    exported = official()
    c2f = {grid: runs(f"c2f_c{grid}_t21") for grid in (32, 64)}
    if exported is None or not exported["dinov3_ade20k"] or not all(c2f.values()):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for grid, style in ((64, "-"), (32, "--")):
        result = c2f[grid]
        assert result is not None
        miou = result.mean("miou")
        gflops = exported["canvit_glimpse_gflops"][str(grid)] * np.arange(1, len(miou) + 1)
        ax.plot(gflops, miou, style, color=COLORS["C2F"], marker="o", markersize=3,
                label=f"CanViT-B, C2F, canvas {grid}×{grid} (glimpses 1-{len(miou)})")
    for variant, color, marker, name in (("vitb16", "#52514e", "s", "DINOv3 ViT-B/16"),
                                         ("vits16", "#a3a29d", "D", "DINOv3 ViT-S/16")):
        points = sorted((r for r in exported["dinov3_ade20k"] if r["variant"] == variant), key=lambda r: r["input_size_px"])
        ax.plot([r["gflops"] for r in points], [r["miou"] for r in points], color=color, marker=marker, markersize=4,
                linewidth=1.5, label=f"{name}, one view at 128-512 px")
        for r, offset, align in ((points[0], (4, -10), "left"), (points[-1], (0, 7), "center")):
            ax.annotate(f"{r['input_size_px']} px", (r["gflops"], r["miou"]), xytext=offset, textcoords="offset points",
                        ha=align, fontsize=7, color=MUTED)
    ax.set(xscale="log", xlabel="forward GFLOPs per scene (CanViT: cumulative over glimpses)", ylabel="ADE20K val mIoU (%)",
           title="Accuracy against compute, released probes")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "ade20k_cost_frontier.png")


def reconstruction() -> None:
    exported = official()
    if exported is None or not exported["reconstruction"]:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for run in sorted(exported["reconstruction"], key=lambda r: list(POLICY_LABELS).index(r["policy"])):
        label = POLICY_LABELS[run["policy"]]
        ax.plot([p["t"] for p in run["per_timestep"]], [p["scene_cos_norm"] for p in run["per_timestep"]],
                color=COLORS[label], label=label)
    ax.axvline(20.5, color=MUTED, linewidth=1, linestyle=":")
    (grid,) = {run["canvas_grid_size"] for run in exported["reconstruction"]}
    ax.set(xlabel="glimpse t", ylabel="cosine similarity to DINOv3 (standardized)",
           title=f"Prediction of the teacher's scene features from the canvas ({grid}×{grid} canvas)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "ade20k_reconstruction.png")


CITYSCAPES = RESULTS / "cityscapes"
DETAIL_STYLES = {  # one blue ramp, darker for more detail; dashed for glimpses all taken from the 1024 px scene
    "s512": ("all glimpses from the 512 px scene", "#9ec5f4", "-"),
    "s512_then_native": ("level 3 from the 1024 px scene", "#104281", "-"),
    "native": ("all glimpses from the 1024 px scene", "#3987e5", "--"),
}


def cityscapes_run(name: str) -> dict | None:
    path = CITYSCAPES / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cityscapes_detail() -> None:
    models = [("CanViT-B (512 px pretraining), probe on 512 px scenes", "c2f_c64_t85_{}_s0"),
              ("CanViT-B fine-tuned on 1024 px scenes (experimental), probe on 1024 px scenes", "s1024exp_c2f_c64_t85_{}_s0")]
    models = [(title, pattern) for title, pattern in models if cityscapes_run(pattern.format("s512"))]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4), sharey=True, squeeze=False)
    for ax, (title, pattern) in zip(axes[0], models, strict=True):
        for detail, (label, color, style) in DETAIL_STYLES.items():
            if run := cityscapes_run(pattern.format(detail)):
                ax.plot(run["miou"], style, color=color, label=f"C2F, {label}")
        eg = cityscapes_run(pattern.replace("c2f", "egc2f").removesuffix("_s0").format("s512_then_native"))
        if eg:
            ax.plot(eg["miou"], color=COLORS["EG-C2F"], label="EG-C2F, level 3 from the 1024 px scene")
        for boundary in (1, 5, 21):
            ax.axvline(boundary - 0.5, color=GRID, linewidth=1, linestyle=":", zorder=0)
        ax.set(xlabel="glimpse t", title=title)
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("Cityscapes val mIoU at 1024 px (%)")
    save(fig, "cityscapes_detail.png")


def cityscapes_classes() -> None:
    low, high = cityscapes_run("c2f_c64_t85_s512_s0"), cityscapes_run("c2f_c64_t85_s512_then_native_s0")
    if low is None or high is None:
        return
    names = low["class_names"]
    gain = 100 * (np.array(high["per_class_iou"][-1]) - np.array(low["per_class_iou"][-1]))
    order = np.argsort(gain)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(np.arange(len(names)), gain[order], color=COLORS["C2F"], height=0.7)
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    t = len(low["miou"]) - 1
    ax.set(xlabel=f"IoU at t = {t}: level 3 from the 1024 px scene − from the 512 px scene (points)",
           title="Which classes gain from real detail (C2F, CanViT-B, canvas 64×64)")
    ax.title.set_fontsize(9)
    save(fig, "cityscapes_classes.png")


def cityscapes_cost() -> None:
    dinov3 = sorted((run for px in (512, 768, 1024) if (run := cityscapes_run(f"dv3b_{px}px"))),
                    key=lambda run: run["gflops"])
    canvit = [(label, run, color, style) for label, name, color, style in [
        ("CanViT-B, C2F", "c2f_c64_t85_s512_then_native_s0", COLORS["C2F"], "-"),
        ("CanViT-B, EG-C2F", "egc2f_c64_t85_s512_then_native", COLORS["EG-C2F"], "-"),
        ("CanViT-B fine-tuned on 1024 px, C2F", "s1024exp_c2f_c64_t85_s512_then_native_s0", COLORS["C2F"], "--"),
    ] if (run := cityscapes_run(name))]
    if not dinov3 or not canvit:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, run, color, style in canvit:
        gflops = run["glimpse_gflops"] * np.arange(1, len(run["miou"]) + 1)
        ax.plot(gflops, run["miou"], style, color=color, label=f"{label} (glimpses 1-{len(run['miou'])})")
    ax.plot([run["gflops"] for run in dinov3], [run["miou"][0] for run in dinov3], color="#52514e", marker="s",
            markersize=4, linewidth=1.5, label="DINOv3 ViT-B/16, one view")
    for run in dinov3:
        ax.annotate(f"{run['probe_config']['scene_size_px']} px", (run["gflops"], run["miou"][0]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    ax.set(xscale="log", xlabel="forward GFLOPs per 1024 px scene (CanViT: cumulative over glimpses)",
           ylabel="Cityscapes val mIoU at 1024 px (%)", title="Accuracy against compute on 1024 px scenes")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "cityscapes_cost_frontier.png")


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
    level3_controls()
    scene_detail()
    canvas_grids()
    cost_frontier()
    reconstruction()
    cityscapes_detail()
    cityscapes_classes()
    cityscapes_cost()
    summary_table()
