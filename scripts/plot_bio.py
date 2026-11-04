"""Figures of the primate-vision experiments (scripts/bio_rollouts.py, bio_reconstruction.py, growing_canvas.py,
train_retina.py), in the style of scripts/plot_results.py.

    .venv/bin/python scripts/plot_bio.py

Each figure is drawn only when its results exist.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import COLORS, GRID, INK, MUTED, label_line_end  # noqa: E402  (also sets the rcParams)

ROOT = Path(__file__).resolve().parents[1]
BIO, CITY, FIGURES = ROOT / "results/bio", ROOT / "results/cityscapes", ROOT / "outputs"
MICRO = "#6250d6"
SAME_VIEW = "#8a8985"
PHASE = "#e87ba4"


def miou(path: Path) -> np.ndarray | None:
    return np.array(json.loads(path.read_text())["miou"]) if path.exists() else None


def microsaccades() -> None:
    """mIoU after every glimpse: the full scene repeated (RFS), a smaller view repeated, the same view with
    microsaccades, the full scene shifted by under a patch (phase only), and C2F for reference."""
    panels = {
        "ADE20K (64×64 canvas)": {
            "C2F": ROOT / "results/c2f_c64_t21_s0.json",
            "RFS: full scene repeated": ROOT / "results/rfs_c64_t21.json",
            "view 7/8 repeated": BIO / "micro_s0.875_j0_c64_t21.json",
            "view 7/8, microsaccades ±1/8": BIO / "micro_s0.875_j0.125_c64_t21.json",
            "full scene, shifts ±½ patch": BIO / "micro_s1_j0.125_out_c64_t21.json",
        },
        "Cityscapes (1024 px scenes)": {
            "C2F": CITY / "c2f_c64_t85_s512_then_native_s0.json",
            "RFS: full scene repeated": CITY / "rfs_c64_t21_s512.json",
            "view 7/8, microsaccades ±1/8": CITY / "micro_s0.875_j0.125_c64_t21_s512.json",
        },
    }
    colors = {"C2F": COLORS["C2F"], "RFS: full scene repeated": COLORS["RFS"], "view 7/8 repeated": SAME_VIEW,
              "view 7/8, microsaccades ±1/8": MICRO, "full scene, shifts ±½ patch": PHASE}
    if not all(p.exists() for p in panels["ADE20K (64×64 canvas)"].values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (title, curves) in zip(axes, panels.items()):
        for label, path in curves.items():
            values = miou(path)
            if values is None:
                continue
            values = values[:21]
            t = np.arange(len(values))
            ax.plot(t, values, "--" if label == "C2F" else "-", color=colors[label], label=label,
                    linewidth=1.6 if label == "C2F" else 2)
            label_line_end(ax, t[-1], values[-1], f"{values[-1]:.1f}", colors[label])
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("glimpse t")
        ax.set_xlim(-0.5, 23.5)
    axes[0].set_ylabel("mIoU (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Microsaccades: views shifted by a fraction of a patch keep improving the canvas; "
                 "the same view repeated degrades it", fontsize=9, x=0.01, ha="left", color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "bio_microsaccades.png", bbox_inches="tight")
    plt.close(fig)


def growing_canvas() -> None:
    """mIoU at the last glimpse against the number of canvas cells: the growing canvas for each budget and tile order,
    and the uniform 32×32 and 64×64 canvases."""
    rows = {order: [] for order in ("random", "entropy")}
    for order in rows:
        for budget in (0, 4, 8, 16):
            path = BIO / f"grow_{order}_b{budget}_t21.json"
            if path.exists():
                r = json.loads(path.read_text())
                rows[order].append((r["canvas_cells"][-1], r["miou"][-1], budget))
    if not any(rows.values()):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    references = {"C2F": (ROOT / "results/c2f_c64_t21_s0.json", "random"),
                  "EG-C2F": (ROOT / "results/egc2f_c64_t21.json", "entropy")}
    for (label, (path, order)), color in zip(references.items(), (COLORS["C2F"], COLORS["EG-C2F"])):
        values = miou(path)
        if values is not None:
            ax.plot([4096], [values[20]], "s", color=color, markersize=6, markerfacecolor="none", markeredgewidth=1.5,
                    label=f"{label}, uniform 64×64")
        if rows[order]:
            cells, values, budgets = zip(*rows[order])
            ax.plot(cells, values, "o-", color=color, markersize=5, label=f"{label}, 32×32 growing where it zooms")
            if order == "random":
                for x, y, b in zip(cells, values, budgets):
                    ax.annotate(f"{b} fine tiles", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7,
                                color=MUTED)
    ax.set_xlabel("canvas cells after 21 glimpses")
    ax.set_ylabel("mIoU at t = 20 (%)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Fine cells only where the eye zoomed in", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "bio_growing_canvas.png", bbox_inches="tight")
    plt.close(fig)


def learned_canvas(name: str = "canvas_frozen") -> None:
    """The learned Gaussian receptive fields of the canvas cells (one ellipse per cell at 2 sigma) over the starting
    grid, for one quadrant of the scene."""
    path = ROOT / f"probes/{name}.pt"
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["canvas_shape_state"]
    centers, sigmas, angles = state["centers"].numpy(), state["log_sigmas"].exp().numpy(), state["angles"].numpy()
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    keep = (centers[:, 0] < 0) & (centers[:, 1] < 0)
    for (y, x), (sy, sx), angle in zip(centers[keep], sigmas[keep], angles[keep]):
        ax.add_patch(Ellipse((x, y), 4 * sx, 4 * sy, angle=np.degrees(angle), facecolor="none", edgecolor=MICRO,
                             linewidth=0.6))
    grid = (np.arange(32) + 0.5) / 32 * 2 - 1
    gy, gx = np.meshgrid(grid[grid < 0], grid[grid < 0], indexing="ij")
    ax.plot(gx.ravel(), gy.ravel(), ".", color=GRID, markersize=2)
    ax.set_xlim(-1, 0)
    ax.set_ylim(0, -1)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: learned receptive fields (2σ), top-left quadrant", fontsize=8, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / f"bio_{name}.png", bbox_inches="tight")
    plt.close(fig)


def ring_radii(centers: np.ndarray) -> list[float]:
    """Mean Chebyshev radius of the patch centers of each of the 4 rings of the starting 8×8 grid."""
    grid = (np.arange(8) + 0.5) / 8 * 2 - 1
    start = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    ring = (np.abs(start).max(axis=1) * 4).astype(int)
    radius = np.abs(centers).max(axis=1)
    return [float(radius[ring == k].mean()) for k in range(4)]


def learned_retinas() -> None:
    """The 64 patches of the learned retinas over the released crop: CanViT frozen or fine-tuned with them, trained
    with zoom (F-IID and R-IID) or without (saccades at scale 0.5); the titles give the spacing between consecutive
    rings of patches, from the center out (0.25 on the released grid)."""
    names = {("frozen", "with zoom"): "retina_frozen_zoom", ("frozen", "without zoom"): "retina_frozen_s0.5",
             ("fine-tuned", "with zoom"): "retina_ft_zoom", ("fine-tuned", "without zoom"): "retina_ft_s0.5"}
    if not all((ROOT / f"probes/{n}.json").exists() for n in names.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))
    for ax, ((model, zoom), name) in zip(axes.ravel(), names.items()):
        retina = json.loads((ROOT / f"probes/{name}.json").read_text())["retina"]
        for (y, x), (hy, hx) in zip(retina["centers"], retina["half_sizes"]):
            ax.add_patch(plt.Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, facecolor=MICRO, alpha=0.12, edgecolor=MICRO,
                                       linewidth=0.8))
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, facecolor="none", edgecolor=INK, linewidth=1.2, linestyle="--"))
        radii = ring_radii(np.array(retina["centers"]))
        spacings = " / ".join(f"{b - a:.3f}" for a, b in zip(radii, radii[1:]))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(1.2, -1.2)
        ax.set_aspect("equal")
        ax.set_title(f"CanViT {model}, {zoom}\nring spacing {spacings}", fontsize=8, loc="left", color=INK)
    fig.suptitle("Learned retinas: 64 patches (shaded) over the released crop (dashed). Without zoom, the spacing "
                 "grows outward and the field widens", fontsize=8.5, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "bio_learned_retinas.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    microsaccades()
    growing_canvas()
    learned_canvas()
    learned_retinas()


if __name__ == "__main__":
    main()
