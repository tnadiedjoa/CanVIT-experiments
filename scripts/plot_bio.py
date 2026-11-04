"""Figures of the primate-vision experiments (scripts/bio_rollouts.py, bio_reconstruction.py, growing_canvas.py,
train_retina.py), in the style of scripts/plot_results.py.

    .venv/bin/python scripts/plot_bio.py

Each figure is drawn only when its results exist. The patchwork example runs the released model on one ADE20K
validation scene, on the CPU.
"""

import json
import math
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


def gradient_fields() -> None:
    """Where the canvas cells would move at the starting grid (train_retina.py --gradient-field): the descent direction
    of the loss on the centers of each 2×2 block of cells, in standard errors over the training steps, on ADE20K and
    on Cityscapes."""
    names = {"ADE20K": "gradfield_ade32", "Cityscapes": "gradfield_city32"}
    if not all((ROOT / f"probes/{n}.pt").exists() for n in names.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, (title, name) in zip(axes, names.items()):
        steps = torch.load(ROOT / f"probes/{name}.pt", map_location="cpu", weights_only=False)["gradient_steps"]["centers"]
        S, N, _ = steps.shape
        G = math.isqrt(N)
        blocks = steps.view(S, G // 2, 2, G // 2, 2, 2).mean(dim=(2, 4))  # [S, G/2, G/2, (row, col)]
        z = -blocks.mean(0) / (blocks.std(0) / S ** 0.5)  # descent, in standard errors
        centers = (np.arange(G // 2) + 0.5) / (G // 2) * 2 - 1
        x, y = np.meshgrid(centers, centers)
        strength = z.norm(dim=-1).numpy()
        ax.quiver(x, y, z[..., 1].numpy(), z[..., 0].numpy(), strength, cmap="viridis", clim=(0, 10), angles="xy",
                  scale_units="xy", scale=60, width=0.004)
        share = (z.abs() > 3).float().mean().item()
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(1.05, -1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}: {100 * share:.0f}% of components beyond 3 standard errors", fontsize=8.5, loc="left",
                     color=INK)
    fig.suptitle("Pull on the canvas cells at the released grid (arrow length: standard errors; pure noise: 0.3% beyond 3)",
                 fontsize=8.5, x=0.01, ha="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "bio_gradient_fields.png", bbox_inches="tight")
    plt.close(fig)


def patchwork() -> None:
    """Glimpses shaped by uncertainty (bio_vision.Patchwork): mIoU after every glimpse from C2F's level 2 on, and at
    t = 20 against the side of the blocks of patches a glimpse gathers (8: whole level-2 tiles)."""
    curves = {"C2F": ROOT / "results/c2f_c64_t21_s0.json", "EG-C2F": ROOT / "results/egc2f_c64_t21.json",
              "patchwork by uncertainty": BIO / "entropy_patchwork_c64_t21.json",
              "the same, Writes kept in its patches": BIO / "entropy_patchwork_local5_c64_t21.json",
              "patchwork at random": BIO / "random_patchwork_c64_t21.json"}
    if not all(path.exists() for path in curves.values()):
        return
    colors = {"C2F": COLORS["C2F"], "EG-C2F": COLORS["EG-C2F"], "patchwork by uncertainty": MICRO,
              "the same, Writes kept in its patches": PHASE, "patchwork at random": SAME_VIEW}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for label, path in curves.items():
        values = miou(path)[4:21]
        t = np.arange(4, 21)
        axes[0].plot(t, values, "--" if label == "C2F" else "-", color=colors[label], label=label,
                     linewidth=1.6 if label == "C2F" else 2)
        label_line_end(axes[0], t[-1], values[-1], f"{values[-1]:.1f}", colors[label])
    axes[0].set_xlabel("glimpse t (level 2 from t = 5)")
    axes[0].set_ylabel("mIoU (%)")
    axes[0].set_xlim(3.5, 22.5)
    axes[0].set_xticks([5, 10, 15, 20])
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[0].set_title("ADE20K, 64×64 canvas", fontsize=9, loc="left", color=INK)
    for order, color, reference in (("entropy", MICRO, BIO / "entropy_patchwork_b8_c64_t21.json"),
                                    ("random", SAME_VIEW, ROOT / "results/c2f_c64_t21_s0.json")):
        paths = [BIO / f"{order}_patchwork{'' if b == 1 else f'_b{b}'}_c64_t21.json" for b in (1, 2, 4)] + [reference]
        points = [(b, miou(path)[20]) for b, path in zip((1, 2, 4, 8), paths) if path.exists()]
        blocks, values = zip(*points)
        axes[1].plot(blocks, values, "o-", color=color, markersize=5,
                     label="by uncertainty (8: EG-C2F's order)" if order == "entropy" else "at random (8: C2F)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([1, 2, 4, 8], ["1×1", "2×2", "4×4", "8×8"])
    axes[1].set_xlabel("blocks of patches in a glimpse")
    axes[1].set_ylabel("mIoU at t = 20 (%)")
    axes[1].legend(fontsize=7.5, loc="lower right")
    axes[1].set_title("The more a glimpse is split, the worse", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "bio_patchwork.png", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def patchwork_example(scene: int = 4) -> None:
    """One validation scene after C2F's levels 0 and 1: the probe's uncertainty, the 64 patches the next glimpse
    gathers there, and that glimpse."""
    from ade20k_rollouts import load_validation_scenes, normalize
    from bio_vision import FreePositions, Patchwork, patchwork_glimpse, take_glimpse, UniformRetina
    from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, released_ade20k_probe
    from canvit_pytorch.model import CanViTForSemanticSegmentation
    from canvit_pytorch.policies import make_policy
    from canvit_pytorch.policies.entropy import predictive_entropy
    from canvit_pytorch.preprocess import imagenet_denormalize

    cache = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    if not cache.exists():
        return
    torch.manual_seed(0)
    device = torch.device("cpu")
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=FLAGSHIP, probe_repo=released_ade20k_probe("in21k", scene_size_px=512, canvas_grid_size=64)).eval()
    free = FreePositions(seg.canvit)
    images_u8, _ = load_validation_scenes(cache, size_px=512)
    image = normalize(images_u8[scene:scene + 1])
    retina = UniformRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=seg.canvit.patch_size, device=device)
    policy = make_policy("coarse_to_fine", batch_size=1, device=device, num_glimpses=21, canvas_grid_size=64)
    state = seg.init_state(batch_size=1, canvas_grid_size=64)
    for t in range(Patchwork.FIRST):
        viewpoint = policy.step(t, state)
        glimpse, free.glimpse = take_glimpse(image, viewpoint, retina)
        state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
    logits = seg.logits(state.canvas)
    entropy = predictive_entropy(logits.float())[0]
    pw = Patchwork(batch_size=1, device=device, glimpse_grid=RELEASED_GLIMPSE_SIZE_PX // seg.canvit.patch_size,
                   order="entropy")
    _, positions = pw.step(logits)
    glimpse = patchwork_glimpse(image, positions, half_side=pw.half_side, patch_px=seg.canvit.patch_size)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    picture = imagenet_denormalize(image[0]).permute(1, 2, 0).numpy()
    axes[0].imshow(picture, extent=(-1, 1, 1, -1))
    axes[0].set_title("scene", fontsize=9, loc="left", color=INK)
    axes[1].imshow(entropy.numpy(), extent=(-1, 1, 1, -1), cmap="magma")
    axes[1].set_title("probe uncertainty after 5 glimpses", fontsize=9, loc="left", color=INK)
    for ax in axes[:2]:
        for y, x in positions[0].tolist():
            ax.add_patch(plt.Rectangle((x - pw.half_side, y - pw.half_side), 2 * pw.half_side, 2 * pw.half_side,
                                       facecolor="none", edgecolor="white" if ax is axes[1] else MICRO, linewidth=0.8))
    axes[2].imshow(imagenet_denormalize(glimpse[0]).permute(1, 2, 0).numpy())
    axes[2].set_title("glimpse 5: its 64 patches", fontsize=9, loc="left", color=INK)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "bio_patchwork_example.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    microsaccades()
    growing_canvas()
    learned_canvas()
    learned_retinas()
    gradient_fields()
    patchwork()
    patchwork_example()


if __name__ == "__main__":
    main()
