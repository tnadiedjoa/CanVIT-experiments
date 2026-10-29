"""A C2F rollout: where CanViT looks, what its canvas holds after each glimpse, and what each glimpse changed.

Follows the layout of the paper's figure 1: the scene with its numbered
viewpoints, then per glimpse the crop, the canvas (PCA protocol of appendix G,
one basis and one color range for the whole rollout) and the Δ map, the cosine
dissimilarity between the layer-normalized canvas before and after the glimpse.

    .venv/bin/python scripts/c2f_rollout.py
    .venv/bin/python scripts/c2f_rollout.py --image path/to/image.jpg --canvas-grid-size 64
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from PIL import Image

from canvit_pytorch import CanViTForPretraining
from canvit_pytorch.episode import run_episode
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.policies import make_policy
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess
from canvit_pytorch.viewpoint import crop_box_px, sample_at_viewpoint
from canvit_pytorch.viz.pca import color_limits, fit_pca, project, to_rgb
from canvit_pytorch.viz.web import canvas_change

UPSTREAM_TEST_DATA = Path(__file__).resolve().parents[1] / "external/CanViT/canvit-pytorch/test_data"
GLIMPSE_COLOR = "#4080d0"  # the paper figures' glimpse blue
COLUMNS = 10


@dataclass(frozen=True)
class Rollout:
    image: Path = UPSTREAM_TEST_DATA / "Places365_IMG_9600.jpeg"
    num_glimpses: int = 20
    canvas_grid_size: int = RELEASED_CANVAS_GRID_SIZE
    seed: int = 0
    """Seeds the order of the tiles within each quadtree level."""
    output: Path = Path("outputs/c2f_rollout.png")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def quadtree_level(scale: float) -> int:
    return round(-math.log2(scale))


def draw_viewpoints(ax: plt.Axes, boxes: list[list[float]], scales: list[float]) -> None:
    """Numbered crop boxes; labels sit at a different spot per level so that nested boxes stay readable."""
    for t, ((y0, x0, y1, x1), scale) in enumerate(zip(boxes, scales, strict=True)):
        level = quadtree_level(scale)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=GLIMPSE_COLOR, linewidth=2.5 - 0.6 * level))
        x, y = ((x0 + 6, y0 + 22), (x0 + 6, y0 + 52))[level] if level < 2 else ((x0 + x1) / 2, (y0 + y1) / 2)
        ax.text(x, y, str(t), color="white", fontsize=7, fontweight="bold", ha="left" if level < 2 else "center",
                va="center", bbox={"facecolor": GLIMPSE_COLOR, "linewidth": 0, "pad": 1.2})


@torch.inference_mode()
def main(cfg: Rollout) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = CanViTForPretraining.from_pretrained(FLAGSHIP).to(device).eval()
    image = preprocess(RELEASED_SCENE_SIZE_PX)(Image.open(cfg.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0).to(device)

    policy = make_policy("coarse_to_fine", batch_size=1, device=device, num_glimpses=cfg.num_glimpses,
                         canvas_grid_size=cfg.canvas_grid_size)
    initial = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
    steps = run_episode(canvit=model.canvit, images=image, policy=policy, num_glimpses=cfg.num_glimpses,
                        glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX, initial_state=initial)

    canvases_np = [model.canvit.canvas_patches(step.state.canvas)[0].float().cpu().numpy() for step in steps]
    before = [model.canvit.canvas_patches(initial.canvas)[0].float().cpu().numpy(), *canvases_np[:-1]]
    deltas = np.stack([canvas_change(b, a) for b, a in zip(before, canvases_np, strict=True)])
    basis = fit_pca(*canvases_np)
    limits = color_limits(*(project(basis, c) for c in canvases_np))
    delta_max = float(np.quantile(deltas[1:] if len(deltas) > 1 else deltas, 0.99))

    grid = cfg.canvas_grid_size
    blocks = math.ceil(cfg.num_glimpses / COLUMNS)
    fig = plt.figure(figsize=(1.5 * (COLUMNS + 4), 1.55 * 3 * blocks))
    spec = fig.add_gridspec(3 * blocks, COLUMNS + 4, hspace=0.25, wspace=0.08)

    scene_ax = fig.add_subplot(spec[:3, :4])
    scene_ax.imshow(imagenet_denormalize(image[0]).permute(1, 2, 0).cpu().numpy())
    boxes = [crop_box_px(step.viewpoint, image_size_px=RELEASED_SCENE_SIZE_PX)[0].tolist() for step in steps]
    draw_viewpoints(scene_ax, boxes, [step.viewpoint.scales[0].item() for step in steps])
    scene_ax.set_title(f"C2F, {cfg.num_glimpses} glimpses, canvas {grid}×{grid}", fontweight="bold")
    scene_ax.set_xticks([])
    scene_ax.set_yticks([])

    curve_ax = fig.add_subplot(spec[4:, :3] if blocks > 1 else spec[2, :3])
    curve_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    curve_ax.plot(range(cfg.num_glimpses), deltas.mean(axis=1), marker="o", markersize=3, color=GLIMPSE_COLOR)
    for t, step in enumerate(steps[1:], start=1):
        if quadtree_level(step.viewpoint.scales[0].item()) != quadtree_level(steps[t - 1].viewpoint.scales[0].item()):
            curve_ax.axvline(t - 0.5, color="0.7", linewidth=1, linestyle="--")
    curve_ax.set_xlabel("glimpse t")
    curve_ax.set_ylabel("mean Δ (t=0: vs. initial canvas)")
    curve_ax.set_yscale("log")
    curve_ax.grid(alpha=0.3)

    for t, step in enumerate(steps):
        block, col = divmod(t, COLUMNS)
        glimpse_ax, canvas_ax, delta_ax = (fig.add_subplot(spec[3 * block + r, col + 4]) for r in range(3))
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=step.viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        glimpse_ax.imshow(imagenet_denormalize(glimpse[0]).permute(1, 2, 0).cpu().numpy())
        glimpse_ax.set_title(f"t={t}", fontsize=9)
        canvas_ax.imshow(to_rgb(project(basis, canvases_np[t]), limits).reshape(grid, grid, 3), interpolation="nearest")
        delta_ax.imshow(deltas[t].reshape(grid, grid), cmap="magma", vmin=0, vmax=delta_max, interpolation="nearest")
        if col == 0:
            for ax, label in ((glimpse_ax, "glimpse"), (canvas_ax, "canvas"), (delta_ax, "Δ")):
                ax.set_ylabel(label, fontsize=9)
        for ax in (glimpse_ax, canvas_ax, delta_ax):
            ax.set_xticks([])
            ax.set_yticks([])

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {cfg.output}")
    print("mean Δ per glimpse:", " ".join(f"{d:.4f}" for d in deltas.mean(axis=1)))


if __name__ == "__main__":
    main(tyro.cli(Rollout))
