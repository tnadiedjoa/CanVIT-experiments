"""What C2F's level 3 does to the canvas, with released Writes and with Writes kept in the glimpse's crop.

Runs C2F for 85 glimpses (to level 3, 64×64 canvas) twice on the same ADE20K validation scenes with the same tile
order: with the released Writes, and with the Write residuals outside the crop dropped from glimpse 21 on
(scripts/local_writes.py). For each shown scene: the scene, its labels, the released probe's prediction and the canvas
(PCA protocol of appendix G, one basis and one color range per scene) at t = 20, and at t = 84 for both Writes.

The scenes are chosen among the first validation images by the pixel accuracy lost from t = 20 to t = 84 with
released Writes: the two largest losses and the two closest to the median loss.

    .venv/bin/python -P scripts/level3_canvas.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from torch import Tensor

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import predicted_labels
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX, released_ade20k_probe
from canvit_pytorch.model import CanViTForSemanticSegmentation
from canvit_pytorch.policies import make_policy
from canvit_pytorch.viz.pca import color_limits, fit_pca, project, to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import load_validation_scenes, normalize  # noqa: E402
from local_writes import LocalWrites  # noqa: E402

LEVEL_3_START, LAST = 21, 84
SHOWN = (20, LAST)


@dataclass(frozen=True)
class Figure:
    num_candidates: int = 64
    """Validation images among which the shown scenes are chosen."""
    canvas_grid_size: int = 64
    seed: int = 0
    batch_size: int = 8
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    output: Path = Path("outputs/ade20k_level3_canvas.png")
    device: str = "cuda"


@torch.inference_mode()
def rollout(seg: CanViTForSemanticSegmentation, images_u8: Tensor, cfg: Figure, *, local: bool,
            device: torch.device) -> dict[int, tuple[Tensor, Tensor]]:
    """C2F to t = 84 from the same seed, so that the same scenes in the same batches get the same tile order:
    {t: ([N, S, S] predicted labels, [N, G*G, D] float16 canvas patches)} for t in SHOWN."""
    torch.manual_seed(cfg.seed)
    local_writes = (LocalWrites(seg.canvit, canvas_grid_size=cfg.canvas_grid_size, device=device,
                                start=LEVEL_3_START) if local else None)
    kept: dict[int, list[tuple[Tensor, Tensor]]] = {t: [] for t in SHOWN}
    for first in range(0, len(images_u8), cfg.batch_size):
        images = normalize(images_u8[first:first + cfg.batch_size].to(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy = make_policy("coarse_to_fine", batch_size=images.shape[0], device=device, num_glimpses=LAST + 1,
                                 canvas_grid_size=cfg.canvas_grid_size)
            state = seg.init_state(batch_size=images.shape[0], canvas_grid_size=cfg.canvas_grid_size)
            for t in range(LAST + 1):
                viewpoint = policy.step(t, state)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                if t in kept:
                    labels = predicted_labels(seg.logits(state.canvas), size_px=RELEASED_SCENE_SIZE_PX)
                    kept[t].append((labels.cpu(), seg.canvit.canvas_patches(state.canvas).half().cpu()))
    if local_writes is not None:
        local_writes.remove()
    return {t: (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])) for t, parts in kept.items()}


def pixel_accuracy(predicted: Tensor, labels: Tensor) -> Tensor:
    valid = labels != ade20k.IGNORE_LABEL
    return ((predicted == labels) & valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)


def colorize(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return np.where((labels == ade20k.IGNORE_LABEL)[..., None], 0, palette[np.minimum(labels, len(palette) - 1)])


def main(cfg: Figure) -> None:
    device = torch.device(cfg.device)
    probe_repo = released_ade20k_probe("in21k", scene_size_px=RELEASED_SCENE_SIZE_PX,
                                       canvas_grid_size=cfg.canvas_grid_size)
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=probe_repo)
    seg = seg.to(device).eval()
    images_u8, labels_u8 = load_validation_scenes(cfg.cache, size_px=RELEASED_SCENE_SIZE_PX)
    images_u8, labels = images_u8[:cfg.num_candidates], labels_u8[:cfg.num_candidates].long()

    released = rollout(seg, images_u8, cfg, local=False, device=device)
    loss = pixel_accuracy(released[20][0], labels) - pixel_accuracy(released[LAST][0], labels)
    order = loss.argsort(descending=True).tolist()
    median = loss.median()
    by_median = sorted(range(len(loss)), key=lambda i: abs(loss[i] - median).item())
    chosen = order[:2] + [i for i in by_median if i not in order[:2]][:2]
    local = rollout(seg, images_u8, cfg, local=True, device=device)
    assert torch.equal(local[20][0], released[20][0]), "the two rollouts differ before level 3"

    palette = np.random.default_rng(0).integers(40, 256, size=(ade20k.NUM_CLASSES, 3), dtype=np.uint8)
    grid = cfg.canvas_grid_size
    columns = ["scene", "labels", "prediction, t = 20", "t = 84, released Writes", "t = 84, local Writes",
               "canvas, t = 20", "t = 84, released Writes", "t = 84, local Writes"]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.1 * len(columns), 2.25 * len(chosen)))
    for row, (index, axs) in enumerate(zip(chosen, axes)):
        predictions = [released[20][0][index], released[LAST][0][index], local[LAST][0][index]]
        accuracies = [pixel_accuracy(p[None], labels[index][None]).item() for p in predictions]
        canvases = [snapshot[1][index].float().numpy() for snapshot in (released[20], released[LAST], local[LAST])]
        basis = fit_pca(*canvases)
        limits = color_limits(*(project(basis, c) for c in canvases))
        axs[0].imshow(images_u8[index].permute(1, 2, 0).numpy())
        axs[1].imshow(colorize(labels[index].numpy(), palette))
        for ax, predicted, accuracy in zip(axs[2:5], predictions, accuracies):
            ax.imshow(colorize(predicted.numpy(), palette))
            ax.set_xlabel(f"pixel accuracy {100 * accuracy:.1f}%", fontsize=8)
        for ax, canvas in zip(axs[5:], canvases):
            ax.imshow(to_rgb(project(basis, canvas), limits).reshape(grid, grid, 3), interpolation="nearest")
        axs[0].set_ylabel("largest losses" if row < 2 else "median loss", fontsize=9)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], columns):
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"C2F on ADE20K, 64×64 canvas: level 3 (glimpses 21-84) with released and local Writes "
                 f"(scenes chosen among the first {cfg.num_candidates} validation images by accuracy lost with "
                 "released Writes)", fontsize=9)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=130, bbox_inches="tight")
    print(f"Wrote {cfg.output}; scenes {chosen}, accuracy lost from t = 20 to t = 84 with released Writes: "
          + " ".join(f"{100 * loss[i]:.1f}" for i in chosen) + f" (median {100 * median:.1f})")


if __name__ == "__main__":
    main(tyro.cli(Figure))
