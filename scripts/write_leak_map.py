"""Where one glimpse's Canvas Attention Writes land, on one ADE20K scene (released CanViT-B, 64×64 canvas, C2F).

For a level-1 glimpse (a quadrant) and the first level-3 glimpse (a tile 1/8 of the scene wide), the figure shows the
crop on the scene, the norm of the summed Write residual in every canvas cell (log scale, one color range for both
glimpses), and the attention weight of every cell on backbone register 3, mean over heads and over the three Writes.
Outside the crop, cells still receive a residual, and their attention goes almost only to that register: the leak
and its attention sink that scripts/write_leakage.py measures over 256 scenes.

    .venv/bin/python -P scripts/write_leak_map.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import make_policy
from canvit_pytorch.viewpoint import crop_box_px

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import load_validation_scenes, normalize  # noqa: E402
from local_writes import PREFIX_TOKENS, write_parts  # noqa: E402

SINK = PREFIX_TOKENS.index("backbone register 3")
SHOWN = {1: "level 1 (glimpse 1)", 21: "level 3 (glimpse 21)"}


@dataclass(frozen=True)
class Figure:
    image_index: int = 46
    """Index of the ADE20K validation scene; 46 is the first scene of outputs/ade20k_level3_canvas.png."""
    canvas_grid_size: int = 64
    seed: int = 0
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    output: Path = Path("outputs/ade20k_write_leak_map.png")
    device: str = "cuda"


@torch.inference_mode()
def main(cfg: Figure) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    canvit = CanViTForPretraining.from_pretrained(FLAGSHIP).canvit.to(device).eval()
    images_u8, _ = load_validation_scenes(cfg.cache, size_px=RELEASED_SCENE_SIZE_PX)
    image_u8 = images_u8[cfg.image_index:cfg.image_index + 1]
    image = normalize(image_u8.to(device))
    grid, registers = cfg.canvas_grid_size, canvit.config.num_canvas_registers

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []  # per Write: (residual [G*G, D], sink weight [G*G])

    def capture(module, args, kwargs, output):
        sink_weight = write_parts(module, **kwargs)[2][0, registers:, SINK]
        captured.append((output[0, registers:].float(), sink_weight))

    handles = [write.register_forward_hook(capture, with_kwargs=True) for write in canvit.canvas_writes]
    policy = make_policy("coarse_to_fine", batch_size=1, device=device, num_glimpses=max(SHOWN) + 1,
                         canvas_grid_size=grid)
    state = canvit.init_state(batch_size=1, canvas_grid_size=grid)
    maps = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for t in range(max(SHOWN) + 1):
            viewpoint = policy.step(t, state)
            captured.clear()
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
            state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
            if t in SHOWN:
                residual = sum(r for r, _ in captured).norm(dim=-1).view(grid, grid).cpu().numpy()
                sink = torch.stack([w for _, w in captured]).mean(dim=0).view(grid, grid).cpu().numpy()
                box = crop_box_px(viewpoint, image_size_px=RELEASED_SCENE_SIZE_PX)[0].tolist()
                maps[t] = (residual, sink, box)
    for handle in handles:
        handle.remove()

    norm = LogNorm(vmin=min(m[0].min() for m in maps.values()), vmax=max(m[0].max() for m in maps.values()))
    fig, axes = plt.subplots(len(SHOWN), 3, figsize=(10.5, 3.6 * len(SHOWN)))
    scene = image_u8[0].permute(1, 2, 0).numpy()
    for row, (t, title) in zip(axes, SHOWN.items()):
        residual, sink, (y0, x0, y1, x1) = maps[t]
        cell = RELEASED_SCENE_SIZE_PX / grid
        row[0].imshow(scene)
        row[0].set_title(f"{title}: the crop", fontsize=9)
        im_residual = row[1].imshow(residual, norm=norm, cmap="magma", interpolation="nearest")
        row[1].set_title("norm of the Write residual per canvas cell", fontsize=9)
        im_sink = row[2].imshow(sink, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        row[2].set_title("attention on backbone register 3", fontsize=9)
        for ax, scale in zip(row, (1, 1 / cell, 1 / cell)):
            ax.add_patch(Rectangle((x0 * scale - 0.5 * (scale != 1), y0 * scale - 0.5 * (scale != 1)),
                                   (x1 - x0) * scale, (y1 - y0) * scale, fill=False, edgecolor="#4080d0", linewidth=2))
            ax.set_xticks([])
            ax.set_yticks([])
        inside = np.zeros_like(residual, dtype=bool)
        inside[int(round(y0 / cell)):int(round(y1 / cell)), int(round(x0 / cell)):int(round(x1 / cell))] = True
        row[1].set_xlabel(f"mean outside / inside: {residual[~inside].mean() / residual[inside].mean():.2f}", fontsize=8)
        row[2].set_xlabel(f"mean outside {sink[~inside].mean():.2f}, inside {sink[inside].mean():.2f}", fontsize=8)
        fig.colorbar(im_residual, ax=row[1], fraction=0.046)
        fig.colorbar(im_sink, ax=row[2], fraction=0.046)
    fig.suptitle(f"Where a glimpse writes: C2F on ADE20K validation scene {cfg.image_index}, 64×64 canvas", fontsize=10)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=130, bbox_inches="tight")
    print(f"Wrote {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Figure))
