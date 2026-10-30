"""How much of a glimpse's Canvas Attention Writes lands outside its crop, along C2F's levels on ADE20K validation
scenes (512 px, the released CanViT-B, 64×64 canvas, 85 glimpses down to level 3).

A Write lets every canvas cell attend to the glimpse's tokens, with no spatial mask, so a glimpse also changes the
cells it did not see. Per glimpse, the residual a cell receives is the sum of the three Writes' residuals; the script
reports its mean L2 norm in the cells inside and outside the crop, per C2F level. Over level 3 (64 tiles, each seen
once), it also sums each cell's residuals from the 63 glimpses of other tiles and from the glimpse of its own tile,
and reports the mean norm of both sums: the net leaked change, after cancellations, against the cell's own update.

    .venv/bin/python -P scripts/write_leakage.py --output results/write_leakage.json
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch import Tensor

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import make_policy
from canvit_pytorch.preprocess import imagenet_normalize
from canvit_pytorch.viewpoint import grid_coords

log = logging.getLogger(__name__)

LEVEL_STARTS = (0, 1, 5, 21)
"""First glimpse of each C2F level (1, 4, 16 and 64 tiles)."""


@dataclass(frozen=True)
class Measurement:
    output: Path
    num_scenes: int = 256
    """The first validation scenes, in the dataset's order."""
    canvas_grid_size: int = 64
    seed: int = 0
    batch_size: int = 16
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    """uint8 validation scenes at 512 px, as written by scripts/ade20k_rollouts.py."""
    device: str = "cuda"


def main(m: Measurement) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(m.seed)
    device = torch.device(m.device)
    canvit = CanViTForPretraining.from_pretrained(FLAGSHIP).canvit.to(device).eval()
    num_registers, grid = canvit.config.num_canvas_registers, m.canvas_grid_size
    scenes = torch.load(m.cache, weights_only=True)["images"][:m.num_scenes]
    assert scenes.shape[-1] == RELEASED_SCENE_SIZE_PX and len(scenes) == m.num_scenes, scenes.shape
    cell_centers = grid_coords(size=grid, device=device)

    residuals: list[Tensor] = []
    handles = [write.register_forward_hook(lambda module, args, output: residuals.append(output[:, num_registers:].float()))
               for write in canvit.canvas_writes]
    num_glimpses, num_levels = LEVEL_STARTS[-1] + 4 ** (len(LEVEL_STARTS) - 1), len(LEVEL_STARTS)
    inside_norm, outside_norm = torch.zeros(num_levels, dtype=torch.float64), torch.zeros(num_levels, dtype=torch.float64)
    glimpse_counts = torch.zeros(num_levels, dtype=torch.float64)
    leaked_sum_norm = own_sum_norm = canvas_norm = 0.0
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for first in range(0, m.num_scenes, m.batch_size):
            images = imagenet_normalize(scenes[first:first + m.batch_size].to(device).float() / 255)
            batch_size = images.shape[0]
            policy = make_policy("coarse_to_fine", batch_size=batch_size, device=device, num_glimpses=num_glimpses,
                                 canvas_grid_size=grid)
            state = canvit.init_state(batch_size=batch_size, canvas_grid_size=grid)
            leaked = torch.zeros(batch_size, grid * grid, canvit.config.canvas_dim, device=device)
            own = torch.zeros_like(leaked)
            for t in range(num_glimpses):
                viewpoint = policy.step(t, state)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                residuals.clear()
                state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                residual = sum(residuals)  # [B, G*G, D]: the three Writes' residuals into each cell
                offsets = (cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs()
                inside = (offsets <= viewpoint.scales.view(-1, 1, 1, 1)).all(dim=-1).flatten(1)  # [B, G*G]
                norms = residual.norm(dim=-1)
                level = sum(t >= start for start in LEVEL_STARTS) - 1
                inside_norm[level] += norms[inside].mean().double().cpu() * batch_size
                if (~inside).any():
                    outside_norm[level] += norms[~inside].mean().double().cpu() * batch_size
                glimpse_counts[level] += batch_size
                if level == num_levels - 1:
                    leaked += residual * (~inside).unsqueeze(-1)
                    own += residual * inside.unsqueeze(-1)
            leaked_sum_norm += leaked.norm(dim=-1).mean(dim=1).sum().item()
            own_sum_norm += own.norm(dim=-1).mean(dim=1).sum().item()
            canvas_norm += state.canvas[:, num_registers:].float().norm(dim=-1).mean(dim=1).sum().item()
            log.info("%d/%d scenes", first + batch_size, m.num_scenes)
    for handle in handles:
        handle.remove()

    levels = [{"level": level, "first_glimpse": LEVEL_STARTS[level], "glimpses": 4 ** level,
               "residual_norm_inside": (inside_norm[level] / glimpse_counts[level]).item(),
               "residual_norm_outside": (outside_norm[level] / glimpse_counts[level]).item() if level else None,
               "outside_over_inside": (outside_norm[level] / inside_norm[level]).item() if level else None}
              for level in range(num_levels)]
    result = {
        "measurement": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(m).items()},
        "pretrained_repo": FLAGSHIP,
        "levels": levels,
        "level3_leaked_sum_norm": leaked_sum_norm / m.num_scenes,
        "level3_own_sum_norm": own_sum_norm / m.num_scenes,
        "level3_leaked_over_own": leaked_sum_norm / own_sum_norm,
        "final_canvas_patch_norm": canvas_norm / m.num_scenes,
    }
    m.output.parent.mkdir(parents=True, exist_ok=True)
    m.output.write_text(json.dumps(result, indent=1))
    for level in levels[1:]:
        log.info("level %d: residual norm per cell inside %.0f, outside %.0f (ratio %.3f)", level["level"],
                 level["residual_norm_inside"], level["residual_norm_outside"], level["outside_over_inside"])
    log.info("level 3, summed per cell: leaked %.0f, own %.0f (ratio %.2f); final canvas patch norm %.0f",
             result["level3_leaked_sum_norm"], result["level3_own_sum_norm"], result["level3_leaked_over_own"],
             result["final_canvas_patch_norm"])


if __name__ == "__main__":
    main(tyro.cli(Measurement))
