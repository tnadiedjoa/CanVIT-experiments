"""How much of a glimpse's Canvas Attention Writes lands outside its crop, along C2F's levels on ADE20K validation
scenes (512 px, the released CanViT-B, 64×64 canvas, 85 glimpses down to level 3).

A Write lets every canvas cell attend to the glimpse's tokens, with no spatial mask, so a glimpse also changes the
cells it did not see. Per glimpse, the residual a cell receives is the sum of the three Writes' residuals; the script
reports its mean L2 norm in the cells inside and outside the crop, per C2F level. Over level 3 (64 tiles, each seen
once), it also sums each cell's residuals from the 63 glimpses of other tiles and from the glimpse of its own tile,
and reports the mean norm of both sums: the net leaked change, after cancellations, against the cell's own update.
Two more statistics say where the leak goes: the residual norm outside the crop by distance from it, in tile widths,
and the share of the leaked sum's energy in its mean over the canvas, i.e. in one vector added to every cell.

Each Write's residual is also split by key (scripts/local_writes.py, write_parts): the part from the glimpse's prefix
tokens (VPE, recurrent CLS, registers), whose keys carry no position, and the part from its patch tokens; the script
reports the attention weight on the prefix and the norm of both parts, inside and outside the crop.

    .venv/bin/python -P scripts/write_leakage.py --output results/write_leakage.json
"""

import json
import logging
import sys
from collections import defaultdict
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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from local_writes import write_parts  # noqa: E402

log = logging.getLogger(__name__)

LEVEL_STARTS = (0, 1, 5, 21)
"""First glimpse of each C2F level (1, 4, 16 and 64 tiles)."""
DISTANCE_BINS = (0, 0.5, 1, 2, 4, 8)
"""Edges of the bins of distance from the crop to an outside cell's center (Chebyshev), in widths of the crop."""


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
    parts: list[tuple[Tensor, Tensor, Tensor]] = []  # per Write, cells only: prefix part, patch part, prefix weight

    def capture(module, args, kwargs, output: Tensor) -> None:
        residuals.append(output[:, num_registers:].float())
        parts.append(tuple(part[:, num_registers:] for part in write_parts(module, **kwargs)))

    handles = [write.register_forward_hook(capture, with_kwargs=True) for write in canvit.canvas_writes]
    num_glimpses, num_levels = LEVEL_STARTS[-1] + 4 ** (len(LEVEL_STARTS) - 1), len(LEVEL_STARTS)
    inside_norm, outside_norm = torch.zeros(num_levels, dtype=torch.float64), torch.zeros(num_levels, dtype=torch.float64)
    glimpse_counts = torch.zeros(num_levels, dtype=torch.float64)
    edges = torch.tensor(DISTANCE_BINS, device=device)
    by_distance = torch.zeros(num_levels, len(DISTANCE_BINS) - 1, dtype=torch.float64, device=device)
    cells_by_distance = torch.zeros_like(by_distance)
    split = defaultdict(lambda: torch.zeros(num_levels, dtype=torch.float64))  # "<quantity>_<inside|outside>"
    per_write = torch.zeros(2, num_levels, len(canvit.canvas_writes), dtype=torch.float64)  # inside, outside norms
    leaked_sum_norm = own_sum_norm = canvas_norm = common_fraction = 0.0
    leaked_parts_norm, leaked_parts_common = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for first in range(0, m.num_scenes, m.batch_size):
            images = imagenet_normalize(scenes[first:first + m.batch_size].to(device).float() / 255)
            batch_size = images.shape[0]
            policy = make_policy("coarse_to_fine", batch_size=batch_size, device=device, num_glimpses=num_glimpses,
                                 canvas_grid_size=grid)
            state = canvit.init_state(batch_size=batch_size, canvas_grid_size=grid)
            leaked = torch.zeros(batch_size, grid * grid, canvit.config.canvas_dim, device=device)
            own = torch.zeros_like(leaked)
            leaked_prefix, leaked_patches = torch.zeros_like(leaked), torch.zeros_like(leaked)
            for t in range(num_glimpses):
                viewpoint = policy.step(t, state)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                residuals.clear()
                parts.clear()
                state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                residual = sum(residuals)  # [B, G*G, D]: the three Writes' residuals into each cell
                offsets = (cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs()
                inside = (offsets <= viewpoint.scales.view(-1, 1, 1, 1)).all(dim=-1).flatten(1)  # [B, G*G]
                norms = residual.norm(dim=-1)
                gap = (offsets - viewpoint.scales.view(-1, 1, 1, 1)).amax(dim=-1).flatten(1) / (2 * viewpoint.scales.view(-1, 1))
                distance_bin = torch.bucketize(gap, edges, right=True) - 1  # -1 inside the crop, beyond the last edge too
                level = sum(t >= start for start in LEVEL_STARTS) - 1
                inside_norm[level] += norms[inside].mean().double().cpu() * batch_size
                if (~inside).any():
                    outside_norm[level] += norms[~inside].mean().double().cpu() * batch_size
                glimpse_counts[level] += batch_size
                prefix, patches = sum(p[0] for p in parts), sum(p[1] for p in parts)
                prefix_weight = sum(p[2] for p in parts) / len(parts)
                for name, values in (("prefix_weight", prefix_weight), ("prefix_norm", prefix.norm(dim=-1)),
                                     ("patch_norm", patches.norm(dim=-1))):
                    split[f"{name}_inside"][level] += values[inside].mean().double().cpu() * batch_size
                    if (~inside).any():
                        split[f"{name}_outside"][level] += values[~inside].mean().double().cpu() * batch_size
                for i, write_residual in enumerate(residuals):
                    write_norms = write_residual.norm(dim=-1)
                    per_write[0, level, i] += write_norms[inside].mean().double().cpu() * batch_size
                    if (~inside).any():
                        per_write[1, level, i] += write_norms[~inside].mean().double().cpu() * batch_size
                for b in range(len(DISTANCE_BINS) - 1):
                    in_bin = (distance_bin == b) & ~inside
                    by_distance[level, b] += norms[in_bin].double().sum()
                    cells_by_distance[level, b] += in_bin.sum()
                if level == num_levels - 1:
                    leaked += residual * (~inside).unsqueeze(-1)
                    own += residual * inside.unsqueeze(-1)
                    leaked_prefix += prefix * (~inside).unsqueeze(-1)
                    leaked_patches += patches * (~inside).unsqueeze(-1)
            leaked_sum_norm += leaked.norm(dim=-1).mean(dim=1).sum().item()
            own_sum_norm += own.norm(dim=-1).mean(dim=1).sum().item()
            common_fraction += (leaked.mean(dim=1).square().sum(dim=-1) / leaked.square().sum(dim=-1).mean(dim=1)).sum().item()
            for i, part in enumerate((leaked_prefix, leaked_patches)):
                leaked_parts_norm[i] += part.norm(dim=-1).mean(dim=1).sum().item()
                leaked_parts_common[i] += (part.mean(dim=1).square().sum(dim=-1) / part.square().sum(dim=-1).mean(dim=1)).sum().item()
            canvas_norm += state.canvas[:, num_registers:].float().norm(dim=-1).mean(dim=1).sum().item()
            log.info("%d/%d scenes", first + batch_size, m.num_scenes)
    for handle in handles:
        handle.remove()

    levels = [{"level": level, "first_glimpse": LEVEL_STARTS[level], "glimpses": 4 ** level,
               "residual_norm_inside": (inside_norm[level] / glimpse_counts[level]).item(),
               "residual_norm_outside": (outside_norm[level] / glimpse_counts[level]).item() if level else None,
               "outside_over_inside": (outside_norm[level] / inside_norm[level]).item() if level else None,
               "outside_by_distance": [{"crop_widths": [lo, hi], "cells": int(cells_by_distance[level, b].item()),
                                        "outside_over_inside": (by_distance[level, b] / cells_by_distance[level, b]).item()
                                        / (inside_norm[level] / glimpse_counts[level]).item()}
                                       for b, (lo, hi) in enumerate(zip(DISTANCE_BINS, DISTANCE_BINS[1:]))
                                       if cells_by_distance[level, b] > 0],
               **{name: (values[level] / glimpse_counts[level]).item() for name, values in sorted(split.items())
                  if level or name.endswith("_inside")},
               "per_write_outside_over_inside": (per_write[1, level] / per_write[0, level]).tolist() if level else None}
              for level in range(num_levels)]
    result = {
        "measurement": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(m).items()},
        "pretrained_repo": FLAGSHIP,
        "levels": levels,
        "level3_leaked_sum_norm": leaked_sum_norm / m.num_scenes,
        "level3_own_sum_norm": own_sum_norm / m.num_scenes,
        "level3_leaked_over_own": leaked_sum_norm / own_sum_norm,
        "level3_leaked_common_fraction": common_fraction / m.num_scenes,
        "level3_leaked_prefix_sum_norm": leaked_parts_norm[0].item() / m.num_scenes,
        "level3_leaked_patch_sum_norm": leaked_parts_norm[1].item() / m.num_scenes,
        "level3_leaked_prefix_common_fraction": leaked_parts_common[0].item() / m.num_scenes,
        "level3_leaked_patch_common_fraction": leaked_parts_common[1].item() / m.num_scenes,
        "final_canvas_patch_norm": canvas_norm / m.num_scenes,
    }
    m.output.parent.mkdir(parents=True, exist_ok=True)
    m.output.write_text(json.dumps(result, indent=1))
    for level in levels[1:]:
        log.info("level %d: residual norm per cell inside %.0f, outside %.0f (ratio %.3f)", level["level"],
                 level["residual_norm_inside"], level["residual_norm_outside"], level["outside_over_inside"])
        log.info("  by distance from the crop: %s", ", ".join(
            f"{d['crop_widths'][0]}-{d['crop_widths'][1]} widths {d['outside_over_inside']:.3f}" for d in level["outside_by_distance"]))
        log.info("  attention on the prefix inside %.2f, outside %.2f; prefix part norm inside %.0f, outside %.0f; patch "
                 "part norm inside %.0f, outside %.0f; per Write outside/inside %s", level["prefix_weight_inside"],
                 level["prefix_weight_outside"], level["prefix_norm_inside"], level["prefix_norm_outside"],
                 level["patch_norm_inside"], level["patch_norm_outside"],
                 " ".join(f"{r:.3f}" for r in level["per_write_outside_over_inside"]))
    log.info("level 3, summed per cell: leaked %.0f, own %.0f (ratio %.2f), %.0f%% of the leak's energy in its canvas "
             "mean; final canvas patch norm %.0f", result["level3_leaked_sum_norm"], result["level3_own_sum_norm"],
             result["level3_leaked_over_own"], 100 * result["level3_leaked_common_fraction"], result["final_canvas_patch_norm"])
    log.info("level 3 leak by part: prefix sum norm %.0f (%.0f%% common), patch sum norm %.0f (%.0f%% common)",
             result["level3_leaked_prefix_sum_norm"], 100 * result["level3_leaked_prefix_common_fraction"],
             result["level3_leaked_patch_sum_norm"], 100 * result["level3_leaked_patch_common_fraction"])


if __name__ == "__main__":
    main(tyro.cli(Measurement))
