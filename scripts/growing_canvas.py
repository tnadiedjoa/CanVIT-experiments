"""ADE20K mIoU after every glimpse with bio_vision.GrowingCanvas: a 32×32 canvas that recruits 64×64-density cells in
the crops of the first glimpses at C2F's level 2 (where the eye zooms to the scene's native resolution). Coarse cells
are decoded by the released 32×32 probe, fine cells by the released 64×64 probe; the segmentation is the fine cells'
where they exist. budget 0 is the 32×32 canvas, budget 16 covers the scene like the 64×64 canvas. With the entropy
order, the level's tiles are visited from the most uncertain under the coarse canvas (EG-C2F), so the fine cells go
where the scene is hardest.

    .venv/bin/python -P scripts/growing_canvas.py --budget 4 --output results/bio/grow_c2f_b4_t21.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import tyro

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX, released_ade20k_probe
from canvit_pytorch.model import CanViTForSemanticSegmentation
from canvit_pytorch.policies import make_policy
from canvit_pytorch.policies.entropy import predictive_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import ScoreGuidedC2F, load_validation_scenes, normalize  # noqa: E402
from bio_vision import FreePositions, GrowingCanvas, UniformRetina, take_glimpse  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    output: Path
    budget: int = 4
    order: Literal["random", "entropy"] = "random"
    num_glimpses: int = 21
    max_scale: float = 0.25
    seed: int = 0
    batch_size: int = 8
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    num_images: int | None = None
    device: str = "cuda"


@torch.inference_mode()
def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    scene_px, T, Gc, Gf = RELEASED_SCENE_SIZE_PX, exp.num_glimpses, 32, 64
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=FLAGSHIP, probe_repo=released_ade20k_probe("in21k", scene_size_px=scene_px, canvas_grid_size=Gc))
    seg = seg.to(device).eval()
    fine_probe = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=FLAGSHIP, probe_repo=released_ade20k_probe("in21k", scene_size_px=scene_px,
                                                                   canvas_grid_size=Gf)).probe.to(device).eval()
    retina = UniformRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=seg.canvit.patch_size, device=device)
    free = FreePositions(seg.canvit)
    grow = GrowingCanvas(seg.canvit, coarse_grid_size=Gc, fine_grid_size=Gf, max_scale=exp.max_scale,
                         budget=exp.budget, device=device)
    R = seg.canvit.config.num_canvas_registers
    images_u8, labels_u8 = load_validation_scenes(exp.cache, size_px=scene_px)
    images_u8, labels_u8 = images_u8[:exp.num_images], labels_u8[:exp.num_images]

    def coarse_logits(state) -> torch.Tensor:
        cells = state.canvas[:, R:R + Gc * Gc].float().view(-1, Gc, Gc, state.canvas.shape[-1])
        return seg.probe(cells)  # [B, C, Gc, Gc]

    mious = [ade20k.MeanIoU(device=device) for _ in range(T)]
    cells = torch.zeros(T, dtype=torch.float64)
    start = time.monotonic()
    for first in range(0, len(images_u8), exp.batch_size):
        images = normalize(images_u8[first:first + exp.batch_size].to(device))
        labels = labels_u8[first:first + exp.batch_size].to(device).long()
        B = images.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if exp.order == "random":
                policy = make_policy("coarse_to_fine", batch_size=B, device=device, num_glimpses=T, canvas_grid_size=Gc)
            else:
                policy = ScoreGuidedC2F(batch_size=B, device=device, num_glimpses=T, canvas_grid_size=Gc,
                                        cell_score=lambda state: predictive_entropy(coarse_logits(state)))
            state = seg.init_state(batch_size=B, canvas_grid_size=Gc)
            free.canvas = grow.reset(B)
            for t in range(T):
                viewpoint = policy.step(t, state)
                state, free.canvas = grow.recruit(state, viewpoint)
                glimpse, free.glimpse = take_glimpse(images, viewpoint, retina)
                state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                coarse = F.interpolate(coarse_logits(state), size=(scene_px, scene_px), mode="bilinear",
                                       align_corners=False)
                if grow.num_cells > Gc * Gc:
                    fine_cells = state.canvas[:, R + Gc * Gc:].float().unsqueeze(1)  # [B, 1, n, D]
                    fine_logits = fine_probe(fine_cells)[:, :, 0].transpose(1, 2)  # [B, n, C]
                    fine, mask = grow.fine_map(fine_logits)
                    fine = F.interpolate(fine, size=(scene_px, scene_px), mode="bilinear", align_corners=False)
                    mask = F.interpolate(mask, size=(scene_px, scene_px), mode="bilinear", align_corners=False)
                    coarse = torch.where(mask > 0.5, fine / mask.clamp(min=1e-6), coarse)
                mious[t].update(coarse.argmax(dim=1), labels)
                cells[t] += grow.num_cells * B
        log.info("%d/%d scenes", first + B, len(images_u8))

    n = len(images_u8)
    result = {"experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
              "miou": [100 * m.compute() for m in mious], "canvas_cells": (cells / n).tolist(), "n_images": n,
              "wall_time_seconds": time.monotonic() - start}
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("mIoU by glimpse: %s", " ".join(f"{v:.2f}" for v in result["miou"]))


if __name__ == "__main__":
    main(tyro.cli(Experiment))
