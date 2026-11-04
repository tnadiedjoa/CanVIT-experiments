"""ADE20K validation mIoU after every glimpse with the primate-vision mechanisms of scripts/bio_vision.py: foveated or
learned retinas, repetition suppression in the Writes, object-shaped Writes, a canvas spread by a density, and
microsaccades. Everything else is scripts/ade20k_rollouts.py's (released model, probes, scenes and metric), so the
released settings give the same numbers.

    .venv/bin/python -P scripts/bio_rollouts.py --policy repeated_full_scene --num-glimpses 85 --canvas-grid-size 64 \
        --novelty project --novelty-strength 1 --output results/bio/rfs_c64_t85_novelty_project1.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn.functional as F
import tyro

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import predicted_labels
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX, released_ade20k_probe
from canvit_pytorch.model import CanViT, CanViTForSemanticSegmentation
from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import load_validation_scenes, normalize  # noqa: E402
from local_writes import LocalWrites  # noqa: E402
from bio_vision import (AdaptiveCanvas, FoveatedRetina, FreePositions, GaussianCanvas, LearnedPeriods,  # noqa: E402
                        NoveltyWrites, ObjectWrites, PatchRetina, UniformRetina, WriteFootprint, boundary_density,
                        c2f_then_jittered_level2, image_edge_density, microsaccades, take_glimpse)

log = logging.getLogger(__name__)

BioPolicy = PolicyName | Literal["microsaccades", "c2f_then_jittered_level2"]
Density = Literal["oracle_boundaries", "glimpse_edges", "entropy"]
"""What spreads an adaptive canvas's cells: the ground truth's boundaries (an upper reference, not a method); the
intensity edges of the first, full-scene glimpse (set before it, from its pixels alone); or the probe's entropy after
the first glimpse (the uniform canvas is then resampled at the new cells)."""


@dataclass(frozen=True)
class Experiment:
    output: Path
    policy: BioPolicy = "coarse_to_fine"
    num_glimpses: int = 21
    canvas_grid_size: int = 64
    probe_canvas_grid_size: int | None = None
    fovea: float | None = None
    """When set, glimpses through bio_vision.FoveatedRetina with this fovea constant (smaller is more foveated)."""
    trained: Path | None = None
    """When set, a training of scripts/train_retina.py: its retina (bio_vision.PatchRetina), its learned canvas shape
    (bio_vision.GaussianCanvas) if any, and its CanViT weights if it fine-tuned them."""
    shape_readout_px: int = 128
    """With a learned canvas shape: the segmentation logits are read through the cells' Gaussians on a grid of this
    many points per side, then upsampled bilinearly to the scene as released."""
    novelty: Literal["project", "gate"] | None = None
    novelty_strength: float = 1.0
    novelty_from: int = 1
    object_writes_threshold: float | None = None
    object_writes_from: int = 21
    local_writes_from: int | None = None
    """When set, from this glimpse on, the Writes are kept in the glimpse's crop (local_writes.LocalWrites, the README's
    fix for C2F's level 3)."""
    density: Density | None = None
    density_mix: float = 0.5
    microsaccade_scale: float = 0.875
    microsaccade_jitter: float = 0.125
    """microsaccades: views of this scale, centers jittered by at most this much; c2f_then_jittered_level2: the
    jitter of the level-2 revisits (1/32 is half a level-2 patch)."""
    microsaccade_outside: bool = False
    """microsaccades: let the jittered views leave the valid box (their part beyond the scene is black)."""
    microsaccade_sequence: Literal["random", "halton"] = "random"
    microsaccade_first_full_scene: bool = True
    """microsaccades: start with the full scene; with --no-microsaccade-first-full-scene and a jitter of 1, every
    glimpse is at a random valid center, the fixed-scale saccades of scripts/train_retina.py's --scale."""
    seed: int = 0
    batch_size: int = 8
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    num_images: int | None = None
    device: str = "cuda"


def build_policy(exp: Experiment, batch_size: int, device: torch.device, canvas_logits=None):
    if exp.policy in get_args(PolicyName):
        return make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                           canvas_grid_size=exp.canvas_grid_size, canvas_logits=canvas_logits)
    if exp.policy == "microsaccades":
        return microsaccades(batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                             scale=exp.microsaccade_scale, jitter=exp.microsaccade_jitter,
                             first_full_scene=exp.microsaccade_first_full_scene, outside=exp.microsaccade_outside,
                             sequence=exp.microsaccade_sequence)
    return c2f_then_jittered_level2(batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                                    jitter=exp.microsaccade_jitter)


def load_retina(exp: Experiment, canvit: CanViT, device: torch.device, free: FreePositions):
    """(retina, learned canvas shape or None); a training also loads its fine-tuned CanViT weights, if any, into
    canvit."""
    patch_px = canvit.patch_size
    if exp.trained is not None:
        saved = torch.load(exp.trained, map_location=device, weights_only=False)
        retina = PatchRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=patch_px, device=device)
        retina.load_state_dict(saved["retina"])
        if saved.get("canvit") is not None:
            canvit.load_state_dict(saved["canvit"])
        if saved.get("periods_state") is not None:
            free.periods = LearnedPeriods(canvit, device)
            free.periods.load_state_dict(saved["periods_state"])
            free.periods.requires_grad_(False)
        shape = None
        if saved.get("canvas_shape_state") is not None:
            config = saved["config"]
            assert exp.canvas_grid_size == 32, "the canvas shapes are learned for the 32×32 teacher grid of pretraining"
            shape = GaussianCanvas(canvas_grid_size=config.get("canvas_grid_size") or 32, canvas_dim=canvit.canvas_dim,
                                   device=device, adaptive=config["canvas_adaptive"],
                                   coarse_grid_size=config.get("canvas_coarse_grid_size"),
                                   surround=config.get("canvas_surround", False))
            shape.load_state_dict(saved["canvas_shape_state"])
            shape.requires_grad_(False)
            if saved.get("write_footprint_state") is not None:
                footprint = WriteFootprint(canvit, shape, free)
                footprint.load_state_dict(saved["write_footprint_state"])
                footprint.requires_grad_(False)
        return retina.requires_grad_(False), shape
    if exp.fovea is not None:
        return FoveatedRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=patch_px, device=device, fovea=exp.fovea), None
    return UniformRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=patch_px, device=device), None


@torch.inference_mode()
def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    scene_px, T, G = RELEASED_SCENE_SIZE_PX, exp.num_glimpses, exp.canvas_grid_size
    probe_repo = released_ade20k_probe("in21k", scene_size_px=scene_px,
                                       canvas_grid_size=exp.probe_canvas_grid_size or G)
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=probe_repo)
    seg = seg.to(device).eval()
    free = FreePositions(seg.canvit)
    retina, shape = load_retina(exp, seg.canvit, device, free)
    shape_points = grid_coords(size=exp.shape_readout_px, device=device).flatten(0, 1)
    novelty = (NoveltyWrites(seg.canvit, mode=exp.novelty, strength=exp.novelty_strength, start=exp.novelty_from,
                             num_glimpses=T, device=device) if exp.novelty is not None else None)
    objects = (ObjectWrites(seg.canvit, canvas_grid_size=G, threshold=exp.object_writes_threshold,
                            start=exp.object_writes_from, num_glimpses=T, device=device)
               if exp.object_writes_threshold is not None else None)
    local_writes = (LocalWrites(seg.canvit, canvas_grid_size=G, device=device, start=exp.local_writes_from)
                    if exp.local_writes_from is not None else None)
    images_u8, labels_u8 = load_validation_scenes(exp.cache, size_px=scene_px)
    images_u8, labels_u8 = images_u8[:exp.num_images], labels_u8[:exp.num_images]

    mious = [ade20k.MeanIoU(device=device) for _ in range(T)]
    sums = {name: torch.zeros(T, dtype=torch.float64, device=device)
            for name in ("canvas_patch_norm", "delta_previous", "entropy")}
    start = time.monotonic()
    for first in range(0, len(images_u8), exp.batch_size):
        images = normalize(images_u8[first:first + exp.batch_size].to(device))
        labels = labels_u8[first:first + exp.batch_size].to(device).long()
        B = images.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy = build_policy(exp, B, device, canvas_logits=seg.logits)
            state = (seg.init_state(batch_size=B, canvas_grid_size=G) if shape is None
                     else shape.init_state(seg.canvit, B))
            canvas = None
            if exp.density == "oracle_boundaries":
                canvas = AdaptiveCanvas(boundary_density(labels, blur_px=33), canvas_grid_size=G, mix=exp.density_mix)
            free.canvas = None if canvas is None else canvas.positions()
            if shape is not None:
                free.canvas = shape.reset(B)
            previous = normalized_cells(seg, state.canvas)
            for t in range(T):
                viewpoint = policy.step(t, state)
                glimpse, free.glimpse = take_glimpse(images, viewpoint, retina)
                if t == 0 and exp.density == "glimpse_edges":
                    edges = F.interpolate(image_edge_density(glimpse).unsqueeze(1), size=(scene_px, scene_px),
                                          mode="bilinear", align_corners=False)[:, 0]
                    canvas = AdaptiveCanvas(edges, canvas_grid_size=G, mix=exp.density_mix)
                    free.canvas = canvas.positions()
                if novelty is not None:
                    novelty.before_glimpse(t)
                if objects is not None:
                    objects.before_glimpse(t, viewpoint, state.canvas)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                if t == 0 and exp.density == "entropy":
                    entropy = predictive_entropy(seg.logits(state.canvas)).float()  # [B, G, G]
                    density = F.interpolate(entropy.unsqueeze(1), size=(scene_px, scene_px), mode="bilinear",
                                            align_corners=False)[:, 0]
                    canvas = AdaptiveCanvas(density, canvas_grid_size=G, mix=exp.density_mix)
                    R = seg.canvit.config.num_canvas_registers
                    resampled = canvas.resample_uniform(seg.canvit.canvas_patches(state.canvas))
                    state = RecurrentState(canvas=torch.cat([state.canvas[:, :R], resampled], dim=1),
                                           recurrent_cls=state.recurrent_cls)
                    free.canvas = canvas.positions()
                if shape is not None and shape.adaptive and t == 0:
                    state, free.canvas = shape.adapt(state, seg.canvit)
                if shape is not None:
                    patches = seg.canvit.canvas_patches(state.canvas).float().unsqueeze(1)  # [B, 1, N, D]
                else:
                    patches = seg.canvit.canvas_patch_grid(state.canvas).float()
                logits = seg.probe(patches)
                if shape is not None:
                    n = exp.shape_readout_px
                    read = shape.read(logits.flatten(2).transpose(1, 2), shape_points)  # [B, n*n, C]
                    predicted = predicted_labels(read.transpose(1, 2).reshape(B, -1, n, n), size_px=scene_px)
                elif canvas is None:
                    predicted = predicted_labels(logits, size_px=scene_px)
                else:
                    predicted = torch.cat([canvas_part.read_back(chunk, scene_px).argmax(dim=1)
                                           for canvas_part, chunk in zip(split(canvas, 2), logits.split(2))])
                mious[t].update(predicted, labels)
                current = normalized_cells(seg, state.canvas)
                sums["canvas_patch_norm"][t] += patches.norm(dim=-1).mean(dim=(1, 2)).sum().double()
                sums["delta_previous"][t] += (1 - F.cosine_similarity(current, previous, dim=-1)).mean(dim=1).sum().double()
                sums["entropy"][t] += predictive_entropy(logits).mean(dim=(1, 2)).sum().double()
                previous = current
        log.info("%d/%d scenes", first + B, len(images_u8))

    n = len(images_u8)
    result = {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_repo": probe_repo,
        "miou": [100 * m.compute() for m in mious],
        **{name: (values / n).tolist() for name, values in sums.items()},
        "n_images": n,
        "wall_time_seconds": time.monotonic() - start,
    }
    if novelty is not None:
        result["novelty_suppressed"] = (novelty.suppressed / n).tolist()
    if objects is not None:
        result["object_writes_reached"] = (objects.reached / n).tolist()
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("mIoU by glimpse: %s", " ".join(f"{v:.2f}" for v in result["miou"]))
    log.info("Wrote %s in %.0f s", exp.output, result["wall_time_seconds"])


def normalized_cells(seg: CanViTForSemanticSegmentation, canvas: torch.Tensor) -> torch.Tensor:
    """[B, N, D] layer-normalized canvas cells, float32 (any number of cells, not only a square grid)."""
    cells = seg.canvit.canvas_patches(canvas).float()
    return F.layer_norm(cells, (cells.shape[-1],))


def split(canvas: AdaptiveCanvas, size: int) -> list[AdaptiveCanvas]:
    """The adaptive canvas of each chunk of `size` scenes, to read back logits a few scenes at a time."""
    parts = []
    for first in range(0, canvas.rows.shape[0], size):
        part = AdaptiveCanvas.__new__(AdaptiveCanvas)
        part.rows, part.cols, part.G = canvas.rows[first:first + size], canvas.cols[first:first + size], canvas.G
        part.row_positions = canvas.row_positions[first:first + size]
        part.col_positions = canvas.col_positions[first:first + size]
        parts.append(part)
    return parts


if __name__ == "__main__":
    main(tyro.cli(Experiment))
