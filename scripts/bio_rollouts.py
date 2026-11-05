"""ADE20K validation mIoU after every glimpse with the primate-vision mechanisms of scripts/bio_vision.py: foveated or
learned retinas, repetition suppression in the Writes, object-shaped Writes, a canvas spread by a density,
microsaccades, and glimpses shaped by uncertainty. Everything else is scripts/ade20k_rollouts.py's (released model,
probes, scenes and metric), so the released settings give the same numbers.

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
from canvit_pytorch.benchmarks.ade20k.metrics import per_image_confusion
from canvit_pytorch.hub.repos import (PRETRAINED, RELEASED_GLIMPSE_SIZE_PX, RELEASED_PROBE_STEPS, RELEASED_SCENE_SIZE_PX,
                                      ade20k_probe_name, hub_repo)
from canvit_pytorch.model import CanViT, CanViTForSemanticSegmentation
from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.policies.entropy import predictive_entropy
from canvit_pytorch.pretrain.ablations import ABLATIONS
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import load_validation_scenes, normalize  # noqa: E402
from local_writes import LocalWrites, SinkGatedWrites  # noqa: E402
from bio_vision import (AdaptiveCanvas, FoveatedRetina, FreePositions, GaussianCanvas, LearnedPeriods,  # noqa: E402
                        MaskedWrites, NoveltyWrites, ObjectWrites, PatchRetina, Patchwork, UniformRetina, WriteFootprint,
                        boundary_density, c2f_then_jittered_level2, image_edge_density, microsaccades,
                        patchwork_glimpse, take_glimpse)

log = logging.getLogger(__name__)

BioPolicy = PolicyName | Literal["microsaccades", "c2f_then_jittered_level2", "c2f_then_jittered_level3",
                                  "entropy_patchwork", "random_patchwork"]
"""entropy_patchwork and random_patchwork: C2F's levels 0 and 1, then glimpses gathering level-2 patches by
uncertainty or at random (bio_vision.Patchwork)."""
Density = Literal["oracle_boundaries", "glimpse_edges", "entropy"]
"""What spreads an adaptive canvas's cells: the ground truth's boundaries (an upper reference, not a method); the
intensity edges of the first, full-scene glimpse (set before it, from its pixels alone); or the probe's entropy after
the first glimpse (the uniform canvas is then resampled at the new cells)."""


@dataclass(frozen=True)
class Experiment:
    output: Path
    model: str = "in21k"
    """The CanViT and its released ADE20K probe: a pretraining dataset of canvit_pytorch.hub.repos.PRETRAINED (in21k,
    the flagship, or in1k) or abl-<slug>, an ablation of canvit_pytorch.pretrain.ablations (probes at 32×32 only)."""
    per_image_at: tuple[int, ...] = ()
    """Glimpses (0-based) at which to save each image's per-class intersections and unions, for paired bootstraps,
    next to the output (.per_image.pt)."""
    eval_stride_after_20: int = 1
    """After glimpse 20, the mIoU only every this many glimpses (and at the last); the others are null. The scene-size
    upsampling of the logits costs about as much as the glimpse itself."""
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
    object_writes_late_threshold: float | None = None
    object_writes_space: Literal["canvas", "probs"] = "canvas"
    """What the object Writes compare: the layer-normalized canvas cells, or the probe's class probabilities on them."""
    object_writes_top_fraction: float | None = None
    object_writes_late_top_fraction: float | None = None
    """When set, the object Writes reach the most similar this fraction of the outside cells at each glimpse (the late
    one from glimpse 21), instead of the cells above a similarity threshold: a reach that does not depend on the model."""
    object_writes_max_confidence: float | None = None
    """When set, the object Writes reach an outside cell only if the probe's top class probability there is below this:
    the similar cells the model may still get wrong (writing into cells it gets right harms them)."""
    object_writes_connected: bool = False
    """When set, the object Writes reach only the similar outside cells joined to the crop by a chain of similar,
    8-adjacent cells: grouping that spreads from the attended place along the object (incremental grouping)."""
    """When set, the object Writes' threshold from glimpse 21 on (C2F's level 3)."""
    local_writes_from: int | None = None
    """When set, from this glimpse on, the Writes are kept in the glimpse's crop (local_writes.LocalWrites, the README's
    fix for C2F's level 3); with a patchwork, in its patches."""
    masked_writes: Literal["random", "oracle", "predicted"] | None = None
    """The controls of the object Writes (bio_vision.MaskedWrites), from glimpse masked_writes_from on: outside the
    crop, random cells in the share that the object Writes of masked_writes_reach (a result file) reached at each
    glimpse, the cells of a ground-truth class present in the crop, or the cells whose predicted class (the probe's
    argmax on the canvas before the glimpse) is predicted in the crop."""
    masked_writes_from: int = 5
    oracle_cells: Literal["all", "correct", "wrong"] = "all"
    """With the oracle: only the outside cells the probe classifies right (or wrong) before the glimpse."""
    masked_writes_reach: Path | None = None
    sink_writes_threshold: float | None = None
    """When set, from glimpse sink_writes_from on, a cell keeps a Write's residual only if it puts less than this share
    of its attention on the glimpse's prefix tokens (local_writes.SinkGatedWrites; sink_writes_token: on that token
    only)."""
    sink_writes_from: int = 1
    sink_writes_token: int | None = None
    readout: Literal["current", "mean_probs"] = "current"
    """current: the probe's prediction on the canvas after each glimpse; mean_probs: the mean of its class
    probabilities over the glimpses from readout_from on, per cell (evidence accumulated over the glimpses, with no
    viewpoint and no change to the Writes: the readout-side alternative to restricting them)."""
    readout_from: int = 0
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
    patchwork_block: int = 1
    """entropy_patchwork and random_patchwork: the side of the blocks of patches they gather (bio_vision.Patchwork)."""
    seed: int = 0
    batch_size: int = 8
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    num_images: int | None = None
    device: str = "cuda"


def pretrained_repo_of(model: str) -> str:
    """Hub repo of an Experiment.model: a PRETRAINED key or abl-<slug>."""
    return ABLATIONS[model.removeprefix("abl-")].released_repo if model.startswith("abl-") else PRETRAINED[model]


def build_policy(exp: Experiment, batch_size: int, device: torch.device, canvas_logits=None):
    if exp.policy.endswith("_patchwork"):  # C2F until the patchwork starts
        return make_policy("coarse_to_fine", batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                           canvas_grid_size=exp.canvas_grid_size)
    if exp.policy in get_args(PolicyName):
        return make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                           canvas_grid_size=exp.canvas_grid_size, canvas_logits=canvas_logits)
    if exp.policy == "microsaccades":
        return microsaccades(batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                             scale=exp.microsaccade_scale, jitter=exp.microsaccade_jitter,
                             first_full_scene=exp.microsaccade_first_full_scene, outside=exp.microsaccade_outside,
                             sequence=exp.microsaccade_sequence)
    return c2f_then_jittered_level2(batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                                    jitter=exp.microsaccade_jitter, level=3 if exp.policy.endswith("3") else 2)


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
                                   surround=config.get("canvas_surround", False),
                                   warp=config.get("canvas_warp", False))
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
    pretrained_repo = pretrained_repo_of(exp.model)
    probe_repo = hub_repo(ade20k_probe_name(exp.model, scene_size_px=scene_px,
                                            canvas_grid_size=exp.probe_canvas_grid_size or G, steps=RELEASED_PROBE_STEPS))
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=pretrained_repo, probe_repo=probe_repo)
    seg = seg.to(device).eval()
    free = FreePositions(seg.canvit)
    retina, shape = load_retina(exp, seg.canvit, device, free)
    assert exp.readout == "current" or (shape is None and exp.density is None), "mean_probs reads the uniform canvas"
    shape_points = grid_coords(size=exp.shape_readout_px, device=device).flatten(0, 1)
    novelty = (NoveltyWrites(seg.canvit, mode=exp.novelty, strength=exp.novelty_strength, start=exp.novelty_from,
                             num_glimpses=T, device=device) if exp.novelty is not None else None)
    objects = (ObjectWrites(seg.canvit, canvas_grid_size=G, threshold=exp.object_writes_threshold,
                            start=exp.object_writes_from, num_glimpses=T, device=device,
                            late_threshold=exp.object_writes_late_threshold,
                            embed=((lambda canvas: seg.logits(canvas).flatten(2).transpose(1, 2).float().softmax(-1))
                                   if exp.object_writes_space == "probs" else None),
                            confidence=((lambda canvas: seg.logits(canvas).float().softmax(dim=1).amax(dim=1).flatten(1))
                                        if exp.object_writes_max_confidence is not None else None),
                            max_confidence=exp.object_writes_max_confidence,
                            top_fraction=exp.object_writes_top_fraction,
                            late_top_fraction=exp.object_writes_late_top_fraction,
                            connected=exp.object_writes_connected)
               if exp.object_writes_threshold is not None else None)
    local_writes = (LocalWrites(seg.canvit, canvas_grid_size=G, device=device, start=exp.local_writes_from)
                    if exp.local_writes_from is not None else None)
    masked = None
    if exp.masked_writes is not None:
        reach = (json.loads(exp.masked_writes_reach.read_text())["object_writes_reached"]
                 if exp.masked_writes == "random" and exp.masked_writes_reach is not None else None)
        predict = ((lambda canvas: seg.logits(canvas).argmax(dim=1).flatten(1))
                   if exp.masked_writes == "predicted" else None)
        masked = MaskedWrites(seg.canvit, canvas_grid_size=G, start=exp.masked_writes_from, num_glimpses=T,
                              device=device, reach=reach, oracle=exp.masked_writes == "oracle", seed=exp.seed,
                              predict=predict, oracle_cells=exp.oracle_cells,
                              classify=lambda canvas: seg.logits(canvas).argmax(dim=1).flatten(1))
    sink_writes = (SinkGatedWrites(seg.canvit, threshold=exp.sink_writes_threshold, start=exp.sink_writes_from,
                                   canvas_grid_size=G, num_glimpses=T, device=device, token=exp.sink_writes_token)
                   if exp.sink_writes_threshold is not None else None)
    images_u8, labels_u8 = load_validation_scenes(exp.cache, size_px=scene_px)
    images_u8, labels_u8 = images_u8[:exp.num_images], labels_u8[:exp.num_images]

    mious = [ade20k.MeanIoU(device=device) for _ in range(T)]
    per_image = {t: ([], []) for t in exp.per_image_at}
    evaluated = {t for t in range(T) if t <= 20 or (t - 20) % exp.eval_stride_after_20 == 0 or t == T - 1}
    evaluated |= set(exp.per_image_at)
    sums = {name: torch.zeros(T, dtype=torch.float64, device=device)
            for name in ("canvas_patch_norm", "delta_previous", "entropy")}
    start = time.monotonic()
    for first in range(0, len(images_u8), exp.batch_size):
        images = normalize(images_u8[first:first + exp.batch_size].to(device))
        labels = labels_u8[first:first + exp.batch_size].to(device).long()
        B = images.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy = build_policy(exp, B, device, canvas_logits=seg.logits)
            patchwork = None
            if exp.policy.endswith("_patchwork"):
                order = "entropy" if exp.policy == "entropy_patchwork" else "random"
                patchwork = Patchwork(batch_size=B, device=device, order=order, block=exp.patchwork_block,
                                      glimpse_grid=RELEASED_GLIMPSE_SIZE_PX // seg.canvit.patch_size)
            state = (seg.init_state(batch_size=B, canvas_grid_size=G) if shape is None
                     else shape.init_state(seg.canvit, B))
            canvas = None
            if exp.density == "oracle_boundaries":
                canvas = AdaptiveCanvas(boundary_density(labels, blur_px=33), canvas_grid_size=G, mix=exp.density_mix)
            free.canvas = None if canvas is None else canvas.positions()
            if shape is not None:
                free.canvas = shape.reset(B)
            previous = normalized_cells(seg, state.canvas)
            if masked is not None and masked.oracle and masked.predict is None:
                masked.set_labels(labels, ade20k.IGNORE_LABEL)
            for t in range(T):
                if patchwork is not None and t >= Patchwork.FIRST:
                    viewpoint, free.glimpse = patchwork.step(seg.logits(state.canvas))
                    glimpse = patchwork_glimpse(images, free.glimpse, half_side=patchwork.half_side,
                                                patch_px=seg.canvit.patch_size)
                else:
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
                if sink_writes is not None:
                    sink_writes.before_glimpse(t, viewpoint)
                if masked is not None:
                    masked.before_glimpse(t, viewpoint, state.canvas)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                    if patchwork is not None and t >= Patchwork.FIRST and local_writes.mask is not None:
                        inside = patchwork.inside(local_writes.cell_centers.flatten(0, 1), free.glimpse)
                        registers = inside.new_ones(B, seg.canvit.config.num_canvas_registers)
                        local_writes.mask = torch.cat([registers, inside], dim=1).unsqueeze(-1)
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
                if exp.readout == "mean_probs" and t >= exp.readout_from:
                    probs = logits.float().softmax(dim=1)
                    prob_sum = probs if t == exp.readout_from else prob_sum + probs
                readout = prob_sum if exp.readout == "mean_probs" and t >= exp.readout_from else logits
                if t not in evaluated:
                    pass
                elif shape is not None:
                    n = exp.shape_readout_px
                    read = shape.read(logits.flatten(2).transpose(1, 2), shape_points)  # [B, n*n, C]
                    predicted = predicted_labels(read.transpose(1, 2).reshape(B, -1, n, n), size_px=scene_px)
                elif canvas is None:
                    predicted = predicted_labels(readout, size_px=scene_px)
                else:
                    predicted = torch.cat([canvas_part.read_back(chunk, scene_px).argmax(dim=1)
                                           for canvas_part, chunk in zip(split(canvas, 2), logits.split(2))])
                if t in evaluated:
                    mious[t].update(predicted, labels)
                if t in per_image:
                    intersection, union, _ = per_image_confusion(predicted, labels, ade20k.NUM_CLASSES)
                    per_image[t][0].append(intersection.cpu())
                    per_image[t][1].append(union.cpu())
                current = normalized_cells(seg, state.canvas)
                sums["canvas_patch_norm"][t] += patches.norm(dim=-1).mean(dim=(1, 2)).sum().double()
                sums["delta_previous"][t] += (1 - F.cosine_similarity(current, previous, dim=-1)).mean(dim=1).sum().double()
                sums["entropy"][t] += predictive_entropy(logits).mean(dim=(1, 2)).sum().double()
                previous = current
        log.info("%d/%d scenes", first + B, len(images_u8))

    n = len(images_u8)
    result = {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "pretrained_repo": pretrained_repo,
        "probe_repo": probe_repo,
        "miou": [100 * m.compute() if t in evaluated else None for t, m in enumerate(mious)],
        **{name: (values / n).tolist() for name, values in sums.items()},
        "n_images": n,
        "wall_time_seconds": time.monotonic() - start,
    }
    if novelty is not None:
        result["novelty_suppressed"] = (novelty.suppressed / n).tolist()
    if objects is not None:
        result["object_writes_reached"] = (objects.reached / n).tolist()
    if masked is not None:
        result["masked_writes_reached"] = (masked.reached / n).tolist()
    if sink_writes is not None:
        result["sink_writes_kept_inside"] = (sink_writes.kept_inside / n).tolist()
        result["sink_writes_kept_outside"] = (sink_writes.kept_outside / n).tolist()
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    if per_image:
        torch.save({t: {"intersection": torch.cat(i), "union": torch.cat(u)} for t, (i, u) in per_image.items()},
                   exp.output.with_suffix(".per_image.pt"))
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("mIoU by glimpse: %s", " ".join("-" if v is None else f"{v:.2f}" for v in result["miou"]))
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
