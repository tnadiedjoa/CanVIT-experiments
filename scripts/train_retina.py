"""Learn where each glimpse patch looks (bio_vision.PatchRetina: a center and half sizes per patch, 192 numbers), with
CanViT frozen or fine-tuned with it, on the paper's objective and rollouts; or fine-tune CanViT alone through a fixed
retina (uniform: the control; foveated: bio_vision.foveal_warp).

The question is the one of Cheung, Weiss & Olshausen (ICLR 2017, "Emergence of foveal image sampling from learning to
attend in visual scenes"): a learned sampling lattice became foveated when their model could only move its glimpse,
and stayed uniform when it could also zoom. CanViT zooms (the paper's random policies draw a scale per glimpse); with
`scale` set, every glimpse has that scale instead, at a random center, as a primate eye that moves but does not zoom.

Steps follow the paper's pretraining: one policy per step (F-IID or R-IID, or fixed-scale saccades), a random rollout
length (chunks of 2 glimpses, stop with probability 0.5), truncated BPTT over each chunk, patch and CLS distillation
losses against the DINOv3 teacher on ADE20K training scenes at 512 px with the 32×32 canvas. The same seed draws the
same scenes, policies and lengths, so runs compare step by step. At the end, the canvas's cosine similarity to the
teacher's standardized patch features is measured on validation scenes after every glimpse of fixed rollouts.

    .venv/bin/python -P scripts/train_retina.py --learn-retina --output probes/retina_frozen.pt
"""

import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import make_policy
from canvit_pytorch.pretrain.loss import DistillationTargets, distillation_losses
from canvit_pytorch.pretrain.step import sample_rollout_length
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from ade20k_rollouts import load_validation_scenes, normalize  # noqa: E402
from bio_vision import (FreePositions, GaussianCanvas, LearnedPeriods, PatchRetina, WriteFootprint,  # noqa: E402
                        microsaccades, take_glimpse)
from train_write_gate import TrainingScenes  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Training:
    output: Path
    learn_retina: bool = False
    finetune_canvit: bool = False
    fovea: float | None = None
    """When set, the retina starts as this foveal warp (bio_vision.foveal_warp) instead of the uniform grid."""
    scale: float | None = None
    """When set, every glimpse has this scale at a uniformly random valid center (no zoom), in place of the paper's
    F-IID and R-IID policies."""
    learn_canvas: bool = False
    """Learn the canvas's shape (bio_vision.GaussianCanvas: a center and an elliptical Gaussian per cell), read at the
    teacher's patch centers for the patch loss."""
    canvas_adaptive: bool = False
    """With learn_canvas: the cells also move after the first glimpse by their own content (GaussianCanvas)."""
    canvas_grid_size: int | None = None
    """With learn_canvas: cells per side of the starting grid (the teacher's 32 by default; fewer for a canvas on a
    budget, still read at the teacher's 32×32 patch centers)."""
    canvas_coarse_grid_size: int | None = None
    """With learn_canvas: a second, coarser grid of cells with wider Gaussians."""
    canvas_fixed_centers: bool = False
    """With learn_canvas: only the Gaussians' shapes learn, the cells stay on their grid (the control)."""
    canvas_surround: bool = False
    """With learn_canvas: each cell also has a learned inhibitory surround (GaussianCanvas)."""
    canvas_write_footprint: bool = False
    """With learn_canvas: each cell's Gaussian also shapes what it receives in the Writes (bio_vision.WriteFootprint)."""
    canvas_lr: float = 1e-3
    learn_periods: bool = False
    """Learn SR-RoPE's wavelengths (bio_vision.LearnedPeriods), with periods_lr."""
    periods_lr: float = 1e-3
    microsaccade_fraction: float = 0.0
    """Fraction of the steps whose rollout is microsaccades (the full scene, then views of scale 7/8 shifted by up to
    1/8, bio_vision.microsaccades) instead of F-IID or R-IID: the model learns to fuse views that differ by a fraction
    of a patch."""
    num_steps: int = 1000
    batch_size: int = 16
    retina_lr: float = 3e-3
    canvit_lr: float = 1e-5
    warmup_steps: int = 20
    grad_clip_norm: float = 1.0
    tbptt_chunk_glimpses: int = 2
    stop_probability: float = 0.5
    max_glimpses: int = 16
    validation_images: int = 256
    validation_glimpses: int = 10
    seed: int = 0
    log_every: int = 25
    device: str = "cuda"


def glimpse_step(model: CanViTForPretraining, state, images, viewpoint, t: int, retina, free: FreePositions,
                 shape: GaussianCanvas | None):
    """One glimpse through the retina, with the canvas cells where `shape` puts them."""
    if shape is not None:
        free.canvas = shape.reset(images.shape[0]) if t == 0 else shape.positions()
    glimpse, free.glimpse = take_glimpse(images, viewpoint, retina)
    state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
    if shape is not None and shape.adaptive and t == 0:
        state, free.canvas = shape.adapt(state, model.canvit)
    return state


def predicted_patches(model: CanViTForPretraining, state, shape: GaussianCanvas | None, points: torch.Tensor):
    """The canvas's prediction of the teacher's standardized patch features, at the teacher's patch centers."""
    cells = model.predict_teacher_patches(state.canvas)
    return cells if shape is None else shape.read(cells, points)


def rollout_policy(cfg: Training, *, batch_size: int, num_glimpses: int, device: torch.device, grid: int,
                   name: str):
    if name == "microsaccades":
        return microsaccades(batch_size=batch_size, device=device, num_glimpses=num_glimpses, scale=0.875,
                             jitter=0.125)
    if cfg.scale is not None:
        return microsaccades(batch_size=batch_size, device=device, num_glimpses=num_glimpses, scale=cfg.scale,
                             jitter=1.0, first_full_scene=False)
    return make_policy(name, batch_size=batch_size, device=device, num_glimpses=num_glimpses, canvas_grid_size=grid)


@torch.no_grad()
def validate(cfg: Training, model: CanViTForPretraining, teacher, retina, free: FreePositions,
             device: torch.device, shape: GaussianCanvas | None = None) -> dict[str, list[float]]:
    """Mean cosine similarity of the canvas's (and CLS token's) predictions to the teacher's standardized features
    after every glimpse, on the first validation scenes, for each policy of the evaluation."""
    images_u8, _ = load_validation_scenes(Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt"), size_px=512)
    grid = model.teacher_patch_grid
    names = ["full_then_random", "random", "coarse_to_fine"] if cfg.scale is None else ["saccades"]
    points = grid_coords(size=grid, device=device).flatten(0, 1)
    out: dict[str, list[float]] = {}
    generator_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(1234)
    for name in names:
        patch_cos = torch.zeros(cfg.validation_glimpses, dtype=torch.float64)
        cls_cos = torch.zeros(cfg.validation_glimpses, dtype=torch.float64)
        count = 0
        for first in range(0, cfg.validation_images, 16):
            images = normalize(images_u8[first:min(first + 16, cfg.validation_images)].to(device))
            count += images.shape[0]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                target = teacher(images)
                targets = DistillationTargets.standardize(model, patches=target.patches.float(), cls=target.cls.float())
                policy = rollout_policy(cfg, batch_size=images.shape[0], num_glimpses=cfg.validation_glimpses,
                                        device=device, grid=grid, name=name)
                state = (model.init_state(batch_size=images.shape[0], canvas_grid_size=grid) if shape is None
                         else shape.init_state(model.canvit, images.shape[0]))
                for t in range(cfg.validation_glimpses):
                    viewpoint = policy.step(t, state)
                    state = glimpse_step(model, state, images, viewpoint, t, retina, free, shape)
                    patches = predicted_patches(model, state, shape, points)
                    cls = model.predict_teacher_cls(state.recurrent_cls).float()
                    patch_cos[t] += F.cosine_similarity(patches.float(), targets.patches, dim=-1).mean().double().item() * images.shape[0]
                    cls_cos[t] += F.cosine_similarity(cls, targets.cls, dim=-1).mean().double().item() * images.shape[0]
        out[f"{name}_patch_cos"] = (patch_cos / count).tolist()
        out[f"{name}_cls_cos"] = (cls_cos / count).tolist()
        log.info("validation %s: patch cos %s", name, " ".join(f"{v:.4f}" for v in out[f"{name}_patch_cos"]))
    torch.set_rng_state(generator_state[0])
    torch.cuda.set_rng_state(generator_state[1])
    return out


def main(cfg: Training) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model = CanViTForPretraining.from_pretrained(FLAGSHIP).to(device).eval().requires_grad_(False)
    teacher = load_teacher(TEACHER_REPO, device)
    grid = model.teacher_patch_grid
    retina = PatchRetina(glimpse_px=RELEASED_GLIMPSE_SIZE_PX, patch_px=model.canvit.patch_size, device=device,
                         fovea=cfg.fovea).requires_grad_(cfg.learn_retina)
    free = FreePositions(model.canvit)
    shape = (GaussianCanvas(canvas_grid_size=cfg.canvas_grid_size or grid, canvas_dim=model.canvit.canvas_dim,
                            device=device, adaptive=cfg.canvas_adaptive, coarse_grid_size=cfg.canvas_coarse_grid_size,
                            learn_centers=not cfg.canvas_fixed_centers, surround=cfg.canvas_surround)
             if cfg.learn_canvas else None)
    footprint = WriteFootprint(model.canvit, shape, free) if shape is not None and cfg.canvas_write_footprint else None
    points = grid_coords(size=grid, device=device).flatten(0, 1)  # the teacher's patch centers
    groups = []
    if shape is not None:
        groups.append({"params": [p for p in shape.parameters() if p.requires_grad], "lr": cfg.canvas_lr})
    if footprint is not None:
        groups.append({"params": list(footprint.parameters()), "lr": cfg.canvas_lr})
    periods = LearnedPeriods(model.canvit, device) if cfg.learn_periods else None
    if periods is not None:
        free.periods = periods
        groups.append({"params": list(periods.parameters()), "lr": cfg.periods_lr})
    if cfg.learn_retina:
        groups.append({"params": list(retina.parameters()), "lr": cfg.retina_lr})
    if cfg.finetune_canvit:
        model.canvit.requires_grad_(True)
        groups.append({"params": list(model.canvit.parameters()), "lr": cfg.canvit_lr})
    assert groups, "learn the retina or the canvas's shape, fine-tune CanViT, or several"
    params = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.Adam(groups)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(cfg.warmup_steps, 1)))
    loader = DataLoader(TrainingScenes("ade20k", size_px=grid * teacher.patch_size), batch_size=cfg.batch_size,
                        shuffle=True, num_workers=4, drop_last=True, generator=torch.Generator().manual_seed(cfg.seed))
    batches = iter(loader)
    history: list[dict] = []
    start = time.monotonic()
    for step in range(cfg.num_steps):
        try:
            images = next(batches)
        except StopIteration:
            batches = iter(loader)
            images = next(batches)
        images = images.to(device)
        with torch.no_grad():
            target = teacher(images)
        targets = DistillationTargets.standardize(model, patches=target.patches.float(), cls=target.cls.float())
        name = random.choice(["full_then_random", "random"])
        if cfg.microsaccade_fraction > 0 and random.random() < cfg.microsaccade_fraction:
            name = "microsaccades"
        num_glimpses = min(cfg.max_glimpses, sample_rollout_length(tbptt_chunk_glimpses=cfg.tbptt_chunk_glimpses,
                                                                   stop_probability=cfg.stop_probability))
        policy = rollout_policy(cfg, batch_size=images.shape[0], num_glimpses=num_glimpses, device=device, grid=grid,
                                name=name)
        state = (model.init_state(batch_size=images.shape[0], canvas_grid_size=grid) if shape is None
                 else shape.init_state(model.canvit, images.shape[0]))
        optimizer.zero_grad()
        chunk_loss, total = torch.zeros((), device=device), 0.0
        for t in range(num_glimpses):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                viewpoint = policy.step(t, state)
                state = glimpse_step(model, state, images, viewpoint, t, retina, free, shape)
                if shape is None:
                    losses = distillation_losses(model, state, targets, enable_teacher_patch_loss=True,
                                                 enable_teacher_cls_loss=True)
                else:
                    losses = {"teacher_patch_loss": F.mse_loss(predicted_patches(model, state, shape, points),
                                                               targets.patches),
                              "teacher_cls_loss": F.mse_loss(model.predict_teacher_cls(state.recurrent_cls).float(),
                                                             targets.cls)}
            loss = torch.stack(list(losses.values())).sum().float()
            chunk_loss = chunk_loss + loss
            total += loss.item()
            if (t + 1) % cfg.tbptt_chunk_glimpses == 0 or t + 1 == num_glimpses:
                (chunk_loss / num_glimpses).backward()
                chunk_loss = torch.zeros((), device=device)
                state = state.detach()
                if shape is not None:
                    shape.detach()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        optimizer.step()
        schedule.step()
        history.append({"step": step, "policy": name if cfg.scale is None else "saccades",
                        "num_glimpses": num_glimpses, "loss": total / num_glimpses})
        if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
            rings = retina.describe()["mean_half_size_by_ring"]
            canvas = "" if shape is None else ", canvas " + " ".join(
                f"{k} {[round(x, 3) for x in v] if isinstance(v, list) else round(v, 4)}" for k, v in shape.describe().items())
            log.info("step %d (%s, %d glimpses): loss %.4f, mean of the last %d %.4f, half size by ring %s%s, %.0f s",
                     step, history[-1]["policy"], num_glimpses, total / num_glimpses, min(len(history), 100),
                     sum(h["loss"] for h in history[-100:]) / min(len(history), 100),
                     " ".join(f"{r:.4f}" for r in rings), canvas, time.monotonic() - start)
    model.eval()
    validation = validate(cfg, model, teacher, retina, free, device, shape)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
               "retina": retina.describe(), "validation": validation, "history": history,
               "canvas_shape": None if shape is None else shape.describe(),
               "write_footprint": None if footprint is None else footprint.describe(),
               "periods": None if periods is None else periods.describe()}
    torch.save({**summary, "retina": {k: v.detach().cpu() for k, v in retina.state_dict().items()},
                "retina_layout": retina.describe(),
                "canvas_shape_state": (None if shape is None
                                       else {k: v.detach().cpu() for k, v in shape.state_dict().items()}),
                "periods_state": None if periods is None else {k: v.detach().cpu() for k, v in periods.state_dict().items()},
                "write_footprint_state": (None if footprint is None
                                          else {k: v.detach().cpu() for k, v in footprint.state_dict().items()}),
                "canvit": ({k: v.detach().cpu() for k, v in model.canvit.state_dict().items()}
                           if cfg.finetune_canvit else None)}, cfg.output)
    cfg.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    log.info("Saved %s", cfg.output)


if __name__ == "__main__":
    main(tyro.cli(Training))
