"""NYU Depth v2: linear depth probes on frozen features (CanViT's canvas, DINOv3's patch features), then the test
depth error of CanViT after every glimpse of a rollout, and of DINOv3 at several input sizes.

The labeled set has 1449 images (640×480, depth in meters with holes filled) and an official split: 795 training
images, of which the last 95 select the probe's step, and 654 test images. A scene is the central 480×480 square of an
image, resized to the scene size (512 px, as in pretraining). Errors are computed against the depth at 480 px, on the
pixels with a depth in (0.001, 10] m inside the crop of Eigen et al. (rows 45-470 of the image; its columns 41-600
contain the whole square), averaged per image over the test images.

The probe is the segmentation head of the paper (canvit_pytorch.probes.SegmentationProbe) with one logit per depth
bin: 256 bins whose centers are spread uniformly over (0, 10] m. The predicted depth is the softmax-weighted mean of
the centers, upsampled bilinearly, trained with the scale-invariant log loss. Training scenes are random 480 px
squares of the training images, flipped horizontally with probability 1/2; canvas probes are trained on 10 R-IID
glimpses per scene and selected on 10 F-IID glimpses, as for Cityscapes (scripts/cityscapes.py).

    .venv/bin/python -P scripts/nyu_depth.py train-probe --features canvas --output probes/nyu_depth_canvas_c64.pt
    .venv/bin/python -P scripts/nyu_depth.py rollouts --probe probes/nyu_depth_canvas_c64.pt --output results/nyu_depth/c2f.json
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import Tensor

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.flops import dinov3_flops, glimpse_flops
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import DINOV3_REPOS, load_teacher

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from cityscapes import CanvasMaps, DINOv3Maps, normalize  # noqa: E402
from local_writes import LocalWrites  # noqa: E402

log = logging.getLogger(__name__)

HEIGHT_PX, WIDTH_PX = 480, 640
SQUARE_LEFT_PX = (WIDTH_PX - HEIGHT_PX) // 2
EIGEN_ROWS = slice(45, 471)
"""Rows of the evaluation crop of Eigen et al.; its columns (41-600) contain the central square."""
MIN_DEPTH_M, MAX_DEPTH_M = 1e-3, 10.0
NUM_BINS = 256
SILOG_VARIANCE_FOCUS = 0.85
LABEL_PX = 128
"""Side of the depth maps the training loss compares, to keep 256 bins × the batch within memory."""
NUM_SELECTION_IMAGES = 95

Features = Literal["canvas", "dinov3_vitb16"]
Split = Literal["train", "selection", "test"]


def dataset_root() -> Path:
    return Path(os.environ.get("NYU_ROOT", "/tmp/canvit-nadiedjoa-24/data/nyuv2"))


def load_split(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """[N, 480, 640, 3] uint8 images and [N, 480, 640] float32 depths in meters, decoded from the .mat files into
    memory-mapped caches on the first call."""
    root = dataset_root()
    images_path, depths_path = root / "images_u8.npy", root / "depths_f32.npy"
    if not (images_path.exists() and depths_path.exists()):
        import h5py

        with h5py.File(root / "nyu_depth_v2_labeled.mat", "r") as mat:
            # MATLAB stores the arrays column-major: images [N, 3, W, H], depths [N, W, H].
            images = np.ascontiguousarray(np.asarray(mat["images"]).transpose(0, 3, 2, 1))
            depths = np.ascontiguousarray(np.asarray(mat["depths"], dtype=np.float32).transpose(0, 2, 1))
        assert images.shape == (1449, HEIGHT_PX, WIDTH_PX, 3) and images.dtype == np.uint8, images.shape
        np.save(images_path, images), np.save(depths_path, depths)
        log.info("Decoded %d images into %s", len(images), root)
    from scipy.io import loadmat

    splits = loadmat(root / "splits.mat")
    train, test = splits["trainNdxs"].ravel() - 1, splits["testNdxs"].ravel() - 1
    assert len(train) == 795 and len(test) == 654
    indices = {"train": train[:-NUM_SELECTION_IMAGES], "selection": train[-NUM_SELECTION_IMAGES:], "test": test}[split]
    images, depths = np.load(images_path, mmap_mode="r"), np.load(depths_path, mmap_mode="r")
    return images[np.sort(indices)], depths[np.sort(indices)]


def evaluation_scenes(images: np.ndarray, depths: np.ndarray, first: int, count: int, size_px: int,
                      device: torch.device) -> tuple[Tensor, Tensor]:
    """The central squares of images first .. first+count-1: [B, 3, S, S] normalized scenes at size_px, and
    [B, 480, 480] depths with NaN where the error is not computed."""
    square = slice(SQUARE_LEFT_PX, SQUARE_LEFT_PX + HEIGHT_PX)
    scenes = normalize(torch.from_numpy(np.ascontiguousarray(images[first:first + count, :, square])).to(device))
    scenes = F.interpolate(scenes, size=(size_px, size_px), mode="bilinear", align_corners=False, antialias=True)
    target = torch.from_numpy(np.ascontiguousarray(depths[first:first + count, :, square])).to(device)
    valid = (target > MIN_DEPTH_M) & (target <= MAX_DEPTH_M)
    valid[:, :EIGEN_ROWS.start], valid[:, EIGEN_ROWS.stop:] = False, False
    return scenes, target.masked_fill(~valid, math.nan)


def training_scenes(images: np.ndarray, depths: np.ndarray, *, batch_size: int, size_px: int,
                    generator: torch.Generator, device: torch.device) -> tuple[Tensor, Tensor]:
    """Random 480 px squares of random training images, flipped with probability 1/2: [B, 3, S, S] scenes and
    [B, LABEL_PX, LABEL_PX] depths (nearest), NaN outside (0.001, 10] m."""
    indices = torch.randint(len(images), (batch_size,), generator=generator).tolist()
    lefts = torch.randint(WIDTH_PX - HEIGHT_PX + 1, (batch_size,), generator=generator).tolist()
    flips = (torch.rand(batch_size, generator=generator) < 0.5).tolist()
    image_batch = np.stack([images[i, :, left:left + HEIGHT_PX] for i, left in zip(indices, lefts)])
    depth_batch = np.stack([depths[i, :, left:left + HEIGHT_PX] for i, left in zip(indices, lefts)])
    scenes = normalize(torch.from_numpy(image_batch).to(device))
    target = torch.from_numpy(depth_batch).to(device)
    flip = torch.tensor(flips, device=device)
    scenes = torch.where(flip[:, None, None, None], scenes.flip(-1), scenes)
    target = torch.where(flip[:, None, None], target.flip(-1), target)
    scenes = F.interpolate(scenes, size=(size_px, size_px), mode="bilinear", align_corners=False, antialias=True)
    target = F.interpolate(target.unsqueeze(1), size=(LABEL_PX, LABEL_PX), mode="nearest").squeeze(1)
    return scenes, target.masked_fill((target <= MIN_DEPTH_M) | (target > MAX_DEPTH_M), math.nan)


BIN_CENTERS_M = torch.linspace(MAX_DEPTH_M / NUM_BINS, MAX_DEPTH_M, NUM_BINS)


def predicted_depth(probe: SegmentationProbe, features: Tensor, size_px: int) -> Tensor:
    """[B, size_px, size_px] depth in meters from [B, g, g, D] features, in float32 also under autocast."""
    with torch.autocast(device_type=features.device.type, enabled=False):
        probabilities = probe(features).softmax(dim=1)  # [B, NUM_BINS, g, g]
        depth = torch.einsum("bkhw,k->bhw", probabilities, BIN_CENTERS_M.to(probabilities.device))
        return F.interpolate(depth.unsqueeze(1), size=(size_px, size_px), mode="bilinear",
                             align_corners=False).squeeze(1)


def silog_loss(prediction: Tensor, target: Tensor) -> Tensor:
    valid = ~target.isnan()
    d = prediction[valid].clamp(min=MIN_DEPTH_M).log() - target[valid].log()
    return torch.sqrt((d ** 2).mean() - SILOG_VARIANCE_FOCUS * d.mean() ** 2)


class DepthErrors:
    """Per-image depth errors summed over images: RMSE (m), absolute relative error, δ < 1.25 (%)."""

    NAMES = ("rmse", "abs_rel", "delta1")

    def __init__(self) -> None:
        self.sums, self.count = torch.zeros(len(self.NAMES), dtype=torch.float64), 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        for p, g in zip(prediction.float(), target):
            valid = ~g.isnan()
            p, g = p[valid].clamp(MIN_DEPTH_M, MAX_DEPTH_M), g[valid]
            ratio = torch.maximum(p / g, g / p)
            self.sums += torch.stack([((p - g) ** 2).mean().sqrt(), ((p - g).abs() / g).mean(),
                                      100 * (ratio < 1.25).float().mean()]).double().cpu()
            self.count += 1

    def compute(self) -> dict[str, float]:
        return {name: value for name, value in zip(self.NAMES, (self.sums / self.count).tolist())}


@dataclass(frozen=True)
class TrainProbe:
    """Train a linear depth probe on frozen features; saves the probe of lowest RMSE on the selection images (on the
    last feature map) with its record."""

    output: Path
    features: Features = "canvas"
    scene_size_px: int = 512
    """Side of the scenes; DINOv3's input size."""
    pretrained_repo: str = FLAGSHIP
    canvas_grid_size: int = 64
    num_glimpses: int = 10
    training_policy: PolicyName = "random"
    selection_policy: PolicyName = "full_then_random"
    num_steps: int = 4000
    batch_size: int = 8
    """8 scenes of 10 glimpses take about 4 GB with a 64×64 canvas, which leaves room for other runs on a 24 GB GPU."""
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 300
    dropout: float = 0.1
    validate_every: int = 500
    seed: int = 0
    device: str = "cuda"


def feature_maps_for(config: TrainProbe, device: torch.device) -> CanvasMaps | DINOv3Maps:
    if config.features == "canvas":
        canvit = CanViTForPretraining.from_pretrained(config.pretrained_repo).canvit.to(device).eval()
        return CanvasMaps(canvit.requires_grad_(False), canvas_grid_size=config.canvas_grid_size,
                          num_glimpses=config.num_glimpses)
    return DINOv3Maps(load_teacher(DINOV3_REPOS[config.features.removeprefix("dinov3_")], device))


@torch.no_grad()
def selection_errors(probe: SegmentationProbe, feature_maps: CanvasMaps | DINOv3Maps, config: TrainProbe,
                     device: torch.device) -> list[dict[str, float]]:
    probe.eval()
    images, depths = load_split("selection")
    errors: list[DepthErrors] = []
    for first in range(0, len(images), config.batch_size):
        scenes, target = evaluation_scenes(images, depths, first, config.batch_size, config.scene_size_px, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            maps = feature_maps(scenes, config.selection_policy)
        errors = errors or [DepthErrors() for _ in maps]
        for error, features in zip(errors, maps, strict=True):
            error.update(predicted_depth(probe, features, HEIGHT_PX), target)
    return [error.compute() for error in errors]


def train_probe(config: TrainProbe) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    feature_maps = feature_maps_for(config, device)
    probe = SegmentationProbe(embed_dim=feature_maps.embed_dim, num_classes=NUM_BINS, dropout=config.dropout,
                              use_ln=not feature_maps.layer_normalized).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)

    def lr_factor(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.num_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    images, depths = load_split("train")
    history: list[dict] = []
    best: dict | None = None
    start = time.monotonic()
    loss_sum, loss_count = 0.0, 0
    for step in range(config.num_steps + 1):
        if step % config.validate_every == 0 or step == config.num_steps:
            errors = selection_errors(probe, feature_maps, config, device)
            history.append({"step": step, "selection_errors": errors, "train_loss": loss_sum / max(1, loss_count),
                            "seconds": time.monotonic() - start})
            loss_sum, loss_count = 0.0, 0
            log.info("step %d: selection RMSE per map %s", step, " ".join(f"{e['rmse']:.3f}" for e in errors))
            if best is None or errors[-1]["rmse"] < best["selection_errors"][-1]["rmse"]:
                best = {"step": step, "selection_errors": errors,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}}
        if step == config.num_steps:
            break
        scenes, target = training_scenes(images, depths, batch_size=config.batch_size, size_px=config.scene_size_px,
                                         generator=generator, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            maps = feature_maps(scenes, config.training_policy)
        probe.train()
        optimizer.zero_grad()
        loss = 0.0
        for features in maps:  # one backward per map: the mean loss's gradient, in a tenth of the memory
            map_loss = silog_loss(predicted_depth(probe, features, LABEL_PX), target) / len(maps)
            map_loss.backward()
            loss += map_loss.item()
        optimizer.step()
        scheduler.step()
        loss_sum, loss_count = loss_sum + loss, loss_count + 1
        if step % 100 == 0:
            log.info("step %d: loss %.4f, lr %.2e, %.0f s", step, loss, scheduler.get_last_lr()[0],
                     time.monotonic() - start)

    assert best is not None
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "embed_dim": feature_maps.embed_dim, "use_ln": not feature_maps.layer_normalized,
        "selected_step": best["step"], "selected_errors": best["selection_errors"],
        "history": history, "state_dict": best["state_dict"],
    }, config.output)
    log.info("Saved the probe of step %d (selection RMSE %.3f on the last map) to %s",
             best["step"], best["selection_errors"][-1]["rmse"], config.output)


def load_probe(path: Path, device: torch.device) -> tuple[SegmentationProbe, dict]:
    record = torch.load(path, weights_only=True)
    probe = SegmentationProbe(embed_dim=record["embed_dim"], num_classes=NUM_BINS, dropout=record["config"]["dropout"],
                              use_ln=record["use_ln"])
    probe.load_state_dict(record["state_dict"])
    return probe.to(device).eval(), record


@dataclass(frozen=True)
class Rollouts:
    """Test depth errors after every glimpse of CanViT, its canvas decoded by a probe trained on its canvas."""

    output: Path
    probe: Path
    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = 21
    canvas_grid_size: int = 64
    local_writes_from: int | None = None
    """When set, from this glimpse on, the Write residuals into the canvas cells outside the glimpse's crop are dropped
    (scripts/local_writes.py)."""
    seed: int = 0
    batch_size: int = 8
    device: str = "cuda"


@torch.inference_mode()
def rollouts(exp: Rollouts) -> None:
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    probe, record = load_probe(exp.probe, device)
    assert record["config"]["features"] == "canvas", exp.probe
    canvit = CanViTForPretraining.from_pretrained(record["config"]["pretrained_repo"]).canvit.to(device).eval()
    local_writes = (LocalWrites(canvit, canvas_grid_size=exp.canvas_grid_size, device=device,
                                start=exp.local_writes_from)
                    if exp.local_writes_from is not None else None)
    images, depths = load_split("test")
    errors = [DepthErrors() for _ in range(exp.num_glimpses)]
    start = time.monotonic()
    for first in range(0, len(images), exp.batch_size):
        scenes, target = evaluation_scenes(images, depths, first, exp.batch_size, record["config"]["scene_size_px"],
                                           device)
        batch_size = scenes.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy = make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                                 canvas_grid_size=exp.canvas_grid_size)
            state = canvit.init_state(batch_size=batch_size, canvas_grid_size=exp.canvas_grid_size)
            for t in range(exp.num_glimpses):
                viewpoint = policy.step(t, state)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                glimpse = sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint,
                                              glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                errors[t].update(predicted_depth(probe, canvit.canvas_patch_grid(state.canvas), HEIGHT_PX), target)
    write_result(exp.output, {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_config": record["config"], "probe_selected_step": record["selected_step"],
        "paper_name": POLICIES[exp.policy].paper_name,
        **{name: [error.compute()[name] for error in errors] for name in DepthErrors.NAMES},
        "glimpse_gflops": glimpse_flops(canvit.config, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX,
                                        canvas_grid_size=exp.canvas_grid_size) / 1e9,
        "n_images": len(images), "wall_time_seconds": time.monotonic() - start,
    })


@dataclass(frozen=True)
class Passive:
    """Test depth errors of DINOv3, the scene resized to the input size of a probe trained on its features."""

    output: Path
    probe: Path
    batch_size: int = 8
    device: str = "cuda"


@torch.inference_mode()
def passive(exp: Passive) -> None:
    device = torch.device(exp.device)
    probe, record = load_probe(exp.probe, device)
    features: Features = record["config"]["features"]
    assert features != "canvas", exp.probe
    input_px = record["config"]["scene_size_px"]
    dinov3 = load_teacher(DINOV3_REPOS[features.removeprefix("dinov3_")], device)
    images, depths = load_split("test")
    errors = DepthErrors()
    start = time.monotonic()
    for first in range(0, len(images), exp.batch_size):
        scenes, target = evaluation_scenes(images, depths, first, exp.batch_size, input_px, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            patches = dinov3.patch_grid(scenes, input_size_px=input_px)
        errors.update(predicted_depth(probe, patches, HEIGHT_PX), target)
    write_result(exp.output, {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_config": record["config"], "probe_selected_step": record["selected_step"],
        **{name: [value] for name, value in errors.compute().items()},
        "gflops": dinov3_flops(dinov3.model.config, input_size_px=input_px) / 1e9,
        "n_images": len(images), "wall_time_seconds": time.monotonic() - start,
    })


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    log.info("RMSE (m): %s", " ".join(f"{v:.3f}" for v in result["rmse"]))
    log.info("δ1 (%%): %s", " ".join(f"{v:.2f}" for v in result["delta1"]))
    log.info("Wrote %s in %.0f s", path, result["wall_time_seconds"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    command = tyro.extras.subcommand_cli_from_dict(
        {"train-probe": TrainProbe, "rollouts": Rollouts, "passive": Passive}, description=__doc__.split("\n\n")[0])
    match command:
        case TrainProbe():
            train_probe(command)
        case Rollouts():
            rollouts(command)
        case Passive():
            passive(command)


if __name__ == "__main__":
    main()
