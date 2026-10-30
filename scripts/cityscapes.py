"""Cityscapes at its native resolution: linear probes on frozen features (CanViT's canvas, DINOv3's patch features)
trained on the training split, then validation mIoU of CanViT rollouts that look closer than the 512 px scenes of
pretraining, and of DINOv3 at several input sizes.

Validation scenes are the two 1024×1024 halves of each 1024×2048 validation image (1000 scenes), scored against
their labels at 1024 px; the classes are the benchmark's 19, other pixels are ignored. Images and labels are decoded
once into uint8 caches next to the dataset.

Probe training follows the paper's ADE20K protocol (canvit_pytorch.specialize.ade20k: AdamW, linear warmup then
cosine decay, dropout 0.1, scale jitter 0.5-2, random square crops, horizontal flips; canvas probes trained on 10
R-IID glimpses and selected on 10 F-IID glimpses), with fewer steps, which suits a training set 7× smaller than
ADE20K's, and without the photometric distortion and the class-ratio crop rejection.

    CITYSCAPES_ROOT=/path/to/cityscapes .venv/bin/python -P scripts/cityscapes.py train-probe \
        --features canvas --output probes/cityscapes_canvas_s512_c64.pt
    .venv/bin/python -P scripts/cityscapes.py rollouts --probe probes/cityscapes_canvas_s512_c64.pt \
        --detail s512_then_native --output results/cityscapes/c2f_s512_then_native.json
"""

import json
import logging
import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import Tensor

from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.benchmarks.ade20k import MeanIoU
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import predicted_labels
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViT, CanViTForPretraining, CanViTForSemanticSegmentation
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.preprocess import imagenet_normalize
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.teacher import DINOV3_REPOS, DINOv3Teacher, load_teacher

log = logging.getLogger(__name__)

HEIGHT_PX, WIDTH_PX = 1024, 2048
SCENE_PX = HEIGHT_PX
"""Side of a validation scene: half of an image."""
NUM_CLASSES = 19
IGNORE_LABEL = 255
CLASS_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
               "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
LABEL_ID_TO_TRAIN_ID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                        26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
"""cityscapesscripts.helpers.labels: the label id of each evaluated class -> its train id."""
SCALE_JITTER_RANGE = (0.5, 2.0)

Split = Literal["train", "val"]
Features = Literal["canvas", "dinov3_vitb16", "dinov3_vits16"]
Detail = Literal["s512", "s512_then_native", "native"]
"""Where each glimpse is sampled from, for a 1024 px scene:

- s512: the scene downsampled to 512 px with anti-aliasing, the scene size of pretraining and of the paper's probes;
  glimpses smaller than a quarter of the scene are upsampled and add no detail;
- s512_then_native: as s512 while a glimpse covers at least 128 px of the 512 px scene, and from the 1024 px scene
  below that, so C2F's level 3 sees real detail at its native resolution and every earlier glimpse is the same as
  under s512;
- native: every glimpse from the 1024 px scene with the official bilinear sampling, which aliases the coarse glimpses.
"""


def dataset_root() -> Path:
    return Path(os.environ.get("CITYSCAPES_ROOT", "/tmp/canvit-nadiedjoa-24/data/cityscapes"))


def scene_files(root: Path, split: Split) -> list[tuple[Path, Path]]:
    """(image, labelIds map) pairs of a split, sorted by name."""
    images = sorted((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
    assert images, f"no images in {root / 'leftImg8bit' / split}"
    pairs = []
    for image in images:
        label = root / "gtFine" / split / image.parent.name / image.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        assert label.exists(), label
        pairs.append((image, label))
    return pairs


TRAIN_ID_LOOKUP = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
    TRAIN_ID_LOOKUP[label_id] = train_id


def decode(pair: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(pair[0]).convert("RGB"))
    label_ids = np.asarray(Image.open(pair[1]))
    assert image.shape == (HEIGHT_PX, WIDTH_PX, 3) and label_ids.shape == (HEIGHT_PX, WIDTH_PX), pair
    return image, TRAIN_ID_LOOKUP[label_ids]


def load_split(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Memory-mapped [N, H, W, 3] uint8 images and [N, H, W] uint8 train ids, decoded on the first call."""
    root = dataset_root()
    images_path, labels_path = root / f"{split}_images_u8.npy", root / f"{split}_labels_u8.npy"
    if not (images_path.exists() and labels_path.exists()):
        pairs = scene_files(root, split)
        partial = [path.with_suffix(".partial.npy") for path in (images_path, labels_path)]
        images = np.lib.format.open_memmap(partial[0], mode="w+", dtype=np.uint8, shape=(len(pairs), HEIGHT_PX, WIDTH_PX, 3))
        labels = np.lib.format.open_memmap(partial[1], mode="w+", dtype=np.uint8, shape=(len(pairs), HEIGHT_PX, WIDTH_PX))
        with multiprocessing.Pool(min(4, os.cpu_count() or 1)) as pool:
            for i, (image, label) in enumerate(pool.imap(decode, pairs, chunksize=4)):
                images[i], labels[i] = image, label
        images.flush(), labels.flush()
        del images, labels
        partial[0].rename(images_path), partial[1].rename(labels_path)
        log.info("Decoded %d %s images into %s", len(pairs), split, root)
    return np.load(images_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")


def normalize(images: Tensor) -> Tensor:
    """[B, H, W, 3] uint8 -> [B, 3, H, W] ImageNet-normalized float32."""
    return imagenet_normalize(images.permute(0, 3, 1, 2).float() / 255)


def validation_scenes(images: np.ndarray, labels: np.ndarray, first: int, count: int,
                      device: torch.device) -> tuple[Tensor, Tensor]:
    """Validation scenes first .. first+count-1 (scene k is half k % 2 of image k // 2): [B, 3, 1024, 1024]
    normalized scenes and [B, 1024, 1024] labels."""
    indices = range(first, min(first + count, 2 * len(images)))
    columns = [slice((k % 2) * SCENE_PX, (k % 2 + 1) * SCENE_PX) for k in indices]
    scene_images = torch.from_numpy(np.stack([images[k // 2, :, cols] for k, cols in zip(indices, columns)]))
    scene_labels = torch.from_numpy(np.stack([labels[k // 2, :, cols] for k, cols in zip(indices, columns)]))
    return normalize(scene_images.to(device)), scene_labels.to(device).long()


def downsample(scenes: Tensor, labels: Tensor, size_px: int) -> tuple[Tensor, Tensor]:
    """Scenes resized with anti-aliasing (as the official evaluation transforms resize), labels by nearest neighbor."""
    if scenes.shape[-1] == size_px:
        return scenes, labels
    small = F.interpolate(scenes, size=(size_px, size_px), mode="bilinear", align_corners=False, antialias=True)
    small_labels = F.interpolate(labels.unsqueeze(1).float(), size=(size_px, size_px), mode="nearest")
    return small, small_labels.squeeze(1).long()


def augmented_scenes(images: np.ndarray, labels: np.ndarray, *, batch_size: int, size_px: int,
                     generator: torch.Generator, device: torch.device) -> tuple[Tensor, Tensor]:
    """A batch of training scenes [B, 3, S, S] and labels [B, S, S] from random images: a square crop whose side is
    the image height divided by a factor drawn from SCALE_JITTER_RANGE (the scale jitter, around the validation
    scenes' framing), padded with ignored mean-color pixels where it leaves the image, bilinearly resampled to S px,
    flipped horizontally with probability 1/2."""
    indices = torch.randint(len(images), (batch_size,), generator=generator).tolist()
    image_batch = normalize(torch.from_numpy(np.stack([images[i] for i in indices])).to(device))
    label_batch = torch.from_numpy(np.stack([labels[i] for i in indices])).to(device)
    low, high = SCALE_JITTER_RANGE
    factor, u_x, u_y, flip = torch.rand(4, batch_size, generator=generator).to(device)
    side = HEIGHT_PX / (low + (high - low) * factor)
    left, top = u_x * (WIDTH_PX - side), u_y * (HEIGHT_PX - side)  # top < 0 when the crop is taller than the image
    t = (torch.arange(size_px, device=device) + 0.5) / size_px
    t_x = torch.where((flip < 0.5)[:, None], 1 - t, t)  # [B, S]
    x = 2 * (left[:, None] + t_x * side[:, None]) / WIDTH_PX - 1
    y = 2 * (top[:, None] + t * side[:, None]) / HEIGHT_PX - 1
    grid = torch.stack(torch.broadcast_tensors(x[:, None, :], y[:, :, None]), dim=-1)  # [B, S, S, (x, y)]
    scenes = F.grid_sample(image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # Shifted by one so that the zero padding becomes IGNORE_LABEL.
    shifted = F.grid_sample((label_batch + 1).unsqueeze(1).float(), grid, mode="nearest", align_corners=False)
    scene_labels = shifted.squeeze(1).round().long() - 1
    return scenes, scene_labels.masked_fill(scene_labels < 0, IGNORE_LABEL)


class CanvasMaps:
    """The canvas patches [B, G, G, canvas_dim] of a frozen CanViT after each glimpse of a rollout."""

    layer_normalized = False

    def __init__(self, canvit: CanViT, *, canvas_grid_size: int, num_glimpses: int) -> None:
        self.canvit, self.canvas_grid_size, self.num_glimpses = canvit, canvas_grid_size, num_glimpses
        self.embed_dim = canvit.canvas_dim

    @torch.no_grad()
    def __call__(self, scenes: Tensor, policy_name: PolicyName) -> list[Tensor]:
        batch_size = scenes.shape[0]
        policy = make_policy(policy_name, batch_size=batch_size, device=scenes.device, num_glimpses=self.num_glimpses,
                             canvas_grid_size=self.canvas_grid_size)
        state = self.canvit.init_state(batch_size=batch_size, canvas_grid_size=self.canvas_grid_size)
        maps = []
        for t in range(self.num_glimpses):
            viewpoint = policy.step(t, state)
            glimpse = sample_at_viewpoint(spatial=scenes, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
            state = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
            maps.append(self.canvit.canvas_patch_grid(state.canvas))
        return maps


class DINOv3Maps:
    """The patch features [B, g, g, D] of a frozen DINOv3, one map per scene, on scenes already at its input size."""

    layer_normalized = True

    def __init__(self, dinov3: DINOv3Teacher) -> None:
        self.dinov3, self.embed_dim = dinov3, dinov3.embed_dim

    @torch.no_grad()
    def __call__(self, scenes: Tensor, policy_name: PolicyName) -> list[Tensor]:
        return [self.dinov3.patch_grid(scenes, input_size_px=scenes.shape[-1])]


@dataclass(frozen=True)
class TrainProbe:
    """Train a linear Cityscapes probe on frozen features; saves the probe of highest validation mIoU (on the last
    feature map) with its record."""

    output: Path
    features: Features = "canvas"
    scene_size_px: int = 512
    """Side of the training crops and of the downsampled validation scenes; DINOv3's input size."""
    pretrained_repo: str = FLAGSHIP
    canvas_grid_size: int = 64
    num_glimpses: int = 10
    training_policy: PolicyName = "random"
    validation_policy: PolicyName = "full_then_random"
    num_steps: int = 8000
    batch_size: int = 16
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    warmup_start_lr_fraction: float = 1e-6
    dropout: float = 0.1
    validate_every: int = 1000
    validation_batch_size: int = 16
    seed: int = 0
    device: str = "cuda"


def load_feature_maps(config: TrainProbe, device: torch.device) -> CanvasMaps | DINOv3Maps:
    match config.features:
        case "canvas":
            canvit = CanViTForPretraining.from_pretrained(config.pretrained_repo).canvit.to(device).eval()
            return CanvasMaps(canvit.requires_grad_(False), canvas_grid_size=config.canvas_grid_size,
                              num_glimpses=config.num_glimpses)
        case "dinov3_vitb16" | "dinov3_vits16":
            return DINOv3Maps(load_teacher(DINOV3_REPOS[config.features.removeprefix("dinov3_")], device))
        case _:
            assert_never(config.features)


def resize_labels(labels: Tensor, size: torch.Size) -> Tensor:
    if labels.shape[1:] == size:
        return labels
    return F.interpolate(labels.unsqueeze(1).float(), size=tuple(size), mode="nearest").squeeze(1).long()


@torch.no_grad()
def validate(probe: SegmentationProbe, feature_maps: CanvasMaps | DINOv3Maps, config: TrainProbe,
             device: torch.device) -> list[float]:
    """mIoU (%) of each feature map on the validation scenes at scene_size_px, labels at the same size."""
    probe.eval()
    images, labels = load_split("val")
    ious: list[MeanIoU] = []
    for first in range(0, 2 * len(images), config.validation_batch_size):
        scenes, scene_labels = downsample(*validation_scenes(images, labels, first, config.validation_batch_size, device),
                                          config.scene_size_px)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            maps = feature_maps(scenes, config.validation_policy)
        ious = ious or [MeanIoU(device=device, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL) for _ in maps]
        for iou, features in zip(ious, maps, strict=True):
            iou.update(resize_labels(probe(features).argmax(1), scene_labels.shape[1:]), scene_labels)
    return [100 * iou.compute() for iou in ious]


def train_probe(config: TrainProbe) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    feature_maps = load_feature_maps(config, device)
    probe = SegmentationProbe(embed_dim=feature_maps.embed_dim, num_classes=NUM_CLASSES, dropout=config.dropout,
                              use_ln=not feature_maps.layer_normalized).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)

    def lr_factor(step: int) -> float:
        if step < config.warmup_steps:
            start = config.warmup_start_lr_fraction
            return start + (1 - start) * step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.num_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    images, labels = load_split("train")
    history: list[dict] = []
    best: dict | None = None
    start = time.monotonic()
    loss_sum, loss_count = 0.0, 0
    for step in range(config.num_steps + 1):
        if step % config.validate_every == 0 or step == config.num_steps:
            mious = validate(probe, feature_maps, config, device)
            history.append({"step": step, "validation_miou": mious, "train_loss": loss_sum / max(1, loss_count),
                            "seconds": time.monotonic() - start})
            loss_sum, loss_count = 0.0, 0
            log.info("step %d: validation mIoU per map %s", step, " ".join(f"{m:.2f}" for m in mious))
            if best is None or mious[-1] > best["validation_miou"][-1]:
                best = {"step": step, "validation_miou": mious,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}}
        if step == config.num_steps:
            break
        scenes, scene_labels = augmented_scenes(images, labels, batch_size=config.batch_size,
                                                size_px=config.scene_size_px, generator=generator, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            maps = feature_maps(scenes, config.training_policy)
        probe.train()
        logits = [probe(features) for features in maps]
        loss = torch.stack([F.cross_entropy(map_logits, resize_labels(scene_labels, map_logits.shape[2:]),
                                            ignore_index=IGNORE_LABEL) for map_logits in logits]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum, loss_count = loss_sum + loss.item(), loss_count + 1
        if step % 100 == 0:
            log.info("step %d: loss %.4f, lr %.2e, %.0f s", step, loss.item(), scheduler.get_last_lr()[0],
                     time.monotonic() - start)

    assert best is not None
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "embed_dim": feature_maps.embed_dim, "use_ln": not feature_maps.layer_normalized,
        "selected_step": best["step"], "selected_validation_miou": best["validation_miou"],
        "history": history, "state_dict": best["state_dict"],
    }, config.output)
    log.info("Saved the probe of step %d (validation mIoU %.2f on the last map) to %s",
             best["step"], best["validation_miou"][-1], config.output)


def load_probe(path: Path, device: torch.device) -> tuple[SegmentationProbe, dict]:
    record = torch.load(path, weights_only=True)
    probe = SegmentationProbe(embed_dim=record["embed_dim"], num_classes=NUM_CLASSES, dropout=record["config"]["dropout"],
                              use_ln=record["use_ln"])
    probe.load_state_dict(record["state_dict"])
    return probe.to(device).eval(), record


@dataclass(frozen=True)
class Rollouts:
    """Validation mIoU at 1024 px after every glimpse of CanViT, its canvas decoded by a probe trained on its canvas."""

    output: Path
    probe: Path
    detail: Detail = "s512"
    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = 85
    """85 glimpses: C2F down to level 3, whose tiles are 128 px of a 1024 px scene."""
    canvas_grid_size: int = 64
    seed: int = 0
    batch_size: int = 8
    device: str = "cuda"


def sample(scene_1024: Tensor, scene_512: Tensor, viewpoint: Viewpoint, detail: Detail) -> Tensor:
    match detail:
        case "s512":
            spatial = scene_512
        case "s512_then_native":
            scale = viewpoint.scales[0].item()
            assert torch.all(viewpoint.scales == scale), "s512_then_native needs one scale per batch"
            spatial = scene_512 if scale * scene_512.shape[-1] >= RELEASED_GLIMPSE_SIZE_PX else scene_1024
        case "native":
            spatial = scene_1024
        case _:
            assert_never(detail)
    return sample_at_viewpoint(spatial=spatial, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)


def iou_record(ious: list[MeanIoU]) -> dict:
    per_class = [(iou.intersection / iou.union.clamp(min=1)).tolist() for iou in ious]
    return {"miou": [100 * iou.compute() for iou in ious], "per_class_iou": per_class, "class_names": CLASS_NAMES}


@torch.inference_mode()
def rollouts(exp: Rollouts) -> None:
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    probe, record = load_probe(exp.probe, device)
    assert record["config"]["features"] == "canvas", exp.probe
    canvit = CanViTForPretraining.from_pretrained(record["config"]["pretrained_repo"]).canvit
    seg = CanViTForSemanticSegmentation(canvit=canvit, probe=probe).to(device).eval()
    images, labels = load_split("val")
    num_scenes = 2 * len(images)
    ious = [MeanIoU(device=device, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL) for _ in range(exp.num_glimpses)]
    scales = torch.zeros(exp.num_glimpses, dtype=torch.float64)
    start = time.monotonic()
    for first in range(0, num_scenes, exp.batch_size):
        scene_1024, scene_labels = validation_scenes(images, labels, first, exp.batch_size, device)
        scene_512, _ = downsample(scene_1024, scene_labels, 512)
        batch_size = scene_1024.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy = make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                                 canvas_grid_size=exp.canvas_grid_size, canvas_logits=seg.logits)
            state = seg.init_state(batch_size=batch_size, canvas_grid_size=exp.canvas_grid_size)
            for t in range(exp.num_glimpses):
                viewpoint = policy.step(t, state)
                glimpse = sample(scene_1024, scene_512, viewpoint, exp.detail)
                state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                ious[t].update(predicted_labels(seg.logits(state.canvas), size_px=SCENE_PX), scene_labels)
                scales[t] += viewpoint.scales.double().sum().cpu()
        log.info("%d/%d scenes", first + batch_size, num_scenes)
    write_result(exp.output, {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_config": record["config"], "probe_selected_step": record["selected_step"],
        "paper_name": POLICIES[exp.policy].paper_name,
        **iou_record(ious), "mean_scale": (scales / num_scenes).tolist(),
        "n_scenes": num_scenes, "wall_time_seconds": time.monotonic() - start,
    })


@dataclass(frozen=True)
class Passive:
    """Validation mIoU at 1024 px of DINOv3, the scene resized to the input size of a probe trained on its features."""

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
    images, labels = load_split("val")
    num_scenes = 2 * len(images)
    iou = MeanIoU(device=device, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)
    start = time.monotonic()
    for first in range(0, num_scenes, exp.batch_size):
        scene_1024, scene_labels = validation_scenes(images, labels, first, exp.batch_size, device)
        scenes, _ = downsample(scene_1024, scene_labels, input_px)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            patches = dinov3.patch_grid(scenes, input_size_px=input_px)
        iou.update(predicted_labels(probe(patches), size_px=SCENE_PX), scene_labels)
    write_result(exp.output, {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_config": record["config"], "probe_selected_step": record["selected_step"],
        **iou_record([iou]), "n_scenes": num_scenes, "wall_time_seconds": time.monotonic() - start,
    })


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    log.info("mIoU: %s", " ".join(f"{v:.2f}" for v in result["miou"]))
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
