"""ADE20K validation mIoU after every glimpse, with canvas statistics, for the paper's policies and for variants the
paper does not evaluate: other tile orders within C2F's quadtree levels, and anti-aliased glimpse extraction.

The model, probe, scene transform, label decoding and metric are the official ones (canvit_pytorch.benchmarks.ade20k,
canvit_pytorch.evaluate), so the mIoU matches `python -m canvit_pytorch.evaluate ade20k-segmentation-canvit` for the
same policy. The validation scenes are decoded once and kept as uint8 in a cache file.

    ADE20K_ROOT=/path/to/ADEChallengeData2016 .venv/bin/python scripts/ade20k_rollouts.py \
        --policy coarse_to_fine --num-glimpses 21 --canvas-grid-size 64 --output results/c2f_c64.json
"""

import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, assert_never, get_args

import torch
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.ade20k_segmentation import predicted_labels
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX, released_ade20k_probe
from canvit_pytorch.model import CanViTForSemanticSegmentation
from canvit_pytorch.model.canvit import RecurrentState
from canvit_pytorch.policies import POLICIES, FixedSequence, Policy, PolicyName, make_policy
from canvit_pytorch.policies.entropy import predictive_entropy, tile_masks
from canvit_pytorch.policies.quadtree import levels_covering, quadtree_level, quadtree_level_z_order, shuffled_levels
from canvit_pytorch.preprocess import IMAGENET_MEAN, IMAGENET_STD
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from local_writes import LocalWrites, OutsideWrites  # noqa: E402

log = logging.getLogger(__name__)

TileOrder = Literal["low_entropy", "canvas_contrast", "oracle_error"]
"""Orders of the tiles within each C2F level, beyond the paper's random order (C2F) and entropy order (EG-C2F):

- low_entropy: EG-C2F reversed, the tile of lowest mean entropy first; a lower reference for the effect of order;
- canvas_contrast: the tile whose layer-normalized canvas patches differ most from their neighbors, without a probe;
- oracle_error: the tile whose pixels the current segmentation gets wrong most often, read from the ground truth;
  an upper reference, not a policy.
"""
Continuation = Literal["level2_revisits", "full_scene"]
"""What follows C2F's first three levels (21 glimpses, down to tiles at the scene's native resolution) in place of the
finer, upsampled level 3: the level-2 tiles again, each pass in a new random order, or the full scene repeated."""
PolicyChoice = PolicyName | Literal[
    "c2f_low_entropy", "c2f_canvas_contrast", "c2f_oracle_error", "c2f_then_level2_revisits", "c2f_then_full_scene",
]
Sampling = Literal["bilinear", "antialiased"]
"""bilinear: the official extraction (grid_sample, used in pretraining too); antialiased: the crop at its native
resolution, then downsampled with an anti-aliasing filter when it covers more pixels than the glimpse."""

STATISTICS = ("canvas_patch_norm", "delta_previous", "entropy")
"""Per glimpse, averaged over canvas cells and scenes: the L2 norm of the raw canvas patches, the cosine
dissimilarity of each layer-normalized patch to the previous canvas, and the probe's predictive entropy in nats."""


@dataclass(frozen=True)
class Experiment:
    output: Path
    policy: PolicyChoice = "coarse_to_fine"
    num_glimpses: int = 21
    canvas_grid_size: int = 32
    sampling: Sampling = "bilinear"
    probe_canvas_grid_size: int | None = None
    """Canvas grid the released probe was trained on, when it differs from canvas_grid_size (the probe is a 1×1
    convolution, so it applies to any grid)."""
    write_scale: float = 1.0
    """Factor on every Canvas Attention Write residual from the second glimpse on; 1 is the released model. Below 1,
    later glimpses change the canvas less: a training-free probe of the trade-off between keeping and updating."""
    finer_seen_write_exponent: float | None = None
    """When set, a glimpse's Write residual in each canvas cell inside its crop is multiplied by
    min(1, finest scale already seen at that cell / glimpse scale) ** exponent: coarse glimpses barely overwrite cells
    already seen in more detail. Changes nothing for policies that only go from coarse to fine (C2F)."""
    local_writes_from: int | None = None
    """When set, from this glimpse on, the Write residuals into the canvas cells outside the glimpse's crop are dropped
    (the canvas registers keep theirs), so that a glimpse changes only the part of the canvas it saw. The released
    model writes into every cell."""
    outside_writes: OutsideWrites = "drop"
    """With local_writes_from: what happens to the Write residuals outside the crop (OutsideWrites)."""
    local_writes_max_scale: float | None = None
    """With local_writes_from: only glimpses of at most this scale keep their Writes in their crop, and larger ones
    write into every cell, as released; 0.25 is C2F's levels 2 and 3."""
    scene_detail_px: int | None = None
    """When set, each scene is downsampled to this size with antialiasing and upsampled back before the rollout, so no
    glimpse sees detail finer than this resolution; labels keep the full scene size."""
    seed: int = 0
    """Seeds the random policies (C2F's tile order, R-IID's viewpoints)."""
    batch_size: int = 8
    """Scenes per batch; 8 fit a 64×64 canvas in 24 GB, as in the official evaluation docs."""
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    """uint8 validation scenes and decoded labels at the released scene size; built from $ADE20K_ROOT if missing."""
    glimpse_size_px: int = RELEASED_GLIMPSE_SIZE_PX
    """Side of each glimpse: 128 px (8×8 patches) in pretraining and in the paper."""
    num_images: int | None = None
    """When set, only the first validation images, for quick checks."""
    autocast: bool = True
    """bfloat16 autocast, as in the official evaluation; False runs in float32."""
    device: str = "cuda"


class Uint8EvaluationTransform:
    """The official evaluation transform up to its last uint8 step: dividing by 255 and normalizing happen later."""

    def __init__(self, size_px: int) -> None:
        official = ade20k.evaluation_transform(size_px)
        assert isinstance(official, ade20k.dataset.EvaluationTransform)
        resize, to_tensor, normalize = official.image_transform.transforms
        assert isinstance(to_tensor, transforms.ToTensor) and isinstance(normalize, transforms.Normalize)
        self.image_transform = transforms.Compose([resize, transforms.PILToTensor()])
        self.label_transform = official.label_transform

    def __call__(self, image, label_map) -> tuple[Tensor, Tensor]:
        return self.image_transform(image), ade20k.decode_annotation(self.label_transform(label_map).squeeze(0))


def load_validation_scenes(cache: Path, *, size_px: int) -> tuple[Tensor, Tensor]:
    """[N, 3, S, S] uint8 scenes and [N, S, S] uint8 labels (IGNORE_LABEL where unlabeled)."""
    if not cache.exists():
        dataset = ade20k.ADE20kDataset(root=ade20k.dataset_root(), split="validation",
                                       transform=Uint8EvaluationTransform(size_px))
        assert len(dataset) == ade20k.NUM_VALIDATION_IMAGES, len(dataset)
        batches = list(DataLoader(dataset, batch_size=64, num_workers=4))
        cache.parent.mkdir(parents=True, exist_ok=True)
        labels = torch.cat([b[1] for b in batches])
        assert labels.min() >= 0 and labels.max() <= ade20k.IGNORE_LABEL, (labels.min(), labels.max())
        torch.save({"images": torch.cat([b[0] for b in batches]), "labels": labels.to(torch.uint8)}, cache)
    scenes = torch.load(cache, weights_only=True)
    assert scenes["images"].shape[-1] == size_px, (cache, scenes["images"].shape)
    return scenes["images"], scenes["labels"]


def normalize(images: Tensor) -> Tensor:
    """uint8 [B, 3, H, W] -> ImageNet-normalized float32, as ToTensor then Normalize compute it."""
    mean = images.new_tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std


def remove_detail(images: Tensor, *, detail_px: int) -> Tensor:
    """[B, C, S, S] scenes without detail finer than detail_px across: antialiased downsampling, bilinear upsampling."""
    size = images.shape[-1]
    assert detail_px < size, (detail_px, size)
    small = F.interpolate(images, size=(detail_px, detail_px), mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(size, size), mode="bilinear", align_corners=False)


def sample_antialiased(*, spatial: Tensor, viewpoint: Viewpoint, glimpse_size_px: int) -> Tensor:
    """Each viewpoint's crop of [B, C, S, S] at its native resolution, then resized to the glimpse with antialiasing.

    Every viewpoint of the batch must share its scale (true of the quadtree policies and RFS), so that the crops share
    a size. A crop no larger than the glimpse is the official bilinear upsampling.
    """
    scale = viewpoint.scales[0].item()
    assert torch.all(viewpoint.scales == scale), "antialiased sampling needs one scale per batch"
    crop_px = round(scale * spatial.shape[-1])
    if crop_px <= glimpse_size_px:
        return sample_at_viewpoint(spatial=spatial, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px)
    crop = sample_at_viewpoint(spatial=spatial, viewpoint=viewpoint, glimpse_size_px=crop_px)
    resized = F.interpolate(crop.float(), size=(glimpse_size_px, glimpse_size_px), mode="bilinear",
                            align_corners=False, antialias=True)
    return resized.to(spatial.dtype)


def layer_normalized_patches(seg: CanViTForSemanticSegmentation, canvas: Tensor) -> Tensor:
    """[B, G, G, D] float32."""
    patches = seg.canvit.canvas_patch_grid(canvas).float()
    return F.layer_norm(patches, (patches.shape[-1],))


def canvas_contrast(patches: Tensor) -> Tensor:
    """[B, G, G, D] layer-normalized patches -> [B, G, G]: mean cosine dissimilarity to the 4-connected neighbors."""
    unit = F.normalize(patches, dim=-1)
    across_cols = 1 - (unit[:, :, 1:] * unit[:, :, :-1]).sum(-1)  # [B, G, G-1]
    across_rows = 1 - (unit[:, 1:] * unit[:, :-1]).sum(-1)  # [B, G-1, G]
    total = (F.pad(across_cols, (1, 0)) + F.pad(across_cols, (0, 1))
             + F.pad(across_rows, (0, 0, 1, 0)) + F.pad(across_rows, (0, 0, 0, 1)))
    count = (F.pad(torch.ones_like(across_cols), (1, 0)) + F.pad(torch.ones_like(across_cols), (0, 1))
             + F.pad(torch.ones_like(across_rows), (0, 0, 1, 0)) + F.pad(torch.ones_like(across_rows), (0, 0, 0, 1)))
    return total / count


class ScoreGuidedC2F:
    """C2F's quadtree levels in order; within a level, the unvisited tile of highest mean cell score.

    EntropyGuidedC2F (canvit_pytorch.policies.entropy) with the score as a parameter; with the probe's entropy as
    score it chooses the same tiles.
    """

    def __init__(self, *, batch_size: int, device: torch.device, num_glimpses: int, canvas_grid_size: int,
                 cell_score: Callable[[RecurrentState], Tensor]) -> None:
        self.batch_size, self.device, self.cell_score = batch_size, device, cell_score
        self.levels = [quadtree_level(level) for level in range(levels_covering(num_glimpses))]
        self.level_start = [sum(len(tiles) for tiles in self.levels[:i]) for i in range(len(self.levels))]
        self.masks = [tile_masks(tiles, canvas_grid_size=canvas_grid_size, device=device).float() for tiles in self.levels]
        self.visited: Tensor | None = None

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        level = max(i for i, start in enumerate(self.level_start) if t >= start)
        tiles = self.levels[level]
        if level == 0:
            return Viewpoint.full_scene(batch_size=self.batch_size, device=self.device)
        if t == self.level_start[level]:
            self.visited = torch.zeros(self.batch_size, len(tiles), dtype=torch.bool, device=self.device)
        assert self.visited is not None
        masks = self.masks[level]
        score = self.cell_score(state).float()
        # Product then sum, as EntropyGuidedC2F computes it: an einsum would run in bfloat16 under autocast.
        mean_score = (score.unsqueeze(1) * masks.unsqueeze(0)).sum(dim=(2, 3)) / masks.sum(dim=(1, 2)).clamp(min=1)
        chosen = mean_score.masked_fill(self.visited, float("-inf")).argmax(dim=1)
        self.visited.scatter_(1, chosen.unsqueeze(1), True)
        selected = torch.tensor(tiles, device=self.device)[chosen]
        return Viewpoint(centers=selected[:, :2].contiguous(), scales=selected[:, 2].contiguous())


def oracle_error(seg: CanViTForSemanticSegmentation, labels: Tensor, canvas_grid_size: int) -> Callable[[RecurrentState], Tensor]:
    """Cell score: the fraction of the cell's labeled pixels that the segmentation decoded from the canvas gets wrong."""
    labeled = (labels != ade20k.IGNORE_LABEL).float().unsqueeze(1)
    cell_px = labels.shape[-1] // canvas_grid_size

    def score(state: RecurrentState) -> Tensor:
        predictions = predicted_labels(seg.logits(state.canvas), size_px=labels.shape[-1])
        wrong = ((predictions != labels).float().unsqueeze(1) * labeled)
        return (F.avg_pool2d(wrong, cell_px) / F.avg_pool2d(labeled, cell_px).clamp(min=1e-6)).squeeze(1)

    return score


class ScaledWrites:
    """Multiplies the output of every Canvas Attention Write by a factor while active."""

    def __init__(self, seg: CanViTForSemanticSegmentation, factor: float) -> None:
        self.factor, self.active = factor, False
        self.handles = [write.register_forward_hook(self.hook) for write in seg.canvit.canvas_writes] if factor != 1 else []

    def hook(self, module, args, output: Tensor) -> Tensor | None:
        return output * self.factor if self.active else None


def c2f_then(continuation: Continuation, *, batch_size: int, device: torch.device, num_glimpses: int) -> FixedSequence:
    """C2F's levels 0 to 2, enumerated as C2F enumerates them, then the continuation up to num_glimpses."""
    c2f = [quadtree_level_z_order(level) for level in range(3)]
    match continuation:
        case "level2_revisits":
            extra = quadtree_level(2)
        case "full_scene":
            extra = quadtree_level(0)
        case _:
            assert_never(continuation)
    passes = math.ceil(max(0, num_glimpses - 21) / len(extra))
    return FixedSequence(shuffled_levels(c2f + [extra] * passes, batch_size=batch_size, device=device,
                                         num_glimpses=num_glimpses))


class FinerSeenWriteGate:
    """Damps Writes into the canvas cells already seen at a finer scale than the current glimpse's
    (Experiment.finer_seen_write_exponent)."""

    def __init__(self, seg: CanViTForSemanticSegmentation, *, exponent: float, batch_size: int, canvas_grid_size: int,
                 device: torch.device) -> None:
        self.exponent = exponent
        self.num_registers = seg.canvit.config.num_canvas_registers
        self.cell_centers = grid_coords(size=canvas_grid_size, device=device)  # [G, G, 2]
        self.finest = torch.full((batch_size, canvas_grid_size, canvas_grid_size), math.inf, device=device)
        self.factor: Tensor | None = None  # [B, num_registers + G*G, 1] for the glimpse being taken
        self.handles = [write.register_forward_hook(self.hook) for write in seg.canvit.canvas_writes]

    def inside(self, viewpoint: Viewpoint) -> Tensor:
        """[B, G, G]: the cells whose centers lie in each viewpoint's crop."""
        offsets = (self.cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs()
        return (offsets <= viewpoint.scales.view(-1, 1, 1, 1)).all(dim=-1)

    def before_glimpse(self, viewpoint: Viewpoint) -> None:
        inside = self.inside(viewpoint)
        ratio = (self.finest / viewpoint.scales.view(-1, 1, 1)).clamp(max=1) ** self.exponent
        cells = torch.where(inside, ratio, torch.ones_like(ratio)).flatten(1)
        registers = torch.ones(cells.shape[0], self.num_registers, device=cells.device)
        self.factor = torch.cat([registers, cells], dim=1).unsqueeze(-1)

    def after_glimpse(self, viewpoint: Viewpoint) -> None:
        scales = viewpoint.scales.view(-1, 1, 1).expand_as(self.finest)
        self.finest = torch.where(self.inside(viewpoint), torch.minimum(self.finest, scales), self.finest)

    def hook(self, module, args, output: Tensor) -> Tensor:
        assert self.factor is not None
        return output * self.factor

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def build_policy(exp: Experiment, seg: CanViTForSemanticSegmentation, *, labels: Tensor, device: torch.device) -> Policy:
    batch_size = labels.shape[0]
    if exp.policy in get_args(PolicyName):
        return make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                           canvas_grid_size=exp.canvas_grid_size, canvas_logits=seg.logits)
    if exp.policy in ("c2f_then_level2_revisits", "c2f_then_full_scene"):
        continuation: Continuation = exp.policy.removeprefix("c2f_then_")  # type: ignore[assignment]
        return c2f_then(continuation, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses)
    order: TileOrder = exp.policy.removeprefix("c2f_")  # type: ignore[assignment]
    match order:
        case "low_entropy":
            cell_score = lambda state: -predictive_entropy(seg.logits(state.canvas))  # noqa: E731
        case "canvas_contrast":
            cell_score = lambda state: canvas_contrast(layer_normalized_patches(seg, state.canvas))  # noqa: E731
        case "oracle_error":
            cell_score = oracle_error(seg, labels, exp.canvas_grid_size)
        case _:
            assert_never(order)
    return ScoreGuidedC2F(batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                          canvas_grid_size=exp.canvas_grid_size, cell_score=cell_score)


def sampler(sampling: Sampling) -> Callable[..., Tensor]:
    match sampling:
        case "bilinear":
            return sample_at_viewpoint
        case "antialiased":
            return sample_antialiased
        case _:
            assert_never(sampling)


@torch.inference_mode()
def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    scene_px, num_glimpses = RELEASED_SCENE_SIZE_PX, exp.num_glimpses
    probe_grid = exp.probe_canvas_grid_size or exp.canvas_grid_size
    probe_repo = released_ade20k_probe("in21k", scene_size_px=scene_px, canvas_grid_size=probe_grid)
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=FLAGSHIP, probe_repo=probe_repo)
    seg = seg.to(device).eval()
    images_u8, labels_u8 = load_validation_scenes(exp.cache, size_px=scene_px)
    images_u8, labels_u8 = images_u8[:exp.num_images], labels_u8[:exp.num_images]
    sample = sampler(exp.sampling)
    scaled_writes = ScaledWrites(seg, exp.write_scale)
    local_writes = (LocalWrites(seg.canvit, canvas_grid_size=exp.canvas_grid_size, device=device,
                                outside=exp.outside_writes, max_scale=exp.local_writes_max_scale)
                    if exp.local_writes_from is not None else None)

    mious = [ade20k.MeanIoU(device=device) for _ in range(num_glimpses)]
    sums = {name: torch.zeros(num_glimpses, dtype=torch.float64, device=device) for name in STATISTICS}
    scales = torch.zeros(num_glimpses, dtype=torch.float64)
    start = time.monotonic()
    for first in range(0, len(images_u8), exp.batch_size):
        images = normalize(images_u8[first:first + exp.batch_size].to(device))
        if exp.scene_detail_px is not None:
            images = remove_detail(images, detail_px=exp.scene_detail_px)
        labels = labels_u8[first:first + exp.batch_size].to(device).long()
        batch_size = images.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=exp.autocast):
            policy = build_policy(exp, seg, labels=labels, device=device)
            state = seg.init_state(batch_size=batch_size, canvas_grid_size=exp.canvas_grid_size)
            gate = (FinerSeenWriteGate(seg, exponent=exp.finer_seen_write_exponent, batch_size=batch_size,
                                       canvas_grid_size=exp.canvas_grid_size, device=device)
                    if exp.finer_seen_write_exponent is not None else None)
            previous = layer_normalized_patches(seg, state.canvas)
            for t in range(num_glimpses):
                viewpoint = policy.step(t, state)
                scaled_writes.active = t >= 1
                glimpse = sample(spatial=images, viewpoint=viewpoint, glimpse_size_px=exp.glimpse_size_px)
                if gate is not None:
                    gate.before_glimpse(viewpoint)
                if local_writes is not None:
                    assert exp.local_writes_from is not None
                    local_writes.restrict_to(viewpoint if t >= exp.local_writes_from else None)
                state = seg.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                if gate is not None:
                    gate.after_glimpse(viewpoint)
                logits = seg.logits(state.canvas)
                mious[t].update(predicted_labels(logits, size_px=scene_px), labels)
                current = layer_normalized_patches(seg, state.canvas)
                raw = seg.canvit.canvas_patch_grid(state.canvas).float()
                sums["canvas_patch_norm"][t] += raw.norm(dim=-1).mean(dim=(1, 2)).sum().double()
                sums["delta_previous"][t] += (1 - F.cosine_similarity(current, previous, dim=-1)).mean(dim=(1, 2)).sum().double()
                sums["entropy"][t] += predictive_entropy(logits).mean(dim=(1, 2)).sum().double()
                scales[t] += viewpoint.scales.double().sum().cpu()
                previous = current
            if gate is not None:
                gate.remove()
        log.info("%d/%d scenes", first + batch_size, len(images_u8))

    n = len(images_u8)
    result = {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "probe_repo": probe_repo,
        "paper_name": POLICIES[exp.policy].paper_name if exp.policy in POLICIES else exp.policy,
        "miou": [100 * m.compute() for m in mious],
        **{name: (values / n).tolist() for name, values in sums.items()},
        "mean_scale": (scales / n).tolist(),
        "n_images": n,
        "wall_time_seconds": time.monotonic() - start,
        "versions": {"torch": torch.__version__},
    }
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("mIoU by glimpse: %s", " ".join(f"{v:.2f}" for v in result["miou"]))
    log.info("Wrote %s in %.0f s", exp.output, result["wall_time_seconds"])


if __name__ == "__main__":
    main(tyro.cli(Experiment))
