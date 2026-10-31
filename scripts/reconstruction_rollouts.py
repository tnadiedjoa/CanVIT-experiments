"""How closely CanViT's canvas predicts its DINOv3 teacher's features of the scene after every glimpse, with the
training-free changes to the Writes of scripts/local_writes.py: the paper's reconstruction metric, which needs no task
probe. The scenes (ADE20K validation, 512 px), teacher, standardization and cosine similarities are those of
`python -m canvit_pytorch.evaluate reconstruction`, so with the released Writes and a deterministic policy (RFS) the
result matches the official CLI's. The canvas is the teacher's 32×32 patch grid, the only size the metric accepts.

    .venv/bin/python -P scripts/reconstruction_rollouts.py --policy coarse_to_fine --local-writes-from 21 \
        --output results/reconstruction/c2f_c32_t85_s0_local21.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.data import DataLoader, Subset

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.reconstruction import METRICS, Scenes, cosine_similarities
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import POLICIES, PolicyName, make_policy
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from local_writes import DroppedWriteToken, LearnedWriteGate, LocalWrites, OutsideWrites  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    output: Path
    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = 85
    local_writes_from: int | None = None
    """When set, from this glimpse on, the Write residuals outside the glimpse's crop change (outside_writes)."""
    outside_writes: OutsideWrites = "drop"
    local_writes_max_scale: float | None = None
    """With local_writes_from: only glimpses of at most this scale keep their Writes in their crop."""
    local_writes_margin: float = 0.0
    """With local_writes_from: cells within this many crop widths of the crop also keep their Writes."""
    unseen_writes: bool = False
    """With local_writes_from: cells that no glimpse has covered yet also keep their Writes."""
    dropped_write_token: int | None = None
    """When set, from glimpse dropped_write_token_from on, every Write residual loses the part that comes from this
    key token of the glimpse (an index into local_writes.PREFIX_TOKENS; 5 is backbone register 3, the attention sink
    of the canvas cells far from the glimpse)."""
    dropped_write_token_from: int = 1
    write_gate: Path | None = None
    """When set, a learned gate on the Writes (local_writes.LearnedWriteGate) saved by scripts/train_write_gate.py."""
    seed: int = 0
    """Seeds the random policies (C2F's tile order, R-IID's viewpoints); the official CLI does not seed them."""
    batch_size: int = 16
    num_images: int | None = None
    """When set, only the first validation images, for quick checks."""
    device: str = "cuda"


def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert not POLICIES[exp.policy].needs_segmentation_probe, exp.policy
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    model = CanViTForPretraining.from_pretrained(FLAGSHIP).to(device).eval()
    teacher = load_teacher(TEACHER_REPO, device)
    grid = model.teacher_patch_grid
    dataset = Scenes(root=ade20k.dataset_root(), size_px=grid * teacher.patch_size)
    if exp.num_images is not None:
        dataset = Subset(dataset, range(exp.num_images))
    local_writes = (LocalWrites(model.canvit, canvas_grid_size=grid, device=device, start=exp.local_writes_from,
                                outside=exp.outside_writes, max_scale=exp.local_writes_max_scale,
                                margin=exp.local_writes_margin, unseen_writes=exp.unseen_writes)
                    if exp.local_writes_from is not None else None)
    dropped_token = (DroppedWriteToken(model.canvit, token=exp.dropped_write_token)
                     if exp.dropped_write_token is not None else None)
    write_gate = None
    if exp.write_gate is not None:
        write_gate = LearnedWriteGate(model.canvit, canvas_grid_size=grid, device=device)
        write_gate.load_state_dict(torch.load(exp.write_gate, weights_only=True)["state_dict"])

    sums = torch.zeros(exp.num_glimpses, len(METRICS), dtype=torch.float64)
    start = time.monotonic()
    loader = DataLoader(dataset, batch_size=exp.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch, images in enumerate(loader):
            images = images.to(device)
            batch_size = images.shape[0]
            with torch.autocast(device_type=device.type, enabled=False):
                target = teacher(images)
            policy = make_policy(exp.policy, batch_size=batch_size, device=device, num_glimpses=exp.num_glimpses,
                                 canvas_grid_size=grid)
            state = model.canvit.init_state(batch_size=batch_size, canvas_grid_size=grid)
            for t in range(exp.num_glimpses):
                viewpoint = policy.step(t, state)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                if write_gate is not None:
                    write_gate.before_glimpse(t, viewpoint)
                if dropped_token is not None:
                    dropped_token.active = t >= exp.dropped_write_token_from
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                state = model.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                scene_norm, scene_raw = cosine_similarities(
                    prediction=model.predict_teacher_patches(state.canvas), raw_target=target.patches.float(),
                    standardizer=model.teacher_patch_standardizer,
                )
                cls_norm, cls_raw = cosine_similarities(
                    prediction=model.predict_teacher_cls(state.recurrent_cls), raw_target=target.cls.float(),
                    standardizer=model.teacher_cls_standardizer,
                )
                sums[t] += torch.tensor([scene_raw, cls_raw, scene_norm, cls_norm], dtype=torch.float64) * batch_size
            if batch % 10 == 0:
                log.info("%d/%d images", (batch + 1) * exp.batch_size, len(dataset))

    means = sums / len(dataset)
    result = {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "pretrained_repo": FLAGSHIP,
        "per_timestep": [{"t": t, **dict(zip(METRICS, row.tolist(), strict=True))} for t, row in enumerate(means)],
        "n_images": len(dataset),
        "wall_time_seconds": time.monotonic() - start,
    }
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("scene_cos_norm by glimpse: %s", " ".join(f"{row['scene_cos_norm']:.4f}" for row in result["per_timestep"]))


if __name__ == "__main__":
    main(tyro.cli(Experiment))
