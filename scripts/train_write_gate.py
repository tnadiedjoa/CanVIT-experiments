"""Learn a change to CanViT's Canvas Attention Writes (local_writes.LEARNED_WRITES: how far they reach outside the
glimpse, a key with a zero value to attend to instead of the sink register, or the Writes' own parameters), with the
rest of the released model frozen, on the paper's pretraining objective but with long rollouts.

Pretraining rollouts are short (F-IID and R-IID, 4 glimpses on average, canvit_pytorch.pretrain.config), so the Writes
never had to stay useful over long sequences of small glimpses, where their leak outside the crop degrades the canvas
(README). Here every step runs one long rollout per scene and the new parameters (a few per Write and head) learn from
the paper's distillation loss: after every glimpse, the canvas predicts the DINOv3 teacher's standardized patch
features of the whole scene, and the recurrent CLS token its CLS token (mean squared errors, summed). Gradients flow
through chunks of consecutive glimpses (truncated BPTT, as in pretraining).

Scenes are ADE20K or Cityscapes training images (labels unused) at 512 px with the 32×32 canvas of pretraining; the
validation scenes are never seen.

    .venv/bin/python -P scripts/train_write_gate.py --output probes/write_gate_c2f_riid.pt
"""

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import PolicyName, make_policy
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.pretrain.loss import DistillationTargets, distillation_losses
from canvit_pytorch.pretrain.step import sample_rollout_length
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from local_writes import LEARNED_WRITES, LocalWrites  # noqa: E402

log = logging.getLogger(__name__)


class TrainingScenes(Dataset[Tensor]):
    """Training images of a dataset, without labels, transformed as the evaluation scenes."""

    def __init__(self, dataset: Literal["ade20k", "cityscapes"], *, size_px: int) -> None:
        match dataset:
            case "ade20k":
                self.paths = sorted((ade20k.dataset_root() / "images" / "training").glob("*.jpg"))
            case "cityscapes":
                root = Path(os.environ.get("CITYSCAPES_ROOT", "/tmp/canvit-nadiedjoa-24/data/cityscapes"))
                self.paths = sorted((root / "leftImg8bit" / "train").glob("*/*_leftImg8bit.png"))
        assert self.paths, dataset
        self.transform = preprocess(size_px)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(Image.open(self.paths[index]).convert("RGB"))
        assert isinstance(image, Tensor)
        return image


@dataclass(frozen=True)
class Training:
    output: Path
    kind: Literal["gate", "null", "writes", "canvas_attention", "canvit"] = "gate"
    """gate: local_writes.LearnedWriteGate (how far Writes reach); null: local_writes.LearnedNullWrite (a key with a
    zero value that cells can attend to instead of the sink register); writes: local_writes.FinetunedWrites (the
    Writes' own parameters); canvas_attention: local_writes.FinetunedCanvasAttention (the Reads' and Writes');
    canvit: local_writes.FinetunedCanViT (all of CanViT's)."""
    scenes: Literal["ade20k", "cityscapes"] = "ade20k"
    policies: tuple[PolicyName, ...] = ("coarse_to_fine", "random")
    """One policy per step, drawn uniformly: C2F to level 3, and R-IID, whose large glimpses should keep writing."""
    num_glimpses: int = 85
    tbptt_chunk_glimpses: int = 4
    stop_probability: float | None = None
    """If set, rollouts have the random length of pretraining instead (chunks of tbptt_chunk_glimpses glimpses, then
    stop with this probability, at most num_glimpses): the control that trains as long, but on short rollouts."""
    num_steps: int = 300
    batch_size: int = 8
    lr: float = 1e-2
    warmup_steps: int = 0
    """Linear warmup of the learning rate from 0."""
    grad_clip_norm: float | None = None
    cls_loss: bool = True
    """Also train on the CLS term of the objective. Local Writes help the canvas but hurt the recurrent CLS token's
    prediction (README), so a change that only acts on canvas cells can be trained on the patch term alone."""
    local_writes_from: int | None = None
    """When set, the Writes are kept in the glimpse's crop from this glimpse on during training
    (local_writes.LocalWrites), so that the learned change adapts to the rule; evaluate with the same rule."""
    local_writes_margin: float = 0.0
    local_writes_max_scale: float | None = None
    seed: int = 0
    log_every: int = 10
    device: str = "cuda"


def main(cfg: Training) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model = CanViTForPretraining.from_pretrained(FLAGSHIP).to(device).eval().requires_grad_(False)
    teacher = load_teacher(TEACHER_REPO, device)
    grid = model.teacher_patch_grid
    gate = LEARNED_WRITES[cfg.kind](model.canvit, canvas_grid_size=grid, device=device)
    local_writes = (LocalWrites(model.canvit, canvas_grid_size=grid, device=device, start=cfg.local_writes_from,
                                margin=cfg.local_writes_margin, max_scale=cfg.local_writes_max_scale)
                    if cfg.local_writes_from is not None else None)
    optimizer = torch.optim.Adam(gate.parameters(), lr=cfg.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lambda step: min(1.0, (step + 1) / max(cfg.warmup_steps, 1)))
    scenes = TrainingScenes(cfg.scenes, size_px=grid * teacher.patch_size)
    loader = DataLoader(scenes, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
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
        policy_name = random.choice(cfg.policies)
        num_glimpses = cfg.num_glimpses
        if cfg.stop_probability is not None:
            num_glimpses = min(num_glimpses, sample_rollout_length(tbptt_chunk_glimpses=cfg.tbptt_chunk_glimpses,
                                                                   stop_probability=cfg.stop_probability))
        policy = make_policy(policy_name, batch_size=images.shape[0], device=device, num_glimpses=num_glimpses,
                             canvas_grid_size=grid)
        state = model.init_state(batch_size=images.shape[0], canvas_grid_size=grid)
        optimizer.zero_grad()
        chunk_loss, total = torch.zeros((), device=device), 0.0
        for t in range(num_glimpses):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                viewpoint = policy.step(t, state)
                gate.before_glimpse(t, viewpoint)
                if local_writes is not None:
                    local_writes.before_glimpse(t, viewpoint)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                losses = distillation_losses(model, state, targets, enable_teacher_patch_loss=True,
                                             enable_teacher_cls_loss=cfg.cls_loss)
            loss = torch.stack(list(losses.values())).sum().float()
            chunk_loss = chunk_loss + loss
            total += loss.item()
            if (t + 1) % cfg.tbptt_chunk_glimpses == 0 or t + 1 == num_glimpses:
                (chunk_loss / num_glimpses).backward()
                chunk_loss = torch.zeros((), device=device)
                state = state.detach()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(gate.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        schedule.step()
        history.append({"step": step, "policy": policy_name, "num_glimpses": num_glimpses,
                        "loss": total / num_glimpses})
        if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
            described = gate.describe()
            log.info("step %d (%s, %d glimpses): loss %.4f, %s, %.0f s", step, policy_name, num_glimpses,
                     total / num_glimpses,
                     ", ".join(f"{name} {[round(sum(row) / len(row), 4) for row in values]}"
                               for name, values in described.items()), time.monotonic() - start)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": cfg.kind, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "state_dict": {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                "gate": gate.describe(), "history": history}, cfg.output)
    cfg.output.with_suffix(".json").write_text(json.dumps({"gate": gate.describe(), "history": history}, indent=1))
    log.info("Saved %s", cfg.output)


if __name__ == "__main__":
    main(tyro.cli(Training))
