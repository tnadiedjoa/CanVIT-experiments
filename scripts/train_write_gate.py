"""Learn a small change to CanViT's Canvas Attention Writes (local_writes.LEARNED_WRITES: how far they reach outside
the glimpse, or a key with a zero value to attend to instead of the sink register), with the rest of the released
model frozen, on the paper's pretraining objective but with long rollouts.

Pretraining rollouts are short (F-IID and R-IID, 4 glimpses on average, canvit_pytorch.pretrain.config), so the Writes
never had to stay useful over long sequences of small glimpses, where their leak outside the crop degrades the canvas
(README). Here every step runs one long rollout per scene and the new parameters (a few per Write and head) learn from
the paper's distillation loss: after every glimpse, the canvas predicts the DINOv3 teacher's standardized patch
features of the whole scene, and the recurrent CLS token its CLS token (mean squared errors, summed). Gradients flow
through chunks of consecutive glimpses (truncated BPTT, as in pretraining).

Scenes are ADE20K training images (labels unused) at 512 px with the 32×32 canvas of pretraining; the validation
scenes are never seen.

    .venv/bin/python -P scripts/train_write_gate.py --output probes/write_gate_c2f_riid.pt
"""

import json
import logging
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
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from local_writes import LEARNED_WRITES  # noqa: E402

log = logging.getLogger(__name__)


class TrainingScenes(Dataset[Tensor]):
    """ADE20K training images, without labels, transformed as the evaluation scenes."""

    def __init__(self, *, root: Path, size_px: int) -> None:
        self.paths = sorted((root / "images" / "training").glob("*.jpg"))
        assert self.paths, root
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
    kind: Literal["gate", "null"] = "gate"
    """gate: local_writes.LearnedWriteGate (how far Writes reach); null: local_writes.LearnedNullWrite (a key with a
    zero value that cells can attend to instead of the sink register)."""
    policies: tuple[PolicyName, ...] = ("coarse_to_fine", "random")
    """One policy per step, drawn uniformly: C2F to level 3, and R-IID, whose large glimpses should keep writing."""
    num_glimpses: int = 85
    tbptt_chunk_glimpses: int = 4
    num_steps: int = 300
    batch_size: int = 8
    lr: float = 1e-2
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
    optimizer = torch.optim.Adam(gate.parameters(), lr=cfg.lr)
    scenes = TrainingScenes(root=ade20k.dataset_root(), size_px=grid * teacher.patch_size)
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
        policy = make_policy(policy_name, batch_size=images.shape[0], device=device, num_glimpses=cfg.num_glimpses,
                             canvas_grid_size=grid)
        state = model.init_state(batch_size=images.shape[0], canvas_grid_size=grid)
        optimizer.zero_grad()
        chunk_loss, total = torch.zeros((), device=device), 0.0
        for t in range(cfg.num_glimpses):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                viewpoint = policy.step(t, state)
                gate.before_glimpse(t, viewpoint)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                state = model(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                losses = distillation_losses(model, state, targets, enable_teacher_patch_loss=True,
                                             enable_teacher_cls_loss=True)
            loss = torch.stack(list(losses.values())).sum().float()
            chunk_loss = chunk_loss + loss
            total += loss.item()
            if (t + 1) % cfg.tbptt_chunk_glimpses == 0 or t + 1 == cfg.num_glimpses:
                (chunk_loss / cfg.num_glimpses).backward()
                chunk_loss = torch.zeros((), device=device)
                state = state.detach()
        optimizer.step()
        history.append({"step": step, "policy": policy_name, "loss": total / cfg.num_glimpses})
        if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
            described = gate.describe()
            log.info("step %d (%s): loss %.4f, %s, %.0f s", step, policy_name, total / cfg.num_glimpses,
                     ", ".join(f"{name} {[round(sum(row) / len(row), 3) for row in values]}"
                               for name, values in described.items()), time.monotonic() - start)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": cfg.kind, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "state_dict": {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                "gate": gate.describe(), "history": history}, cfg.output)
    cfg.output.with_suffix(".json").write_text(json.dumps({"gate": gate.describe(), "history": history}, indent=1))
    log.info("Saved %s", cfg.output)


if __name__ == "__main__":
    main(tyro.cli(Training))
