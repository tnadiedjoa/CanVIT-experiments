"""Whether the Writes' leak outside the crop is visible to CanViT's readouts, on C2F rollouts of ADE20K validation
scenes, for any released pretraining (`--model`, as in scripts/bio_rollouts.py).

A readout decodes a cell x as A·LN(x) + b, with A the pretraining head's matrix (to the teacher's 768-d features) or
the ADE20K probe's (LayerNorm, BatchNorm, 1×1 convolution to 150 classes), each scaled by its norms' gains. Up to
LN's scale, a residual r changes the readout by A·C·r, with C the centering of LN. Its visibility is
‖A C r‖² / ‖r‖², divided by its mean over random directions (‖A C‖_F² / D): 1 for a residual the readout weighs
like a random one, near 0 for one it does not see. The script reports, at level 3 (glimpses 21-84), the visibility of
the per-glimpse residual in the cells outside the crop and inside it, and the cosine between an outside cell's residuals
at consecutive glimpses (1 when the leak adds the same vector every time); and per glimpse the visibility of the
canvas itself (centered) and its norm, with released Writes or, with `--local-writes-from`, Writes kept in the crop.
It also reports, unweighted, the share of the residuals' energy in each readout's row space
(local_writes.readout_subspace; `random_share_*` is that of a random direction, the rank over D).

    .venv/bin/python -P scripts/leak_visibility.py --output results/leak_visibility/in21k_released.json
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch import Tensor

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.hub.repos import (RELEASED_GLIMPSE_SIZE_PX, RELEASED_PROBE_STEPS, RELEASED_SCENE_SIZE_PX,
                                      ade20k_probe_name, hub_repo)
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.policies import make_policy
from canvit_pytorch.preprocess import imagenet_normalize
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from bio_rollouts import pretrained_repo_of  # noqa: E402
from local_writes import LocalWrites, readout_subspace  # noqa: E402

log = logging.getLogger(__name__)
LEVEL3 = 21


@dataclass(frozen=True)
class Measurement:
    output: Path
    model: str = "in21k"
    num_scenes: int = 128
    canvas_grid_size: int = 32
    local_writes_from: int | None = None
    batch_size: int = 16
    seed: int = 0
    cache: Path = Path("/tmp/canvit-nadiedjoa-24/data/ade20k_val_s512.pt")
    """uint8 validation scenes at 512 px, as written by scripts/ade20k_rollouts.py."""
    device: str = "cuda"


def readout_matrix(scale: Tensor, weight: Tensor) -> Tensor:
    """[out, D] matrix of a readout A·LN(x) with LN's centering folded in, and normalized so that a random direction
    has visibility 1."""
    D = weight.shape[1]
    A = (weight.float() * scale.float()[None]) @ (torch.eye(D, device=weight.device) - 1 / D)
    return A / (A.square().sum() / D).sqrt()


def visibility(A: Tensor, r: Tensor) -> Tensor:
    """[...] visibility of the vectors r [..., D] to the normalized readout matrix A."""
    return (r @ A.T).square().sum(-1) / r.square().sum(-1).clamp(min=1e-12)


def main(m: Measurement) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(m.seed)
    device = torch.device(m.device)
    pretrained_repo = pretrained_repo_of(m.model)
    pretrained = CanViTForPretraining.from_pretrained(pretrained_repo).to(device).eval()
    canvit, G = pretrained.canvit, m.canvas_grid_size
    probe_repo = hub_repo(ade20k_probe_name(m.model, scene_size_px=RELEASED_SCENE_SIZE_PX, canvas_grid_size=G,
                                            steps=RELEASED_PROBE_STEPS))
    probe = SegmentationProbe.from_pretrained(probe_repo).to(device).eval()
    head = pretrained.teacher_patch_readout
    bn_scale = probe.bn.weight / (probe.bn.running_var + probe.bn.eps).sqrt()
    readouts = {"pretraining_head": readout_matrix(head.norm.weight, head.proj.weight),
                "ade20k_probe": readout_matrix(probe.ln.weight * bn_scale, probe.conv.weight[:, :, 0, 0])}
    bases = {"pretraining_head": readout_subspace(head.norm.weight.detach(), head.proj.weight.detach()),
             "ade20k_probe": readout_subspace(probe.ln.weight.detach() * bn_scale.detach(), probe.conv.weight[:, :, 0, 0])}
    num_registers = canvit.config.num_canvas_registers
    scenes = torch.load(m.cache, weights_only=True)["images"][:m.num_scenes]
    cell_centers = grid_coords(size=G, device=device)
    local = (LocalWrites(canvit, canvas_grid_size=G, device=device, start=m.local_writes_from)
             if m.local_writes_from is not None else None)
    residuals: list[Tensor] = []
    handles = [w.register_forward_hook(lambda mod, args, out: residuals.append(out[:, num_registers:].float()))
               for w in canvit.canvas_writes]
    T = 85
    sums = {k: torch.zeros(T, dtype=torch.float64) for k in
            [f"canvas_visibility_{n}" for n in readouts] + ["canvas_norm"]}
    level3 = {k: 0.0 for k in [f"{w}_visibility_{n}" for w in ("outside", "inside") for n in readouts]
              + [f"{w}_share_{n}" for w in ("outside", "inside") for n in readouts]
              + ["outside_cos_previous", "outside_norm", "inside_norm"]}
    count3 = 0
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for first in range(0, m.num_scenes, m.batch_size):
            images = imagenet_normalize(scenes[first:first + m.batch_size].to(device).float() / 255)
            B = images.shape[0]
            policy = make_policy("coarse_to_fine", batch_size=B, device=device, num_glimpses=T, canvas_grid_size=G)
            state = canvit.init_state(batch_size=B, canvas_grid_size=G)
            previous = None
            for t in range(T):
                viewpoint = policy.step(t, state)
                if local is not None:
                    local.before_glimpse(t, viewpoint)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint,
                                              glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
                residuals.clear()
                state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                cells = state.canvas[:, num_registers:].float()
                sums["canvas_norm"][t] += cells.norm(dim=-1).mean(1).sum().item()
                for n, A in readouts.items():
                    sums[f"canvas_visibility_{n}"][t] += visibility(A, cells).mean(1).sum().item()
                if t < LEVEL3:
                    continue
                r = sum(residuals)  # [B, G*G, D]
                offsets = (cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs()
                inside = (offsets <= viewpoint.scales.view(-1, 1, 1, 1)).all(dim=-1).flatten(1)
                for where, mask in (("outside", ~inside), ("inside", inside)):
                    level3[f"{where}_norm"] += r.norm(dim=-1)[mask].mean().item() * B
                    for n, A in readouts.items():
                        level3[f"{where}_visibility_{n}"] += visibility(A, r)[mask].mean().item() * B
                        with torch.autocast(device_type=device.type, enabled=False):
                            share = (r @ bases[n]).square().sum(-1) / r.square().sum(-1).clamp(min=1e-12)
                        level3[f"{where}_share_{n}"] += share[mask].mean().item() * B
                if previous is not None:
                    both = ~inside & previous[1]
                    level3["outside_cos_previous"] += F.cosine_similarity(r, previous[0], dim=-1)[both].mean().item() * B
                previous = (r, ~inside)
                count3 += B
            log.info("%d/%d scenes", first + B, m.num_scenes)
    for h in handles:
        h.remove()
    n_prev = count3 - m.num_scenes  # no previous residual at the first level-3 glimpse
    result = {
        "measurement": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(m).items()},
        "pretrained_repo": pretrained_repo, "probe_repo": probe_repo,
        "level3": {**{k: v / (n_prev if k == "outside_cos_previous" else count3) for k, v in level3.items()},
                   **{f"random_share_{n}": U.shape[1] / U.shape[0] for n, U in bases.items()}},
        **{k: (v / m.num_scenes).tolist() for k, v in sums.items()},
    }
    m.output.parent.mkdir(parents=True, exist_ok=True)
    m.output.write_text(json.dumps(result, indent=1))
    log.info("level 3: %s", ", ".join(f"{k} {v:.3f}" for k, v in result["level3"].items()))
    for n in readouts:
        v = result[f"canvas_visibility_{n}"]
        log.info("canvas visibility to the %s at t = 0, 20, 40, 84: %s", n, " ".join(f"{v[t]:.3f}" for t in (0, 20, 40, 84)))


if __name__ == "__main__":
    main(tyro.cli(Measurement))
