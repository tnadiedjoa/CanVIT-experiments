"""ImageNet-1k top-1 accuracy after every glimpse, with the released Writes or the training-free Write restrictions of
scripts/local_writes.py (local) and scripts/bio_vision.py (object), as scripts/bio_rollouts.py does on ADE20K.

The classifier reads the recurrent CLS token, which sees the canvas only through the Reads: this asks whether the
canvas drift of long rollouts also reaches a global readout. Scenes as the paper evaluates them (short side resized
to 512 px, center crop); IMAGENET_VAL holds the validation images, one subdirectory per class. Every `stride`-th image
is used; each image's correctness at every glimpse is saved next to the output (.per_image.pt) for paired bootstraps.

    IMAGENET_VAL=/tmp/canvit-nadiedjoa-24/data/imagenet_val .venv/bin/python -P scripts/imagenet_rollouts.py \
        --num-glimpses 85 --stride 5 --local-writes-from 21 --output results/imagenet/c2f_c32_t85_local21.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from canvit_pytorch import sample_at_viewpoint
from canvit_pytorch.benchmarks import imagenet
from canvit_pytorch.hub import repos
from canvit_pytorch.model import CanViTForImageClassification
from canvit_pytorch.policies import PolicyName, make_policy

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from bio_vision import ObjectWrites  # noqa: E402
from local_writes import LocalWrites  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    output: Path
    classifier: Literal["finetuned", "finetuned-nnx", "frozen"] = "finetuned"
    """finetuned: the flagship fine-tuned end to end (repos.FINETUNED_IN1K); finetuned-nnx: the same recipe's JAX
    export (repos.FINETUNED_IN1K_NNX["in21k"]); frozen: the pretrained flagship with its fused DINOv3 CLS probe."""
    policy: PolicyName = "coarse_to_fine"
    num_glimpses: int = 21
    canvas_grid_size: int = repos.RELEASED_CANVAS_GRID_SIZE
    stride: int = 5
    """Every stride-th validation image (in ImageFolder's order), so stride 5 keeps 10 000 images, 10 per class."""
    local_writes_from: int | None = None
    object_writes_threshold: float | None = None
    object_writes_from: int = 21
    object_writes_late_threshold: float | None = None
    batch_size: int = 32
    num_workers: int = 3
    seed: int = 0
    device: str = "cuda"


def load(classifier: str) -> CanViTForImageClassification:
    if classifier == "finetuned":
        return CanViTForImageClassification.from_pretrained(repos.FINETUNED_IN1K)
    if classifier == "finetuned-nnx":
        return CanViTForImageClassification.from_pretrained(repos.FINETUNED_IN1K_NNX["in21k"])
    return CanViTForImageClassification.from_pretrained_with_probe(pretrained_repo=repos.FLAGSHIP,
                                                                   probe_repo=repos.DINOV3_VITB16_IN1K_PROBE)


@torch.inference_mode()
def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(exp.seed)
    device, T, G = torch.device(exp.device), exp.num_glimpses, exp.canvas_grid_size
    model = load(exp.classifier).to(device).eval()
    dataset = imagenet.validation_set(imagenet.validation_dir(), size_px=repos.RELEASED_SCENE_SIZE_PX)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), exp.stride))
    loader = torch.utils.data.DataLoader(subset, batch_size=exp.batch_size, num_workers=exp.num_workers,
                                         pin_memory=True)
    local = (LocalWrites(model.canvit, canvas_grid_size=G, device=device, start=exp.local_writes_from)
             if exp.local_writes_from is not None else None)
    objects = (ObjectWrites(model.canvit, canvas_grid_size=G, threshold=exp.object_writes_threshold,
                            start=exp.object_writes_from, num_glimpses=T, device=device,
                            late_threshold=exp.object_writes_late_threshold)
               if exp.object_writes_threshold is not None else None)
    correct = torch.zeros(len(subset), T, dtype=torch.bool)
    # Evidence accumulated over the glimpses, read at every glimpse: the mean of the class probabilities so far, and
    # the sum of the log-probabilities (independent evidence, as in sequential sampling models of decisions).
    correct_mean = torch.zeros_like(correct)
    correct_logsum = torch.zeros_like(correct)
    labels_all = torch.zeros(len(subset), dtype=torch.int16)
    start, done = time.monotonic(), 0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            B = images.shape[0]
            policy = make_policy(exp.policy, batch_size=B, device=device, num_glimpses=T, canvas_grid_size=G)
            state = model.init_state(batch_size=B, canvas_grid_size=G)
            prob_sum = torch.zeros(B, imagenet.NUM_CLASSES, device=device)
            logp_sum = torch.zeros_like(prob_sum)
            for t in range(T):
                viewpoint = policy.step(t, state)
                glimpse = sample_at_viewpoint(spatial=images, viewpoint=viewpoint,
                                              glimpse_size_px=repos.RELEASED_GLIMPSE_SIZE_PX)
                if local is not None:
                    local.before_glimpse(t, viewpoint)
                if objects is not None:
                    objects.before_glimpse(t, viewpoint, state.canvas)
                state = model.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                log_probs = model.classify(state.recurrent_cls).float().log_softmax(dim=-1)
                prob_sum += log_probs.exp()
                logp_sum += log_probs
                for target, scores in ((correct, log_probs), (correct_mean, prob_sum), (correct_logsum, logp_sum)):
                    target[done:done + B, t] = (scores.argmax(dim=-1) == labels).cpu()
            labels_all[done:done + B] = labels.cpu().to(torch.int16)
            done += B
            if done % (20 * exp.batch_size) < exp.batch_size:
                log.info("%d/%d images, top-1 at the last glimpse so far %.2f", done, len(subset),
                         100 * correct[:done, -1].float().mean().item())
    top1 = (100 * correct.float().mean(dim=0)).tolist()
    result = {"experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
              "top1": top1, "top1_mean_probs": (100 * correct_mean.float().mean(dim=0)).tolist(),
              "top1_sum_log_probs": (100 * correct_logsum.float().mean(dim=0)).tolist(),
              "n_images": len(subset), "wall_time_seconds": time.monotonic() - start}
    if objects is not None:
        result["object_writes_reached"] = (objects.reached / len(subset)).tolist()
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"correct": correct, "correct_mean_probs": correct_mean, "correct_sum_log_probs": correct_logsum,
                "labels": labels_all}, exp.output.with_suffix(".per_image.pt"))
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("Top-1 by glimpse: %s", " ".join(f"{v:.2f}" for v in top1))


if __name__ == "__main__":
    main(tyro.cli(Experiment))
