"""Paired bootstrap of the difference between two runs on the same images: ADE20K mIoU (from the per-image
intersections and unions that scripts/bio_rollouts.py saves with --per-image-at) or ImageNet top-1 (from the
per-image correctness of scripts/imagenet_rollouts.py). Each resample draws the images with replacement, the same
draw for both runs, and recomputes the dataset-level metric of each, so the interval accounts for which images the
benchmark happens to hold.

    .venv/bin/python -P scripts/paired_bootstrap.py results/a.json results/b.json --glimpse 84
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass(frozen=True)
class Comparison:
    baseline: tyro.conf.Positional[Path]
    other: tyro.conf.Positional[Path]
    glimpse: int = 84
    """0-based glimpse (for ADE20K, one of the runs' --per-image-at glimpses)."""
    resamples: int = 4000
    seed: int = 0


def miou(intersection: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    """[..., C] summed counts -> [...] mean IoU over the classes present, as ade20k.MeanIoU computes it (in %)."""
    present = union > 0
    iou = torch.where(present, intersection / union.clamp(min=1e-8), torch.zeros_like(intersection))
    return 100 * iou.sum(-1) / present.sum(-1)


def load(path: Path, glimpse: int):
    saved = torch.load(path.with_suffix(".per_image.pt"), weights_only=True)
    if "correct" in saved:
        return "top1", saved["correct"][:, glimpse].double()
    counts = saved[glimpse]
    return "miou", (counts["intersection"].double(), counts["union"].double())


def main(c: Comparison) -> dict:
    kind, a = load(c.baseline, c.glimpse)
    kind_b, b = load(c.other, c.glimpse)
    assert kind == kind_b, (kind, kind_b)
    n = a.shape[0] if kind == "top1" else a[0].shape[0]
    generator = torch.Generator().manual_seed(c.seed)
    draws = torch.randint(n, (c.resamples, n), generator=generator)
    weights = torch.zeros(c.resamples, n, dtype=torch.float64).scatter_add_(
        1, draws, torch.ones(c.resamples, n, dtype=torch.float64))  # [R, N]: how often each image is drawn
    if kind == "top1":
        point = [100 * x.mean().item() for x in (a, b)]
        boot = [100 * (weights @ x) / n for x in (a, b)]
    else:
        point = [miou(i.sum(0), u.sum(0)).item() for i, u in (a, b)]
        boot = [miou(weights @ i, weights @ u) for i, u in (a, b)]
    delta = boot[1] - boot[0]
    low, high = torch.quantile(delta, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    result = {"metric": kind, "glimpse": c.glimpse, "n_images": n, "baseline": point[0], "other": point[1],
              "delta": point[1] - point[0], "ci95": [low, high], "p_delta_le_0": (delta <= 0).double().mean().item()}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main(tyro.cli(Comparison))
