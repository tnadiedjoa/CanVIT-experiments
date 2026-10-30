# Experiments with CanViT

Experiments with the released [CanViT](https://github.com/m2b3/CanViT) (Canvas Vision Transformer,
arXiv:2603.22570) active-vision model: reproducing its ADE20K results and testing it in settings the paper does not
cover. Everything here uses the published `canvit-pytorch` package and the released checkpoints from the Hugging
Face Hub; nothing is retrained.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install "canvit-pytorch[demo,evaluate]==0.3.0"
git clone https://github.com/m2b3/CanViT external/CanViT   # the paper's exported results, sample images
```

ADE20K (`ADEChallengeData2016`) goes wherever `ADE20K_ROOT` points, Cityscapes (`gtFine_trainvaltest.zip` and
`leftImg8bit_trainvaltest.zip`, unzipped) wherever `CITYSCAPES_ROOT` points; the queue runners set the defaults.
DINOv3 is gated on the Hugging Face Hub: request access, then `hf auth login`.

## Contents

| Path | What it does |
| --- | --- |
| `scripts/c2f_rollout.py` | One scene under C2F: numbered viewpoints, the canvas after each glimpse (the paper's PCA protocol) and the per-glimpse canvas change. |
| `scripts/ade20k_rollouts.py` | ADE20K validation mIoU after every glimpse, with canvas statistics, for the paper's policies and for variants: other tile orders within C2F's levels, C2F continued past level 2 by revisits, anti-aliased glimpses, scenes without fine detail, a probe read on a larger canvas, scaled Write residuals, Writes damped where the canvas was already seen at a finer scale. Uses the official model, probe, transform and metric. |
| `scripts/cityscapes.py` | Cityscapes at its native 1024 px: linear probes trained on frozen features (CanViT's canvas, DINOv3's patch features), then validation mIoU of C2F rollouts down to glimpses finer than the 512 px scenes of pretraining, and of DINOv3 at several input sizes. |
| `scripts/run_queue.sh`, `scripts/run_official_queue.sh`, `scripts/run_commands.sh`, `queues/` | Run a list of experiments one after another, skipping finished ones: `ade20k_rollouts.py` runs, official `canvit_pytorch.evaluate` runs, and any Python command. |
| `scripts/plot_results.py` | Figures in `outputs/` and the table `results/summary.md` from the saved results. |
| `scripts/canvas_attention_cost.py` | FLOPs and GPU latency of a glimpse with asymmetric and QKVO Canvas Attention. |
| `exercises/` | Canvas Attention and SR-RoPE to write from the paper, with a checker against the released model. |

## Checks

- On RFS with a 64×64 canvas, `scripts/ade20k_rollouts.py` gives the same mIoU after each of the 21 glimpses as
  `python -m canvit_pytorch.evaluate ade20k-segmentation-canvit`.
- Its tile-ordered C2F with the probe's entropy as score chooses the same viewpoints as the official EG-C2F.

Results are in `results/` (one JSON file per run, with its configuration) and summarized in `results/summary.md`.
