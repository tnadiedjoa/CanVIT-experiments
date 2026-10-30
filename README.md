# Experiments with CanViT

Experiments with the released [CanViT](https://github.com/m2b3/CanViT) (Canvas Vision Transformer,
arXiv:2603.22570) active-vision model: reproducing its ADE20K results and testing it in settings the paper does not
cover. Everything here uses the published `canvit-pytorch` package and the released checkpoints from the Hugging
Face Hub; nothing is retrained.

## Results

ADE20K validation (2,000 images, 512 px scenes), with the released CanViT-B
(`canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`) and its released linear probes, all frozen.
mIoU is in %, and glimpses are counted from t = 0. Runs use bfloat16 autocast on one RTX 3090. Every condition is in
[`results/summary.md`](results/summary.md).

![C2F rollout on one scene](outputs/c2f_rollout_canvas64.png)

*C2F on one scene. Left: the viewpoints in visiting order. Right, for each glimpse: the glimpse, the canvas (PCA of
the layer-normalized patches, as in the paper's appendix G) and the canvas change since the previous step.*

### Reproduction

The rollout script gives the same mIoU after each of the 21 glimpses as the official CLI (RFS, 64×64 canvas). On the
64×64 canvas, it stays within 0.43 points of the paper's exported curves for all six policies. The largest gaps are
for the two seed-dependent policies, R-IID (0.43) and F2C (0.37). C2F, averaged over 5 seeds, is within 0.17 points
(0.19 on the 32×32 canvas).

The DINOv3 baselines with the released probes give 28.83 for ViT-B/16 at 128 px and 47.19 at 512 px. The paper
reports 28.8 and 47.2.

![Measured curves against the paper's](outputs/ade20k_reproduction.png)

### Accuracy against compute

![mIoU against forward GFLOPs](outputs/ade20k_cost_frontier.png)

On 512 px scenes, C2F stays on the frontier of the two DINOv3 models up to about 100 GFLOPs:

- One glimpse (20.8 GFLOPs, 39.6%) beats DINOv3 ViT-B at 192 px (26.3 GFLOPs, 35.9%).
- Five glimpses (104 GFLOPs, 44.7%) match ViT-B at 384 px (112 GFLOPs, 44.8%).

Past that point, C2F flattens: 21 glimpses (438 GFLOPs) reach 45.9%, below DINOv3 ViT-B at 512 px (47.2%,
215 GFLOPs). A glimpse costs the same whatever the scene size, so the comparison that matters is on scenes larger
than 512 px. That comparison is in progress, on Cityscapes at 1024 px.

### Long rollouts: repetition, not length, degrades the canvas

Pretraining rollouts average four glimpses. The table follows four policies over 85 glimpses, on a 64×64 canvas:

| Policy (seeds) | t = 1 | t = 20 | t = 84 |
|---|---|---|---|
| RFS (deterministic) | 41.06 | 39.32 | 37.55 |
| C2F (2) | 41.22 | 45.91 | 43.65 |
| F-IID (2) | 41.56 | 45.02 | 44.95 |
| R-IID (2) | 27.46 | 43.35 | 45.20 |

- **Random viewpoints do not drift.** F-IID peaks at 45.19 at t = 40, and R-IID is still rising at t = 84.
- **Repeating the full view does.** RFS falls from 41.12 at t = 2 to 37.55. Meanwhile the raw canvas norm grows
  linearly, because Writes add unnormalized residuals.
- **C2F peaks at the end of level 2 (t = 20).** It then loses 2.3 points over level 3, whose tiles are upsampled 2×
  from the 512 px scene.

Three controls continue the same 21 C2F glimpses differently (seed 0). Change in mIoU from t = 20 to t = 84:

- level 3: −2.28;
- level-2 tiles again: −1.24;
- the full scene again: −5.77 (down to 40.17).

![C2F continued past level 2](outputs/ade20k_level3_controls.png)

A probe-free metric orders the policies the same way. It is the paper's reconstruction metric (official CLI): how
well a 32×32 canvas predicts DINOv3's features of the scene, as a cosine similarity.

- RFS falls from 0.800 (t = 1) to 0.753.
- C2F falls from 0.837 (t = 20) to 0.803.
- F-IID and R-IID end near 0.82.

The canvas norm, the change per glimpse and the probe entropy for these rollouts are in
[`outputs/ade20k_long_rollouts.png`](outputs/ade20k_long_rollouts.png).

![Reconstruction of the teacher's features over 85 glimpses](outputs/ade20k_reconstruction.png)

### Most of C2F's gain does not need new pixels

This test removes the detail finer than 128 px from the scenes (antialiased downsampling, then bilinear upsampling),
so no later glimpse holds information the first one lacks. Gain of C2F from t = 0 to t = 20 (seed 0):

| Scene | Gain (points) |
|---|---|
| original 512 px | +6.35 |
| detail of 256 px | +5.58 |
| detail of 128 px | +4.12 |

Without any new detail, C2F keeps two thirds of its gain.

A 128×128 canvas read by the 64×64 probe adds 0.2 points at t = 20 (46.15 vs 45.94), for twice the FLOPs per
glimpse.

![C2F on scenes without fine detail](outputs/ade20k_scene_detail.png)

### Tile order within C2F's levels

Mean mIoU over glimpses 1-20:

| Order of the tiles in each level | 64×64 | 32×32 |
|---|---|---|
| lowest probe entropy first (EG-C2F reversed) | 44.50 | 43.00 |
| random, the paper's C2F (5 seeds, ± sd) | 44.92 ± 0.08 | 43.33 ± 0.04 |
| highest canvas contrast first (no probe) | 45.20 | 43.54 |
| highest probe entropy first, the paper's EG-C2F | 45.21 | 43.66 |
| most wrongly labeled pixels first (oracle: reads the labels) | 45.34 | 43.79 |

**Canvas contrast** needs no task head. A canvas cell's score is the mean cosine dissimilarity of its
layer-normalized patch to its four neighbours, and a tile's score is the mean over its cells.

- It matches EG-C2F on the 64×64 canvas and comes 0.12 points below it on the 32×32 canvas.
- The oracle bounds what any order gains over random at about 0.4-0.5 points.

![Tile orders](outputs/ade20k_tile_orders.png)

### Training-free changes that did not help

- **Anti-aliased glimpse extraction** (the crop at its native resolution, then an antialiased resize) lowers mIoU:
  - −0.48 at t = 0 on the 64×64 canvas;
  - −0.17 for C2F and −0.46 for RFS at t = 20.

  The model was pretrained on aliased bilinear glimpses. See
  [`outputs/ade20k_antialiasing.png`](outputs/ade20k_antialiasing.png).
- **Halving every Write residual after the first glimpse** slows RFS's decline (+0.30 at t = 84), but costs C2F up
  to 0.89 points (−0.50 at t = 20).
- **Damping Writes where the canvas was already seen at a finer scale.** Inside the glimpse, Writes are multiplied by
  min(1, finest scale seen / glimpse scale)^k.
  - It is an exact no-op for C2F.
  - It hurts F2C at t = 20: −0.94, −1.69 and −2.62 for k = 0.5, 1 and 2 (seed 0).
  - It halves the loss from repeating the full scene after C2F (43.10 vs 40.17 at t = 84), which again points at
    repetition rather than coarseness.

### Cost of the asymmetric Canvas Attention

The released model has no learned projections on the canvas side; the QKVO ablation has projections on both sides.

| Canvas | GFLOPs per glimpse, asymmetric / QKVO | Latency at batch size 32, asymmetric / QKVO |
|---|---|---|
| 64×64 | 20.8 / 124.3 | 103 / 158 ms |
| 128×128 | 42.3 / 455.0 | 361 / 580 ms |

FLOPs are analytic (`canvit_pytorch.flops`). Latency is measured on an RTX 3090 with bfloat16 autocast. At batch
size 1, both variants take 15-23 ms, a time bounded by kernel launches.

### Limits

- One dataset.
- Deterministic policies (RFS, EG-C2F and the scored tile orders) ran once; the others ran over the seeds shown.
  Variants of C2F use seed 0.
- Nothing is retrained, so the conclusions hold for the released model.

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
| `scripts/export_official.py` | Collects the official CLI's runs (DINOv3 baselines, reconstruction) into `results/official.json`, with the FLOPs of each model. |
| `exercises/` | Canvas Attention and SR-RoPE to write from the paper, with a checker against the released model. |

## Checks

- On RFS with a 64×64 canvas, `scripts/ade20k_rollouts.py` gives the same mIoU after each of the 21 glimpses as
  `python -m canvit_pytorch.evaluate ade20k-segmentation-canvit`.
- Its tile-ordered C2F with the probe's entropy as score chooses the same viewpoints as the official EG-C2F.

Results are in `results/` (one JSON file per run, with its configuration) and summarized in `results/summary.md`.
