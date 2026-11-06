"""A small canvas model trained from scratch, to test whether a persistent memory written by softmax cross-attention
drifts on rollouts longer than its training ones, and whether a Write that can write nothing does not.

The model copies CanViT's design at a small scale. A 64×64 scene is seen through glimpses (y, x, s): the square
[x−s, x+s] × [y−s, y+s], resampled to 16×16 px and cut into 4×4 patches of 4 px. A 16×16 canvas of 128-d cells covers
the scene. Four transformer blocks process each glimpse's patch tokens plus three positionless prefix tokens (a
viewpoint embedding and two registers); after blocks 1 and 3 the glimpse Reads the canvas, after blocks 2 and 4 the
canvas Writes from the glimpse. Positions enter as a 2D RoPE in scene coordinates. A linear head decodes each cell into
its 4×4 pixels; the loss is the pixel error on the whole scene after every glimpse. Training uses rollouts of 4
glimpses (half F-IID, half R-IID), as CanViT's pretraining averages about four. The scale ratios match CanViT's:
C2F's level 2 is the native resolution and level 3 is upsampled 2×.

The Writes differ in how a cell may decline a glimpse:

- softmax: CanViT's Write; the attention weights sum to 1 over the glimpse's tokens, prefix included;
- out: softmax with an output projection, as in a standard cross-attention (Perceiver), which CanViT drops to save
  compute: the projection could map the prefix's values to zero;
- null: softmax with one more key per head, positionless and learned, whose value is zero (Miller's "softmax plus
  one" with a learned key), so a cell can attend to nothing;
- null_open: the same, plus a learned logit bias per head on the null key, set at log(19) (the number of glimpse
  tokens) so that at initialization half of each cell's attention goes to the null key: writing nothing is the default;
- sigmoid: independent sigmoid weights with a learned bias (Ramapuram et al., 2025), which can all be near 0;
- gate: softmax times a sigmoid gate per head from the cell's content (Gated Attention, Qiu et al., 2025);
- delta: softmax, but the cell moves toward the attention output instead of adding it: residual = g (o − cell), with
  a sigmoid gate g per head from the normed cell and o (the delta rule of DeltaNet, Schlag et al., 2021, and a
  GRU-like convex update), so writing the same output again changes the cell less and less.

With `--hidden-penalty λ` the loss also charges λ times the squared norm of the Writes' residuals in the directions the
pixel readout cannot see (defined below): the part of a write that the pixel loss alone leaves free. Since LayerNorm
makes the canvas's scale free, a model can meet that penalty by shrinking the whole canvas; `--hidden-relative`
divides it by each cell's squared norm. `--outside-penalty λ` charges instead, outside the glimpse's crop only, the
part of the Writes' residual that a LayerNorm would see (all but its mean over the channels), relative to the cell's;
with `--penalty-cells all` it charges every cell, a cost on writing at all that needs no knowledge of the crop.

With `--loss seen` the training loss covers only the cells seen so far (the union of the glimpses' crops), so the
model has nothing to gain from writing into the others; with the default `--loss scene` it is trained, as CanViT is,
to predict the whole scene after every glimpse, unseen parts included.

At evaluation, `--test-writes local` keeps every Write in the glimpse's crop from glimpse `--test-from` on, and
`object` also into the cells whose layer-normed content has cosine similarity ≥ `--object-threshold` with a crop cell.
`visible` and `invisible` keep, outside the crop, only the part of each residual that the pixel readout can see (its
projection on the readout's row space, defined below) or only the part it cannot see: which of the two parts makes the
drift.

`analyze` looks at a model's C2F rollouts. At level 3 (glimpses 21-84), for the residual a cell receives from a glimpse
(both Writes), inside and outside the crop: the share of its energy that the readout can see (the readout decodes
head(LN(cell)), so up to LN's scale a residual r moves it by A·r with A = head.weight · diag(LN gain) · centering; the
share is that of r in A's row space, 48 of 128 dimensions, so 0.375 for a random direction), and, outside, the cosine
with the same cell's residual from the previous glimpse and with its mean over level 3; the same cosine for the
visible and hidden parts separately; and, per part, how much the outside residuals add up over level 3: the norm of
their sum over the sum of their norms (1 when every glimpse adds the same direction, about 1/8 for 64 random ones);
and the share of the outside residual's energy along the constant direction (1, ..., 1), the one direction that every
LayerNorm reading the canvas removes exactly (1/128 for a random direction). After every glimpse, the mean norm of
the cells after removing their mean (what LayerNorm divides by) is also reported.
After every glimpse: the regression slope of the predicted on the true pixels (after removing each image's mean; 1 is
the right contrast), their correlation, and the predictions' standard deviation.

    .venv/bin/python -P scripts/sink_memory.py train --write softmax --output probes/sink_memory/softmax_s0.pt
    .venv/bin/python -P scripts/sink_memory.py evaluate --checkpoint probes/sink_memory/softmax_s0.pt \
        --policy coarse_to_fine --output results/sink_memory/softmax_s0_c2f.json
    .venv/bin/python -P scripts/sink_memory.py analyze --checkpoint probes/sink_memory/softmax_s0.pt \
        --output results/sink_memory/softmax_s0_analysis.json
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import Tensor

log = logging.getLogger("sink_memory")
ADE = Path("/tmp/canvit-nadiedjoa-24/data/ADEChallengeData2016/images")
CACHE = Path("/tmp/canvit-nadiedjoa-24/data/sink_memory")
SCENE_PX, GLIMPSE_PX, PATCH_PX, GRID = 64, 16, 4, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WriteKind = Literal["softmax", "out", "null", "null_open", "sigmoid", "gate", "delta"]


def scenes(split: Literal["training", "validation"]) -> Tensor:
    """uint8 [N, 3, 64, 64]: the ADE20K images, resized (shorter side) and center-cropped; cached."""
    path = CACHE / f"{split}_{SCENE_PX}.pt"
    if path.exists():
        return torch.load(path)
    files = sorted((ADE / split).glob("*.jpg"))
    out = torch.empty(len(files), 3, SCENE_PX, SCENE_PX, dtype=torch.uint8)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        k = SCENE_PX / min(w, h)
        im = im.resize((max(SCENE_PX, round(w * k)), max(SCENE_PX, round(h * k))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - SCENE_PX) // 2, (h - SCENE_PX) // 2
        im = im.crop((left, top, left + SCENE_PX, top + SCENE_PX))
        out[i] = torch.from_numpy(np.array(im)).permute(2, 0, 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def grid_centers(n: int, device) -> Tensor:
    """[n] centers of n equal intervals of [-1, 1]."""
    return (torch.arange(n, device=device) * 2 + 1) / n - 1


def cell_positions(device) -> Tensor:
    """[G*G, 2] (y, x) centers of the canvas cells."""
    c = grid_centers(GRID, device)
    return torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).view(-1, 2)


def glimpse(images: Tensor, vp: Tensor) -> Tensor:
    """images [B, 3, H, W], viewpoints [B, 3] (y, x, s) -> [B, 3, 16, 16], bilinear."""
    u = grid_centers(GLIMPSE_PX, images.device)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    gy = vp[:, 0, None, None] + vp[:, 2, None, None] * yy
    gx = vp[:, 1, None, None] + vp[:, 2, None, None] * xx
    return F.grid_sample(images, torch.stack([gx, gy], -1), mode="bilinear", align_corners=False)


def patch_positions(vp: Tensor) -> Tensor:
    """[B, 16, 2] scene positions (y, x) of a glimpse's 4×4 patch centers."""
    n = GLIMPSE_PX // PATCH_PX
    u = grid_centers(n, vp.device)
    off = torch.stack(torch.meshgrid(u, u, indexing="ij"), -1).view(-1, 2)
    return vp[:, None, :2] + vp[:, None, 2:3] * off


class Rope:
    """2D rotary embedding in scene coordinates: half of each head's dimensions rotate with y, half with x, at
    frequencies π·2^k (wavelengths 2 down to 2/2^(n-1) scene units). Positionless tokens are not rotated."""

    def __init__(self, head_dim: int, device):
        n = head_dim // 4
        self.freqs = math.pi * 2.0 ** torch.arange(n, device=device, dtype=torch.float32)

    def __call__(self, x: Tensor, pos: Tensor | None) -> Tensor:
        """x [B, h, N, d]; pos [B, N, 2] or None (rotates nothing). Rotates the last pos.shape[1] tokens of x."""
        if pos is None:
            return x
        n = pos.shape[1]
        head, tail = x[..., : x.shape[2] - n, :], x[..., x.shape[2] - n :, :]
        ang = (pos[:, None, :, :, None] * self.freqs).flatten(-2)  # [B, 1, n, 2 * len(freqs)]
        cos, sin = ang.cos(), ang.sin()
        t = tail.float()
        a, b = t[..., ::2], t[..., 1::2]
        rot = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(x.dtype)
        return torch.cat([head, rot], dim=2)


class Attention(nn.Module):
    """Multi-head attention from `queries` to `keys`, with RoPE on the positional tokens (the last ones)."""

    def __init__(self, dim: int, heads: int, kv_dim: int | None = None, out: bool = True):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim)
        self.o = nn.Linear(dim, dim) if out else nn.Identity()

    def split(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], self.heads, self.dh).transpose(1, 2)

    def qkv(self, queries, keys, rope, q_pos, k_pos):
        q = rope(self.split(self.q(queries)), q_pos)
        k, v = self.kv(keys).chunk(2, dim=-1)
        return q, rope(self.split(k), k_pos), self.split(v)

    def forward(self, queries, keys, rope, q_pos, k_pos):
        q, k, v = self.qkv(queries, keys, rope, q_pos, k_pos)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).flatten(2))


class Write(Attention):
    """The canvas cells (queries) attend to the glimpse's tokens (prefix first, then patches); no output projection,
    as in CanViT (except the `out` kind). Returns the residual [B, N, D] and, per cell, the attention mass on the
    prefix tokens."""

    def __init__(self, dim: int, heads: int, kind: WriteKind, n_prefix: int):
        super().__init__(dim, heads, out=kind == "out")
        self.kind, self.n_prefix = kind, n_prefix
        if kind in ("null", "null_open"):
            self.null_key = nn.Parameter(torch.randn(heads, 1, self.dh) * self.dh**-0.5)
        if kind == "null_open":
            self.null_bias = nn.Parameter(torch.full((heads, 1, 1), math.log(16 + n_prefix)))
        if kind == "sigmoid":
            self.bias = nn.Parameter(torch.full((heads, 1, 1), -math.log(16 + n_prefix)))
        if kind == "gate":
            self.gate = nn.Linear(dim, heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 3.0)
        if kind == "delta":
            self.gate = nn.Linear(2 * dim, heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, cells, tokens, rope, cell_pos, patch_pos, raw=None):
        """cells: the normed canvas (queries); raw: the canvas itself, which the delta Write moves toward its output."""
        q, k, v = self.qkv(cells, tokens, rope, cell_pos, patch_pos)
        logits = (q.float() @ k.float().transpose(-1, -2)) * self.dh**-0.5  # [B, h, N, n_prefix + P]
        if self.kind == "sigmoid":
            w = torch.sigmoid(logits + self.bias)
        elif self.kind in ("null", "null_open"):
            null = (q.float() @ self.null_key.float().transpose(-1, -2)) * self.dh**-0.5  # [B, h, N, 1]
            if self.kind == "null_open":
                null = null + self.null_bias
            w = torch.cat([logits, null], -1).softmax(-1)[..., :-1]
        else:
            w = logits.softmax(-1)
        out = w.to(v.dtype) @ v  # [B, h, N, dh]
        if self.kind == "gate":
            out = out * torch.sigmoid(self.gate(cells)).transpose(1, 2).unsqueeze(-1).to(out.dtype)
        if self.kind == "delta":
            o = out.transpose(1, 2).flatten(2)  # [B, N, D]
            g = torch.sigmoid(self.gate(torch.cat([cells, o.to(cells.dtype)], -1))).transpose(1, 2).unsqueeze(-1)
            out = g.to(out.dtype) * (out - self.split(raw).to(out.dtype))
        prefix_mass = w[..., : self.n_prefix].sum(-1).mean(1)  # [B, N]
        return self.o(out.transpose(1, 2).flatten(2)), prefix_mass


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, rope, pos):
        h = self.n1(x)
        x = x + self.attn(h, h, rope, pos, pos)
        return x + self.mlp(self.n2(x))


@dataclass
class ModelConfig:
    write: WriteKind = "softmax"
    dim: int = 128
    heads: int = 4
    registers: int = 2


class CanvasModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.embed = nn.Linear(3 * PATCH_PX * PATCH_PX, d)
        self.vpe = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.registers = nn.Parameter(torch.randn(cfg.registers, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(4))
        self.read_norm_q, self.read_norm_kv = nn.ModuleList(nn.LayerNorm(d) for _ in range(2)), nn.ModuleList(
            nn.LayerNorm(d) for _ in range(2))
        self.reads = nn.ModuleList(Attention(d, cfg.heads) for _ in range(2))
        self.write_norm_q, self.write_norm_kv = nn.ModuleList(nn.LayerNorm(d) for _ in range(2)), nn.ModuleList(
            nn.LayerNorm(d) for _ in range(2))
        self.writes = nn.ModuleList(Write(d, cfg.heads, cfg.write, 1 + cfg.registers) for _ in range(2))
        self.canvas_init = nn.Parameter(torch.zeros(1, 1, d))
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3 * PATCH_PX * PATCH_PX)
        self.n_prefix = 1 + cfg.registers

    def init_canvas(self, batch: int) -> Tensor:
        return self.canvas_init.expand(batch, GRID * GRID, -1).clone()

    def decode(self, canvas: Tensor) -> Tensor:
        """[B, G*G, D] -> normalized pixels [B, 3, 64, 64]."""
        B = canvas.shape[0]
        p = self.head(self.head_norm(canvas)).view(B, GRID, GRID, 3, PATCH_PX, PATCH_PX)
        return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, SCENE_PX, SCENE_PX)

    def step(self, canvas: Tensor, images: Tensor, vp: Tensor, rope: Rope, cells: Tensor,
             write_mask=None, outside_projection=None) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """One glimpse: returns the new canvas, the Writes' residuals and their prefix masses. write_mask [B, N] (True =
        keep) drops the other cells' residuals (test-time rules), or with outside_projection [D, D] projects them."""
        B = images.shape[0]
        g = glimpse(images, vp)
        patches = g.unfold(2, PATCH_PX, PATCH_PX).unfold(3, PATCH_PX, PATCH_PX)  # [B, 3, 4, 4, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * PATCH_PX * PATCH_PX)
        pos = patch_positions(vp)
        prefix = torch.cat([self.vpe(vp)[:, None], self.registers.expand(B, -1, -1)], 1)
        x = torch.cat([prefix, self.embed(patches)], 1)
        cell_pos = cells.expand(B, -1, -1)
        residuals, masses = [], []
        for i, block in enumerate(self.blocks):
            x = block(x, rope, pos)
            j = i // 2
            if i % 2 == 0:  # Read after blocks 1 and 3
                x = x + self.reads[j](self.read_norm_q[j](x), self.read_norm_kv[j](canvas), rope, pos, cell_pos)
            else:  # Write after blocks 2 and 4
                r, m = self.writes[j](self.write_norm_q[j](canvas), self.write_norm_kv[j](x), rope, cell_pos, pos,
                                      raw=canvas)
                if write_mask is not None:
                    keep = write_mask[..., None]
                    r = (torch.where(keep, r, r @ outside_projection.to(r.dtype)) if outside_projection is not None
                         else r * keep.to(r.dtype))
                canvas = canvas + r
                residuals.append(r)
                masses.append(m)
        return canvas, residuals, masses


def normalize(u8: Tensor) -> Tensor:
    return (u8.float() / 255 - MEAN.to(u8.device)) / STD.to(u8.device)


def random_viewpoints(batch: int, device, full_first: bool, t: int) -> Tensor:
    """R-IID (CanViT's appendix D.1): A ~ U[0, (1 - 0.05)^2], s = 1 - sqrt(A), center uniform in the valid box;
    with full_first, glimpse 0 is the full scene (F-IID)."""
    a = torch.rand(batch, device=device) * (1 - 0.05) ** 2
    s = 1 - a.sqrt()
    c = (torch.rand(batch, 2, device=device) * 2 - 1) * (1 - s[:, None])
    vp = torch.cat([c, s[:, None]], 1)
    if full_first and t == 0:
        vp = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch, 3).clone()
    return vp


def c2f_viewpoints(batch: int, device, generator: torch.Generator) -> list[Tensor]:
    """C2F: levels 0-3 (1 + 4 + 16 + 64 = 85 glimpses), tiles in a random order within each level, per scene."""
    out = []
    for level in range(4):
        n, s = 2**level, 2.0**-level
        c = grid_centers(n, device)
        tiles = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).view(-1, 2)
        order = torch.stack([torch.randperm(n * n, generator=generator) for _ in range(batch)]).to(device)
        for k in range(n * n):
            out.append(torch.cat([tiles[order[:, k]], torch.full((batch, 1), s, device=device)], 1))
    return out


@dataclass
class Train:
    output: Path
    write: WriteKind = "softmax"
    registers: int = 2
    steps: int = 20000
    batch_size: int = 64
    glimpses: int = 4
    loss: Literal["scene", "seen"] = "scene"
    hidden_penalty: float = 0.0
    """When > 0, adds this times the mean squared norm of the part of the Writes' residuals that the readout cannot see
    (orthogonal to visible_basis), over every cell and both Writes: a cost on what the pixel loss is blind to."""
    outside_penalty: float = 0.0
    """When > 0, adds this times, over the cells outside the glimpse's crop and both Writes, the squared norm of the
    residual after removing its mean over the channels, divided by the cell's own (centered, not differentiated):
    everything a LayerNorm reading the canvas would see of the forced write. Writing along the constant direction,
    which every LayerNorm removes, stays free."""
    penalty_cells: Literal["outside", "all"] = "outside"
    """Where `--outside-penalty` charges: outside the crop, or in every cell (a cost on writing at all, which needs no
    knowledge of where the glimpse is)."""
    hidden_relative: bool = False
    """Divides that squared norm, per cell, by the cell's squared norm after the glimpse (not differentiated), so the
    penalty cannot be met by shrinking the whole canvas, which LayerNorm makes free."""
    lr: float = 1e-3
    warmup: int = 500
    seed: int = 0


@dataclass
class Evaluate:
    checkpoint: Path
    output: Path
    policy: Literal["coarse_to_fine", "random", "full_then_random", "repeated_full_scene", "fine_to_coarse"] = (
        "coarse_to_fine")
    num_glimpses: int = 85
    test_writes: Literal["released", "local", "object", "visible", "invisible"] = "released"
    test_from: int = 21
    object_threshold: float = 0.9
    num_images: int = 2000
    batch_size: int = 100
    seed: int = 0


def train(cfg: Train) -> None:
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    data = scenes("training").to(dev)
    model = CanvasModel(ModelConfig(write=cfg.write, registers=cfg.registers)).to(dev)
    rope, cells = Rope(model.cfg.dim // model.cfg.heads, dev), cell_positions(dev)[None]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / cfg.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(i, cfg.steps) / cfg.steps)))
    start, history = time.time(), []
    for step in range(cfg.steps):
        idx = torch.randint(len(data), (cfg.batch_size,), device=dev)
        images = data[idx]
        images = torch.where(torch.rand(cfg.batch_size, 1, 1, 1, device=dev) < 0.5, images, images.flip(-1))
        x = normalize(images)
        full_first = bool(torch.rand(()) < 0.5)
        canvas, loss, hidden = model.init_canvas(cfg.batch_size), 0.0, 0.0
        seen = torch.zeros(cfg.batch_size, GRID * GRID, dtype=torch.bool, device=dev)
        outside_cost = 0.0
        if cfg.hidden_penalty > 0:
            with torch.no_grad():
                U = visible_basis(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(cfg.glimpses):
                vp = random_viewpoints(cfg.batch_size, dev, full_first, t)
                canvas, residuals, _ = model.step(canvas, x, vp, rope, cells)
                if cfg.hidden_penalty > 0:
                    scale = (canvas.detach().float().square().sum(-1).clamp(min=1e-6) if cfg.hidden_relative
                             else torch.ones((), device=dev))
                    for r in residuals:
                        r = r.float()
                        hidden = hidden + ((r - (r @ U) @ U.T).square().sum(-1) / scale).mean() / len(residuals)
                if cfg.outside_penalty > 0:
                    outside = (cells - vp[:, None, :2]).abs().amax(-1) > vp[:, None, 2]  # [B, N]
                    if cfg.penalty_cells == "all":
                        outside = torch.ones_like(outside)
                    c = canvas.detach().float()
                    own = (c - c.mean(-1, keepdim=True)).square().sum(-1).clamp(min=1e-6)
                    for r in residuals:
                        r = r.float()
                        ratio = (r - r.mean(-1, keepdim=True)).square().sum(-1) / own
                        outside_cost = outside_cost + (ratio * outside).sum() / outside.sum().clamp(min=1) / len(residuals)
                if cfg.loss == "scene":
                    loss = loss + F.mse_loss(model.decode(canvas).float(), x)
                else:
                    seen |= (cells - vp[:, None, :2]).abs().amax(-1) <= vp[:, None, 2]
                    m = seen.view(-1, 1, GRID, GRID).repeat_interleave(PATCH_PX, 2).repeat_interleave(PATCH_PX, 3)
                    err = (model.decode(canvas).float() - x).square() * m
                    loss = loss + err.sum() / (3 * m.sum()).clamp(min=1)
        loss, hidden, outside_cost = loss / cfg.glimpses, hidden / cfg.glimpses, outside_cost / cfg.glimpses
        opt.zero_grad(set_to_none=True)
        (loss + cfg.hidden_penalty * hidden + cfg.outside_penalty * outside_cost).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == cfg.steps - 1:
            history.append({"step": step, "loss": loss.item(), "hidden": float(hidden), "outside": float(outside_cost)})
            log.info("step %d loss %.4f hidden %.3f outside %.3f %.0f s", step, loss.item(), float(hidden),
                     float(outside_cost), time.time() - start)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg) | {"output": str(cfg.output)}, "model": asdict(model.cfg),
                "state_dict": model.state_dict(), "history": history}, cfg.output)
    log.info("saved %s", cfg.output)


def visible_basis(model: CanvasModel) -> Tensor:
    """[D, rank] orthonormal basis of the directions the pixel readout sees: the row space of head.weight times
    head_norm's gain, after LayerNorm's centering. A residual orthogonal to it leaves the decoded pixels unchanged up
    to LayerNorm's rescaling."""
    D = model.cfg.dim
    A = (model.head.weight * model.head_norm.weight[None]).float() @ (torch.eye(D, device=model.head.weight.device) - 1 / D)
    U, S, _ = torch.linalg.svd(A.T, full_matrices=False)
    return U[:, S > S[0] * 1e-4]


@torch.inference_mode()
def evaluate(cfg: Evaluate) -> None:
    dev = torch.device("cuda")
    saved = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
    model = CanvasModel(ModelConfig(**saved["model"])).to(dev).eval()
    model.load_state_dict(saved["state_dict"])
    rope, cells = Rope(model.cfg.dim // model.cfg.heads, dev), cell_positions(dev)[None]
    projection = None
    if cfg.test_writes in ("visible", "invisible"):
        U = visible_basis(model)
        projection = U @ U.T if cfg.test_writes == "visible" else torch.eye(model.cfg.dim, device=dev) - U @ U.T
    data = scenes("validation")[: cfg.num_images]
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    T = 85 if cfg.policy in ("coarse_to_fine", "fine_to_coarse") else cfg.num_glimpses
    sums = {k: torch.zeros(T, dtype=torch.float64) for k in ("mse", "residual_inside", "residual_outside",
                                                            "prefix_inside", "prefix_outside", "canvas_norm",
                                                            "reached")}
    counts = torch.zeros(T, dtype=torch.float64)
    for first in range(0, len(data), cfg.batch_size):
        x = normalize(data[first:first + cfg.batch_size].to(dev))
        B = x.shape[0]
        vps = (c2f_viewpoints(B, dev, gen) if cfg.policy in ("coarse_to_fine", "fine_to_coarse") else None)
        if cfg.policy == "fine_to_coarse":
            vps = vps[::-1]
        canvas = model.init_canvas(B)
        for t in range(T):
            if vps is not None:
                vp = vps[t]
            elif cfg.policy == "repeated_full_scene":
                vp = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(B, 3)
            else:
                vp = random_viewpoints(B, dev, cfg.policy == "full_then_random", t)
            inside = ((cells - vp[:, None, :2]).abs().amax(-1) <= vp[:, None, 2])  # [B, N]
            mask = None
            if cfg.test_writes != "released" and t >= cfg.test_from:
                mask = inside.clone()
                if cfg.test_writes == "object":
                    c = F.normalize(F.layer_norm(canvas.float(), (canvas.shape[-1],)), dim=-1)
                    sim = c @ c.transpose(1, 2)  # [B, N, N]
                    sim = sim.masked_fill(~inside[:, None, :], -1).amax(-1)
                    mask |= sim >= cfg.object_threshold
            with torch.autocast("cuda", dtype=torch.bfloat16):
                canvas, residuals, masses = model.step(canvas, x, vp, rope, cells, write_mask=mask,
                                                       outside_projection=projection)
                pred = model.decode(canvas).float()
            sums["mse"][t] += F.mse_loss(pred, x, reduction="none").mean((1, 2, 3)).sum().item()
            r = sum(res.float().norm(dim=-1) for res in residuals)  # [B, N]
            m = sum(masses) / len(masses)
            out = ~inside
            n_in, n_out = inside.sum(1).clamp(min=1), out.sum(1).clamp(min=1)
            sums["residual_inside"][t] += ((r * inside).sum(1) / n_in).sum().item()
            sums["residual_outside"][t] += ((r * out).sum(1) / n_out).sum().item()
            sums["prefix_inside"][t] += ((m * inside).sum(1) / n_in).sum().item()
            sums["prefix_outside"][t] += ((m * out).sum(1) / n_out).sum().item()
            sums["canvas_norm"][t] += canvas.float().norm(dim=-1).mean(1).sum().item()
            reached = (mask & out).sum(1) / n_out if mask is not None else (out.sum(1) > 0).float()
            sums["reached"][t] += reached.sum().item()
            counts[t] += B
    result = {k: (v / counts).tolist() for k, v in sums.items()}
    result["psnr"] = [10 * math.log10(1 / m) for m in result["mse"]]  # on normalized pixels: a relative scale
    result["config"] = {k: str(v) for k, v in asdict(cfg).items()}
    result["train_config"] = saved["config"]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=1))
    mse = result["mse"]
    log.info("mse at t = 0, 4, 20, 84: %s", " ".join(f"{mse[t]:.4f}" for t in (0, 4, 20, min(84, T - 1))))
    print("mse by glimpse:", " ".join(f"{mse[t]:.4f}" for t in range(0, T, max(1, T // 12))))


@dataclass
class Analyze:
    checkpoint: Path
    output: Path
    num_images: int = 200
    seed: int = 0


@torch.inference_mode()
def analyze(cfg: Analyze) -> None:
    dev = torch.device("cuda")
    saved = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
    model = CanvasModel(ModelConfig(**saved["model"])).to(dev).eval()
    model.load_state_dict(saved["state_dict"])
    rope, cells = Rope(model.cfg.dim // model.cfg.heads, dev), cell_positions(dev)[None]
    D = model.cfg.dim
    U = visible_basis(model)
    x = normalize(scenes("validation")[: cfg.num_images].to(dev))
    B = x.shape[0]
    vps = c2f_viewpoints(B, dev, torch.Generator().manual_seed(cfg.seed))
    canvas = model.init_canvas(B)
    P = U @ U.T
    level3 = {k: [] for k in ("visible_share_outside", "visible_share_inside", "cos_previous_outside",
                              "cos_previous_outside_visible", "cos_previous_outside_hidden", "mean_share_outside")}
    curves = {k: [] for k in ("mse", "slope", "correlation", "prediction_std", "centered_norm")}
    outside_residuals, previous = [], None
    for t, vp in enumerate(vps):
        inside = (cells - vp[:, None, :2]).abs().amax(-1) <= vp[:, None, 2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canvas, residuals, _ = model.step(canvas, x, vp, rope, cells)
            pred = model.decode(canvas).float()
        pc, xc = pred - pred.mean((2, 3), keepdim=True), x - x.mean((2, 3), keepdim=True)
        curves["mse"].append((pred - x).square().mean().item())
        curves["slope"].append(((pc * xc).sum() / xc.square().sum()).item())
        curves["correlation"].append(((pc * xc).sum() / (pc.square().sum() * xc.square().sum()).sqrt()).item())
        curves["prediction_std"].append(pc.square().mean().sqrt().item())
        c = canvas.float()
        curves["centered_norm"].append((c - c.mean(-1, keepdim=True)).norm(dim=-1).mean().item())
        if t < 21:
            continue
        r = sum(res.float() for res in residuals)  # [B, N, D]
        share = (r @ U).square().sum(-1) / r.square().sum(-1).clamp(min=1e-12)
        level3["visible_share_outside"].append(share[~inside].mean().item())
        level3["visible_share_inside"].append(share[inside].mean().item())
        mean_share = r.mean(-1).square() * D / r.square().sum(-1).clamp(min=1e-12)
        level3["mean_share_outside"].append(mean_share[~inside].mean().item())
        if previous is not None:
            both = ~inside & previous[1]
            level3["cos_previous_outside"].append(F.cosine_similarity(r, previous[0], dim=-1)[both].mean().item())
            for part, project in (("visible", lambda v: v @ P), ("hidden", lambda v: v - v @ P)):
                cos = F.cosine_similarity(project(r), project(previous[0]), dim=-1)[both].mean().item()
                level3[f"cos_previous_outside_{part}"].append(cos)
        previous = (r, ~inside)
        outside_residuals.append((r.cpu(), (~inside).cpu()))
    total = sum(r * o[..., None] for r, o in outside_residuals)
    mean = total / sum(o[..., None].float() for _, o in outside_residuals).clamp(min=1)
    cos_mean = [F.cosine_similarity(r, mean, dim=-1)[o].mean().item() for r, o in outside_residuals]
    accumulation = {}  # per cell, norm of the summed outside residuals over the sum of their norms (1: all aligned)
    for part, project in (("visible", lambda v: v @ P.cpu()), ("hidden", lambda v: v - v @ P.cpu())):
        summed = sum(project(r) * o[..., None] for r, o in outside_residuals)
        norms = sum(project(r).norm(dim=-1) * o for r, o in outside_residuals)
        accumulation[f"accumulation_outside_{part}"] = (summed.norm(dim=-1) / norms.clamp(min=1e-12)).mean().item()
    result = {"level3": {k: sum(v) / len(v) for k, v in level3.items()}
              | {"cos_mean_outside": sum(cos_mean) / len(cos_mean), "visible_share_random": U.shape[1] / D}
              | accumulation,
              **curves, "config": {k: str(v) for k, v in asdict(cfg).items()}, "train_config": saved["config"]}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=1))
    print("level 3:", " ".join(f"{k} {v:.3f}" for k, v in result["level3"].items()))
    print("slope by glimpse (t = 4, 20, 40, 84):", " ".join(f"{curves['slope'][t]:.2f}" for t in (4, 20, 40, 84)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd = tyro.extras.subcommand_cli_from_dict({"train": Train, "evaluate": Evaluate, "analyze": Analyze})
    {Train: train, Evaluate: evaluate, Analyze: analyze}[type(cmd)](cmd)
