"""Training-free restrictions of CanViT's Canvas Attention Writes to the glimpse's crop, shared by the rollout scripts.

A Write lets every canvas cell attend to the glimpse's tokens, so a glimpse also changes the cells it did not see
(scripts/write_leakage.py measures how much). LocalWrites hooks the Writes and changes the residual of the cells
outside the crop; write_parts splits a Write's residual into the part from the glimpse's prefix tokens (VPE, recurrent
CLS, registers), whose keys carry no position, and the part from its patch tokens. DroppedWriteToken removes, in every
cell, the part that comes from one key token: canvas cells far from the glimpse attend almost only to backbone
register 3, which works as an attention sink whose value is not zero.
"""

import math
from typing import Literal

import torch
from torch import Tensor

from canvit_pytorch import Viewpoint
from canvit_pytorch.model import CanViT
from canvit_pytorch.model.attention.base import CanvasAttention
from canvit_pytorch.model.multihead import from_multihead, to_multihead
from canvit_pytorch.model.rope import RoPE, apply_2d_rope
from canvit_pytorch.viewpoint import grid_coords

OutsideWrites = Literal["drop", "center", "keep_prefix", "keep_patches"]
"""What local Writes do with the residual into a canvas cell outside the glimpse's crop:

- drop: remove it;
- center: subtract its mean over those cells, which removes the one vector the glimpse adds to every cell it did not
  see and keeps how the residual varies across them;
- keep_prefix: keep only the part from the glimpse's prefix tokens, whose keys carry no position;
- keep_patches: keep only the part from the glimpse's patch tokens.
"""


def write_parts(write: CanvasAttention, *, query: Tensor, kv: Tensor, query_rope: RoPE,
                kv_rope: RoPE) -> tuple[Tensor, Tensor, Tensor]:
    """A Write's residual [B, N_query, D] split by key, in float32: (part from the prefix tokens, part from the patch
    tokens, attention weight on each prefix token averaged over heads [B, N_query, n_prefix]). The two parts sum to
    the output."""
    heads = write.num_heads
    q = apply_2d_rope(to_multihead(write.q_map(write.ln_q(query)), heads), query_rope).float()
    kv = write.ln_kv(kv)
    k = apply_2d_rope(to_multihead(write.k_map(kv), heads), kv_rope).float()
    v = to_multihead(write.v_map(kv), heads).float()
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)  # [B, heads, N_query, N_kv]
    n_prefix = k.shape[2] - kv_rope.sin.shape[2]
    prefix = write.o_map(from_multihead(weights[..., :n_prefix] @ v[:, :, :n_prefix]))
    patches = write.o_map(from_multihead(weights[..., n_prefix:] @ v[:, :, n_prefix:]))
    return prefix, patches, weights[..., :n_prefix].mean(dim=1)


PREFIX_TOKENS = ("VPE", "recurrent CLS", *(f"backbone register {i}" for i in range(5)))
"""The glimpse's prefix tokens, in their order among a Write's keys."""


def token_part(write: CanvasAttention, token: int, *, query: Tensor, kv: Tensor, query_rope: RoPE,
               kv_rope: RoPE) -> Tensor:
    """The part [B, N_query, D] of a Write's residual that comes from key token `token`, in float32."""
    heads = write.num_heads
    q = apply_2d_rope(to_multihead(write.q_map(write.ln_q(query)), heads), query_rope).float()
    kv = write.ln_kv(kv)
    k = apply_2d_rope(to_multihead(write.k_map(kv), heads), kv_rope).float()
    v = to_multihead(write.v_map(kv[:, token:token + 1]), heads).float()
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)[..., token:token + 1]
    return write.o_map(from_multihead(weights @ v))


class LocalWrites:
    """Changes the Canvas Attention Write residuals into the canvas cells whose centers lie outside the current
    glimpse's crop (OutsideWrites), from glimpse `start` on; the canvas registers keep theirs. With max_scale, only
    glimpses of at most that scale are restricted, and larger ones write into every cell, as released. With margin,
    the cells within that many crop widths of the crop (Chebyshev distance) count as inside. With unseen_writes, the
    cells that no glimpse has covered yet also keep their residuals."""

    def __init__(self, canvit: CanViT, *, canvas_grid_size: int, device: torch.device, start: int,
                 outside: OutsideWrites = "drop", max_scale: float | None = None, margin: float = 0.0,
                 unseen_writes: bool = False) -> None:
        self.start, self.outside, self.max_scale, self.margin = start, outside, max_scale, margin
        self.unseen_writes = unseen_writes
        self.num_registers = canvit.config.num_canvas_registers
        self.cell_centers = grid_coords(size=canvas_grid_size, device=device)  # [G, G, 2]
        self.seen: Tensor | None = None  # [B, G*G], cells inside the crop of an earlier glimpse of the rollout
        self.mask: Tensor | None = None  # [B, num_registers + G*G, 1] for the glimpse being taken
        self.handles = [write.register_forward_hook(self.hook, with_kwargs=True) for write in canvit.canvas_writes]

    def before_glimpse(self, t: int, viewpoint: Viewpoint) -> None:
        """Sets the restriction for glimpse t of a rollout; t = 0 starts a new one."""
        offsets = (self.cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs().amax(dim=-1).flatten(1)
        scales = viewpoint.scales.view(-1, 1)
        covered = offsets <= scales
        if t == 0:
            self.seen = torch.zeros_like(covered)
        assert self.seen is not None
        inside = offsets <= scales * (1 + 2 * self.margin)
        if self.max_scale is not None:
            inside |= scales > self.max_scale
        if self.unseen_writes:
            inside |= ~self.seen
        self.seen |= covered
        if t < self.start:
            self.mask = None
            return
        registers = inside.new_ones(inside.shape[0], self.num_registers)
        self.mask = torch.cat([registers, inside], dim=1).unsqueeze(-1)

    def hook(self, module: CanvasAttention, args, kwargs: dict, output: Tensor) -> Tensor | None:
        if self.mask is None:
            return None
        outside = (~self.mask).float()
        match self.outside:
            case "drop":
                return output * self.mask
            case "center":
                mean = (output.float() * outside).sum(dim=1, keepdim=True) / outside.sum(dim=1, keepdim=True).clamp(min=1)
                return output - (outside * mean).to(output.dtype)
            case "keep_prefix" | "keep_patches":
                # The kept part is recomputed in float32: under bfloat16 autocast, the Write's own attention differs
                # from float32 by about 13%, so subtracting the other part from its output would not isolate one.
                prefix, patches = write_parts(module, **kwargs)[:2]
                kept = prefix if self.outside == "keep_prefix" else patches
                return (output * self.mask + outside * kept).to(output.dtype)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class DroppedWriteToken:
    """Removes from every Write residual, in every canvas cell, the part that comes from one key token of the glimpse
    (an index into PREFIX_TOKENS), while active. It needs no viewpoint, unlike LocalWrites."""

    def __init__(self, canvit: CanViT, *, token: int) -> None:
        self.token, self.active = token, False
        self.handles = [write.register_forward_hook(self.hook, with_kwargs=True) for write in canvit.canvas_writes]

    def hook(self, module: CanvasAttention, args, kwargs: dict, output: Tensor) -> Tensor | None:
        if not self.active:
            return None
        return (output.float() - token_part(module, self.token, **kwargs)).to(output.dtype)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
