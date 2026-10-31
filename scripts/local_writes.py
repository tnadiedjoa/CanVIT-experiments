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
import torch.nn.functional as F
from torch import Tensor, nn

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


class LearnedWriteGate(nn.Module):
    """A learned, soft version of local Writes. Each Write's residual into a canvas cell is multiplied, per head h, by
    exp(-rate_h · max(0, d − reach_h)), where d is the Chebyshev distance from the cell's center to the glimpse's crop
    in crop widths (0 inside the crop); the canvas registers keep their residuals. rate = softplus(raw_rate) and
    reach = softplus(raw_reach) are learned per Write and head; everything else in CanViT stays frozen. The initial
    rate (softplus(-4) ≈ 0.018) leaves the released Writes almost unchanged."""

    def __init__(self, canvit: CanViT, *, canvas_grid_size: int, device: torch.device) -> None:
        super().__init__()
        num_writes, num_heads = len(canvit.canvas_writes), canvit.canvas_writes[0].num_heads
        self.raw_rate = nn.Parameter(torch.full((num_writes, num_heads), -4.0, device=device))
        self.raw_reach = nn.Parameter(torch.zeros(num_writes, num_heads, device=device))
        self.num_registers = canvit.config.num_canvas_registers
        self.cell_centers = grid_coords(size=canvas_grid_size, device=device)  # [G, G, 2]
        self.distance: Tensor | None = None  # [B, G*G, 1], for the glimpse being taken
        self.handles = [write.register_forward_hook(self.hook(index), with_kwargs=True)
                        for index, write in enumerate(canvit.canvas_writes)]

    def before_glimpse(self, t: int, viewpoint: Viewpoint) -> None:
        offsets = (self.cell_centers.unsqueeze(0) - viewpoint.centers.view(-1, 1, 1, 2)).abs().amax(dim=-1).flatten(1)
        scales = viewpoint.scales.view(-1, 1)
        self.distance = ((offsets - scales) / (2 * scales)).clamp(min=0).unsqueeze(-1).float()

    def gates(self, index: int) -> Tensor:
        """[B, G*G, heads] factors of Write `index` for the current glimpse."""
        assert self.distance is not None, "call before_glimpse first"
        rate, reach = F.softplus(self.raw_rate[index]), F.softplus(self.raw_reach[index])
        return torch.exp(-rate * (self.distance - reach).clamp(min=0))

    def hook(self, index: int):
        def apply(module: CanvasAttention, args, kwargs: dict, output: Tensor) -> Tensor:
            B, N, D = output.shape
            heads = self.raw_rate.shape[1]
            cells = output[:, self.num_registers:].float().view(B, N - self.num_registers, heads, D // heads)
            gated = (cells * self.gates(index).unsqueeze(-1)).view(B, N - self.num_registers, D)
            return torch.cat([output[:, :self.num_registers].float(), gated], dim=1).to(output.dtype)
        return apply

    def describe(self) -> dict[str, list[list[float]]]:
        return {"rate": F.softplus(self.raw_rate).tolist(), "reach": F.softplus(self.raw_reach).tolist()}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class LearnedNullWrite(nn.Module):
    """A key with a zero value added to every Write, so that a canvas cell can attend to it and write nothing, instead
    of attending to backbone register 3, whose value is not zero. Its key (per head, without position) and a logit
    bias per head are learned; everything else in CanViT stays frozen. With the initial bias (-3), the zero-value key
    takes a few percent of each cell's attention, enough for gradients to flow."""

    def __init__(self, canvit: CanViT, *, canvas_grid_size: int, device: torch.device) -> None:
        super().__init__()
        writes = canvit.canvas_writes
        num_heads = writes[0].num_heads
        head_dim = canvit.canvas_dim // num_heads
        self.null_key = nn.Parameter(torch.zeros(len(writes), num_heads, head_dim, device=device))
        self.null_bias = nn.Parameter(torch.full((len(writes), num_heads), -3.0, device=device))
        self.writes = list(writes)
        self.released_forwards = [write.forward for write in writes]
        for index, write in enumerate(writes):
            write.forward = self.forward_with_null(write, index)

    def forward_with_null(self, write: CanvasAttention, index: int):
        def forward(*, query: Tensor, kv: Tensor, query_rope: RoPE, kv_rope: RoPE) -> Tensor:
            heads = write.num_heads
            q = apply_2d_rope(to_multihead(write.q_map(write.ln_q(query)), heads), query_rope)
            kv = write.ln_kv(kv)
            k = apply_2d_rope(to_multihead(write.k_map(kv), heads), kv_rope).to(q.dtype)
            v = to_multihead(write.v_map(kv), heads).to(q.dtype)
            out = F.scaled_dot_product_attention(q, k, v)  # the released Write, same kernel
            # A zero-value key with logit l0 rescales that output by the attention mass left to the real keys,
            # sigmoid(logsumexp(real logits) - l0), computed in float32.
            scale = q.shape[-1] ** -0.5
            real = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) * scale, dim=-1, keepdim=True)
            null = (q.float() @ self.null_key[index].view(1, heads, -1, 1)) * scale
            null = null + self.null_bias[index].view(1, heads, 1, 1)
            out = (out.float() * torch.sigmoid(real - null)).to(out.dtype)
            return write.o_map(from_multihead(out))
        return forward

    def before_glimpse(self, t: int, viewpoint: Viewpoint) -> None:
        pass

    def describe(self) -> dict[str, list[list[float]]]:
        return {"null_bias": self.null_bias.tolist(), "null_key_norm": self.null_key.norm(dim=-1).tolist()}

    def remove(self) -> None:
        for write, forward in zip(self.writes, self.released_forwards):
            write.forward = forward


LEARNED_WRITES = {"gate": LearnedWriteGate, "null": LearnedNullWrite}


def load_learned_writes(canvit: CanViT, path, *, canvas_grid_size: int, device: torch.device) -> nn.Module:
    """The learned Write change saved by scripts/train_write_gate.py, hooked into canvit."""
    record = torch.load(path, weights_only=True)
    module = LEARNED_WRITES[record.get("kind", "gate")](canvit, canvas_grid_size=canvas_grid_size, device=device)
    module.load_state_dict(record["state_dict"])
    return module
