"""A solution to exercises/canvas_attention.py. Read it only after trying."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sr_rope(positions: Tensor, *, head_dim: int, rope_base: float) -> tuple[Tensor, Tensor]:
    """[B, N, 2] (row, col) scene positions -> (sin, cos) of the rotation angles, each [B, 1, N, head_dim].

    head_dim / 4 wavelengths, geometric from 1 to rope_base scene units. Each axis gets head_dim / 4 angles; the
    head_dim / 2 angles of both axes rotate the pairs (x_i, x_{i + head_dim / 2}).
    """
    n = head_dim // 4
    wavelengths = rope_base ** (torch.arange(n, dtype=torch.float32) / n)
    angles = 2 * math.pi * positions.unsqueeze(-1) / wavelengths  # [B, N, 2, n]: rows first, then columns
    angles = angles.flatten(-2).repeat(1, 1, 2)  # [B, N, head_dim]
    return angles.sin().unsqueeze(1), angles.cos().unsqueeze(1)


def rotate(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Rotate the last N tokens of [B, heads, N_total, head_dim]; the tokens before them have no position."""
    n_unrotated = x.shape[2] - sin.shape[2]
    head, tail = x[:, :, :n_unrotated], x[:, :, n_unrotated:]
    first, second = tail.chunk(2, dim=-1)
    return torch.cat([head, tail * cos + torch.cat([-second, first], dim=-1) * sin], dim=2)


def heads(x: Tensor, num_heads: int) -> Tensor:
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def attend(module, *, query: Tensor, kv: Tensor, query_positions: Tensor, kv_positions: Tensor, rope_base: float) -> Tensor:
    """LayerNorm both sides, project where the module learned a projection, rotate queries and keys, attend."""
    q = module.q_map(module.ln_q(query))
    kv = module.ln_kv(kv)
    k, v = module.k_map(kv), module.v_map(kv)
    head_dim = q.shape[-1] // module.num_heads
    q_sin, q_cos = sr_rope(query_positions, head_dim=head_dim, rope_base=rope_base)
    k_sin, k_cos = sr_rope(kv_positions, head_dim=head_dim, rope_base=rope_base)
    q = rotate(heads(q, module.num_heads), q_sin, q_cos)
    k = rotate(heads(k, module.num_heads), k_sin, k_cos)
    out = F.scaled_dot_product_attention(q, k, heads(v, module.num_heads))
    return module.o_map(out.transpose(1, 2).flatten(2))


def canvas_attention_read(module, *, glimpse_tokens: Tensor, canvas: Tensor, glimpse_positions: Tensor,
                          canvas_positions: Tensor, rope_base: float) -> Tensor:
    """Glimpse tokens [B, N_g, D_backbone] query the canvas [B, N_can, D_can]; returns the glimpse-side residual."""
    return attend(module, query=glimpse_tokens, kv=canvas, query_positions=glimpse_positions,
                  kv_positions=canvas_positions, rope_base=rope_base)


def canvas_attention_write(module, *, canvas: Tensor, glimpse_tokens: Tensor, canvas_positions: Tensor,
                           glimpse_positions: Tensor, rope_base: float) -> Tensor:
    """Canvas tokens query the glimpse tokens; returns the canvas-side residual."""
    return attend(module, query=canvas, kv=glimpse_tokens, query_positions=canvas_positions,
                  kv_positions=glimpse_positions, rope_base=rope_base)
