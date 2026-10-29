"""Exercise: Canvas Attention (Read and Write) and SR-RoPE, written from the paper alone (arXiv:2603.22570,
appendix A pseudocode and the section on Canvas Attention).

Replace each NotImplementedError, then check against the released CanViT-B:

    .venv/bin/python exercises/check_canvas_attention.py

`module` is the released CanvasAttentionRead or CanvasAttentionWrite: use only its weights (`ln_q`, `ln_kv`,
`q_map`, `k_map`, `v_map`, `o_map`, where a missing projection is an identity) and `num_heads`. Positions are
(row, col) scene coordinates in [-1, 1]². Questions to settle on the way: which tokens get no rotation, how the
angles of the two axes share a head, and which side of each attention has learned projections.
A solution is in canvas_attention_solution.py.
"""

from torch import Tensor


def sr_rope(positions: Tensor, *, head_dim: int, rope_base: float) -> tuple[Tensor, Tensor]:
    """[B, N, 2] (row, col) scene positions -> (sin, cos) of the rotation angles, each [B, 1, N, head_dim]."""
    raise NotImplementedError


def canvas_attention_read(module, *, glimpse_tokens: Tensor, canvas: Tensor, glimpse_positions: Tensor,
                          canvas_positions: Tensor, rope_base: float) -> Tensor:
    """Glimpse tokens [B, N_g, D_backbone] (VPE, recurrent CLS, registers, then patches) query the canvas
    [B, N_can, D_can] (registers, then patches); returns the residual added to the glimpse tokens."""
    raise NotImplementedError


def canvas_attention_write(module, *, canvas: Tensor, glimpse_tokens: Tensor, canvas_positions: Tensor,
                           glimpse_positions: Tensor, rope_base: float) -> Tensor:
    """Canvas tokens query the glimpse tokens; returns the residual added to the canvas."""
    raise NotImplementedError
