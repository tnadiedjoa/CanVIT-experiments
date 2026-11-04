"""Mechanisms of primate vision that CanViT lacks, as changes to the released model that need no training (or train a
few parameters, scripts/train_retina.py), shared by scripts/bio_rollouts.py.

- FreePositions: CanViT's forward with the glimpse patches and canvas cells at given scene positions. SR-RoPE computes
  every attention from positions alone (the backbone infers the zoom from the patches' spacing), so neither grid has
  to be uniform.
- Retinas (where each glimpse pixel comes from): UniformRetina is the released square crop; FoveatedRetina warps it
  like a primate retina, fine at the center and coarse in the periphery (the receptive fields of V1 grow linearly with
  eccentricity, so the cortical map is log-polar), with the same 64 patches; PatchRetina gives every patch a center
  and a size of its own, the parameters of a retina that can be learned.
- NoveltyWrites (repetition suppression, predictive coding): a Write's residual into a cell loses the part the cell
  already holds, so a repeated view stops changing the canvas. A released Write cannot write nothing (its residual is
  a convex combination of value vectors), which is why repeating the full scene degrades the canvas (README).
- ObjectWrites (object-based attention): outside the glimpse's crop, a Write reaches only the cells whose content
  resembles a cell inside it, so it spreads along the seen object instead of into the whole canvas.
- AdaptiveCanvas (cortical magnification on the canvas side): the canvas cells are spread by a density over the scene
  instead of uniformly, with a separable warp; the segmentation is read back through the warp.
- Microsaccades: views of the same region shifted by a fraction of a patch, instead of the same view repeated.
"""

import functools
import math
import types
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch import Viewpoint
from canvit_pytorch.model import CanViT
from canvit_pytorch.model.attention.base import CanvasAttention
from canvit_pytorch.model.canvit import CanViTOutput, RecurrentState
from canvit_pytorch.model.multihead import from_multihead as from_heads
from canvit_pytorch.model.multihead import to_multihead as to_heads
from canvit_pytorch.model.rope import apply_2d_rope as apply_rope
from canvit_pytorch.model.rope import compute_2d_rope, make_rope_periods
from canvit_pytorch.policies import FixedSequence
from canvit_pytorch.policies.quadtree import quadtree_level, quadtree_level_z_order, shuffled_levels
from canvit_pytorch.viewpoint import grid_coords, viewpoint_grid_coords


# Free positions


class FreePositions:
    """Makes canvit's forward place the glimpse patches at `glimpse` [B, g*g, 2] and the canvas cells at `canvas`
    [B, G*G, 2] (scene coordinates, (row, col)); None keeps the released uniform grid."""

    def __init__(self, canvit: CanViT) -> None:
        self.glimpse: Tensor | None = None
        self.canvas: Tensor | None = None
        self.periods: "LearnedPeriods | None" = None  # when set, the RoPE wavelengths come from it
        self.canvit = canvit
        canvit.forward = types.MethodType(functools.partial(_forward, positions=self), canvit)

    def remove(self) -> None:
        del self.canvit.forward


def _forward(self: CanViT, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint,
             positions: FreePositions) -> CanViTOutput:
    """CanViT.forward, with the positions of `positions`."""
    B, _, height, width = glimpse.shape
    glimpse_grid = height // self.patch_size
    patches = self.backbone.patch_embed(glimpse)
    prefix = [state.recurrent_cls, self.backbone_registers.expand(B, -1, -1)]
    if self.vpe is not None:
        prefix.insert(0, self.vpe(viewpoint).unsqueeze(1).to(patches.dtype))
    n_prefix = sum(t.shape[1] for t in prefix)
    tokens = torch.cat([*prefix, patches], dim=1)

    spec = self.backbone.spec
    if positions.periods is not None:
        backbone_periods, canvas_periods = positions.periods.backbone(), positions.periods.canvas()
    else:
        backbone_periods = make_rope_periods(head_dim=spec.head_dim, base=spec.rope_base, device=glimpse.device)
        canvas_periods = make_rope_periods(head_dim=self.config.canvas_head_dim, base=spec.rope_base,
                                           device=glimpse.device)
    glimpse_positions = positions.glimpse
    if glimpse_positions is None:
        glimpse_positions = viewpoint_grid_coords(viewpoint, size=glimpse_grid).flatten(1, 2)
    glimpse_rope = compute_2d_rope(positions=glimpse_positions.float(), periods=backbone_periods)
    glimpse_rope_canvas = compute_2d_rope(positions=glimpse_positions.float(), periods=canvas_periods)

    canvas = state.canvas
    canvas_positions = positions.canvas
    if canvas_positions is None:
        canvas_grid = math.isqrt(canvas.shape[1] - self.config.num_canvas_registers)
        canvas_positions = grid_coords(size=canvas_grid, device=canvas.device).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
    canvas_rope = compute_2d_rope(positions=canvas_positions.float(), periods=canvas_periods)

    read_after_block = dict(zip(self.read_after_blocks, self.canvas_reads))
    write_after_block = dict(zip(self.write_after_blocks, self.canvas_writes))
    for i, block in enumerate(self.backbone.blocks):
        tokens = block(tokens, glimpse_rope)
        if (read := read_after_block.get(i)) is not None:
            tokens = tokens + read(query=tokens, kv=canvas, query_rope=glimpse_rope_canvas, kv_rope=canvas_rope)
        if (write := write_after_block.get(i)) is not None:
            canvas = canvas + write(query=canvas, kv=tokens, query_rope=canvas_rope, kv_rope=glimpse_rope_canvas)

    cls_index = 1 if self.vpe is not None else 0
    return CanViTOutput(
        state=RecurrentState(canvas=canvas, recurrent_cls=tokens[:, cls_index : cls_index + 1].contiguous()),
        glimpse_patches=tokens[:, n_prefix:].contiguous(),
        vpe=tokens[:, 0] if self.vpe is not None else None,
    )


class LearnedPeriods(nn.Module):
    """SR-RoPE's wavelengths as parameters (log scale), starting at the released ones (geometric from 1 to 100 scene
    units; the finest is half the scene). RoPE codes a position by sinusoids at several spatial scales, as the grid
    cells of the entorhinal cortex do with modules of growing spacing; the finest module bounds how precisely a
    position is told apart, which matters for small glimpses (level 3's patches are 1/32 of a scene unit apart)."""

    def __init__(self, canvit: CanViT, device: torch.device) -> None:
        super().__init__()
        spec = canvit.backbone.spec
        self.log_backbone = nn.Parameter(make_rope_periods(head_dim=spec.head_dim, base=spec.rope_base,
                                                           device=device).log())
        self.log_canvas = nn.Parameter(make_rope_periods(head_dim=canvit.config.canvas_head_dim, base=spec.rope_base,
                                                         device=device).log())

    def backbone(self) -> Tensor:
        return self.log_backbone.exp()

    def canvas(self) -> Tensor:
        return self.log_canvas.exp()

    def describe(self) -> dict[str, list[float]]:
        return {"backbone_periods": self.backbone().detach().tolist(), "canvas_periods": self.canvas().detach().tolist()}


# Retinas


def local_offsets(size: int, *, device: torch.device | None) -> Tensor:
    """Pixel centers of a size×size patch, in units of the patch's half side: [size, size, 2]."""
    return grid_coords(size=size, device=device)


class UniformRetina:
    """The released glimpse: a uniform grid of patches over the square crop."""

    def __init__(self, *, glimpse_px: int, patch_px: int, device: torch.device) -> None:
        self.pixels = grid_coords(size=glimpse_px, device=device)  # [px, px, 2], crop units
        self.patches = grid_coords(size=glimpse_px // patch_px, device=device).flatten(0, 1)  # [g*g, 2]

    def offsets(self) -> tuple[Tensor, Tensor]:
        """(pixel offsets [px, px, 2], patch centers [g*g, 2]) in crop units: [-1, 1]² is the viewpoint's crop."""
        return self.pixels, self.patches


def foveal_warp(offsets: Tensor, fovea: float) -> Tensor:
    """Moves crop offsets along rays from the crop's center: Chebyshev radius ρ ∈ [0, 1] goes to
    fovea·((1 + 1/fovea)^ρ − 1), so the crop's border stays where it is, a patch's size grows linearly with its
    eccentricity (the primate retina and V1, whose map of the visual field is log-polar), and the central patches are
    1 / (fovea·ln(1 + 1/fovea)) times finer than the released ones (1.44 at fovea 1, 2.5 at 0.25, 4.2 at 0.1)."""
    radius = offsets.double().abs().amax(dim=-1, keepdim=True)
    warped = fovea * torch.expm1(radius * math.log1p(1 / fovea))  # float64: (1 + 1/fovea)^ρ − 1 cancels otherwise
    return (offsets.double() * warped / radius.clamp(min=1e-12)).to(offsets.dtype)


class FoveatedRetina(UniformRetina):
    """The released crop, warped by foveal_warp: same border, same 64 patches, fine at the center."""

    def __init__(self, *, glimpse_px: int, patch_px: int, device: torch.device, fovea: float) -> None:
        super().__init__(glimpse_px=glimpse_px, patch_px=patch_px, device=device)
        self.pixels = foveal_warp(self.pixels, fovea)
        self.patches = foveal_warp(self.patches, fovea)


class PatchRetina(nn.Module):
    """Every patch has its own center and half sizes (row, col), in crop units, starting as the released uniform grid
    (or from a foveal warp of it): a retina whose layout can be learned. A patch may leave the crop, a peripheral
    view at low resolution; pixels outside the scene are black, as in the released extraction."""

    def __init__(self, *, glimpse_px: int, patch_px: int, device: torch.device, fovea: float | None = None) -> None:
        super().__init__()
        g = glimpse_px // patch_px
        centers = grid_coords(size=g, device=device).flatten(0, 1)
        half_sizes = torch.full_like(centers, 1 / g)
        if fovea is not None:
            # Half sizes from the warp's spacing: the distance between the warped edges of each uniform patch.
            edges = torch.linspace(-1, 1, g + 1, device=device)
            lo = foveal_warp(torch.stack(torch.meshgrid(edges[:-1], edges[:-1], indexing="ij"), -1).flatten(0, 1), fovea)
            hi = foveal_warp(torch.stack(torch.meshgrid(edges[1:], edges[1:], indexing="ij"), -1).flatten(0, 1), fovea)
            centers, half_sizes = (lo + hi) / 2, (hi - lo) / 2
        self.centers = nn.Parameter(centers)
        self.log_half_sizes = nn.Parameter(half_sizes.log())
        self.local = local_offsets(patch_px, device=device)  # [p, p, 2]
        self.g, self.patch_px = g, patch_px

    def offsets(self) -> tuple[Tensor, Tensor]:
        g, p = self.g, self.patch_px
        pixels = self.centers.view(g * g, 1, 1, 2) + self.log_half_sizes.exp().view(g * g, 1, 1, 2) * self.local
        pixels = pixels.view(g, g, p, p, 2).permute(0, 2, 1, 3, 4).reshape(g * p, g * p, 2)
        return pixels, self.centers

    def describe(self) -> dict[str, list]:
        sizes = self.log_half_sizes.exp().detach()
        radius = self.centers.detach().abs().amax(dim=-1)
        return {"centers": self.centers.detach().cpu().tolist(), "half_sizes": sizes.cpu().tolist(),
                "mean_half_size_by_ring": [sizes[(radius > r / self.g * 2 - 1e-3) & (radius < (r + 1) / self.g * 2)]
                                           .mean().item() for r in range(self.g // 2)]}


def take_glimpse(images: Tensor, viewpoint: Viewpoint, retina) -> tuple[Tensor, Tensor]:
    """(glimpse [B, 3, px, px], patch positions [B, g*g, 2] in scene coordinates) through the retina; bilinear, as the
    released extraction (sample_at_viewpoint)."""
    pixels, patches = retina.offsets()
    B = images.shape[0]
    centers, scales = viewpoint.centers.view(B, 1, 1, 2), viewpoint.scales.view(B, 1, 1, 1)
    xy = (centers + scales * pixels.unsqueeze(0)).flip(-1)
    glimpse = F.grid_sample(images.float(), xy.float(), mode="bilinear", align_corners=False).to(images.dtype)
    return glimpse, centers.view(B, 1, 2) + scales.view(B, 1, 1) * patches.unsqueeze(0)


# Writes


class NoveltyWrites:
    """Repetition suppression: from glimpse `start` on, the residual of every Write into a canvas cell loses its part
    along what the cell held before the glimpse (both centered over channels, as LayerNorm sees them).

    - project: r − strength·max(0, ⟨r, c⟩)/|c|²·c, the redundant component of the residual r along the cell c;
    - gate: r·(1 − cos(r, c))^strength, clamped to [0, 1]: a cell is written in proportion to how new the residual is.

    The canvas registers keep their residuals. `suppressed` accumulates, per glimpse, the mean over cells of
    |removed| / |r|."""

    def __init__(self, canvit: CanViT, *, mode: Literal["project", "gate"], strength: float, start: int,
                 num_glimpses: int, device: torch.device) -> None:
        self.mode, self.strength, self.start = mode, strength, start
        self.num_registers = canvit.config.num_canvas_registers
        self.t, self.reference = 0, None
        self.suppressed = torch.zeros(num_glimpses, dtype=torch.float64, device=device)
        self.handles = [write.register_forward_hook(functools.partial(self.hook, index=i), with_kwargs=True)
                        for i, write in enumerate(canvit.canvas_writes)]

    def before_glimpse(self, t: int) -> None:
        self.t, self.reference = t, None

    def hook(self, module: CanvasAttention, args, kwargs: dict, output: Tensor, *, index: int) -> Tensor | None:
        if self.t < self.start:
            return None
        if index == 0:
            query = kwargs["query"].float()
            self.reference = query - query.mean(dim=-1, keepdim=True)
        assert self.reference is not None
        r, c = output.float(), self.reference
        centered = r - r.mean(dim=-1, keepdim=True)
        dot = (centered * c).sum(dim=-1, keepdim=True)
        match self.mode:
            case "project":
                new = r - self.strength * dot.clamp(min=0) / c.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12) * c
            case "gate":
                cos = dot / (centered.norm(dim=-1, keepdim=True) * c.norm(dim=-1, keepdim=True)).clamp(min=1e-12)
                new = r * (1 - cos).clamp(0, 1) ** self.strength
        new[:, :self.num_registers] = r[:, :self.num_registers]
        cells = slice(self.num_registers, None)
        removed = (r - new)[:, cells].norm(dim=-1) / r[:, cells].norm(dim=-1).clamp(min=1e-12)
        self.suppressed[self.t] += removed.mean(dim=1).sum().double() / len(self.handles)
        return new.to(output.dtype)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class ObjectWrites:
    """Object-based attention: from glimpse `start` on, a Write reaches a canvas cell outside the crop only when the
    cell's layer-normalized content has a cosine similarity of at least `threshold` with one of the cells inside the
    crop, as they were before the glimpse; the canvas registers keep their residuals. `reached` accumulates, per
    glimpse, the fraction of the outside cells that are written."""

    def __init__(self, canvit: CanViT, *, canvas_grid_size: int, threshold: float, start: int, num_glimpses: int,
                 device: torch.device) -> None:
        self.canvit, self.threshold, self.start = canvit, threshold, start
        self.num_registers = canvit.config.num_canvas_registers
        self.cell_centers = grid_coords(size=canvas_grid_size, device=device).flatten(0, 1)  # [G*G, 2]
        self.mask: Tensor | None = None
        self.reached = torch.zeros(num_glimpses, dtype=torch.float64, device=device)
        self.handles = [write.register_forward_hook(self.hook) for write in canvit.canvas_writes]

    def before_glimpse(self, t: int, viewpoint: Viewpoint, canvas: Tensor) -> None:
        if t < self.start:
            self.mask = None
            return
        offsets = (self.cell_centers.unsqueeze(0) - viewpoint.centers.unsqueeze(1)).abs().amax(dim=-1)
        inside = offsets <= viewpoint.scales.view(-1, 1)  # [B, G*G]
        cells = F.layer_norm(self.canvit.canvas_patches(canvas).float(), (canvas.shape[-1],))
        cells = F.normalize(cells, dim=-1)
        similar = torch.zeros_like(inside)
        for b in range(cells.shape[0]):
            seen = cells[b, inside[b]]  # [n_inside, D]
            similar[b] = (cells[b] @ seen.T).amax(dim=1) >= self.threshold
        written = inside | similar
        outside = (~inside).float()
        self.reached[t] += ((written.float() * outside).sum(dim=1) / outside.sum(dim=1).clamp(min=1)).sum().double()
        registers = written.new_ones(written.shape[0], self.num_registers)
        self.mask = torch.cat([registers, written], dim=1).unsqueeze(-1)

    def hook(self, module: CanvasAttention, args, output: Tensor) -> Tensor | None:
        return None if self.mask is None else output * self.mask

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


# Learned canvas shape


class GaussianCanvas(nn.Module):
    """The canvas cells as learned Gaussian receptive fields (the population receptive field model of fMRI, Dumoulin &
    Wandell 2008, fits one per voxel of visual cortex): each cell has a center, its position for SR-RoPE in the Reads
    and Writes, and an elliptical shape (two half widths and an angle) that says which part of the scene it stands for.
    The scene's features at a point are the cells' predictions averaged with weights proportional to each cell's
    Gaussian there. It starts as a uniform grid with narrow Gaussians (sigma = 0.3 cell spacings): on the released
    grid, reading at the cell centers is then almost the identity (each cell keeps 98.5% of the weight at its center).

    - canvas_grid_size: the starting grid, G×G cells (fewer than the teacher's patches for a canvas on a budget);
    - coarse_grid_size: a second, coarser grid of cells with wider Gaussians on top (receptive fields of several
      sizes, as along the ventral stream from V1 to V4 and IT);
    - learn_centers = False: only the shapes learn, the control that separates moving the cells from smoothing;
    - adaptive: the canvas also reshapes itself for each scene: after the first glimpse, every cell moves by up to
      max_shift fine spacings in a direction read linearly from its own layer-normalized content (zero at the start),
      and the canvas is resampled at the new centers through the cells' Gaussians;
    - surround: each cell also has an inhibitory surround, a wider Gaussian (3 times at the start) subtracted with a
      learned weight (about 0 at the start), the center-surround receptive fields of retinal ganglion cells and the
      LGN: the reading at p is A + alpha·(A − S), A and S the averages under the centers and the surrounds."""

    def __init__(self, *, canvas_grid_size: int, canvas_dim: int, device: torch.device, adaptive: bool = False,
                 max_shift: float = 2.0, sigma: float = 0.3, coarse_grid_size: int | None = None,
                 learn_centers: bool = True, surround: bool = False) -> None:
        super().__init__()
        grids = [canvas_grid_size] + ([coarse_grid_size] if coarse_grid_size is not None else [])
        centers = torch.cat([grid_coords(size=g, device=device).flatten(0, 1) for g in grids])
        spacings = torch.cat([torch.full((g * g,), 2 / g, device=device) for g in grids])
        self.centers = nn.Parameter(centers.clone(), requires_grad=learn_centers)  # [N, 2]
        self.log_sigmas = nn.Parameter((sigma * spacings).log().unsqueeze(1).repeat(1, 2))  # [N, 2]
        self.angles = nn.Parameter(torch.zeros(centers.shape[0], device=device))
        self.register_buffer("initial_centers", centers.clone(), persistent=False)
        self.register_buffer("spacings", spacings, persistent=False)
        self.adaptive = adaptive
        self.shift = nn.Linear(canvas_dim, 2, device=device) if adaptive else None
        if self.shift is not None:
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        self.surround = surround
        if surround:
            self.log_surround_scales = nn.Parameter(torch.full((centers.shape[0],), math.log(3.0), device=device))
            self.raw_surround_weights = nn.Parameter(torch.full((centers.shape[0],), -4.0, device=device))
        self.G, self.spacing = canvas_grid_size, 2 / canvas_grid_size
        self.max_shift = max_shift * self.spacing
        self.num_fine = canvas_grid_size ** 2
        self.offset: Tensor | None = None  # [B, N, 2], this rollout's displacement by content (adaptive)
        self.batch_size = 0

    @property
    def num_cells(self) -> int:
        return self.centers.shape[0]

    def init_state(self, canvit: CanViT, batch_size: int) -> RecurrentState:
        """The state before any glimpse, with one canvas cell per Gaussian (canvit.init_state broadcasts its learned
        initial patch to a square grid; here to any number of cells)."""
        registers = canvit.init_canvas_registers.expand(batch_size, -1, -1)
        patches = canvit.init_canvas_patch.expand(batch_size, self.num_cells, -1)
        return RecurrentState(canvas=torch.cat([registers, patches], dim=1),
                              recurrent_cls=canvit.init_recurrent_cls.expand(batch_size, -1, -1))

    def reset(self, batch_size: int) -> Tensor:
        """Starts a rollout; returns the cell centers [B, N, 2] for FreePositions.canvas."""
        self.batch_size, self.offset = batch_size, None
        return self.positions()

    def positions(self) -> Tensor:
        """The centers of the rollout in progress, [B, N, 2]."""
        centers = self.centers.unsqueeze(0).expand(self.batch_size, -1, -1)
        return centers if self.offset is None else (centers + self.offset).clamp(-1, 1)

    def detach(self) -> None:
        """At a truncated-BPTT boundary: later glimpses still train the centers and shapes, not the content shift."""
        if self.offset is not None:
            self.offset = self.offset.detach()

    def log_weights(self, d: Tensor, widen: Tensor | None = None) -> Tensor:
        """Offsets [..., N, 2] from each cell's center -> the log of its Gaussian there (up to a constant), float32;
        widen [N] multiplies the half widths (the surrounds)."""
        cos, sin = self.angles.cos(), self.angles.sin()
        u = cos * d[..., 0] + sin * d[..., 1]
        v = -sin * d[..., 0] + cos * d[..., 1]
        sigmas = self.log_sigmas.exp()
        if widen is not None:
            sigmas = sigmas * widen.unsqueeze(1)
        return -0.5 * ((u / sigmas[:, 0]) ** 2 + (v / sigmas[:, 1]) ** 2)

    def weights(self, points: Tensor, centers: Tensor) -> Tensor:
        """[P, 2] scene points, [B, N, 2] centers -> [B, P, N], each row summing to 1, in float32."""
        d = points.float().view(1, -1, 1, 2) - centers.float().unsqueeze(1)
        return torch.softmax(self.log_weights(d), dim=-1)

    def read(self, cells: Tensor, points: Tensor, *, chunk: int = 4096) -> Tensor:
        """Per-cell values [B, N, C] read at scene points [P, 2]: [B, P, C], in float32."""
        centers = self.positions()
        out = []
        for i in range(0, points.shape[0], chunk):
            weights = self.weights(points[i:i + chunk], centers)
            center = weights @ cells.float()
            if self.surround:
                d = points[i:i + chunk].float().view(1, -1, 1, 2) - centers.float().unsqueeze(1)
                surround = torch.softmax(self.log_weights(d, self.log_surround_scales.exp()), dim=-1) @ cells.float()
                alpha = weights @ F.softplus(self.raw_surround_weights).unsqueeze(1)  # [B, P, 1]
                center = center + alpha * (center - surround)
            out.append(center)
        return torch.cat(out, dim=1)

    def adapt(self, state: RecurrentState, canvit: CanViT) -> tuple[RecurrentState, Tensor]:
        """After the first glimpse: moves the cells by their content and resamples the canvas at the new centers."""
        assert self.shift is not None
        R = canvit.config.num_canvas_registers
        cells = canvit.canvas_patches(state.canvas)
        content = F.layer_norm(cells.float(), (cells.shape[-1],))
        old = self.positions()
        self.offset = self.max_shift * torch.tanh(self.shift(content))
        moved = self.positions()
        d = moved.float().unsqueeze(2) - old.float().unsqueeze(1)  # [B, new, old, 2]
        resampled = torch.softmax(self.log_weights(d), dim=-1) @ cells.float()
        state = RecurrentState(canvas=torch.cat([state.canvas[:, :R], resampled.to(state.canvas.dtype)], dim=1),
                               recurrent_cls=state.recurrent_cls)
        return state, moved

    def describe(self) -> dict:
        """For the fine cells, by Chebyshev ring of the starting grid from the center outward (8 rings): the mean
        displacement from the start and the mean half width, in spacings of their grid; the same means for the
        coarse cells, if any."""
        n, G = self.num_fine, self.G
        moved = (self.centers.detach() - self.initial_centers).norm(dim=-1) / self.spacings
        sigmas = self.log_sigmas.detach().exp() / self.spacings.unsqueeze(1)
        ring = (self.initial_centers[:n].abs().amax(dim=-1) * G / 2).floor().clamp(max=G // 2 - 1) // max(1, G // 16)
        rings = range(int(ring.max().item()) + 1)
        out = {"mean_displacement_spacings": moved[:n].mean().item(),
               "mean_sigma_spacings": sigmas[:n].mean().item(),
               "sigma_by_ring": [sigmas[:n][ring == r].mean().item() for r in rings],
               "displacement_by_ring": [moved[:n][ring == r].mean().item() for r in rings],
               "anisotropy": (sigmas[:n].amax(dim=1) / sigmas[:n].amin(dim=1)).mean().item()}
        if self.num_cells > n:
            out["coarse_mean_sigma_spacings"] = sigmas[n:].mean().item()
            out["coarse_mean_displacement_spacings"] = moved[n:].mean().item()
        if self.shift is not None:
            out["shift_weight_norm"] = self.shift.weight.detach().norm().item()
        if self.surround:
            out["surround_weight"] = F.softplus(self.raw_surround_weights.detach()).mean().item()
            out["surround_scale"] = self.log_surround_scales.detach().exp().mean().item()
        return out


class WriteFootprint(nn.Module):
    """What each canvas cell listens to: in every Write, a cell's attention logit to a glimpse patch loses
    beta·m/2, where m is the patch's squared Mahalanobis distance to the cell under the cell's own Gaussian widened by
    kappa (one beta per Write and head, one kappa per Write, both learned; the prefix tokens and the canvas registers
    are unaffected). A receptive field then shapes both what a cell stands for (GaussianCanvas) and what it receives.
    It starts close to the released Writes (beta 0.018, kappa 16) and computes the Writes' attention explicitly in
    float32."""

    def __init__(self, canvit: CanViT, shape: GaussianCanvas, free: FreePositions) -> None:
        super().__init__()
        writes = canvit.canvas_writes
        device = shape.centers.device
        self.log_kappas = nn.Parameter(torch.full((len(writes),), math.log(16.0), device=device))
        self.raw_betas = nn.Parameter(torch.full((len(writes), writes[0].num_heads), -4.0, device=device))
        object.__setattr__(self, "shape", shape)  # not a submodule: its parameters belong to the GaussianCanvas
        self.free = free
        self.num_registers = canvit.config.num_canvas_registers
        self.writes = list(writes)
        for index, write in enumerate(writes):
            write.forward = self.forward_with_footprint(write, index)

    def bias(self, index: int, n_query: int, n_kv: int) -> Tensor:
        """[B, heads, n_query, n_kv] additive logits."""
        assert self.free.glimpse is not None
        patches = self.free.glimpse.float()  # [B, P, 2]
        centers = self.shape.positions().float()  # [B, N, 2]
        d = (patches.unsqueeze(1) - centers.unsqueeze(2)) / self.log_kappas[index].exp()  # [B, N, P, 2]
        # the cell's Gaussian is indexed by the cell (dim 1): move it last for log_weights, then back
        m = -2 * self.shape.log_weights(d.transpose(1, 2)).transpose(1, 2)  # [B, N, P]
        betas = F.softplus(self.raw_betas[index]).view(1, -1, 1, 1)
        B, N, P = m.shape
        out = torch.zeros(B, betas.shape[1], n_query, n_kv, device=m.device)
        out[:, :, self.num_registers:, n_kv - P:] = -0.5 * betas * m.unsqueeze(1)
        return out

    def forward_with_footprint(self, write: CanvasAttention, index: int):
        def forward(*, query: Tensor, kv: Tensor, query_rope, kv_rope) -> Tensor:
            heads = write.num_heads
            q = apply_rope(to_heads(write.q_map(write.ln_q(query)), heads), query_rope).float()
            kv = write.ln_kv(kv)
            k = apply_rope(to_heads(write.k_map(kv), heads), kv_rope).float()
            v = to_heads(write.v_map(kv), heads).float()
            logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            logits = logits + self.bias(index, q.shape[2], k.shape[2])
            out = torch.softmax(logits, dim=-1) @ v
            return write.o_map(from_heads(out.to(query.dtype)))
        return forward

    def describe(self) -> dict:
        return {"kappa": self.log_kappas.detach().exp().tolist(),
                "beta": F.softplus(self.raw_betas.detach()).tolist()}

    def remove(self) -> None:
        for write in self.writes:
            del write.forward


# Adaptive canvas


def inverse_cdf(density: Tensor, quantiles: Tensor) -> Tensor:
    """Scene coordinates in [-1, 1] at the given quantiles of a positive 1D density over N equal bins of [-1, 1]:
    density [B, N], quantiles [Q] -> [B, Q]."""
    B, N = density.shape
    cdf = F.pad(density.cumsum(dim=1) / density.sum(dim=1, keepdim=True), (1, 0))  # [B, N + 1], at bin edges
    edges = torch.linspace(-1, 1, N + 1, device=density.device)
    q = quantiles.view(1, -1).expand(B, -1).contiguous()
    upper = torch.searchsorted(cdf, q).clamp(1, N)
    lo, hi = cdf.gather(1, upper - 1), cdf.gather(1, upper)
    frac = ((q - lo) / (hi - lo).clamp(min=1e-12)).clamp(0, 1)
    return edges[upper - 1] + frac * (edges[upper] - edges[upper - 1])


class AdaptiveCanvas:
    """A G×G canvas whose rows and columns are spread by a density over the scene (cortical magnification), instead
    of uniformly: the cell (i, j) sits at the (i + ½)/G quantile of the row density and the (j + ½)/G quantile of the
    column density, both the marginals of a 2D density mixed with the uniform one (mix = 0 is the released canvas).
    The canvas index coordinate of a scene point is its quantile, so a map over the cells is read back at any scene
    point by bilinear interpolation in index space, which for mix = 0 is the released bilinear upsampling."""

    def __init__(self, density: Tensor, *, canvas_grid_size: int, mix: float) -> None:
        B, H, W = density.shape
        rows = density.mean(dim=2)
        cols = density.mean(dim=1)
        rows = (1 - mix) + mix * rows / rows.mean(dim=1, keepdim=True)
        cols = (1 - mix) + mix * cols / cols.mean(dim=1, keepdim=True)
        self.rows, self.cols = rows, cols
        quantiles = (torch.arange(canvas_grid_size, device=density.device, dtype=torch.float32) + 0.5) / canvas_grid_size
        self.row_positions, self.col_positions = inverse_cdf(rows, quantiles), inverse_cdf(cols, quantiles)  # [B, G]
        self.G = canvas_grid_size

    def positions(self) -> Tensor:
        """Cell centers [B, G*G, 2] in scene coordinates, row-major."""
        B, G = self.row_positions.shape
        return torch.stack([self.row_positions.view(B, G, 1).expand(B, G, G),
                            self.col_positions.view(B, 1, G).expand(B, G, G)], dim=-1).flatten(1, 2)

    def index_coordinates(self, size_px: int) -> Tensor:
        """[B, size_px, size_px, 2] grid_sample coordinates (x, y) into the cell grid of each scene pixel's center."""
        def quantile(density: Tensor) -> Tensor:  # [B, N] -> [B, size_px], the CDF at the pixel centers
            N = density.shape[1]
            cdf = F.pad(density.cumsum(dim=1) / density.sum(dim=1, keepdim=True), (1, 0)).unsqueeze(1)  # [B, 1, N+1]
            centers = (torch.arange(size_px, device=density.device, dtype=torch.float32) + 0.5) / size_px * N
            lower = centers.floor().long().clamp(max=N - 1)
            frac = centers - lower
            return cdf[:, 0, lower] * (1 - frac) + cdf[:, 0, lower + 1] * frac
        y, x = quantile(self.rows) * 2 - 1, quantile(self.cols) * 2 - 1
        B = y.shape[0]
        return torch.stack([x.view(B, 1, size_px).expand(B, size_px, size_px),
                            y.view(B, size_px, 1).expand(B, size_px, size_px)], dim=-1)

    def read_back(self, cell_map: Tensor, size_px: int) -> Tensor:
        """[B, C, G, G] over the cells -> [B, C, size_px, size_px] over the scene."""
        return F.grid_sample(cell_map.float(), self.index_coordinates(size_px), mode="bilinear",
                             padding_mode="border", align_corners=False)

    def resample_uniform(self, patches: Tensor) -> Tensor:
        """Canvas patches [B, G*G, D] of a uniform G×G canvas, resampled at this canvas's cells: [B, G*G, D]."""
        B, _, D = patches.shape
        grid = patches.transpose(1, 2).reshape(B, D, self.G, self.G).float()
        xy = self.positions().flip(-1).view(B, self.G, self.G, 2)
        out = F.grid_sample(grid, xy, mode="bilinear", padding_mode="border", align_corners=False)
        return out.flatten(2).transpose(1, 2).to(patches.dtype)


def boundary_density(labels: Tensor, *, blur_px: int) -> Tensor:
    """[B, H, W] labels -> the density of their boundaries (a pixel whose right or lower neighbor differs), box-blurred
    over blur_px, plus a floor so no region is empty."""
    edges = torch.zeros_like(labels, dtype=torch.float32)
    edges[:, :, :-1] += (labels[:, :, 1:] != labels[:, :, :-1]).float()
    edges[:, :-1, :] += (labels[:, 1:, :] != labels[:, :-1, :]).float()
    density = F.avg_pool2d(edges.unsqueeze(1), blur_px, stride=1, padding=blur_px // 2, count_include_pad=False)
    density = density[:, 0, :labels.shape[1], :labels.shape[2]]
    return density + 1e-3 * density.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)


def image_edge_density(glimpse: Tensor) -> Tensor:
    """[B, 3, px, px] full-scene glimpse -> the magnitude of its intensity gradient [B, px, px], a retina-like map of
    local contrast that needs nothing but the first glimpse's pixels."""
    gray = glimpse.float().mean(dim=1, keepdim=True)
    gy = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 0, 0, 1)).abs()
    gx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0)).abs()
    density = F.avg_pool2d(gx + gy, 9, stride=1, padding=4, count_include_pad=False)[:, 0]
    return density + 1e-3 * density.mean(dim=(1, 2), keepdim=True)


# Microsaccades


def halton(index: int, base: int) -> float:
    """The index-th term (from 1) of the van der Corput sequence in this base, in [0, 1)."""
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def microsaccades(*, batch_size: int, device: torch.device, num_glimpses: int, scale: float, jitter: float,
                  first_full_scene: bool = True, outside: bool = False,
                  sequence: Literal["random", "halton"] = "random") -> FixedSequence:
    """The full scene, then views of scale `scale` centered at offsets of at most `jitter` (clamped to the valid box
    unless `outside`, where the view's part beyond the scene is black); jitter = 0 is the same smaller view repeated,
    the control. With scale 1, outside and a jitter below a patch, every view shows the whole scene with another
    sampling phase: a test of super-resolution from fixational eye movements. The offsets are uniformly random, or
    (halton) the 2D Halton sequence in bases 2 and 3, rotated by a random offset per scene: they cover the square
    evenly from the first views on."""
    viewpoints = [Viewpoint.full_scene(batch_size=batch_size, device=device)] if first_full_scene else []
    limit = jitter if outside else min(jitter, 1 - scale)
    rotation = torch.rand(batch_size, 2, device=device)
    while len(viewpoints) < num_glimpses:
        if sequence == "random":
            unit = torch.rand(batch_size, 2, device=device)
        else:
            k = len(viewpoints) + 1
            unit = (rotation + torch.tensor([halton(k, 2), halton(k, 3)], device=device)) % 1
        centers = (unit * 2 - 1) * limit
        viewpoints.append(Viewpoint(centers=centers, scales=torch.full((batch_size,), scale, device=device)))
    return FixedSequence(viewpoints)


def c2f_then_jittered_level2(*, batch_size: int, device: torch.device, num_glimpses: int, jitter: float) -> FixedSequence:
    """C2F's levels 0 to 2, then the level-2 tiles again, each pass in a new random order and each view shifted by a
    uniform random offset of at most `jitter` per axis (clamped to the valid box): microsaccades in place of the
    upsampled level 3. jitter = 0 is the README's c2f_then_level2_revisits."""
    levels = [quadtree_level_z_order(level) for level in range(3)]
    passes = math.ceil(max(0, num_glimpses - 21) / 16)
    viewpoints = shuffled_levels(levels + [quadtree_level(2)] * passes, batch_size=batch_size, device=device,
                                 num_glimpses=num_glimpses)
    out = []
    for t, viewpoint in enumerate(viewpoints):
        if t >= 21 and jitter > 0:
            limit = 1 - viewpoint.scales.view(-1, 1)
            shift = (torch.rand(batch_size, 2, device=device) * 2 - 1) * jitter
            viewpoint = Viewpoint(centers=(viewpoint.centers + shift).clamp(-limit, limit), scales=viewpoint.scales)
        out.append(viewpoint)
    return FixedSequence(out)


# Growing canvas


class GrowingCanvas:
    """A coarse uniform canvas that recruits fine cells where the eye zooms in (use-dependent cortical magnification:
    the cortical map of a trained finger grows, Merzenich and colleagues): each of the first `budget` glimpses of scale
    at most max_scale adds the cells of a uniform fine_grid×fine_grid grid whose centers lie in its crop and are not
    in the canvas yet. A new cell starts as the coarse canvas bilinearly interpolated at its center, the context it
    inherits; every later Write updates coarse and fine cells alike (canvas cells never interact directly, so any set
    of cells at any positions is a canvas CanViT can use). The segmentation is the fine cells' where they exist and
    the coarse cells' elsewhere."""

    def __init__(self, canvit: CanViT, *, coarse_grid_size: int, fine_grid_size: int, max_scale: float, budget: int,
                 device: torch.device) -> None:
        self.canvit, self.Gc, self.Gf = canvit, coarse_grid_size, fine_grid_size
        self.max_scale, self.budget = max_scale, budget
        self.coarse_positions = grid_coords(size=coarse_grid_size, device=device).flatten(0, 1)  # [Gc², 2]
        self.fine_positions = grid_coords(size=fine_grid_size, device=device).flatten(0, 1)  # [Gf², 2]
        self.present: Tensor | None = None  # [B, Gf²] fine cells in the canvas
        self.index: Tensor | None = None  # [B, n] their fine-grid indices, in canvas order
        self.recruited = 0

    def reset(self, batch_size: int) -> Tensor:
        device = self.coarse_positions.device
        self.present = torch.zeros(batch_size, self.Gf ** 2, dtype=torch.bool, device=device)
        self.index = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        self.recruited = 0
        return self.positions()

    def positions(self) -> Tensor:
        """[B, Gc² + n, 2]: the coarse cells, then the recruited fine cells."""
        assert self.index is not None
        B = self.index.shape[0]
        return torch.cat([self.coarse_positions.unsqueeze(0).expand(B, -1, -1), self.fine_positions[self.index]], dim=1)

    @property
    def num_cells(self) -> int:
        assert self.index is not None
        return self.Gc ** 2 + self.index.shape[1]

    def recruit(self, state: RecurrentState, viewpoint: Viewpoint) -> tuple[RecurrentState, Tensor]:
        """Before a glimpse: adds its fine cells if it qualifies; returns the state and the cell positions."""
        assert self.present is not None and self.index is not None
        if self.recruited >= self.budget or bool((viewpoint.scales > self.max_scale).any()):
            return state, self.positions()
        offsets = (self.fine_positions.unsqueeze(0) - viewpoint.centers.unsqueeze(1)).abs().amax(dim=-1)
        new = (offsets <= viewpoint.scales.view(-1, 1)) & ~self.present  # [B, Gf²]
        counts = new.sum(dim=1)
        assert bool((counts == counts[0]).all()), "every scene of the batch must recruit as many cells"
        added = new.nonzero()[:, 1].view(new.shape[0], -1)  # [B, n_new], ascending per scene
        R = self.canvit.config.num_canvas_registers
        coarse = state.canvas[:, R:R + self.Gc ** 2]
        B, _, D = coarse.shape
        grid = coarse.transpose(1, 2).reshape(B, D, self.Gc, self.Gc).float()
        xy = self.fine_positions[added].flip(-1).unsqueeze(1)  # [B, 1, n_new, 2]
        inherited = F.grid_sample(grid, xy, mode="bilinear", padding_mode="border", align_corners=False)[:, :, 0]
        state = RecurrentState(canvas=torch.cat([state.canvas, inherited.transpose(1, 2).to(state.canvas.dtype)], dim=1),
                               recurrent_cls=state.recurrent_cls)
        self.present |= new
        self.index = torch.cat([self.index, added], dim=1)
        self.recruited += 1
        return state, self.positions()

    def fine_map(self, fine_logits: Tensor) -> tuple[Tensor, Tensor]:
        """[B, n, C] logits of the recruited cells -> ([B, C, Gf, Gf] logits on the fine grid, [B, 1, Gf, Gf] mask)."""
        assert self.index is not None and self.present is not None
        B, n, C = fine_logits.shape
        full = fine_logits.new_zeros(B, self.Gf ** 2, C)
        full.scatter_(1, self.index.unsqueeze(-1).expand(-1, -1, C), fine_logits)
        mask = self.present.float().view(B, 1, self.Gf, self.Gf)
        return full.transpose(1, 2).reshape(B, C, self.Gf, self.Gf), mask
