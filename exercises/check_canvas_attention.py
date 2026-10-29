"""Checks a from-scratch Canvas Attention Read and Write, with SR-RoPE, against the released CanViT-B.

The released model takes a glimpse of the full scene, then a second glimpse while hooks record the inputs and outputs
of its first Read and first Write. The functions under test get the module (for its weights), the token streams and
the raw scene positions, and must return the same residuals.

    .venv/bin/python exercises/check_canvas_attention.py                      # exercises/canvas_attention.py
    .venv/bin/python exercises/check_canvas_attention.py --module exercises/canvas_attention_solution.py
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from PIL import Image

from canvit_pytorch import CanViTForPretraining, Viewpoint, sample_at_viewpoint
from canvit_pytorch.hub.repos import FLAGSHIP, RELEASED_CANVAS_GRID_SIZE, RELEASED_GLIMPSE_SIZE_PX, RELEASED_SCENE_SIZE_PX
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.viewpoint import grid_coords, viewpoint_grid_coords

IMAGE = Path(__file__).resolve().parents[1] / "external/CanViT/canvit-pytorch/test_data/Places365_IMG_9600.jpeg"
TOLERANCE = 1e-4


@dataclass(frozen=True)
class Check:
    module: Path = Path(__file__).with_name("canvas_attention.py")


@torch.inference_mode()
def main(check: Check) -> None:
    spec = importlib.util.spec_from_file_location("attempt", check.module)
    assert spec is not None and spec.loader is not None, check.module
    attempt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attempt)

    canvit = CanViTForPretraining.from_pretrained(FLAGSHIP).canvit.eval()
    scene = preprocess(RELEASED_SCENE_SIZE_PX)(Image.open(IMAGE).convert("RGB")).unsqueeze(0)
    viewpoints = [Viewpoint.full_scene(batch_size=1, device=None),
                  Viewpoint(centers=torch.tensor([[-0.3, 0.2]]), scales=torch.tensor([0.4]))]
    recorded: dict[str, tuple[dict, torch.Tensor]] = {}
    hooks = [module.register_forward_hook(lambda m, args, kwargs, out, name=name: recorded.__setitem__(name, (kwargs, out)),
                                          with_kwargs=True)
             for name, module in (("read", canvit.canvas_reads[0]), ("write", canvit.canvas_writes[0]))]
    state = canvit.init_state(batch_size=1, canvas_grid_size=RELEASED_CANVAS_GRID_SIZE)
    for viewpoint in viewpoints:  # the hooks keep the second glimpse's call
        glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX)
        state = canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
    for hook in hooks:
        hook.remove()

    glimpse_grid = RELEASED_GLIMPSE_SIZE_PX // canvit.patch_size
    glimpse_positions = viewpoint_grid_coords(viewpoints[-1], size=glimpse_grid).flatten(1, 2)
    canvas_positions = grid_coords(size=RELEASED_CANVAS_GRID_SIZE, device=None).flatten(0, 1).unsqueeze(0)
    base = canvit.backbone.spec.rope_base

    read_inputs, read_output = recorded["read"]
    mine = attempt.canvas_attention_read(
        canvit.canvas_reads[0], glimpse_tokens=read_inputs["query"], canvas=read_inputs["kv"],
        glimpse_positions=glimpse_positions, canvas_positions=canvas_positions, rope_base=base,
    )
    report("Canvas Attention Read", mine, read_output)
    write_inputs, write_output = recorded["write"]
    mine = attempt.canvas_attention_write(
        canvit.canvas_writes[0], canvas=write_inputs["query"], glimpse_tokens=write_inputs["kv"],
        canvas_positions=canvas_positions, glimpse_positions=glimpse_positions, rope_base=base,
    )
    report("Canvas Attention Write", mine, write_output)


def report(name: str, mine: torch.Tensor, released: torch.Tensor) -> None:
    assert mine.shape == released.shape, f"{name}: shape {tuple(mine.shape)}, released {tuple(released.shape)}"
    error = (mine - released).abs().max().item()
    print(f"{name}: max |difference| = {error:.2e} -> {'OK' if error < TOLERANCE else 'MISMATCH'}")


if __name__ == "__main__":
    main(tyro.cli(Check))
