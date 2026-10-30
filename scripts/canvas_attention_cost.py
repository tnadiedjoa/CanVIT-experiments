"""Cost of one CanViT-B glimpse with the released asymmetric Canvas Attention and with learned projections on both
sides (the paper's QKVO ablation), across canvas grids: analytic FLOPs (canvit_pytorch.flops) and measured latency.

Latency follows the protocol of canvit_pytorch.evaluate.latency (untimed warmup, then timed iterations between device
synchronizations), at each batch size, on randomly initialized weights, which do not change the cost. Run on an idle
GPU.

    .venv/bin/python scripts/canvas_attention_cost.py --output results/canvas_attention_cost.json
"""

import dataclasses
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_core.config import CanViTConfig

from canvit_pytorch.flops import canvas_attention_read_flops, canvas_attention_write_flops, glimpse_flops
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX
from canvit_pytorch.model import CanViT
from canvit_pytorch.viewpoint import Viewpoint


@dataclass(frozen=True)
class Cost:
    output: Path
    canvas_grid_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    batch_sizes: tuple[int, ...] = (1, 32)
    """1 is the paper's latency protocol, launch-bound on a GPU; a large batch shows the compute cost."""
    device: str = "cuda"
    warmup_iters: int = 5
    timed_iters: int = 50


@torch.inference_mode()
def latency_ms(canvit: CanViT, *, canvas_grid_size: int, batch_size: int, device: torch.device, cost: Cost) -> list[float]:
    glimpse = torch.randn(batch_size, 3, RELEASED_GLIMPSE_SIZE_PX, RELEASED_GLIMPSE_SIZE_PX, device=device)
    viewpoint = Viewpoint(centers=torch.zeros(batch_size, 2, device=device),
                          scales=torch.full((batch_size,), 0.5, device=device))
    state = canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
    times = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i in range(cost.warmup_iters + cost.timed_iters):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
            torch.cuda.synchronize(device)
            if i >= cost.warmup_iters:
                times.append(1000 * (time.perf_counter() - start))
    return times


def main(cost: Cost) -> None:
    device = torch.device(cost.device)
    rows = []
    for projections in ("asymmetric", "qkvo"):
        config = dataclasses.replace(CanViTConfig(), canvas_projections=projections)
        canvit = CanViT(config).to(device).eval()
        for grid in cost.canvas_grid_sizes:
            geometry = dict(glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX, canvas_grid_size=grid)
            for batch_size in cost.batch_sizes:
                times = latency_ms(canvit, canvas_grid_size=grid, batch_size=batch_size, device=device, cost=cost)
                rows.append({
                    "canvas_projections": projections, "canvas_grid_size": grid, "batch_size": batch_size,
                    "glimpse_gflops": glimpse_flops(config, **geometry) / 1e9,
                    "read_write_pair_gflops": (canvas_attention_read_flops(config, **geometry)
                                               + canvas_attention_write_flops(config, **geometry)) / 1e9,
                    "latency_ms_min": min(times), "latency_ms_median": statistics.median(times),
                })
                print(rows[-1])
        del canvit
        torch.cuda.empty_cache()
    cost.output.parent.mkdir(parents=True, exist_ok=True)
    cost.output.write_text(json.dumps({"device": torch.cuda.get_device_name(device), "precision": "amp-bf16",
                                       "rows": rows}, indent=1))
    print(f"Wrote {cost.output}")


if __name__ == "__main__":
    main(tyro.cli(Cost))
