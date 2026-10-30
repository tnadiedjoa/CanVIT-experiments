"""Collects the runs of the official evaluation CLI (`python -m canvit_pytorch.evaluate`, saved as .pt files in
results/official/ by scripts/run_official_queue.sh) into results/official.json, with the analytic forward FLOPs of
each model (canvit_pytorch.flops), so that plots and tables read plain JSON.

    HF_TOKEN_PATH=~/.cache/huggingface/token .venv/bin/python -P scripts/export_official.py
"""

import json
from pathlib import Path

import torch
from canvit_core.config import CanViTConfig
from transformers import AutoConfig

from canvit_pytorch.flops import dinov3_flops, glimpse_flops
from canvit_pytorch.hub.repos import RELEASED_GLIMPSE_SIZE_PX

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "results" / "official"


def per_glimpse(mious: dict[str, float]) -> list[float]:
    """{"t0": fraction, ...} -> mIoU in % in glimpse order."""
    return [100 * mious[f"t{t}"] for t in range(len(mious))]


def main() -> None:
    runs = {path.stem: torch.load(path, weights_only=False) for path in sorted(OFFICIAL.glob("*.pt"))}
    dinov3, reconstruction, canvit = [], [], []
    dinov3_configs: dict[str, AutoConfig] = {}
    for name, run in runs.items():
        metadata = {k: v for k, v in run["metadata"].items() if k != "provenance"}
        config = metadata["config"]
        if "dinov3_repo" in metadata:
            repo = metadata["dinov3_repo"]
            dinov3_configs.setdefault(repo, AutoConfig.from_pretrained(repo))
            (miou,) = per_glimpse(run["mious"])
            dinov3.append({"name": name, "variant": config["variant"], "input_size_px": config["input_size_px"],
                           "miou": miou, "gflops": dinov3_flops(dinov3_configs[repo], input_size_px=config["input_size_px"]) / 1e9,
                           "dinov3_repo": repo, "probe_repo": metadata["probe_repo"], "n_images": metadata["n_images"]})
        elif "per_timestep" in run:
            reconstruction.append({"name": name, **config["episode"], "pretrained_repo": config["pretrained_repo"],
                                   "per_timestep": run["per_timestep"], "n_images": run["n_images"]})
        else:
            canvit.append({"name": name, **config["episode"], "probe_repo": config["probe_repo"],
                           "miou": per_glimpse(run["mious"]), "n_images": metadata["n_images"]})
    canvit_config = CanViTConfig()
    glimpse_gflops = {grid: glimpse_flops(canvit_config, glimpse_size_px=RELEASED_GLIMPSE_SIZE_PX, canvas_grid_size=grid) / 1e9
                      for grid in (32, 64, 128)}
    output = ROOT / "results" / "official.json"
    output.write_text(json.dumps({"dinov3_ade20k": dinov3, "reconstruction": reconstruction, "canvit_ade20k": canvit,
                                  "canvit_glimpse_gflops": glimpse_gflops}, indent=1))
    print(f"Wrote {output}: {len(dinov3)} DINOv3, {len(reconstruction)} reconstruction, {len(canvit)} CanViT runs")


if __name__ == "__main__":
    main()
