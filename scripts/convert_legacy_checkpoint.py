"""Converts a CanViTForPretraining checkpoint in the canvit-pytorch 0.1 format (such as the Hub's experimental
canvitb16-add-vpe-pretrain-g128px-s1024px-sa1b-dv3b16-experimental, which the current package refuses to load) into
the current format, as a local directory that CanViTForPretraining.from_pretrained accepts.

The weight names are learned from the flagship, which the Hub serves in both formats (the 0.1 format at the revision
canvit_pytorch.hub.repos.OLD_FORMAT_REVISION): each 0.1 tensor is paired with the current tensor equal to it. The
conversion fails unless every weight finds exactly one counterpart and the architecture matches the flagship's.

    .venv/bin/python -P scripts/convert_legacy_checkpoint.py \
        --legacy-repo canvit/canvitb16-add-vpe-pretrain-g128px-s1024px-sa1b-dv3b16-experimental \
        --output /tmp/canvit-nadiedjoa-24/converted/canvitb16-s1024px-sa1b-experimental
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from canvit_pytorch.hub.repos import FLAGSHIP, OLD_FORMAT_REVISION
from canvit_pytorch.model import CanViTForPretraining

STANDARDIZER_FLAGS = {"cls_standardizers.{grid}._initialized": "teacher_cls_standardizer.fitted",
                      "scene_standardizers.{grid}._initialized": "teacher_patch_standardizer.fitted"}
"""Scalar flags, equal in every checkpoint, so they are paired by name instead of by value."""


@dataclass(frozen=True)
class Conversion:
    legacy_repo: str
    output: Path
    reference_repo: str = FLAGSHIP


def checkpoint(repo: str, revision: str | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    return config, load_file(hf_hub_download(repo, "model.safetensors", revision=revision))


def legacy_to_current_names(legacy: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
                            grid: int) -> dict[str, str]:
    flags = {old.format(grid=grid): new for old, new in STANDARDIZER_FLAGS.items()}
    names = dict(flags)
    for name, tensor in legacy.items():
        if name in flags:
            continue
        (match,) = [n for n, t in current.items() if t.shape == tensor.shape and t.dtype == tensor.dtype
                    and torch.equal(t, tensor)]
        names[name] = match
    assert sorted(names.values()) == sorted(current), "the reference's two formats do not pair one to one"
    return names


def main(conversion: Conversion) -> None:
    reference_legacy_config, reference_legacy = checkpoint(conversion.reference_repo, OLD_FORMAT_REVISION)
    reference_config, reference = checkpoint(conversion.reference_repo)
    (reference_grid,) = reference_legacy_config["canvas_patch_grid_sizes"]
    names = legacy_to_current_names(reference_legacy, reference, reference_grid)

    legacy_config, legacy = checkpoint(conversion.legacy_repo)
    (grid,) = legacy_config["canvas_patch_grid_sizes"]
    # Same architecture as the reference, up to options left at their defaults.
    defaults = {"read_scale_anchor": None, "canvas_proj_mode": "asymmetric"}
    extra = {k: v for k, v in legacy_config["model_config"].items() if k not in reference_legacy_config["model_config"]}
    assert extra == {k: defaults[k] for k in extra}, extra
    assert all(legacy_config["model_config"][k] == v for k, v in reference_legacy_config["model_config"].items())
    assert legacy_config["backbone_name"] == reference_legacy_config["backbone_name"]

    rename = lambda name: re.sub(rf"standardizers\.{grid}\.", f"standardizers.{reference_grid}.", name)  # noqa: E731
    converted = {names[rename(name)]: tensor for name, tensor in legacy.items()}
    assert set(converted) == set(reference), set(converted) ^ set(reference)
    config = {**reference_config, "teacher_patch_grid": grid}
    conversion.output.mkdir(parents=True, exist_ok=True)
    (conversion.output / "config.json").write_text(json.dumps(config, indent=2))
    save_file(converted, str(conversion.output / "model.safetensors"))
    model = CanViTForPretraining.from_pretrained(str(conversion.output))  # strict: every weight used, none missing
    print(f"Wrote {conversion.output}: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"teacher patch grid {model.teacher_patch_grid}, converted from {conversion.legacy_repo}")


if __name__ == "__main__":
    main(tyro.cli(Conversion))
