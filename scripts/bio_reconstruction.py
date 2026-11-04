"""The paper's reconstruction metric (cosine similarity of the canvas's and CLS token's predictions to the DINOv3
teacher's features, after every glimpse) with the primate-vision mechanisms of scripts/bio_vision.py; the options are
scripts/bio_rollouts.py's. Scenes, teacher and metric are scripts/reconstruction_rollouts.py's, so the released
settings give its numbers; the canvas is the teacher's 32×32 patch grid. An adaptive canvas's predictions are read
back at the teacher's patch centers through its warp.

    .venv/bin/python -P scripts/bio_reconstruction.py --policy microsaccades --num-glimpses 21 \
        --output results/bio/reconstruction/micro_s0.875_j0.125_c32_t21.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.data import DataLoader, Subset

from canvit_pytorch.benchmarks import ade20k
from canvit_pytorch.evaluate.tasks.reconstruction import METRICS, Scenes, cosine_similarities
from canvit_pytorch.hub.repos import FLAGSHIP
from canvit_pytorch.model import CanViTForPretraining
from canvit_pytorch.teacher import TEACHER_REPO, load_teacher
from canvit_pytorch.viewpoint import grid_coords

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
import bio_rollouts  # noqa: E402
from bio_rollouts import build_policy, load_retina  # noqa: E402
from bio_vision import AdaptiveCanvas, FreePositions, NoveltyWrites, ObjectWrites, image_edge_density, take_glimpse  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment(bio_rollouts.Experiment):
    canvas_grid_size: int = 32
    batch_size: int = 16


def main(exp: Experiment) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(exp.seed)
    device = torch.device(exp.device)
    model = CanViTForPretraining.from_pretrained(FLAGSHIP).to(device).eval()
    teacher = load_teacher(TEACHER_REPO, device)
    grid = model.teacher_patch_grid
    assert exp.canvas_grid_size == grid, "the metric compares the canvas cell by cell with the teacher's patch grid"
    assert exp.density in (None, "glimpse_edges"), "the other densities need labels or a segmentation probe"
    T = exp.num_glimpses
    free = FreePositions(model.canvit)
    retina, shape = load_retina(exp, model.canvit, device, free)
    novelty = (NoveltyWrites(model.canvit, mode=exp.novelty, strength=exp.novelty_strength, start=exp.novelty_from,
                             num_glimpses=T, device=device) if exp.novelty is not None else None)
    objects = (ObjectWrites(model.canvit, canvas_grid_size=grid, threshold=exp.object_writes_threshold,
                            start=exp.object_writes_from, num_glimpses=T, device=device)
               if exp.object_writes_threshold is not None else None)
    dataset = Scenes(root=ade20k.dataset_root(), size_px=grid * teacher.patch_size)
    if exp.num_images is not None:
        dataset = Subset(dataset, range(exp.num_images))

    sums = torch.zeros(T, len(METRICS), dtype=torch.float64)
    start = time.monotonic()
    loader = DataLoader(dataset, batch_size=exp.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch, images in enumerate(loader):
            images = images.to(device)
            B = images.shape[0]
            with torch.autocast(device_type=device.type, enabled=False):
                target = teacher(images)
            policy = build_policy(exp, B, device)
            state = (model.canvit.init_state(batch_size=B, canvas_grid_size=grid) if shape is None
                     else shape.init_state(model.canvit, B))
            canvas, free.canvas = None, None
            if shape is not None:
                free.canvas = shape.reset(B)
            points = grid_coords(size=grid, device=device).flatten(0, 1)
            for t in range(T):
                viewpoint = policy.step(t, state)
                glimpse, free.glimpse = take_glimpse(images, viewpoint, retina)
                if t == 0 and exp.density == "glimpse_edges":
                    size = images.shape[-1]
                    edges = torch.nn.functional.interpolate(image_edge_density(glimpse).unsqueeze(1), size=(size, size),
                                                            mode="bilinear", align_corners=False)[:, 0]
                    canvas = AdaptiveCanvas(edges, canvas_grid_size=grid, mix=exp.density_mix)
                    free.canvas = canvas.positions()
                if novelty is not None:
                    novelty.before_glimpse(t)
                if objects is not None:
                    objects.before_glimpse(t, viewpoint, state.canvas)
                state = model.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint).state
                if shape is not None and shape.adaptive and t == 0:
                    state, free.canvas = shape.adapt(state, model.canvit)
                prediction = model.predict_teacher_patches(state.canvas)  # [B, G*G, D]
                if shape is not None:
                    prediction = shape.read(prediction, points)
                elif canvas is not None:
                    cells = prediction.transpose(1, 2).reshape(B, -1, grid, grid)
                    prediction = canvas.read_back(cells, grid).flatten(2).transpose(1, 2)
                scene_norm, scene_raw = cosine_similarities(
                    prediction=prediction, raw_target=target.patches.float(), standardizer=model.teacher_patch_standardizer,
                )
                cls_norm, cls_raw = cosine_similarities(
                    prediction=model.predict_teacher_cls(state.recurrent_cls), raw_target=target.cls.float(),
                    standardizer=model.teacher_cls_standardizer,
                )
                sums[t] += torch.tensor([scene_raw, cls_raw, scene_norm, cls_norm], dtype=torch.float64) * B
            if batch % 10 == 0:
                log.info("%d/%d images", (batch + 1) * exp.batch_size, len(dataset))

    means = sums / len(dataset)
    result = {
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(exp).items()},
        "pretrained_repo": FLAGSHIP,
        "per_timestep": [{"t": t, **dict(zip(METRICS, row.tolist(), strict=True))} for t, row in enumerate(means)],
        "n_images": len(dataset),
        "wall_time_seconds": time.monotonic() - start,
    }
    exp.output.parent.mkdir(parents=True, exist_ok=True)
    exp.output.write_text(json.dumps(result, indent=1))
    log.info("scene_cos_norm by glimpse: %s", " ".join(f"{row['scene_cos_norm']:.4f}" for row in result["per_timestep"]))


if __name__ == "__main__":
    main(tyro.cli(Experiment))
