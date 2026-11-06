"""A memory of slots written by cross-attention from a stream of token chunks: does the canvas's failure (a softmax
Write cannot write nothing, so a memory trained on short sequences drifts on long ones) carry over to a memory that
has no geometry at all?

Task. A stream is a sequence of chunks of 8 tokens. An informative chunk holds 8 facts, each a token embedding the
sum of a key's and a value's (64 keys, 64 values). A noise chunk holds 8 tokens of a separate noise vocabulary (64 ids),
so a single token tells the two apart. After every chunk the model gives, for every key stored so far, its value:
a query embedding of the key reads the memory (cross-attention over the slots, then an MLP and a linear map to the 64
values). The loss is the cross-entropy over those queries, averaged over the chunks; nothing asks the model to predict
what it has not been shown. Training streams have 4 chunks: the first informative, each other one informative with
probability 1/2 (keys never repeat within a stream). At test (`--policy noise`), the first chunk is informative and
every later one is noise, up to 257 chunks: a memory that ignores what is irrelevant keeps its accuracy flat.
`--policy repeat` shows the informative chunk again at every step instead, the analogue of RFS.

Model. scripts/sink_memory.py's, without positions: 32 memory slots of 128 dims, each with its own learned initial
value; per chunk, the 8 token embeddings and two positionless registers go through four transformer blocks; after blocks 1 and 3 the chunk Reads the memory, after blocks 2 and 4
the memory Writes from the chunk, with the Write kinds of scripts/sink_memory.py (softmax, out, null, null_open,
sigmoid, gate, delta). The registers play the part of CanViT's positionless prefix: a place where a slot's forced
attention can go.

`--penalty λ` charges, as sink_memory's `--outside-penalty`, the Writes' residual after removing its mean over the
channels, squared, relative to the slot's own (centered, not differentiated): with `--penalty-on noise` during noise
chunks only (the analogue of outside the crop: it uses the labels), with `all` during every chunk (a cost on writing
at all, which needs no labels), with `prefix` during every chunk but only on the part of the residual read from the
registers (no labels either, but aimed at the forced write).

At test, `--test-writes abstain` drops every Write during noise chunks, the analogue of local Writes: it shows whether
the writes during noise chunks make the drift. The evaluation reports per chunk the accuracy on the stored keys, the
norm of the slots' summed residual (both Writes) before and after removing its mean over the channels (which every
LayerNorm reading the memory removes), the Writes' attention mass on the registers, and the memory's norm before and after removing each
slot's mean (what a LayerNorm reading the memory divides by).

    .venv/bin/python -P scripts/stream_memory.py train --write softmax --output probes/stream_memory/softmax_s0.pt
    .venv/bin/python -P scripts/stream_memory.py evaluate --checkpoint probes/stream_memory/softmax_s0.pt \
        --output results/stream_memory/softmax_s0_noise.json
"""

import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scripts run with python -P
from sink_memory import Attention, Block, Write, WriteKind  # noqa: E402

log = logging.getLogger("stream_memory")
CHUNK, KEYS, VALUES, NOISE = 8, 64, 64, 64
SLOTS, DIM, HEADS, REGISTERS = 32, 128, 4, 2


def no_rope(x: Tensor, pos) -> Tensor:
    return x


class StreamModel(nn.Module):
    def __init__(self, write: WriteKind):
        super().__init__()
        d = DIM
        self.write_kind = write
        self.key_embed, self.value_embed = nn.Embedding(KEYS, d), nn.Embedding(VALUES, d)
        self.noise_embed = nn.Embedding(NOISE, d)
        self.registers = nn.Parameter(torch.randn(REGISTERS, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, HEADS) for _ in range(4))
        self.read_norm_q, self.read_norm_kv = (nn.ModuleList(nn.LayerNorm(d) for _ in range(2)),
                                               nn.ModuleList(nn.LayerNorm(d) for _ in range(2)))
        self.reads = nn.ModuleList(Attention(d, HEADS) for _ in range(2))
        self.write_norm_q, self.write_norm_kv = (nn.ModuleList(nn.LayerNorm(d) for _ in range(2)),
                                                 nn.ModuleList(nn.LayerNorm(d) for _ in range(2)))
        self.writes = nn.ModuleList(Write(d, HEADS, write, REGISTERS) for _ in range(2))
        self.memory_init = nn.Parameter(torch.randn(SLOTS, d))
        self.query = nn.Embedding(KEYS, d)
        self.out_norm_q, self.out_norm_m, self.out_norm_h = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.out_attn = Attention(d, HEADS)
        self.out_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out_head = nn.Linear(d, VALUES)

    def init_memory(self, batch: int) -> Tensor:
        return self.memory_init[None].expand(batch, -1, -1).clone()

    def answer(self, memory: Tensor, keys: Tensor) -> Tensor:
        """Logits [B, Q, VALUES] of the values of keys [B, Q], read from memory [B, SLOTS, D]."""
        q = self.query(keys)
        h = q + self.out_attn(self.out_norm_q(q), self.out_norm_m(memory), no_rope, None, None)
        h = h + self.out_mlp(h)
        return self.out_head(self.out_norm_h(h))

    def tokens(self, keys: Tensor, values: Tensor, noise: Tensor, informative: Tensor) -> Tensor:
        """Token embeddings [B, CHUNK, D] of one chunk per stream: facts where informative [B], else noise."""
        facts = self.key_embed(keys) + self.value_embed(values)
        return torch.where(informative[:, None, None], facts, self.noise_embed(noise))

    def step(self, memory: Tensor, tokens: Tensor, keep: Tensor | None = None
             ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """One chunk of token embeddings [B, CHUNK, D]: the new memory, the Writes' residuals [B, SLOTS, D] and their
        attention mass on the registers [B, SLOTS]. keep [B] (bool) drops the Writes of the streams where it is
        False."""
        B = tokens.shape[0]
        x = torch.cat([self.registers.expand(B, -1, -1), tokens], 1)
        residuals, masses = [], []
        for i, block in enumerate(self.blocks):
            x = block(x, no_rope, None)
            j = i // 2
            if i % 2 == 0:
                x = x + self.reads[j](self.read_norm_q[j](x), self.read_norm_kv[j](memory), no_rope, None, None)
            else:
                r, m = self.writes[j](self.write_norm_q[j](memory), self.write_norm_kv[j](x), no_rope, None, None,
                                      raw=memory)
                if keep is not None:
                    r = r * keep[:, None, None].to(r.dtype)
                memory = memory + r
                residuals.append(r)
                masses.append(m)
        return memory, residuals, masses


def make_streams(batch: int, chunks: int, informative: Tensor, generator: torch.Generator, device
                 ) -> tuple[Tensor, Tensor, Tensor]:
    """keys, values and noise ids [B, T, CHUNK] of streams whose informative chunks are informative [B, T] (bool; at
    most KEYS / CHUNK per stream); keys never repeat within a stream."""
    keys = torch.argsort(torch.rand(batch, KEYS, generator=generator), dim=1)
    assert int(informative.sum(1).max()) * CHUNK <= KEYS
    # the i-th informative chunk of a stream gets the i-th block of CHUNK keys of its permutation
    rank = (informative.cumsum(1) - 1).clamp(min=0)  # [B, T]
    idx = rank[..., None] * CHUNK + torch.arange(CHUNK)  # [B, T, CHUNK]
    chunk_keys = torch.gather(keys, 1, idx.flatten(1).clamp(max=KEYS - 1)).view(batch, chunks, CHUNK)
    values = torch.randint(VALUES, (batch, chunks, CHUNK), generator=generator)
    noise = torch.randint(NOISE, (batch, chunks, CHUNK), generator=generator)
    return chunk_keys.to(device), values.to(device), noise.to(device)


@dataclass
class Train:
    output: Path
    write: WriteKind = "softmax"
    steps: int = 10000
    batch_size: int = 128
    chunks: int = 4
    """Chunks per training stream; past the 8th informative chunk (64 keys) the rest is noise."""
    penalty: float = 0.0
    """When > 0, adds this times the Writes' residual after removing its mean over the channels, squared, over the
    slot's own (centered, not differentiated), averaged over the slots and both Writes."""
    penalty_on: Literal["noise", "all", "prefix"] = "noise"
    """When the penalty charges: during noise chunks only (it uses the labels), during every chunk (label free), or
    during every chunk but only the part of the residual read from the registers (label free, aimed at the forced
    write)."""
    lr: float = 1e-3
    warmup: int = 500
    seed: int = 0


@dataclass
class Evaluate:
    checkpoint: Path
    output: Path
    policy: Literal["noise", "repeat"] = "noise"
    chunks: int = 257
    test_writes: Literal["released", "abstain"] = "released"
    num_streams: int = 2000
    batch_size: int = 250
    seed: int = 0


def load(checkpoint: Path, device) -> tuple[StreamModel, dict]:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = StreamModel(saved["config"]["write"]).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    return model, saved


def train(cfg: Train) -> None:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    dev = torch.device("cuda")
    model = StreamModel(cfg.write).to(dev)
    for write in model.writes:
        write.keep_prefix_part = cfg.penalty > 0 and cfg.penalty_on == "prefix"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / cfg.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(i, cfg.steps) / cfg.steps)))
    start, history = time.time(), []
    B, T = cfg.batch_size, cfg.chunks
    for step in range(cfg.steps):
        informative = torch.rand(B, T, generator=gen) < 0.5
        informative[:, 0] = True
        informative &= informative.cumsum(1) <= KEYS // CHUNK  # longer streams run out of keys: noise after that
        keys, values, noise = make_streams(B, T, informative, gen, dev)
        informative = informative.to(dev)
        memory = model.init_memory(B)
        loss, cost, correct, asked = 0.0, 0.0, 0.0, 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(T):
                tokens = model.tokens(keys[:, t], values[:, t], noise[:, t], informative[:, t])
                memory, residuals, _ = model.step(memory, tokens)
                if cfg.penalty > 0:
                    charged = ~informative[:, t] if cfg.penalty_on == "noise" else torch.ones_like(informative[:, t])
                    c = memory.detach().float()
                    own = (c - c.mean(-1, keepdim=True)).square().sum(-1).clamp(min=1e-6)  # [B, SLOTS]
                    for r in (residuals if cfg.penalty_on != "prefix" else [w.prefix_part for w in model.writes]):
                        r = r.float()
                        ratio = ((r - r.mean(-1, keepdim=True)).square().sum(-1) / own).mean(1)  # [B]
                        cost = cost + (ratio * charged).sum() / charged.sum().clamp(min=1) / len(residuals)
                stored = informative[:, : t + 1, None].expand(-1, -1, CHUNK).flatten(1)  # [B, (t+1)*CHUNK]
                logits = model.answer(memory, keys[:, : t + 1].flatten(1)).float()
                ce = F.cross_entropy(logits.transpose(1, 2), values[:, : t + 1].flatten(1), reduction="none")
                loss = loss + (ce * stored).sum() / stored.sum()
                correct += ((logits.argmax(-1) == values[:, : t + 1].flatten(1)) & stored).sum().item()
                asked += stored.sum().item()
        loss, cost = loss / T, cost / T
        opt.zero_grad(set_to_none=True)
        (loss + cfg.penalty * cost).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == cfg.steps - 1:
            history.append({"step": step, "loss": loss.item(), "penalty": float(cost), "accuracy": correct / asked})
            log.info("step %d loss %.4f accuracy %.3f penalty %.4f %.0f s", step, loss.item(), correct / asked,
                     float(cost), time.time() - start)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg) | {"output": str(cfg.output)}, "state_dict": model.state_dict(),
                "history": history}, cfg.output)
    log.info("saved %s", cfg.output)


@torch.inference_mode()
def evaluate(cfg: Evaluate) -> None:
    dev = torch.device("cuda")
    model, saved = load(cfg.checkpoint, dev)
    gen = torch.Generator().manual_seed(cfg.seed)
    T = cfg.chunks
    names = ("accuracy", "loss", "residual", "residual_centered", "register_mass", "memory_norm",
             "memory_centered_norm")
    sums = {k: torch.zeros(T, dtype=torch.float64) for k in names}
    for first in range(0, cfg.num_streams, cfg.batch_size):
        B = min(cfg.batch_size, cfg.num_streams - first)
        informative = torch.zeros(B, T, dtype=torch.bool)
        informative[:, 0] = True
        keys, values, noise = make_streams(B, T, informative, gen, dev)
        if cfg.policy == "repeat":
            informative[:] = True
            keys, values = keys[:, :1].expand(-1, T, -1), values[:, :1].expand(-1, T, -1)
        informative = informative.to(dev)
        memory = model.init_memory(B)
        for t in range(T):
            keep = None
            if cfg.test_writes == "abstain" and cfg.policy == "noise" and t > 0:
                keep = torch.zeros(B, dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.tokens(keys[:, t], values[:, t], noise[:, t], informative[:, t])
                memory, residuals, masses = model.step(memory, tokens, keep)
                logits = model.answer(memory, keys[:, 0]).float()
            sums["accuracy"][t] += (logits.argmax(-1) == values[:, 0]).float().mean(1).sum().item()
            sums["loss"][t] += F.cross_entropy(logits.transpose(1, 2), values[:, 0], reduction="none").mean(1).sum().item()
            r = sum(r.float() for r in residuals)  # [B, SLOTS, D]
            sums["residual"][t] += r.norm(dim=-1).mean(1).sum().item()
            sums["residual_centered"][t] += (r - r.mean(-1, keepdim=True)).norm(dim=-1).mean(1).sum().item()
            sums["register_mass"][t] += (sum(masses) / len(masses)).float().mean(1).sum().item()
            m = memory.float()
            sums["memory_norm"][t] += m.norm(dim=-1).mean(1).sum().item()
            sums["memory_centered_norm"][t] += (m - m.mean(-1, keepdim=True)).norm(dim=-1).mean(1).sum().item()
    result = {k: (v / cfg.num_streams).tolist() for k, v in sums.items()}
    result["config"] = {k: str(v) for k, v in asdict(cfg).items()}
    result["train_config"] = {k: str(v) for k, v in saved["config"].items()}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=1))
    acc = result["accuracy"]
    print("accuracy by chunk:", " ".join(f"{acc[t]:.3f}" for t in range(0, T, max(1, T // 12))))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd = tyro.extras.subcommand_cli_from_dict({"train": Train, "evaluate": Evaluate})
    {Train: train, Evaluate: evaluate}[type(cmd)](cmd)
