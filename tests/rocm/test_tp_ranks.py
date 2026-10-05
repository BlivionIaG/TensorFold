"""Tensor parallel on real GPUs (TENSORFOLD_TP_WORLD ranks, TENSORFOLD_TP_MODEL) gives one GPU's greedy ids."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

MODEL = os.environ.get("TENSORFOLD_TP_MODEL", "")
WORLD = int(os.environ.get("TENSORFOLD_TP_WORLD", "2"))
PORT = int(os.environ.get("TENSORFOLD_TP_PORT", "29671"))
PROMPTS = [[(index * 7 + row * 13) % 5000 + 11 for index in range(300)] for row in range(2)]
STEPS = 12

RANK = """
import json, sys, torch
from tensorfold.rocm.build import gfx_name
from tensorfold.rocm.qwen import Engine, activation_dtype, load, slice_for_tp
model_dir, world, rank, port, prompts, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \\
    json.loads(sys.argv[5]), int(sys.argv[6])
rccl = None
if world > 1:
    from tensorfold.rocm.comm import RCCL
    torch.cuda.set_device(rank)
    rccl = RCCL(rank, world, "127.0.0.1", port)
    rccl.ready("startup")
model = load(model_dir)
if rccl is not None:
    slice_for_tp(model, rank, world)
engine = Engine(model, dtype=activation_dtype(gfx_name()), rccl=rccl)
print("IDS " + json.dumps(engine.generate(prompts, steps)), flush=True)
"""


def _launch(world: int, port: int) -> list[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-c", RANK, MODEL, str(world), str(rank), str(port),
                              json.dumps(PROMPTS), str(STEPS)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True) for rank in range(world)]


def _ids(procs: list[subprocess.Popen]) -> list:
    out = []
    for proc in procs:
        text, _ = proc.communicate(timeout=1800)
        lines = [line for line in text.splitlines() if line.startswith("IDS ")]
        assert proc.returncode == 0 and lines, text[-3000:]
        out.append(json.loads(lines[-1][4:]))
    return out


def test_ranks_match_one_gpu():
    if not MODEL:
        pytest.skip("set TENSORFOLD_TP_MODEL")
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} visible GPUs")
    (one,) = _ids(_launch(1, PORT))
    ranks = _ids(_launch(WORLD, PORT))
    assert all(ids == ranks[0] for ids in ranks), "ranks disagree"
    assert ranks[0] == one, f"tp={WORLD} {ranks[0]} vs one GPU {one}"
