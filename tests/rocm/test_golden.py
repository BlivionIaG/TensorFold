"""The auto schedule against a Mac MLX hidden state and last-token logits.

The fixture is produced by ``tests/rocm/export_mlx_golden.py`` on Apple Silicon. Set
``TENSORFOLD_GOLDEN_MODEL`` to that same checkpoint. Row invariance is not this check.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

import numpy as np  # noqa: E402

from tensorfold.rocm.build import gfx_name  # noqa: E402
from tensorfold.rocm.qwen import Engine, activation_dtype, load  # noqa: E402
from tensorfold.rocm.qwen_math import _project, forward_hidden  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_0_8b_prompt.npz"
# Against mlx-lm 0.31.3 on this prompt, schedule auto. The Mac row decoder refuses this tied checkpoint,
# so the reference is mlx-lm's forward.
# W7800 gfx1100 bf16, 2026-10-01: hidden max abs 0.496 (mean 0.050), logit max abs 0.219, top-5 ids matched.
# V620 gfx1030 fp16, 2026-10-02: hidden max abs 0.551, logit max abs 0.163, same argmax.
HIDDEN_ATOL = {torch.bfloat16: 0.55, torch.float16: 0.6}
LOGIT_ATOL = 0.25


def test_auto_matches_the_mlx_golden():
    model_dir = os.environ.get("TENSORFOLD_GOLDEN_MODEL")
    if not FIXTURE.is_file() or not model_dir:
        pytest.skip("export the MLX fixture and set TENSORFOLD_GOLDEN_MODEL")
    data = np.load(FIXTURE)
    model = load(model_dir)
    engine = Engine(model, schedule="auto", dtype=activation_dtype(gfx_name()))
    tokens = torch.tensor(data["tokens"], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        hidden, _ = forward_hidden(model, tokens, None, engine.linear, 0, engine.dtype)
        logits = _project(hidden[:, -1], model.output_head(), engine.linear).float().cpu().numpy()
    got_h = hidden[:, -1].float().cpu().numpy()
    got_l = logits
    hidden_gap = float(np.max(np.abs(got_h - data["hidden"])))
    logit_gap = float(np.max(np.abs(got_l - data["logits"])))
    print(f"golden hidden_max_abs={hidden_gap:.6g} logit_max_abs={logit_gap:.6g}", flush=True)
    assert int(np.argmax(got_l)) == int(np.argmax(np.asarray(data["logits"]).reshape(-1)))
    assert hidden_gap <= HIDDEN_ATOL[engine.dtype]
    assert logit_gap <= LOGIT_ATOL
