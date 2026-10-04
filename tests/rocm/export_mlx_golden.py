"""Write the golden fixture from MLX. Run on Apple Silicon, not on ROCm.

    python tests/rocm/export_mlx_golden.py /path/to/Qwen3.5-0.8B-MLX-8bit

TensorFold's Qwen row decoder refuses a tied embedding, and Qwen3.5-0.8B ties the head to the
embedding. The fixture is mlx-lm's forward of that checkpoint, which is the MLX implementation
that actually runs these weights. ``lane_kernels=off`` cannot load them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROMPT = [151643, 198, 15, 279, 1196]  # a short fixed id list, not a chat string


def _hidden_and_logits(model, tokens):
    """Last-layer hidden states and the last position's logits from an mlx-lm Qwen module."""

    import mlx.core as mx

    language = getattr(model, "language_model", model)
    core = language.model
    cache = language.make_cache()
    hidden = core(tokens, cache=cache)
    tied = bool(getattr(getattr(language, "args", None), "tie_word_embeddings", False))
    if tied:
        logits = core.embed_tokens.as_linear(hidden[:, -1:])
    else:
        logits = language.lm_head(hidden[:, -1:])
    mx.eval(hidden, logits)
    return hidden, logits


def main(model_dir: Path) -> None:
    import mlx.core as mx
    from mlx_lm.utils import load

    model, _tokenizer = load(str(model_dir))
    hidden, logits = _hidden_and_logits(model, mx.array(PROMPT)[None])
    if int(hidden.shape[-1]) < 32 or int(logits.shape[-1]) < 1000:
        raise SystemExit(f"unexpected shapes hidden {tuple(hidden.shape)} logits {tuple(logits.shape)}")
    out = Path(__file__).parent / "fixtures" / "qwen35_0_8b_prompt.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tokens=np.asarray(PROMPT, dtype=np.int64)[None],
        hidden=np.array(hidden[:, -1].astype(mx.float32)),
        logits=np.array(logits.reshape(-1).astype(mx.float32)),
        model_dir=str(model_dir),
    )
    print(f"wrote {out} hidden {tuple(hidden.shape)} logits {tuple(logits.shape)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_mlx_golden.py MODEL_DIR")
    main(Path(sys.argv[1]))
