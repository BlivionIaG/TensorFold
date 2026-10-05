"""Mixed-width MLX conversions: a tensor the config's quantization names keeps its own bits and group."""

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorfold.rocm.model.checkpoint import _packed, _Shards  # noqa: E402


def _file(tmp_path):
    key = "language_model.model.embed_tokens"
    rows, k, bits, group = 8, 256, 4, 64
    safetensors_torch.save_file({
        key + ".weight": torch.zeros((rows, k * bits // 32), dtype=torch.int32),
        key + ".scales": torch.ones((rows, k // group), dtype=torch.bfloat16),
        key + ".biases": torch.zeros((rows, k // group), dtype=torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))
    return key, tmp_path / "model.safetensors"


def test_a_named_tensor_loads_at_its_own_width(tmp_path):
    key, path = _file(tmp_path)
    quant = {"bits": 3, "group_size": 64, "mode": "affine", key: {"bits": 4, "group_size": 64, "mode": "affine"}}
    packed = _packed(_Shards([path], quant=quant), key, 3, 64, torch.device("cpu"))
    assert (packed.bits, packed.group) == (4, 64)


def test_an_override_without_a_mode_is_affine(tmp_path):
    key, path = _file(tmp_path)
    quant = {"bits": 3, "group_size": 64, "mode": "affine", key: {"bits": 4, "group_size": 64}}
    assert _packed(_Shards([path], quant=quant), key, 3, 64, torch.device("cpu")).bits == 4


def test_a_width_the_config_does_not_name_is_refused(tmp_path):
    key, path = _file(tmp_path)
    with pytest.raises(ValueError, match="does not match K"):
        _packed(_Shards([path], quant={"bits": 3, "group_size": 64, "mode": "affine"}), key, 3, 64,
                torch.device("cpu"))


def test_a_hugging_face_gptq_export_is_refused_by_name(tmp_path):
    import json

    from tensorfold.rocm.model.checkpoint import load

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "text_config": {},
                                                      "quantization_config": {"quant_method": "gptq", "bits": 4}}))
    with pytest.raises(ValueError, match="Hugging Face GPTQ"):
        load(tmp_path, torch.device("cpu"))
