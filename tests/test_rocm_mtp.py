from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_mtp_state_default_is_unallocated():
    from tensorfold.rocm.mtp import MTPState

    state = MTPState()
    assert state.k is None
    assert state.v is None
    assert state.pos == 0
    assert state.capacity == 0


def test_mtp_state_reset_with_no_cache_is_safe():
    from tensorfold.rocm.mtp import MTPState

    MTPState().reset()


def test_mtp_head_dataclass_has_expected_fields():
    from tensorfold.rocm.qwen import MTPHead

    fields = {f.name for f in MTPHead.__dataclass_fields__.values()}
    assert fields == {"fc_e_norm", "fc_h_norm", "fc_e", "fc_h",
                      "q_norm", "k_norm", "q", "k", "v", "o",
                      "final_norm", "head",
                      "input_norm", "post_norm", "gate", "up", "down", "moe", "gated"}


def test_text_model_carries_optional_mtp_head():
    from tensorfold.rocm.qwen import TextModel

    assert "mtp" in TextModel.__dataclass_fields__
    assert TextModel.__dataclass_fields__["mtp"].default is None


def test_load_mtp_head_returns_none_when_no_mtp_file(tmp_path, monkeypatch):
    from tensorfold.rocm import qwen as qwen_mod

    spec = qwen_mod.Spec(hidden=4096, intermediate=14336, heads=24, kv_heads=4, head_dim=128,
                         key_heads=4, value_heads=4, key_dim=128, value_dim=128,
                         conv=4, vocab=100000, eps=1e-6, rope_theta=1000000.0, rotary_dim=32,
                         full_every=4, n_layers=2, bits=4, group=64)
    assert qwen_mod.load_mtp_head(tmp_path, spec, bits=4, group=64, device="cpu") is None


def test_load_mtp_head_refuses_partial_mtp_file(tmp_path, monkeypatch):
    from tensorfold.rocm import qwen as qwen_mod

    spec = qwen_mod.Spec(hidden=4096, intermediate=14336, heads=24, kv_heads=4, head_dim=128,
                         key_heads=4, value_heads=4, key_dim=128, value_dim=128,
                         conv=4, vocab=100000, eps=1e-6, rope_theta=1000000.0, rotary_dim=32,
                         full_every=4, n_layers=2, bits=4, group=64)

    class _FakeShards:
        def __init__(self, paths):
            self.paths = paths

        def __iter__(self):
            return iter(self.paths)

        def __contains__(self, key):
            return key == "mtp.q_proj.weight"

        def __getitem__(self, key):
            raise KeyError(key)

        def get_tensor(self, key):
            raise KeyError(key)

    shard = tmp_path / "mtp-4bit.safetensors"
    shard.touch()
    from tensorfold.rocm import checkpoint

    monkeypatch.setattr(checkpoint, "_Shards", lambda paths, strip="", quant=None: _FakeShards(paths))

    with pytest.raises(ValueError, match="incomplete"):
        qwen_mod.load_mtp_head(tmp_path, spec, bits=4, group=64, device="cpu")