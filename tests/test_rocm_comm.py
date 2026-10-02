from __future__ import annotations

import pytest


def test_dtype_enum_has_bf16_and_fp32():
    from tensorfold.rocm import comm

    assert comm._DTYPES[__import__("torch").float32] == 7
    assert comm._DTYPES[__import__("torch").bfloat16] == 9


def test_ops_enum_uses_sum_zero():
    from tensorfold.rocm import comm

    assert comm._OPS["sum"] == 0
    assert "prod" in comm._OPS


def test_library_refuses_when_no_rccl(monkeypatch):
    from tensorfold.rocm import comm

    monkeypatch.setattr(comm.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(comm.ctypes.util, "find_library", lambda name: None)
    monkeypatch.delenv("TF_RCCL_LIB", raising=False)
    with pytest.raises(RuntimeError, match="librccl not found"):
        comm._library()


def test_rccl_rejects_world_one():
    from tensorfold.rocm import comm

    with pytest.raises(ValueError, match="multi-rank"):
        comm.RCCL(rank=0, world=1, master="127.0.0.1", port=29551)


def test_rccl_rejects_rank_out_of_range():
    from tensorfold.rocm import comm

    with pytest.raises(ValueError, match="not in"):
        comm.RCCL(rank=4, world=4, master="127.0.0.1", port=29551)


def test_rccl_rejects_empty_master():
    from tensorfold.rocm import comm

    with pytest.raises(ValueError, match="rank 0"):
        comm.RCCL(rank=0, world=2, master="", port=29551)


def test_rccl_rejects_negative_world():
    from tensorfold.rocm import comm

    with pytest.raises(ValueError, match="multi-rank"):
        comm.RCCL(rank=0, world=0, master="127.0.0.1", port=29551)


def test_gather_ints_local_when_no_rccl():
    from tensorfold.rocm import comm

    fake_torch = type("Torch", (), {"tensor": lambda *args, **kwargs: "values"})
    assert comm.gather_ints(None, fake_torch, [1, 2, 3]) == [[1, 2, 3]]


def test_windows_refuses_librccl(monkeypatch):
    import sys as _sys

    from tensorfold.rocm import comm

    monkeypatch.setattr(comm.os, "name", "nt")
    with pytest.raises(RuntimeError, match="Windows"):
        comm._library()


def test_p2p_default_off_for_discrete():
    from tensorfold.rocm.engine import _resolve_p2p

    class _Props:
        multi_gpu_capable = False
        is_integrated = False

    fake_torch = type("Torch", (), {
        "cuda": type("CUDA", (), {"get_device_properties": staticmethod(lambda index: _Props())}),
    })
    import tensorfold.rocm.engine as engine_mod
    engine_mod.torch = fake_torch
    assert _resolve_p2p("gfx1100", None) is False
    assert _resolve_p2p("gfx1100", True) is True
    assert _resolve_p2p("gfx1100", False) is False


def test_p2p_default_on_for_multi_mgpu_apu():
    from tensorfold.rocm.engine import _resolve_p2p

    class _Props:
        multi_gpu_capable = True
        is_integrated = True

    fake_torch = type("Torch", (), {
        "cuda": type("CUDA", (), {"get_device_properties": staticmethod(lambda index: _Props())}),
    })
    import tensorfold.rocm.engine as engine_mod
    engine_mod.torch = fake_torch
    assert _resolve_p2p("gfx1151", None) is True


def test_cli_extends_tp_choices_to_eight():
    from tensorfold import cli_args

    parser = cli_args.build_parser({"serve": lambda args: None, "pull": lambda args: None,
                                    "models": lambda args: None, "update": lambda args: None,
                                    "info": lambda args: None})
    args = parser.parse_args(["serve", "x", "--tp", "4"])
    assert args.tp == 4
    args = parser.parse_args(["serve", "x", "--tp", "8", "--rank", "7"])
    assert args.tp == 8
    assert args.rank == 7
    with __import__("pytest").raises(SystemExit):
        parser.parse_args(["serve", "x", "--tp", "3"])


def test_family_rocm_engine_rejects_bad_world_size():
    from tensorfold.families import qwen3_5

    with __import__("pytest").raises(ValueError, match="world size"):
        qwen3_5.rocm_engine("/tmp/none", tp=3)
    with __import__("pytest").raises(ValueError, match="not in"):
        qwen3_5.rocm_engine("/tmp/none", tp=2, rank=2)


def test_family_rocm_engine_passes_kwargs(monkeypatch):
    from tensorfold.families import qwen3_5
    from tensorfold.rocm import engine

    seen = {}

    def fake_load(model_dir, **kwargs):
        seen.update(kwargs)
        seen["model_dir"] = str(model_dir)
        return object()

    monkeypatch.setattr(engine.QwenEngine, "load", classmethod(lambda cls, model_dir, **kwargs: fake_load(model_dir, **kwargs)))
    qwen3_5.rocm_engine("/tmp/fake", tp=2, rank=1, master="192.0.2.1", master_port=29552, p2p=True, no_drafts=True)
    assert seen["tp"] == 2
    assert seen["rank"] == 1
    assert seen["master"] == "192.0.2.1"
    assert seen["master_port"] == 29552
    assert seen["p2p"] is True
    assert seen["no_drafts"] is True
    assert seen["model_dir"] == "/tmp/fake"