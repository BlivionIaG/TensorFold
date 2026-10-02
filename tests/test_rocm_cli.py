"""The CLI's ROCm backend choice: a family serves ROCm only through its rocm_engine (any machine)."""

from types import SimpleNamespace

import pytest

from tensorfold import cli, families


def _family(**members):
    return SimpleNamespace(title="Test family", package=SimpleNamespace(**members))


def _engine(*args, **kwargs):
    return None


def test_rocm_needs_the_familys_rocm_engine():
    with pytest.raises(ValueError, match="no ROCm engine"):
        cli._backend("rocm", _family(cuda_engine=_engine))
    assert cli._backend("rocm", _family(rocm_engine=_engine)) == "rocm"


def test_auto_picks_rocm_where_the_amd_driver_is(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.os.path, "exists", lambda path: path == "/dev/kfd")
    assert cli._backend("auto", _family(cuda_engine=_engine, rocm_engine=_engine)) == "rocm"
    assert cli._backend("auto", _family(cuda_engine=_engine)) == "cuda"
    monkeypatch.setattr(cli.os.path, "exists", lambda path: False)
    assert cli._backend("auto", _family(cuda_engine=_engine, rocm_engine=_engine)) == "cuda"


def test_rocm_is_listed_where_a_family_has_it():
    assert families.backends_of(_family(load=_engine, rocm_engine=_engine)) == ("mlx", "rocm")
    assert "ROCm engine" in cli._engines(SimpleNamespace(package=SimpleNamespace(rocm_engine=_engine), lanes=False))
