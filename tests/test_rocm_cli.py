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


@pytest.mark.parametrize("backend", ["cuda", "mlx"])
def test_tp_four_and_eight_are_rocm_only(backend):
    for tp in (4, 8):
        with pytest.raises(ValueError, match=f"--tp {tp} is ROCm only"):
            cli._check_world(SimpleNamespace(tp=tp, rank=0), backend)
    with pytest.raises(ValueError, match="--rank 3 is ROCm only"):
        cli._check_world(SimpleNamespace(tp=2, rank=3), backend)
    cli._check_world(SimpleNamespace(tp=2, rank=1), backend)


def test_rocm_takes_every_world():
    for tp in (1, 2, 4, 8):
        cli._check_world(SimpleNamespace(tp=tp, rank=tp - 1), "rocm")
