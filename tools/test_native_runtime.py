import pytest

from native_runtime import check_requirements, resolved_dependencies


def project(*requirements):
    return {"project": {"dependencies": list(requirements)}}


def test_upstream_mlx_upper_bound():
    upstream = project("mlx>=0.32.2,<0.32.3")
    check_requirements(upstream, {"mlx": "0.32.2"}, {"mlx": "0.32.2"})
    with pytest.raises(RuntimeError, match="installed 0.32.3"):
        check_requirements(upstream, {"mlx": "0.32.3"}, {"mlx": "0.32.2"})


def test_upstream_bump_rejects_old_native_pin_even_with_new_python():
    with pytest.raises(RuntimeError, match="native pin 0.32.2 is incompatible"):
        check_requirements(project("mlx>=0.33"), {"mlx": "0.33.0"}, {"mlx": "0.32.2"})


def test_model_library_pin_is_checked_and_names_are_normalized():
    check_requirements(project("MLX_LM>=0.31.3,<0.32"), {"mlx-lm": "0.31.3"}, {"mlx-lm": "0.31.3"})
    with pytest.raises(RuntimeError, match="native pin 0.31.3 is incompatible"):
        check_requirements(project("mlx-lm>=0.32"), {"mlx-lm": "0.32"}, {"mlx-lm": "0.31.3"})


def test_new_dependency_and_inactive_marker():
    check_requirements(project('missing-package>=1; python_version < "2"'), {}, {})
    with pytest.raises(RuntimeError, match="installed missing"):
        check_requirements(project("new-dependency>=1"), {}, {})


def test_vision_dependency_changes_follow_upstream():
    upstream = project("mlx>=0.32.2")
    upstream["project"]["optional-dependencies"] = {"vision": ["mlx-vlm>=0.7.5,<0.8", "transformers>=5.18,<6"]}
    with pytest.raises(RuntimeError, match="mlx-vlm>=0.7.5"):
        check_requirements(upstream, {"mlx": "0.32.2", "mlx-vlm": "0.7.4", "transformers": "5.17.0"}, {})
    check_requirements(upstream, {"mlx": "0.32.2", "mlx-vlm": "0.7.5", "transformers": "5.18.0"}, {})


def test_direct_source_requirement_needs_review():
    with pytest.raises(RuntimeError, match="direct-source dependencies"):
        check_requirements(project("mlx @ https://example.invalid/mlx.whl"), {"mlx": "0.32.2"}, {"mlx": "0.32.2"})


def test_upstream_resolution_updates_native_pairing_instead_of_freezing_it():
    previous = {"python": {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.31.3"},
                "mlx_revision": "old-commit", "mlx_c_revision": "bridge-commit"}
    versions = {"mlx": "0.33.1", "mlx-metal": "0.33.1", "mlx-lm": "0.32.2"}
    result = resolved_dependencies(previous, versions, "0.32.2")
    assert result["python"] == versions
    assert result["mlx_revision"] == "v0.33.1"
    assert result["rebuild_mlx"]
    # Retry a partial install: MLX may already be new while its C bridge failed.
    assert resolved_dependencies(previous, versions, "0.33.1")["rebuild_mlx"]
    assert previous["mlx_revision"] == "old-commit"
    assert not resolved_dependencies(previous, previous["python"], "0.32.2")["rebuild_mlx"]
    assert resolved_dependencies(previous, previous["python"], None)["rebuild_mlx"]


def test_resolver_rejects_mismatched_mlx_metal_wheels():
    with pytest.raises(RuntimeError, match="versions disagree"):
        resolved_dependencies({}, {"mlx": "0.33.1", "mlx-metal": "0.32.2"}, None)


def test_fixture_reserve_checks_incoming_bytes_before_opening(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import native_runtime as runtime
    storage = runtime.FixtureStorage()
    path = tmp_path / "array.npy"
    path.write_bytes(b"previous fixture")
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=runtime.DISK_RESERVE_BYTES + 8))
    storage.check(path, 8)
    with pytest.raises(OSError, match="preserving 64 GiB"):
        storage.write(path, 9, lambda _: pytest.fail("writer must not run"))
    assert path.read_bytes() == b"previous fixture"
    assert list(tmp_path.iterdir()) == [path]


def test_fixture_budget_accumulates_writes_and_removes_failed_partials(tmp_path):
    from native_runtime import FixtureStorage
    storage = FixtureStorage(max_bytes=10)
    first, second = tmp_path / "first", tmp_path / "second"
    storage.write(first, 6, lambda path: path.write_bytes(b"123456"))
    with pytest.raises(OSError, match="storage limit"):
        storage.write(second, 5, lambda _: pytest.fail("writer must not run"))

    def fail(path):
        path.write_bytes(b"bad")
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        storage.write(first, 4, fail)
    assert first.read_bytes() == b"123456"
    assert list(tmp_path.iterdir()) == [first]
    assert storage.used_bytes == 6
    storage.write(second, 4, lambda path: path.write_bytes(b"7890"))
    assert storage.used_bytes == 10


def test_fixture_guard_covers_numpy_and_imported_checkpoint_writers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import mlx.core as mx
    import numpy as np
    import native_runtime as runtime
    original_numpy, original_mlx = np.save, mx.save_safetensors
    values = np.array([2**60 + 1], dtype=np.uint64)
    with runtime.FixtureStorage() as storage:
        np.save(tmp_path / "array", values, allow_pickle=False)
        mx.save_safetensors(str(tmp_path / "checkpoint.safetensors"), {"value": mx.array(values)})
        storage.save_npz(tmp_path / "compressed.npz", values)
        assert np.array_equal(np.load(tmp_path / "array.npy"), values)
        assert np.array_equal(np.asarray(mx.load(str(tmp_path / "checkpoint.safetensors"))["value"]), values)
        with np.load(tmp_path / "compressed.npz") as archive:
            assert np.array_equal(archive["value"], values)
        monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=runtime.DISK_RESERVE_BYTES))
        for save in (lambda: np.save(tmp_path / "blocked", values),
                     lambda: mx.save_safetensors(str(tmp_path / "blocked.safetensors"), {"value": mx.array(values)}),
                     lambda: storage.save_npz(tmp_path / "blocked.npz", values)):
            with pytest.raises(OSError, match="preserving 64 GiB"):
                save()
        assert len(list(tmp_path.iterdir())) == 3
    assert (np.save, mx.save_safetensors) == (original_numpy, original_mlx)


def test_fixture_guard_restores_writers_after_failure_and_ignores_model_links(tmp_path):
    import mlx.core as mx
    import numpy as np
    from native_runtime import fixture_bytes, fixture_storage, FixtureStorage
    originals = np.save, mx.save_safetensors
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "output"
    output.mkdir()
    link = output / "model.safetensors"
    link.symlink_to(checkpoint)
    (output / "model-directory").symlink_to(tmp_path, target_is_directory=True)
    assert fixture_bytes(output) == 0
    with pytest.raises(ValueError, match="symlink"):
        FixtureStorage().write(link, 0, lambda _: pytest.fail("must preserve model link"))

    @fixture_storage
    def fail():
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        fail()
    assert (np.save, mx.save_safetensors) == originals
    assert link.is_symlink() and checkpoint.read_bytes() == b"weights"
