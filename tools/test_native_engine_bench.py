import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from native_engine_bench import GoldenCapture, checkpoint_identity, compare_capacity, compare_golden, compare_resources, golden_cases, golden_native_command, local_drafter, measured_process, native_synthetic_checkpoint, verify_golden


def test_capture_restores_inherited_methods_after_failure(tmp_path):
    class Model:
        def hidden(self, tokens):
            raise RuntimeError("forward failed")

    model = Model()
    engine = SimpleNamespace()
    capture = GoldenCapture(tmp_path / "trace")
    capture.install(model, engine)
    with pytest.raises(RuntimeError, match="forward failed"):
        model.hidden([1, 2])
    capture.close()
    assert "hidden" not in vars(model)
    assert capture.events[0]["inputs"] == [[1, 2]]
    assert "output" not in capture.events[0]


def test_checkpoint_identity_distinguishes_payload_and_header(tmp_path):
    import struct
    header = b'{"weight":{"shape":[1]}}'
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"weights")
    identity = checkpoint_identity(tmp_path)
    assert identity["files"][0]["hash_scope"] == "header"
    assert identity["files"][0]["sha256"] == hashlib.sha256(header).hexdigest()
    assert identity["files"][0]["bytes"] == path.stat().st_size


def test_gemma_drafter_uses_publisher_cache_and_prefers_model_root(tmp_path):
    models, cache = tmp_path / "models", tmp_path / "cache"
    name = "gemma-4-26B-A4B-it-DFlash"
    cached = cache / "z-lab" / name
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    (cached / "model.safetensors").write_bytes(b"weights")
    linked = models / name
    assert local_drafter("gemma", models, cache) == linked
    assert linked.is_symlink() and linked.resolve() == cached
    linked.unlink()
    linked.mkdir()
    (linked / "config.json").write_text("{}")
    (linked / "model.safetensors").write_bytes(b"weights")
    assert local_drafter("gemma", models, cache) == linked


def test_cached_drafter_does_not_overwrite_an_incomplete_destination(tmp_path):
    models, cache = tmp_path / "models", tmp_path / "cache"
    name = "gemma-4-26B-A4B-it-DFlash"
    cached = cache / "z-lab" / name
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    (cached / "model.safetensors").write_bytes(b"weights")
    destination = models / name
    destination.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        local_drafter("gemma", models, cache)
    assert destination.is_dir() and not destination.is_symlink()


def test_missing_drafter_cannot_silently_pass_draft_coverage(tmp_path):
    with pytest.raises(RuntimeError, match="no download"):
        local_drafter("gemma", tmp_path / "models", tmp_path / "cache")


def test_cases_cover_decode_prefill_chunk_and_pass_boundaries():
    cases = golden_cases(None, synthetic=True)
    assert [len(c["tokens"]) for c in cases[4:]] == [16, 17, 2049, 4097]
    assert all(0 <= t < 256 for c in cases for t in c["tokens"])
    assert cases[0]["tokens"] == cases[2]["tokens"]
    assert cases[1]["tokens"] == cases[3]["tokens"]


@pytest.mark.parametrize("family", ["qwen", "bonsai", "nemotron", "flash", "gemma", "glm", "deepseek"])
@pytest.mark.parametrize("driver", ["cli", "serving"])
@pytest.mark.parametrize("drafts", [False, True])
def test_native_comparison_uses_fixture_inputs_and_supported_runtime_flags(tmp_path, family, driver, drafts):
    args = SimpleNamespace(binary=tmp_path / "tensorfold", resident_ple=False, native_arg=[], native_driver=driver)
    manifest = dict(checkpoint={"path": str(tmp_path / "checkpoint")}, seed=5678, top_k=12, top_p=0.8,
                    drafter={"path": str(tmp_path / "linked-drafter")}, load_options=dict(ple_on_ssd=True))
    case = dict(tokens=[1, 38, 75], temperature=0.7, drafts=drafts, measurements=[{"tokens": [2] * 16}])
    if driver == "serving" and family in ("glm", "deepseek"):
        with pytest.raises(ValueError, match="fitting checkpoint"):
            golden_native_command(args, family, manifest, case, tmp_path / "report.json")
        return
    command = golden_native_command(args, family, manifest, case, tmp_path / "report.json")
    assert command[1] == ("bench-session" if driver == "serving" else "run")
    assert command[command.index("--tokens") + 1] == "1,38,75"
    assert command[command.index("--max-tokens") + 1] == "16"
    assert command[command.index("--seed") + 1] == "5678"
    assert ("--no-copy" in command) == (driver == "serving" or family not in ("gemma", "glm", "deepseek"))
    assert ("--metal-sampling" in command) == (family not in ("qwen", "bonsai"))
    assert ("--ignore-eos" in command) == (driver == "serving" or family in ("gemma", "glm", "deepseek"))
    if driver == "serving":
        assert "--warm-case" not in command
    assert ("--no-drafts" in command) == (not drafts)
    if family in ("qwen", "bonsai", "gemma") and (drafts or driver == "serving"):
        assert command[command.index("--drafter") + 1] == manifest["drafter"]["path"]
    if family == "gemma" and (drafts or driver == "serving"):
        assert command[command.index("--mtp-drafts") + 1] == "15"


def test_native_glm_config_adaptation_preserves_golden_and_weight_payload(tmp_path):
    checkpoint = tmp_path / "golden"
    checkpoint.mkdir()
    original = '{"text_config":{"eos_token_id":1,"hidden_size":4096}}'
    (checkpoint / "config.json").write_text(original)
    (checkpoint / "model.safetensors").write_bytes(b"weight-payload")
    output = tmp_path / "comparison"
    output.mkdir()
    manifest = dict(family="glm", checkpoint={"path": str(checkpoint)})
    adapted = native_synthetic_checkpoint(manifest, output)
    assert (checkpoint / "config.json").read_text() == original
    assert json.loads((adapted / "config.json").read_text())["text_config"] == dict(eos_token_id=[1], hidden_size=4096)
    assert (adapted / "model.safetensors").is_symlink()
    assert (adapted / "model.safetensors").resolve() == checkpoint / "model.safetensors"


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("unknown_case", [False, True])
@pytest.mark.parametrize("capacity_mismatch", [False, True])
def test_comparison_checks_all_cases_before_measuring_and_rejects_regressions(tmp_path, monkeypatch, mismatch, unknown_case, capacity_mismatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    golden = tmp_path / "golden"
    golden.mkdir()
    cases = [dict(name=name, tokens=[3], temperature=0, drafts=False,
                  measurements=[dict(tokens=[8, 9], timing_valid=True, first_token_seconds=2,
                                     decode_seconds=3, total_seconds=5, peak_mlx_bytes=1000)]) for name in ("bad", "good")]
    manifest = dict(complete=True, family="qwen", checkpoint=checkpoint_identity(checkpoint),
                    cases=cases, seed=5678, top_k=12, top_p=0.8, environment={"MLX_MAX_OPS_PER_BUFFER": "200"})
    (golden / "manifest.json").write_text(json.dumps(manifest))
    (golden / "suite.json").write_text(json.dumps(dict(complete=True, models=[
        dict(family="qwen", synthetic=False, manifest="manifest.json")
    ])))
    binary = tmp_path / "binary"
    binary.write_bytes(b"binary")
    calls = []

    def native(command, log, env=None):
        from pathlib import Path
        python = "--python-case-manifest" in command
        if not python:
            assert env["MLX_MAX_OPS_PER_BUFFER"] == "200"
        report = Path(command[command.index("--output" if python else "--report") + 1])
        calls.append(report.stem)
        report.write_text(json.dumps(dict(tokens=[8, 7] if mismatch and "bad" in report.stem else [8, 9],
                                          prefill_seconds=2, first_token_seconds=2, decode_seconds=3,
                                          total_seconds=5, timing_valid=True, peak_mlx_bytes=1000 if python else 1001,
                                          calibration_streams=8 if capacity_mismatch and not python and "bad" in report.stem else 64,
                                          calibration_rows=128)))
        return dict(exit_code=0, process_seconds=1, peak_rss_bytes=2000, peak_footprint_bytes=3000)

    monkeypatch.setattr("native_engine_bench.measured_process", native)
    args = SimpleNamespace(compare_golden=golden, output=tmp_path / "comparison", binary=binary,
                           family=None, case=[], repetitions=1, native_arg=[], native_env=[], resident_ple=False, native_driver="cli")
    if unknown_case:
        args.case = ["good", "missing"]
        with pytest.raises(ValueError, match="Unknown fixture cases: missing"):
            compare_golden(args)
        assert not calls
        return
    with pytest.raises(SystemExit, match="correctness incomplete" if mismatch else "capacity parity incomplete" if capacity_mismatch else "resource parity failed"):
        compare_golden(args)
    assert calls == ["qwen-bad-0-python", "qwen-bad-0", "qwen-good-0-python", "qwen-good-0"] + (
        [] if mismatch or capacity_mismatch else ["qwen-bad-1", "qwen-bad-1-python"]) + ["qwen-good-1", "qwen-good-1-python"]
    result = json.loads((args.output / "comparison.json").read_text())
    assert result["complete"] and result["correctness_passed"] == (not mismatch)
    assert not result["passed"] and not result["memory_passed"]
    assert result["capacity_passed"] == (not capacity_mismatch)
    assert len(result["cases"][0]["native"]) == (0 if mismatch or capacity_mismatch else 1)
    assert len(result["cases"][1]["native"]) == 1


def test_verification_rejects_partial_suite(tmp_path):
    (tmp_path / "suite.json").write_text(json.dumps({"complete": False, "models": []}))
    with pytest.raises(ValueError, match="incomplete"):
        verify_golden(tmp_path)


def test_python_worker_includes_production_shared_workspace_before_warmup(tmp_path, monkeypatch):
    import sys
    from native_engine_bench import python_case_worker
    events = []
    model = object()
    case = dict(name="text-greedy", measurements=[dict(tokens=[8, 9])])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(family="qwen", cases=[case], checkpoint=dict(path="checkpoint"),
                                        load_options={}, engine_options=dict(max_rows=128),
                                        python_source=dict(path="reference", sha256="source"), environment={},
                                        seed=5678, top_k=12, top_p=0.8)))

    class Engine:
        def __init__(self, loaded, **options):
            assert loaded is model and options == dict(max_rows=128)

        def round_working_set(self):
            events.append("workspace")
            return 4096

        def release_rounds(self):
            events.append("release")

    def load(*args, **kwargs):
        events.append("load")
        return model, None

    def run(loaded, *args):
        assert loaded is model
        events.append("request")
        return dict(tokens=[8, 9], calibration_streams=64, calibration_rows=128)

    monkeypatch.setattr("native_runtime.require_mlx", lambda: None)
    monkeypatch.setattr("native_engine_bench.select_python_source", lambda directory, expected: expected)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", SimpleNamespace(LaneEngine=Engine))
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5", SimpleNamespace(load=load))
    monkeypatch.setattr("native_engine_bench.run_golden_case", run)
    output = tmp_path / "result.json"
    python_case_worker(SimpleNamespace(python_case_manifest=manifest, case=[case["name"]], output=output))
    assert events == ["load", "workspace", "release", "request", "request"]
    result = json.loads(output.read_text())
    assert result["round_workspace_bytes"] == 4096
    assert result["calibration_streams"] == 64 and result["calibration_rows"] == 128


@pytest.mark.parametrize("missing", [False, True])
def test_process_memory_retains_lifetime_rss_and_footprint_in_bytes(tmp_path, monkeypatch, missing):
    def run(command, **kwargs):
        assert command == ["/usr/bin/time", "-l", "-o", str(tmp_path / "process.memory.txt"), "benchmark"]
        (tmp_path / "process.memory.txt").write_text(" 4096  maximum resident set size\n" + (
            "" if missing else " 8192  peak memory footprint\n"))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr("native_engine_bench.subprocess.run", run)
    monkeypatch.setattr("native_engine_bench.sys.platform", "darwin")
    if missing:
        with pytest.raises(RuntimeError, match="Missing whole-process"):
            measured_process(["benchmark"], tmp_path / "process.log")
        return
    result = measured_process(["benchmark"], tmp_path / "process.log")
    assert result["exit_code"] == 7
    assert result["peak_rss_bytes"] == 4096
    assert result["peak_footprint_bytes"] == 8192


@pytest.mark.parametrize("metric,extra", [("peak_mlx_bytes", 1), ("peak_rss_bytes", 1), ("peak_footprint_bytes", 1), ("prefill_seconds", 0.001), ("decode_seconds", 0.001)])
def test_resource_parity_rejects_any_memory_or_phase_regression(metric, extra):
    python = dict(peak_mlx_bytes=1000, peak_rss_bytes=2000, peak_footprint_bytes=3000, first_token_seconds=2, decode_seconds=3, total_seconds=5, timing_valid=True)
    native = dict(peak_mlx_bytes=1000, peak_rss_bytes=2000, peak_footprint_bytes=3000, prefill_seconds=2, decode_seconds=3)
    for sample in (python, native):
        sample.update(calibration_streams=64, calibration_rows=128)
    assert compare_resources([python], [native])["passed"]
    native[metric] += extra
    result = compare_resources([python], [native])
    assert not result["passed"]
    assert result["memory_passed"] == (not metric.endswith("bytes"))
    assert result["performance_passed"] == metric.endswith("bytes")


def test_resource_parity_uses_worst_peak_and_median_latency():
    python = dict(peak_mlx_bytes=1000, peak_rss_bytes=2000, peak_footprint_bytes=3000, first_token_seconds=2, decode_seconds=3, total_seconds=5, timing_valid=True)
    native = dict(peak_mlx_bytes=900, peak_rss_bytes=1900, peak_footprint_bytes=2900, prefill_seconds=1, decode_seconds=2)
    outlier = dict(peak_mlx_bytes=1001, peak_rss_bytes=2001, peak_footprint_bytes=3001, prefill_seconds=100, decode_seconds=100)
    for sample in (python, native, outlier):
        sample.update(calibration_streams=64, calibration_rows=128)
    result = compare_resources([python] * 3, [native, native, outlier])
    assert not result["memory_passed"]
    assert result["performance_passed"]
    assert result["metrics"]["peak_rss_bytes"]["native"] == 2001


@pytest.mark.parametrize("invalid", [None, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("metric", ["peak_mlx_bytes", "peak_rss_bytes", "peak_footprint_bytes"])
def test_resource_parity_rejects_invalid_or_missing_measurements(invalid, metric):
    python = dict(peak_mlx_bytes=1000, peak_rss_bytes=2000, peak_footprint_bytes=3000, first_token_seconds=2, decode_seconds=3, total_seconds=5, timing_valid=True)
    native = dict(peak_mlx_bytes=1000, peak_rss_bytes=2000, peak_footprint_bytes=3000, prefill_seconds=2, decode_seconds=3)
    for sample in (python, native):
        sample.update(calibration_streams=64, calibration_rows=128)
    native[metric] = invalid
    assert not compare_resources([python], [native])["passed"]
    assert not compare_resources([python], [])["passed"]
    assert not compare_resources([], [native])["passed"]
    python["timing_valid"] = False
    native[metric] = python[metric]
    assert not compare_resources([python], [native])["passed"]


@pytest.mark.parametrize("field", ["calibration_streams", "calibration_rows"])
@pytest.mark.parametrize("invalid", [None, 0, -1, True, 8, 256, 64.0])
def test_resource_parity_requires_exact_capacity(field, invalid):
    python = dict(calibration_streams=64, calibration_rows=128, peak_mlx_bytes=1000,
                  peak_rss_bytes=2000, peak_footprint_bytes=3000, first_token_seconds=2,
                  decode_seconds=3, total_seconds=5, timing_valid=True)
    native = dict(python, prefill_seconds=1, decode_seconds=1, peak_mlx_bytes=1,
                  peak_rss_bytes=1, peak_footprint_bytes=1)
    native[field] = invalid
    assert not compare_capacity(python, native)["passed"]
    result = compare_resources([python], [native])
    assert not any(result[k] for k in ("capacity_passed", "memory_passed", "performance_passed", "passed"))
    del native[field]
    assert not compare_resources([python], [native])["passed"]


def test_array_storage_retains_integer_precision_and_bf16(tmp_path):
    import mlx.core as mx
    import numpy as np
    capture = GoldenCapture(tmp_path / "trace")
    integer = capture.value(mx.array([2**60 + 1], dtype=mx.uint64))
    bf16 = capture.value(mx.array([0.125, -3.5], dtype=mx.bfloat16))
    with np.load(capture.directory / integer["file"]) as archive:
        assert int(archive["value"][0]) == 2**60 + 1
    with np.load(capture.directory / bf16["file"]) as archive:
        bits = archive["value"]
        assert np.array_equal(np.asarray(mx.array(bits).view(mx.bfloat16).astype(mx.float32)), [0.125, -3.5])
        assert bits.nbytes == 4
    assert bf16["storage_dtype"] == "uint16"
    capture.close()


def test_golden_captures_share_one_budget_and_compress_repeated_values(tmp_path):
    import mlx.core as mx
    from native_runtime import FixtureStorage
    storage = FixtureStorage(max_bytes=2 * 1024**2)
    first = GoldenCapture(tmp_path / "first", storage)
    entry = first.value(mx.zeros((65536,), dtype=mx.bfloat16))
    assert (first.directory / entry["file"]).stat().st_size < 4096
    storage.max_bytes = storage.used_bytes
    second = GoldenCapture(tmp_path / "second", storage)
    with pytest.raises(OSError, match="storage limit"):
        second.value(mx.array([1.0]))
    assert not list(second.directory.iterdir())


def test_golden_deduplicates_exact_arrays_without_changing_dtype_or_shape(tmp_path):
    import mlx.core as mx
    import numpy as np
    capture = GoldenCapture(tmp_path / "trace")
    first = capture.value(mx.array([1, 2], dtype=mx.uint32))
    used = capture.storage.used_bytes
    assert capture.value(mx.array([1, 2], dtype=mx.uint32)) == first
    assert capture.storage.used_bytes == used
    reshaped = capture.value(mx.array([[1, 2]], dtype=mx.uint32))
    different_type = capture.value(mx.array([1, 2], dtype=mx.int32))
    scalar = capture.value(mx.array(1, dtype=mx.uint32))
    assert len({entry["file"] for entry in (first, reshaped, different_type, scalar)}) == 4
    with np.load(capture.directory / scalar["file"]) as archive:
        assert archive["value"].shape == ()
    empty = capture.value(mx.zeros((0, 3), dtype=mx.uint32))
    with np.load(capture.directory / empty["file"]) as archive:
        assert archive["value"].shape == (0, 3)
    capture.close()


def test_python_reference_identity_rejects_changed_code_and_ignores_bytecode(tmp_path, monkeypatch):
    import sys
    from native_engine_bench import python_source_identity, select_python_source
    package = tmp_path / "tensorfold"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("value = 1\n")
    identity = python_source_identity(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    assert select_python_source(tmp_path, identity) == identity
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "__init__.pyc").write_bytes(b"compiled")
    assert python_source_identity(tmp_path) == identity
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        select_python_source(tmp_path, identity)


def test_native_flash_uses_golden_resident_setting(tmp_path):
    args = SimpleNamespace(binary=tmp_path / "tensorfold", resident_ple=False, native_arg=[], native_driver="serving")
    manifest = dict(checkpoint=dict(path="checkpoint"), seed=5678, top_k=12, top_p=0.8,
                    load_options=dict(ple_on_ssd=False))
    case = dict(tokens=[1], temperature=0, drafts=False, measurements=[dict(tokens=[2, 3])])
    assert "--resident-ple" in golden_native_command(args, "flash", manifest, case, tmp_path / "report")


def test_golden_workers_count_previous_families_against_the_suite_limit(tmp_path, monkeypatch):
    import native_engine_bench as bench
    (tmp_path / "previous-family.npz").write_bytes(b"123456")
    output = tmp_path / "next-family"
    output.mkdir()
    monkeypatch.setattr(bench, "FIXTURE_LIMIT_BYTES", 10)

    def capture(args, storage):
        storage.write(args.output / "array.npz", 5, lambda _: pytest.fail("suite budget must reject this write"))

    monkeypatch.setattr(bench, "capture_golden_worker", capture)
    with pytest.raises(OSError, match="storage limit"):
        bench.golden_worker(SimpleNamespace(output=output, storage_root=tmp_path))
    assert not list(output.iterdir())


@pytest.mark.parametrize("dtype,number", [("float32", 1.0), ("bfloat16", 1.0), ("bfloat16", float("inf")), ("bfloat16", float("nan"))])
def test_verification_rejects_modified_or_nonfinite_array_payload(tmp_path, dtype, number):
    import mlx.core as mx
    family = tmp_path / "qwen"
    capture = GoldenCapture(family / "text-greedy")
    value = capture.value(mx.array([number], dtype=getattr(mx, dtype)))
    capture.events = [{"method": method} for method in ("head", "_draw", "hidden")]
    capture.close()
    tokens = [1, 2]
    import struct
    result = dict(tokens=tokens, prompt_tokens=[3], timing_valid=False, snapshots=[{"cache": [value]}],
                  token_sha256=hashlib.sha256(struct.pack("<2I", *tokens)).hexdigest())
    (capture.directory / "result.json").write_text(json.dumps(result))
    case = dict(name="text-greedy", tokens=[3], drafts=False, result="text-greedy/result.json",
                trace="text-greedy/trace.json", measurements=[dict(tokens=tokens, timing_valid=True)])
    (family / "manifest.json").write_text(json.dumps(dict(complete=True, synthetic=False, cases=[case])))
    (tmp_path / "suite.json").write_text(json.dumps(dict(complete=True, models=[
        dict(family="qwen", status="ok", synthetic=False, manifest="qwen/manifest.json")
    ])))
    if not math.isfinite(number):
        with pytest.raises(ValueError, match="Nonfinite fixture array"):
            verify_golden(tmp_path)
        return
    verify_golden(tmp_path)
    with (capture.directory / value["file"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Array hash mismatch"):
        verify_golden(tmp_path)
