import hashlib
import json
from types import SimpleNamespace

import pytest

from native_engine_bench import GoldenCapture, checkpoint_identity, golden_cases, local_drafter, verify_golden


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


def test_verification_rejects_partial_suite(tmp_path):
    (tmp_path / "suite.json").write_text(json.dumps({"complete": False, "models": []}))
    with pytest.raises(ValueError, match="incomplete"):
        verify_golden(tmp_path)


def test_array_storage_retains_integer_precision_and_bf16(tmp_path):
    import mlx.core as mx
    import numpy as np
    capture = GoldenCapture(tmp_path / "trace")
    integer = capture.value(mx.array([2**60 + 1], dtype=mx.uint64))
    bf16 = capture.value(mx.array([0.125, -3.5], dtype=mx.bfloat16))
    assert int(np.load(capture.directory / integer["file"])[0]) == 2**60 + 1
    assert np.array_equal(np.load(capture.directory / bf16["file"]), [0.125, -3.5])
    assert bf16["storage_dtype"] == "float32"
    capture.close()


def test_verification_rejects_modified_array_payload(tmp_path):
    import mlx.core as mx
    family = tmp_path / "qwen"
    capture = GoldenCapture(family / "text-greedy")
    value = capture.value(mx.array([1.0]))
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
    verify_golden(tmp_path)
    with (capture.directory / value["file"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Array hash mismatch"):
        verify_golden(tmp_path)
