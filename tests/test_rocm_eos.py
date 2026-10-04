"""The ROCm engine stops on every end id a checkpoint names, wherever its config puts them."""

import json

import pytest

pytest.importorskip("torch")

from tensorfold.rocm.engine import read_eos  # noqa: E402


def test_top_level_text_config_and_generation_config_are_all_read(tmp_path):
    # Qwen3.6-35B-A3B: <|im_end|> only at the top level and in generation_config, <|endoftext|> in text_config.
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": [248046, 248044],
                                                      "text_config": {"eos_token_id": 248044}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}))
    assert read_eos(tmp_path) == (248046, 248044)


def test_a_single_id_and_a_missing_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 7}}))
    assert read_eos(tmp_path) == (7,)
    assert read_eos(tmp_path / "absent") == (151645,)
