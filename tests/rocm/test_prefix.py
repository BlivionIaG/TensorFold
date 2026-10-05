"""Host checks for the ROCm prompt cache and the per-request stop rule. No GPU."""

from tensorfold.rocm.serving.prefix import PrefixCache, decode_ids, entry_end


def test_entry_end_leaves_the_last_token():
    assert entry_end([4, 5, 6, 7]) == 3
    assert entry_end([4]) == 1


def test_a_hit_must_leave_one_token_and_outlives_a_miss():
    cache = PrefixCache(keep=2)
    cache.add([1, 2], "state", None)
    cache.add([9], "other", None)
    assert cache.longest([1, 2]) is None
    hit = cache.longest([1, 2, 3])
    assert hit[0] == [1, 2]
    cache.add([8, 8, 8], "third", None)
    assert any(entry[0] == [1, 2] for entry in cache.entries)
    assert all(entry[0] != [9] for entry in cache.entries)


def test_each_request_stops_on_its_own_end_token():
    def step(done):
        # ``done`` is the prompt plus tokens already produced. 0 ends that request only.
        return 0 if len(done) >= (5 if done[0] == 1 else 3) else 7

    short = decode_ids([2], step, 10, eos=(0,))
    long = decode_ids([1], step, 10, eos=(0,))
    assert short == [7, 7, 0]
    assert long == [7, 7, 7, 7, 0]


def test_on_tokens_can_stop_before_eos():
    seen = []

    def stop(tokens):
        seen.extend(tokens)
        return len(seen) == 2

    out = decode_ids([1], lambda done: 4, 10, eos=(0,), on_tokens=stop)
    assert out == [4, 4]


def test_mixed_lengths_are_separate_chains():
    from tensorfold.rocm.serving.prefix import request_parents

    parents = request_parents([2, 5, 1])
    assert parents == [[-1, 0], [-1, 0, 1, 2, 3], [-1]]
    try:
        import torch  # noqa: F401

        from tensorfold.cuda.kernels.gdn import plan_host
    except ImportError:
        return
    _entries, starts, slots, rows = plan_host(parents)
    assert starts == [0, 2, 7, 8]
    # A chain's next row is its child, so the state stays in the running row and no slot is parked.
    assert slots == 0
    assert rows == 5
    _entries, _starts, branched, _rows = plan_host([[-1, 0, 0]])
    assert branched == 1


def test_a_resumed_prefix_outlives_the_byte_budget():
    from tensorfold.rocm.serving.prefix import trim_bytes

    cache = PrefixCache(keep=4)
    cache.add([1, 2], "state", 10)
    cache.longest([1, 2, 3])
    cache.add([9], "other", 10)
    cache.add([8], "third", 10)
    trim_bytes(cache, 20)
    assert [entry[0] for entry in cache.entries] == [[1, 2], [8]]


def test_a_follow_up_resumes_at_the_assistant_header():
    import os
    from pathlib import Path

    import pytest

    Tokenizer = pytest.importorskip("tokenizers").Tokenizer
    model = Path(os.environ.get("TENSORFOLD_CHAT_MODEL", ""))
    if not (model / "tokenizer.json").is_file():
        pytest.skip("set TENSORFOLD_CHAT_MODEL to a Qwen checkpoint with tokenizer.json")

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.rocm.serving.prefix import message_points

    points = message_points(model)
    assert points is not None
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    tpl = ChatTemplate(model)

    def encode(messages):
        text = tpl.render(messages, tools=None, enable_thinking=False, extra={})
        return tok.encode(text, add_special_tokens=False).ids

    first = encode([{"role": "user", "content": "Say hi"}])
    second = encode([
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
    ])
    cuts = points(first)
    assert cuts
    cache = PrefixCache(8)
    for cut in cuts:
        cache.add(first[:cut], "state", 1)
    cache.add(first[:entry_end(first)], "end", 1)
    hit = cache.longest(second)
    assert hit is not None and hit[0] == second[:len(hit[0])]


def test_rocm_reads_a_tied_eight_bit_checkpoint():
    from tensorfold.families.qwen3_5 import check_quantization

    config = {"model_type": "qwen3_5", "tie_word_embeddings": True,
              "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    check_quantization(config, "rocm")
    try:
        check_quantization(config, "mlx")
    except ValueError as exc:
        assert "tied embedding" in str(exc)
    else:
        raise AssertionError("the Mac decoder still refuses a tied head")
