"""The Qwen3 MTP head on RDNA: both checkpoint shapes load, the gated attention runs, and the head drafts.

The checkpoint is written here, so the loader's fused ``fc`` split and its dense / routed MLP pick are the
thing under test, and the draft chain is checked for determinism on the greedy path.
"""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tests.rocm.test_moe import _EXPERTS, _GROUP, _HIDDEN, _TOP_K, _WIDTH, _affine, _checkpoint  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.rocm import qwen as qwen_mod  # noqa: E402
from tensorfold.rocm.build import gfx_name  # noqa: E402
from tensorfold.rocm.moe import Routed  # noqa: E402
from tensorfold.rocm.mtp import MTPEngine  # noqa: E402

_BITS = 4
_MLP = {"gate_proj": (_WIDTH, _HIDDEN), "up_proj": (_WIDTH, _HIDDEN), "down_proj": (_HIDDEN, _WIDTH)}


def _put(tensors, name, n, k, seed):
    words, scale, bias = _affine(n, k, _BITS, _GROUP, seed)
    tensors[name + ".weight"] = words
    tensors[name + ".scales"] = scale
    tensors[name + ".biases"] = bias


def _write_qwen3_mtp(root, spec, *, routed, embedded=False, head_proj=False):
    """A Qwen3-layout MTP layer: pre norms, a fused ``fc``, ``layers.0``, and ``mtp.norm``.

    ``embedded`` merges the tensors into the checkpoint's own shard, which is how the Qwen3.5 and Qwen3.8
    conversions keep the head; otherwise it lands in ``mtp-4bit.safetensors`` beside the weights.
    """

    from safetensors.torch import load_file, save_file

    hidden, head_dim, heads, kv_heads = spec.hidden, spec.head_dim, spec.heads, spec.kv_heads
    tensors = {
        "mtp.pre_fc_norm_embedding.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_hidden.weight": torch.full((hidden,), 0.5),
        "mtp.norm.weight": torch.full((hidden,), 1.5),
        "mtp.layers.0.input_layernorm.weight": torch.full((hidden,), 2.0),
        "mtp.layers.0.post_attention_layernorm.weight": torch.full((hidden,), 3.0),
        "mtp.layers.0.self_attn.q_norm.weight": torch.ones(head_dim),
        "mtp.layers.0.self_attn.k_norm.weight": torch.ones(head_dim),
    }
    _put(tensors, "mtp.fc", hidden, 2 * hidden, 900)
    attn = "mtp.layers.0.self_attn."
    _put(tensors, attn + "q_proj", heads * head_dim * 2, hidden, 901)
    _put(tensors, attn + "k_proj", kv_heads * head_dim, hidden, 902)
    _put(tensors, attn + "v_proj", kv_heads * head_dim, hidden, 903)
    _put(tensors, attn + "o_proj", hidden, heads * head_dim, 904)
    mlp = "mtp.layers.0.mlp."
    if routed:
        for stack, count in (("switch_mlp", _EXPERTS), ("shared_expert", 1)):
            for index, (name, (n, k)) in enumerate(_MLP.items()):
                for suffix, part in ((".weight", 0), (".scales", 1), (".biases", 2)):
                    tensors[f"{mlp}{stack}.{name}{suffix}"] = torch.cat(
                        [_affine(n, k, _BITS, _GROUP, 950 + 10 * index + expert)[part][None]
                         for expert in range(count)])
        for name, count, seed in (("gate", _EXPERTS, 800), ("shared_expert_gate", 1, 801)):
            words, scale, bias = _affine(count, hidden, 8, _GROUP, seed)
            tensors[f"{mlp}{name}.weight"] = words
            tensors[f"{mlp}{name}.scales"] = scale
            tensors[f"{mlp}{name}.biases"] = bias
    else:
        for index, (name, (n, k)) in enumerate(_MLP.items()):
            _put(tensors, mlp + name, n, k, 960 + index)
    if head_proj:
        _put(tensors, "mtp.head_proj", spec.vocab, spec.hidden, 970)
    if embedded:
        tensors = {**load_file(str(root / "model.safetensors")), **tensors}
        save_file({key: value.contiguous() for key, value in tensors.items()}, str(root / "model.safetensors"))
        return
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(root / "mtp-4bit.safetensors"))


def _write_flash_next_mtp(root, spec):
    """The older head: two norms, two fc halves, the attention and a final norm, and no MLP of either kind."""

    from safetensors.torch import save_file

    hidden, head_dim, heads, kv_heads = spec.hidden, spec.head_dim, spec.heads, spec.kv_heads
    tensors = {"mtp.norm_e.weight": torch.ones(hidden), "mtp.norm_h.weight": torch.full((hidden,), 0.5),
               "mtp.q_norm.weight": torch.ones(head_dim), "mtp.k_norm.weight": torch.ones(head_dim),
               "mtp.final_norm.weight": torch.full((hidden,), 1.5)}
    _put(tensors, "mtp.fc_e", hidden, hidden, 910)
    _put(tensors, "mtp.fc_h", hidden, hidden, 911)
    _put(tensors, "mtp.q_proj", heads * head_dim, hidden, 912)
    _put(tensors, "mtp.k_proj", kv_heads * head_dim, hidden, 913)
    _put(tensors, "mtp.v_proj", kv_heads * head_dim, hidden, 914)
    _put(tensors, "mtp.o_proj", hidden, heads * head_dim, 915)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(root / "mtp-4bit.safetensors"))


def _loaded(tmp_path, *, routed):
    """The tiny MoE checkpoint with a Qwen3 MTP layer beside it, on the device, with its head and projections."""

    device = torch.device("cuda")
    root = _checkpoint(tmp_path, gptq=False)
    model = qwen_mod.load(root, device)
    _write_qwen3_mtp(root, model.spec, routed=routed)
    head = qwen_mod.load_mtp_head(root, model.spec, _BITS, _GROUP, device)
    assert head is not None
    return model, head, MTPEngine(model, head, linear=qwen_mod.Engine(model).linear), device


def test_the_qwen3_dense_head_loads(tmp_path):
    """The Qwen3 layout is read as gated, with an input and post norm and a dense MLP, and its fc halves split."""

    model, head, _, _ = _loaded(tmp_path, routed=False)
    assert head.gated is True
    assert head.input_norm is not None and head.post_norm is not None
    assert head.moe is None and head.gate is not None and head.up is not None and head.down is not None
    assert head.fc_e.words.shape[1] == model.spec.hidden * _BITS // 32
    assert head.fc_h.words.shape[1] == model.spec.hidden * _BITS // 32
    assert head.q.words.shape[0] == model.spec.heads * model.spec.head_dim * 2


def test_the_qwen3_routed_head_loads(tmp_path):
    """The routed layout puts one MoE block in place of the dense MLP, the shared expert last."""

    _, head, _, _ = _loaded(tmp_path, routed=True)
    assert head.gated is True and head.gate is None and head.up is None and head.down is None
    assert isinstance(head.moe, Routed) and head.moe.top_k == _TOP_K
    assert head.moe.count == _EXPERTS and head.moe.experts.count == _EXPERTS + 1


def test_the_flash_next_head_still_loads(tmp_path):
    """The older shape keeps loading: no norms around the block, no MLP, and an ungated attention."""

    device = torch.device("cuda")
    root = _checkpoint(tmp_path, gptq=False)
    model = qwen_mod.load(root, device)
    _write_flash_next_mtp(root, model.spec)
    head = qwen_mod.load_mtp_head(root, model.spec, _BITS, _GROUP, device)
    assert head is not None and head.gated is False
    assert head.input_norm is None and head.post_norm is None and head.moe is None and head.gate is None
    engine = MTPEngine(model, head, linear=qwen_mod.Engine(model).linear)
    assert engine._heads == model.spec.heads and engine._kv_heads == model.spec.kv_heads


def test_the_gated_head_counts_its_heads_without_the_gate(tmp_path):
    """A gated q carries two head_dims a head, so the engine reads the head count through the gate."""

    model, _, engine, _ = _loaded(tmp_path, routed=True)
    assert engine._heads == model.spec.heads and engine._kv_heads == model.spec.kv_heads


def test_the_head_loads_from_the_checkpoints_own_shards(tmp_path):
    """Qwen3.5 and Qwen3.8 keep ``mtp.*`` among the model's tensors, so a load with no side file finds it."""

    device = torch.device("cuda")
    root = _checkpoint(tmp_path, gptq=False)
    spec = qwen_mod.load(root, torch.device("cpu")).spec
    _write_qwen3_mtp(root, spec, routed=False, embedded=True)
    assert not list(root.glob("mtp*.safetensors"))
    model = qwen_mod.load(root, device)
    assert model.mtp is not None and model.mtp.gated is True and model.mtp.down is not None
    engine = MTPEngine(model, model.mtp, linear=qwen_mod.Engine(model).linear)
    dtype = qwen_mod.activation_dtype(gfx_name())
    hidden = torch.randn(1, 1, spec.hidden, generator=torch.Generator().manual_seed(5)).to(device, dtype)
    chain = engine.draft_chain(hidden, 7, 0, 2, engine.fresh_cache(batch=1, total=64, device=device, dtype=dtype),
                               sampling=Sampling(seed=0, temperature=0.0), dtype=dtype)
    assert len(chain) == 2 and all(0 <= token < spec.vocab for token in chain)

    from tensorfold.families.qwen3_5_moe import rocm_engine

    served = rocm_engine(root, context=64)
    assert served.mtp is not None and served.no_drafts is False and served.mtp_depth > 0


def test_slicing_takes_the_head_rows_and_keeps_a_routed_head_whole(tmp_path):
    """Under tp the head's logits projection takes the rank's rows, the rest stays whole, routed is refused."""

    from tensorfold.rocm.slicing import _slice_mtp

    cpu = torch.device("cpu")
    root = _checkpoint(tmp_path, gptq=False)
    spec = qwen_mod.load(root, cpu).spec
    _write_qwen3_mtp(root, spec, routed=False, head_proj=True)
    head = qwen_mod.load_mtp_head(root, spec, _BITS, _GROUP, cpu)
    rows = head.head.words.shape[0]
    _slice_mtp(head, spec, 1, 2)
    assert head.head.words.shape[0] == rows // 2
    assert head.q.words.shape[0] == spec.heads * spec.head_dim * 2
    assert head.fc_e.words.shape[0] == spec.hidden

    _write_qwen3_mtp(root, spec, routed=True)
    routed = qwen_mod.load_mtp_head(root, spec, _BITS, _GROUP, cpu)
    assert routed.head is None
    experts = routed.moe.experts.count
    _slice_mtp(routed, spec, 0, 2)
    assert routed.moe.remap is None and routed.moe.experts.count == experts      # the head stays whole


def _served(tmp_path, *, no_drafts=False, eos=(0,)):
    """The tiny MoE checkpoint with its own MTP head, served through the real engine on one rank."""

    from tensorfold.rocm.engine import QwenEngine
    from tensorfold.rocm.qwen import Engine as Kernels

    device = torch.device("cuda")
    root = _checkpoint(tmp_path, gptq=False)
    spec = qwen_mod.load(root, torch.device("cpu")).spec
    _write_qwen3_mtp(root, spec, routed=True, embedded=True)
    model = qwen_mod.load(root, device)
    assert model.mtp is not None
    return QwenEngine(model, Kernels(model, schedule="auto"), eos, tp=1, rank=0, rccl=None,
                      no_drafts=no_drafts, mtp_depth=4), model


def _engine(model, kernels, *, eos=(0,), no_drafts=False, depth=4):
    """An engine over weights already loaded, so a second one cannot see a rebuilt model."""

    from tensorfold.rocm.engine import QwenEngine

    return QwenEngine(model, kernels, eos, tp=1, rank=0, rccl=None, no_drafts=no_drafts, mtp_depth=depth)


def test_drafting_equals_serial_token_for_token(tmp_path):
    """The gate: greedy drafting equals no_drafts token for token, and both stop at ``max_tokens``."""

    prompt, sampling, room = [1, 2, 3, 4], Sampling(seed=11, temperature=0.0), 12
    drafted, model = _served(tmp_path)
    serial = _engine(model, drafted.kernels, no_drafts=True, depth=0)
    want, got = [], []
    serial.generate(prompt, room, sampling, want.extend, stop_eos=False)
    drafted.generate(prompt, room, sampling, got.extend, stop_eos=False)
    assert len(want) == room and got == want


def test_draft_false_decodes_serially(tmp_path):
    """``"draft": false`` is the serial reference: the head drafts nothing and the reply is no_drafts' reply."""

    prompt, sampling, room = [1, 2, 3, 4], Sampling(seed=11, temperature=1.0), 12
    drafted, model = _served(tmp_path)
    serial = _engine(model, drafted.kernels, no_drafts=True, depth=0)

    def refuse(*_args, **_kwargs):
        raise AssertionError("draft=False drafted")

    drafted.mtp.draft_chain = refuse
    want, got = [], []
    serial.generate(prompt, room, sampling, want.extend, stop_eos=False)
    drafted.generate(prompt, room, sampling, got.extend, stop_eos=False, draft=False)
    assert len(want) == room and got == want


def test_a_stop_token_ends_generation_at_that_token(tmp_path):
    """An eos ends a drafted run at its first appearance: the third token's id, inside the first drafted round.

    The client-stop test below pins the other arm of the same ``done`` decision.
    """

    prompt, sampling, room = [1, 2, 3, 4], Sampling(seed=11, temperature=0.0), 12
    drafted, model = _served(tmp_path)
    want = []
    drafted.generate(prompt, room, sampling, want.extend, stop_eos=False)
    assert len(want) == room

    seen = []
    _engine(model, drafted.kernels, eos=(want[2],)).generate(prompt, room, sampling, seen.extend)
    assert seen == want[:want.index(want[2]) + 1]


def test_a_client_stop_inside_the_drafts_ends_generation(tmp_path):
    """A client stop landing inside a drafted batch ends generation there, not after the whole round."""

    drafted, _ = _served(tmp_path)
    seen = []

    def on_tokens(tokens):
        seen.extend(tokens)
        return len(seen) >= 3

    drafted.generate([1, 2, 3, 4], 12, Sampling(seed=11, temperature=0.0), on_tokens, stop_eos=False)
    assert len(seen) == 3


def test_a_checkpoint_without_an_mtp_layer_still_loads(tmp_path):
    """An MLX conversion that dropped the head loads as None rather than refusing the checkpoint."""

    root = _checkpoint(tmp_path, gptq=False)
    model = qwen_mod.load(root, torch.device("cpu"))
    assert model.mtp is None


@pytest.mark.parametrize("routed", [False, True])
def test_the_head_drafts_a_deterministic_chain(tmp_path, routed):
    """The head's own forward drives the chain: a greedy chain is stable and the MLP is on its path."""

    model, head, engine, device = _loaded(tmp_path, routed=routed)
    dtype = qwen_mod.activation_dtype(gfx_name())
    hidden = torch.randn(1, 1, model.spec.hidden, generator=torch.Generator().manual_seed(3)).to(device, dtype)
    cache = engine.fresh_cache(batch=1, total=64, device=device, dtype=dtype)
    sampling = Sampling(seed=0, temperature=0.0)
    first = engine.draft_chain(hidden, 7, 0, 3, engine.fresh_cache(batch=1, total=64, device=device, dtype=dtype),
                               sampling=sampling, dtype=dtype)
    second = engine.draft_chain(hidden, 7, 0, 3, engine.fresh_cache(batch=1, total=64, device=device, dtype=dtype),
                                sampling=sampling, dtype=dtype)
    assert len(first) == 3 and first == second
    assert all(0 <= token < model.spec.vocab for token in first)

    logits, _ = engine.forward(hidden, torch.tensor([7], device=device), 0, cache, dtype=dtype)
    assert logits.shape == (1, 1, model.spec.vocab)
    engine.absorb([hidden, hidden], [3, 4], cache, dtype=dtype)
    assert cache["len"] == 3
    head.moe, head.gate, head.up, head.down = (None, None, None, None)
    head.input_norm = head.post_norm = None
    plain, _ = engine.forward(hidden, torch.tensor([7], device=device), 0,
                              engine.fresh_cache(batch=1, total=64, device=device, dtype=dtype), dtype=dtype)
    assert not torch.equal(logits, plain)


def test_the_head_residual_is_the_fc_output(tmp_path):
    """The input norm feeds the head's attention only; the residual is the fc output, as in the MLX layer."""

    from tensorfold.rocm import forward
    from tensorfold.rocm import mtp as mtp_mod
    from tensorfold.rocm.qwen_math import gather_rows, rms_norm

    model, head, engine, device = _loaded(tmp_path, routed=False)
    spec, dtype = model.spec, qwen_mod.activation_dtype(gfx_name())
    hidden = torch.randn(1, 1, spec.hidden, generator=torch.Generator().manual_seed(5)).to(device, dtype)
    token = torch.tensor([7], device=device)
    _, got = engine.forward(hidden, token, 0, engine.fresh_cache(batch=1, total=8, device=device, dtype=dtype),
                            dtype=dtype)

    def project(x, norm, weight):
        return engine.linear(rms_norm(x, norm, spec.eps).view(-1, spec.hidden), weight).view(1, 1, -1)

    emb = gather_rows(model.embed, token, dtype=dtype).view(1, 1, -1)
    x = project(emb, head.fc_e_norm, head.fc_e) + project(hidden, head.fc_h_norm, head.fc_h)
    cache = engine.fresh_cache(batch=1, total=8, device=device, dtype=dtype)
    attended, _ = mtp_mod._attention(head, rms_norm(x, head.input_norm, spec.eps), cache, 0, dtype, engine.linear,
                                     spec, engine._heads, engine._kv_heads)
    x = x + attended
    x = x + forward._mlp(spec, head, rms_norm(x, head.post_norm, spec.eps), engine.linear)
    assert torch.equal(got, rms_norm(x, head.final_norm, spec.eps))
