"""Routed experts on RDNA: the pick is exact, and both weight kinds land on the dequantized product.

The routing reference consumes the very logits the buffer was filled from, so the picks, the weights and
the plan compare exactly. The expert reference dequantizes the pack the test itself wrote, so a mismatch is
the kernel's. The loader test writes a one-layer checkpoint whose routed experts are 4-bit and whose
routers are 8-bit, as Qwen3.6 is, and tells the shared expert apart by its scale.
"""

import json

import pytest
import torch

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm import moe as moe_mod  # noqa: E402
from tensorfold.rocm import qwen as qwen_mod  # noqa: E402
from tensorfold.rocm.build import gfx_name  # noqa: E402
from tensorfold.rocm.experts import AffineExperts, GptqExperts, Plan, route  # noqa: E402

try:
    _GFX = gfx_name()
except RuntimeError:
    pytest.skip("not an RDNA device", allow_module_level=True)
_ACT = qwen_mod.activation_dtype(_GFX)

_HIDDEN, _WIDTH, _EXPERTS, _TOP_K, _GROUP = 32, 32, 4, 2, 32
_BITS = 4


class _Sizes:
    num_experts_per_tok = _TOP_K
    num_experts = _EXPERTS
    moe_intermediate_size = _WIDTH
    hidden_size = _HIDDEN


def _affine(n, k, bits, group, seed):
    """MLX affine packing: ``bits`` consecutive K codes per row and one group scale and bias each ``group``."""

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (n, k), generator=g)
    words = torch.zeros((n, k * bits // 32), dtype=torch.int64)
    for col in range(k):
        bit = col * bits
        word, shift = divmod(bit, 32)
        value = (codes[:, col] & ((1 << bits) - 1)).to(torch.int64)
        words[:, word] |= value << shift
        if shift + bits > 32:
            words[:, word + 1] |= value >> (32 - shift)
    table = torch.rand((n, k // group), generator=g) * 0.05 + 0.02
    return words.to(torch.int32), table, torch.randn((n, k // group), generator=g) * 0.02


def _dequant(words, scale, bias, bits, group):
    codes = words.to(torch.int64) & 0xFFFFFFFF
    per = 32 // bits
    shifts = torch.arange(per, device=words.device, dtype=torch.int64) * bits
    k = scale.shape[-1] * group
    unpacked = ((codes[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], k).to(torch.float32)
    return unpacked * scale.float().repeat_interleave(group, -1) + bias.float().repeat_interleave(group, -1)


def _stack(seed, tag=0.0):
    """``_EXPERTS`` routed stacks with the shared expert last, its scale tagged so it can be told apart."""

    mine = [_affine(_WIDTH, _HIDDEN, _BITS, _GROUP, seed * 10 + expert) for expert in range(_EXPERTS)]
    shared = _affine(_WIDTH, _HIDDEN, _BITS, _GROUP, seed * 10 + 99)
    words = torch.cat([torch.stack([item[0] for item in mine]), shared[0][None]])
    scale = torch.cat([torch.stack([item[1] for item in mine]), (shared[1] + tag)[None]])
    bias = torch.cat([torch.stack([item[2] for item in mine]), shared[2][None]])
    return words, scale, bias


def _affine_experts(tag=0.0):
    def on_device(part):
        return part[0].cuda(), part[1].cuda(), part[2].cuda()

    return AffineExperts(up=on_device(_stack(1, tag)), down=on_device(_stack(2, tag)),
                         gate=on_device(_stack(3, tag)), bits=_BITS, group=_GROUP)


def _routed(experts, router):
    return moe_mod.Routed(router, experts, _TOP_K)


def _pick_reference(logits):
    """The documented pick over one row's fp32 logits: no kernel runs, so the comparison is exact."""

    gates = logits[:, :_EXPERTS]
    order = torch.argsort(gates, dim=1, descending=True, stable=True)[:, :_TOP_K]
    picked = torch.gather(gates, 1, order)
    weights = torch.exp(picked - picked[:, :1])
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(torch.bfloat16).to(torch.float32)
    shared = logits[:, _EXPERTS].to(torch.bfloat16).to(torch.float32)
    return order, weights, torch.sigmoid(shared).to(torch.bfloat16).to(torch.float32)


def test_pick_and_plan_are_exact():
    """The router's top k, its renormalized weights, the shared slot and the grouping are all bit-exact."""

    rows = 6
    gen = torch.Generator().manual_seed(4)
    x = torch.randn((rows, _HIDDEN), generator=gen).cuda()
    router = torch.randn((_EXPERTS + 1, _HIDDEN), generator=gen)
    logits = x.float() @ router.cuda().float().t()

    buf = moe_mod.MoEBuffers(rows, _Sizes(), torch.device("cuda"))
    moe_mod.select_rows(logits, buf, _TOP_K, _EXPERTS)
    order, weights, shared = _pick_reference(logits.cpu())

    assert torch.equal(buf.pick[:rows, :_TOP_K].cpu(), order.to(torch.int32))
    assert torch.equal(buf.wts[:rows, :_TOP_K].cpu(), weights)
    assert torch.equal(buf.wts[:rows, _TOP_K].cpu(), shared)
    assert torch.equal(buf.pick[:rows, _TOP_K].cpu(), torch.full((rows,), _EXPERTS, dtype=torch.int32))

    picks = buf.pick[:rows]
    plan = Plan(rows, _TOP_K + 1, _EXPERTS + 1, torch.device("cuda"))
    route(picks, plan)
    flat = [int(v) for v in picks.reshape(-1).cpu()]
    assert all(int(items[2]) > 0 for items in plan.items[: plan.count].cpu().tolist())
    grouped = {expert: [] for expert in range(_EXPERTS + 1)}
    for expert, first, count in plan.items[: plan.count].cpu().tolist():
        members = [int(m) for m in plan.members[first:first + count].cpu()]
        assert all(flat[member] == expert for member in members)
        grouped[expert] += members
    assert sorted(pair for pairs in grouped.values() for pair in pairs) == sorted(range(rows * (_TOP_K + 1)))


def _moe_reference(x, logits, experts, weight_of):
    """Route one row at a time and sum its slots in order in fp32, as ``combine`` describes."""

    order, weights, shared = _pick_reference(logits)
    rows = x.shape[0]
    out = torch.empty(rows, _HIDDEN)
    for row in range(rows):
        total = torch.zeros(_HIDDEN)
        for slot in range(_TOP_K + 1):
            expert = _EXPERTS if slot == _TOP_K else int(order[row, slot])
            weight = shared[row] if slot == _TOP_K else weights[row, slot]
            gate, up, down = (weight_of(part, expert) for part in (experts.gate, experts.up, experts.down))
            activated = torch.nn.functional.silu(x[row].cpu() @ gate.t()) * (x[row].cpu() @ up.t())
            total += (activated.to(torch.bfloat16).float() @ down.t()) * weight
        out[row] = total.to(torch.bfloat16).float()
    return out


@pytest.mark.parametrize("rows", [1, 5])
def test_affine_experts_match_the_dequantized_product(rows):
    """The affine kind picks the same experts and lands on the product its own pack dequantizes to."""

    gen = torch.Generator().manual_seed(11)
    x = (torch.randn((rows, _HIDDEN), generator=gen) * 0.5).cuda().to(_ACT)
    router = torch.randn((_EXPERTS + 1, _HIDDEN), generator=gen).cuda().to(torch.bfloat16)
    experts = _affine_experts()
    logits = (x.float() @ router.float().t()).cpu()
    got = moe_mod.run(x, _routed(experts, router))

    def weight_of(part, expert):
        return _dequant(part[0][expert].cpu(), part[1][expert].cpu(), part[2][expert].cpu(), _BITS, _GROUP)

    want = _moe_reference(x.cpu().float(), logits, experts, weight_of)
    assert got.shape == (rows, _HIDDEN) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float().cpu(), want, rtol=3e-2, atol=3e-2)


def test_the_shared_expert_is_the_last_slot():
    """Slot ``top_k`` is expert ``E``: tagging that stack's scale alone moves the output."""

    gen = torch.Generator().manual_seed(12)
    x = (torch.randn((3, _HIDDEN), generator=gen) * 0.5).cuda().to(_ACT)
    router = torch.randn((_EXPERTS + 1, _HIDDEN), generator=gen).cuda().to(torch.bfloat16)
    plain = moe_mod.run(x, _routed(_affine_experts(), router))
    tagged = moe_mod.run(x, _routed(_affine_experts(tag=0.5), router))
    assert not torch.equal(plain.cpu(), tagged.cpu())


@pytest.mark.skipif(_GFX != "gfx1030", reason="the W4A16 path is the gfx1030 fp16 dot")
def test_gptq_experts_match_the_fp32_reference():
    """The W4A16 kind takes bf16 activations, stacks gate first, and lands on the int4 product."""

    from test_qgemm import _expert_weight, _pack_experts, _plan

    from tensorfold.rocm.qgemm import moe as gptq_moe

    mats, k, n, group = 2, _HIDDEN, _WIDTH, _GROUP
    codes, zeros, qweight, qzeros, scales = _pack_experts(mats, _EXPERTS + 1, n, k, group, seed=31)
    experts = GptqExperts(up=qweight, up_z=qzeros, up_s=scales, down=qweight[:1], down_z=qzeros[:1],
                          down_s=scales[:1], group=group, v2=False)
    assert experts.swiglu and experts.count == _EXPERTS + 1
    assert (experts.width, experts.dims) == (n, k)

    rows, slots = 4, _TOP_K + 1
    picks = torch.randint(0, _EXPERTS + 1, (rows, slots), generator=torch.Generator().manual_seed(8))
    members, items = _plan(picks, _EXPERTS + 1, 16)
    gen = torch.Generator().manual_seed(13)
    x = (torch.randn((rows, k), generator=gen) * 0.5).cuda().to(torch.bfloat16)
    got = gptq_moe(x, qweight.cuda(), qzeros.cuda(), scales.cuda(), items.cuda(), members.cuda(), rows, slots,
                   epi=2, block_m=4, use_v2_format=False)

    ref = torch.empty(rows * slots, n)
    for pair in range(rows * slots):
        pid = int(members[pair])
        expert = int(picks.reshape(-1)[pid])
        row = x[pid // slots].float().cpu()
        gate = _expert_weight(codes, zeros, scales, 0, expert, 1)
        up = _expert_weight(codes, zeros, scales, 1, expert, 1)
        ref[pid] = (torch.nn.functional.silu(row @ gate.t()) * (row @ up.t())).to(torch.bfloat16).float()
    torch.testing.assert_close(got.float().cpu(), ref, rtol=6e-2, atol=2.5)


def _gptq_projection(n, k, group, seed):
    from test_qgemm import _pack

    _, _, qweight, qzeros, scales = _pack(n, k, group, seed)
    return {".qweight": qweight, ".qzeros": qzeros, ".scales": scales}


def _checkpoint(root, *, gptq):
    """Write a one-layer ``qwen3_5_moe`` checkpoint: 4-bit experts, 8-bit routers, the shared expert last."""

    from safetensors.torch import save_file
    from test_qgemm import _pack_experts

    hidden, width, group, vocab, conv = _HIDDEN, _WIDTH, _GROUP, 48, 4
    heads, head_dim, key_width, value_width, value_heads = 4, 16, 4 * 16, 16 * 16, 16
    text = {"hidden_size": hidden, "num_hidden_layers": 1, "num_attention_heads": heads, "num_key_value_heads": 2,
            "head_dim": head_dim, "linear_num_key_heads": 4, "linear_num_value_heads": value_heads,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": conv,
            "vocab_size": vocab, "rope_parameters": {"partial_rotary_factor": 0.5},
            "full_attention_interval": 2, "layer_types": ["linear_attention"], "num_experts": _EXPERTS,
            "num_experts_per_tok": _TOP_K, "moe_intermediate_size": width, "norm_topk_prob": True}
    cfg = {"model_type": "qwen3_5_moe", "tie_word_embeddings": True, "text_config": text,
           "quantization": {"mode": "gptq" if gptq else "affine", "bits": _BITS, "group_size": group}}
    base = "language_model.model."
    tensors = {f"{base}norm.weight": torch.ones(hidden)}

    def add(key, n, k, seed, affine=False):
        if gptq and not affine:
            tensors.update({key + suffix: value for suffix, value in _gptq_projection(n, k, group, seed).items()})
            return
        words, scale, bias = _affine(n, k, _BITS, group, seed)
        tensors[key + ".weight"], tensors[key + ".scales"], tensors[key + ".biases"] = words, scale, bias

    add(f"{base}embed_tokens", vocab, hidden, 1, affine=True)
    layer = f"{base}layers.0."
    tensors[layer + "input_layernorm.weight"] = torch.ones(hidden)
    tensors[layer + "post_attention_layernorm.weight"] = torch.ones(hidden)
    lin = layer + "linear_attn."
    add(lin + "in_proj_qkv", 2 * key_width + value_width, hidden, 2)
    add(lin + "in_proj_z", value_width, hidden, 3)
    add(lin + "in_proj_a", value_heads, hidden, 4)
    add(lin + "in_proj_b", value_heads, hidden, 5)
    tensors[lin + "conv1d.weight"] = torch.randn(2 * key_width + value_width, conv) * 0.05
    tensors[lin + "A_log"] = torch.randn(value_heads)
    tensors[lin + "dt_bias"] = torch.randn(value_heads)
    tensors[lin + "norm.weight"] = torch.ones(16)
    add(lin + "out_proj", hidden, value_width, 6)

    mlp = layer + "mlp."
    shapes = {"gate_proj": (width, hidden), "up_proj": (width, hidden), "down_proj": (hidden, width)}
    for stack, count in (("switch_mlp", _EXPERTS), ("shared_expert", 1)):
        for index, (name, (n, k)) in enumerate(shapes.items()):
            if gptq:
                parts = _pack_experts(1, count, n, k, group, seed=100 + index)
                for suffix, tensor in zip((".qweight", ".qzeros", ".scales"), parts[2:]):
                    tensors[f"{mlp}{stack}.{name}{suffix}"] = tensor[0]
                continue
            for suffix, part in ((".weight", 0), (".scales", 1), (".biases", 2)):
                tensors[f"{mlp}{stack}.{name}{suffix}"] = torch.cat(
                    [_affine(n, k, _BITS, group, 700 + 10 * index + expert)[part][None] for expert in range(count)])
    def router(name, count, seed):
        if gptq:
            tensors[f"{mlp}{name}.weight"] = torch.randn(count, hidden, generator=torch.Generator().manual_seed(seed))
            return
        words, scale, bias = _affine(count, hidden, 8, group, seed)
        tensors[f"{mlp}{name}.weight"] = words
        tensors[f"{mlp}{name}.scales"] = scale
        tensors[f"{mlp}{name}.biases"] = bias

    router("gate", _EXPERTS, 800)
    router("shared_expert_gate", 1, 801)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps(cfg))
    return root


def test_loader_reads_a_moe_checkpoint(tmp_path):
    """A MoE checkpoint loads as a routed layer: E + 1 experts, the shared one last, the routers 8-bit."""

    model = qwen_mod.load(_checkpoint(tmp_path, gptq=False), torch.device("cpu"))
    layer = model.layers[0]
    assert layer.gate is None and layer.up is None and layer.down is None and layer.moe is not None
    experts = layer.moe.experts
    assert isinstance(experts, AffineExperts) and experts.count == _EXPERTS + 1
    assert experts.swiglu and (experts.width, experts.dims) == (_WIDTH, _HIDDEN)
    assert layer.moe.top_k == _TOP_K and model.spec.experts == _EXPERTS and model.spec.moe_width == _WIDTH
    assert tuple(layer.moe.router.shape) == (_EXPERTS + 1, _HIDDEN) and layer.moe.router.dtype == torch.bfloat16

    stack = experts.up
    assert not torch.equal(stack[0][_EXPERTS], stack[0][_EXPERTS - 1])
    assert float(layer.moe.router[_EXPERTS].abs().sum()) != float(layer.moe.router[_EXPERTS - 1].abs().sum())


def test_loader_dispatches_the_expert_kind(tmp_path):
    """The same family loads a GPTQ checkpoint's experts as the W4A16 kind, gate first and the shared last."""

    model = qwen_mod.load(_checkpoint(tmp_path, gptq=True), torch.device("cpu"))
    experts = model.layers[0].moe.experts
    assert isinstance(experts, GptqExperts) and experts.swiglu and experts.count == _EXPERTS + 1
    assert experts.up.shape[0] == 2 and experts.down.shape[0] == 1 and experts.group == _GROUP
    assert experts.down.shape[1] == _EXPERTS + 1
    assert model.layers[0].moe.router.dtype == torch.bfloat16


def test_the_engine_serves_a_moe_checkpoint(tmp_path):
    """The whole engine runs a MoE stack: a greedy continuation comes back and the experts are on the path."""

    from tensorfold.rocm.qwen import Engine

    model = qwen_mod.load(_checkpoint(tmp_path, gptq=False), torch.device("cuda"))
    engine = Engine(model)
    got = engine.generate([[1, 2, 3]], 2)
    assert len(got) == 1 and len(got[0]) == 2
    assert engine.projections > 0


@pytest.mark.skipif(_GFX != "gfx1030", reason="the W4A16 path is the gfx1030 fp16 dot")
def test_the_engine_serves_gptq_experts(tmp_path):
    """A GPTQ checkpoint's experts run through the same engine, so the dispatch is not just a load-time pick."""

    from tensorfold.rocm.qwen import Engine

    model = qwen_mod.load(_checkpoint(tmp_path, gptq=True), torch.device("cuda"))
    assert isinstance(model.layers[0].moe.experts, GptqExperts)
    got = Engine(model).generate([[1, 2, 3]], 2)
    assert len(got) == 1 and len(got[0]) == 2


def test_the_family_rocm_engine_loads_a_moe_checkpoint(tmp_path):
    """The hook ``tensorfold serve --backend rocm`` calls loads a MoE checkpoint."""

    from tensorfold.families.qwen3_5_moe import rocm_engine

    engine = rocm_engine(_checkpoint(tmp_path, gptq=False), context=64)
    assert isinstance(engine.model.layers[0].moe.experts, AffineExperts)
    assert engine.model.layers[0].moe.top_k == _TOP_K


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_the_ranks_shares_sum_to_the_whole_layer(world, rows):
    """Under tp each rank runs its own experts; the fp32 shares summed give the one-rank layer."""

    from tensorfold.rocm.qwen import _experts_share

    whole = _routed(_affine_experts(), torch.randn(_EXPERTS + 1, _HIDDEN).to(torch.bfloat16).cuda())
    x = torch.randn(rows, _HIDDEN, device="cuda").to(torch.bfloat16)
    want = moe_mod.run(x, whole, prefill=rows > 1).float()
    shares = [moe_mod.run(x, _experts_share(whole, rank, world, "test"), prefill=rows > 1) for rank in range(world)]
    assert all(share.dtype == torch.float32 for share in shares)
    got = torch.stack(shares).sum(dim=0)
    assert torch.allclose(got, want, rtol=1e-2, atol=1e-2), (got - want).abs().max()
    held = [_experts_share(whole, rank, world, "test").experts.count for rank in range(world)]
    assert held == [_EXPERTS // world + 1] + [_EXPERTS // world] * (world - 1)
