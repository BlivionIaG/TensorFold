#!/usr/bin/env python3
"""Developer-only MLX teacher-forced LFM3 text reference, independent of native kernels."""

import argparse
import json
import math
from pathlib import Path


def canonical(name):
    if name.endswith('.weight'):
        name = name[:-7]
    for prefix in ('model.language_model.', 'language_model.', 'text_model.', 'model.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def quantization(config, name):
    spec = config.get('quantization', config.get('quantization_config'))
    if not isinstance(spec, dict):
        raise ValueError('Checkpoint needs affine quantization metadata')
    width = spec
    for key, value in spec.items():
        if isinstance(value, dict) and canonical(key) == canonical(name):
            width = value
            break
    bits, group = width['bits'], width['group_size']
    if bits not in (4, 6, 8) or group not in (32, 64) or width.get('mode', 'affine') != 'affine':
        raise ValueError(f'Unsupported quantization for {name}')
    return bits, group


class Reference:
    def __init__(self, directory, mx, attention='sdpa'):
        self.mx = mx
        self.attention_mode = attention
        self.config = json.loads((directory / 'config.json').read_text())
        self.text = self.config.get('text_config', self.config)
        if self.text.get('model_type') != 'lfm3' or not self.text.get('tie_word_embeddings'):
            raise ValueError('Expected LFM3 with a tied output embedding')
        self.weights = {}
        index = directory / 'model.safetensors.index.json'
        files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else [
            path.name for path in directory.glob('*.safetensors')
        ]
        if not files:
            raise ValueError('No safetensors checkpoint found')
        for name in files:
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Shard path escapes checkpoint directory')
            for key, value in mx.load(str(path)).items():
                if key in self.weights:
                    raise ValueError(f'Duplicate tensor {key}')
                self.weights[key] = value
        wrapped = 'language_model.model.embed_tokens.weight' in self.weights
        bare = 'model.embed_tokens.weight' in self.weights
        if wrapped == bare:
            raise ValueError('Checkpoint must use exactly one text namespace')
        self.prefix = 'language_model.model.' if wrapped else 'model.'

    def round(self, x):
        return x.astype(self.mx.bfloat16)

    def linear(self, x, module):
        bits, group = quantization(self.config, module)
        return self.round(self.mx.quantized_matmul(
            x, self.weights[module + '.weight'], self.weights[module + '.scales'],
            self.weights[module + '.biases'], transpose=True, group_size=group, bits=bits,
        ))

    def embedding(self, tokens):
        mx = self.mx
        name = self.prefix + 'embed_tokens'
        bits, group = quantization(self.config, name)
        ids = mx.array(tokens, dtype=mx.int32)
        return self.round(mx.dequantize(
            mx.take(self.weights[name + '.weight'], ids, axis=0),
            mx.take(self.weights[name + '.scales'], ids, axis=0),
            mx.take(self.weights[name + '.biases'], ids, axis=0), group_size=group, bits=bits,
        ))

    def norm(self, x, name):
        mx = self.mx
        values = x.astype(mx.float32)
        scale = mx.rsqrt(mx.mean(values * values, axis=-1, keepdims=True) + self.text['norm_eps'])
        normalized = self.round(values * scale)
        return self.round(normalized * self.weights[name + '.weight'])

    def convolution(self, x, prefix):
        mx = self.mx
        width = self.text['hidden_size']
        b, gate, values = mx.split(self.linear(x, prefix + 'conv.in_proj'), 3, axis=-1)
        gated = self.round(b * values)
        kernel = self.weights[prefix + 'conv.conv.weight'].astype(mx.float32)[:, :, 0]
        taps = kernel.shape[1]
        padded = mx.pad(gated.astype(mx.float32), ((taps - 1, 0), (0, 0)))
        total = mx.zeros((x.shape[0], width), dtype=mx.float32)
        for tap in range(taps):
            total = total + padded[tap:tap + x.shape[0]] * kernel[:, tap]
        result = self.round(self.round(total) * gate)
        return self.linear(result, prefix + 'conv.out_proj')

    def rotary(self, x, kind):
        mx = self.mx
        dim = self.text['head_dim']
        spec = self.text['rope_parameters'][kind]
        pairs = int(dim * spec.get('partial_rotary_factor', 1.0)) // 2
        frequencies = mx.power(mx.array(spec['rope_theta'], dtype=mx.float32),
                               -mx.arange(0, pairs * 2, 2, dtype=mx.float32) / dim)
        frequencies = mx.concatenate((frequencies, mx.zeros((dim // 2 - pairs,), dtype=mx.float32)))
        angles = mx.arange(x.shape[0], dtype=mx.float32)[:, None] * frequencies[None, :]
        angles = mx.concatenate((angles, angles), axis=-1)
        cosine = self.round(mx.cos(angles)).astype(mx.float32)[:, None, :]
        sine = self.round(mx.sin(angles)).astype(mx.float32)[:, None, :]
        values = x.astype(mx.float32)
        first, second = mx.split(values, 2, axis=-1)
        rotated = mx.concatenate((-second, first), axis=-1)
        return self.round(values * cosine + rotated * sine)

    def attention(self, x, prefix, kind):
        mx = self.mx
        length = x.shape[0]
        heads, kv, dim = (self.text[key] for key in ('num_attention_heads', 'num_key_value_heads', 'head_dim'))
        q = self.linear(x, prefix + 'self_attn.q_proj').reshape(length, heads, dim)
        k = self.linear(x, prefix + 'self_attn.k_proj').reshape(length, kv, dim)
        v = self.linear(x, prefix + 'self_attn.v_proj').reshape(length, kv, dim)
        q = self.rotary(self.norm(q, prefix + 'self_attn.q_norm'), kind).transpose(1, 0, 2)[None]
        k = self.rotary(self.norm(k, prefix + 'self_attn.k_norm'), kind).transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        positions = mx.arange(length)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        if kind == 'sliding_attention':
            allowed = allowed & (distance < self.text['sliding_window'])
        if self.attention_mode == 'sdpa':
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(dim), mask=allowed)
        else:
            k = mx.repeat(k, heads // kv, axis=1)
            v = mx.repeat(v, heads // kv, axis=1)
            scores = self.round(mx.matmul(q, k.swapaxes(-1, -2)))
            scores = self.round(scores * (1 / math.sqrt(dim)))
            scores = mx.where(allowed, scores.astype(mx.float32), -float('inf'))
            probabilities = self.round(mx.softmax(scores, axis=-1))
            output = mx.matmul(probabilities, v)
        output = self.round(output[0].transpose(1, 0, 2))
        logits = self.linear(x, prefix + 'self_attn.gate_proj')
        gate = self.round(mx.sigmoid(logits.astype(mx.float32)))
        output = self.round(output * gate[:, :, None]).reshape(length, heads * dim)
        return self.linear(output, prefix + 'self_attn.o_proj')

    def run(self, tokens, dump=None, last_only=False):
        mx = self.mx
        hidden = self.embedding(tokens)
        if dump:
            dump('embedding', hidden)
        for index, kind in enumerate(self.text['layer_types']):
            prefix = self.prefix + f'layers.{index}.'
            x = self.norm(hidden, prefix + 'operator_norm')
            result = self.convolution(x, prefix) if kind == 'conv' else self.attention(x, prefix, kind)
            hidden = self.round(hidden + self.norm(result, prefix + 'post_operator_norm'))
            x = self.norm(hidden, prefix + 'ffn_norm')
            gate = self.linear(x, prefix + 'feed_forward.w1')
            up = self.linear(x, prefix + 'feed_forward.w3')
            silu = self.round(gate.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32)))
            result = self.linear(self.round(silu * up), prefix + 'feed_forward.w2')
            hidden = self.round(hidden + self.norm(result, prefix + 'post_ffn_norm'))
            mx.eval(hidden)
            if dump:
                dump(f'layer_{index:02d}', hidden)
        final = self.norm(hidden[-1:] if last_only else hidden, self.prefix + 'embedding_norm')
        if dump:
            dump('final_norm', final)
        return self.linear(final, self.prefix + 'embed_tokens')


def write_bf16(path, values, mx, np):
    floats = np.asarray(values.astype(mx.float32))
    raw = (floats.view(np.uint32) >> 16).astype('<u2')
    path.write_bytes(raw.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--tokens', required=True, type=Path, help='One-dimensional integer NumPy .npy token IDs')
    parser.add_argument('--output', required=True, type=Path, help='Little-endian BF16 logits, rows then vocabulary')
    parser.add_argument('--last-only', action='store_true')
    parser.add_argument('--attention', choices=('sdpa', 'eager'), default='sdpa')
    parser.add_argument('--intermediates-dir', type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    tokens = np.load(args.tokens, allow_pickle=False)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or tokens.size == 0:
        parser.error('Token file must contain a nonempty one-dimensional integer array')
    model = Reference(args.model_dir, mx, args.attention)
    if tokens.min() < 0 or tokens.max() >= model.text['vocab_size']:
        parser.error('Token IDs exceed the checkpoint vocabulary')
    if tokens.size > model.text['max_position_embeddings']:
        parser.error('Token sequence exceeds the checkpoint context')
    intermediates = {}

    def dump(name, values):
        write_bf16(args.intermediates_dir / (name + '.bf16'), values, mx, np)
        intermediates[name] = {'shape': list(values.shape), 'dtype': 'BF16'}

    if args.intermediates_dir:
        args.intermediates_dir.mkdir(parents=True, exist_ok=True)
    logits = model.run(tokens, dump if args.intermediates_dir else None, args.last_only)
    mx.eval(logits)
    write_bf16(args.output, logits, mx, np)
    metadata = {
        'shape': list(logits.shape), 'dtype': 'BF16', 'byte_order': 'little',
        'attention': args.attention, 'token_count': int(tokens.size), 'intermediates': intermediates,
    }
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
