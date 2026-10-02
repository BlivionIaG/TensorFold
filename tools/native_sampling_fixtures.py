"""Generate small GPU sampler/top-k oracles using the original Python Metal kernels."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from tensorfold.engine.exact_sampling import Sampling, choose
from tensorfold.engine import gpu_sampling, topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9876)
    arrays, cases = {}, []
    for vocab in (31, 4097):
        raw = rng.standard_normal((4, vocab)).astype(np.float32) * 3
        raw[1] = 0  # tied scores: deterministic token-ID order
        raw[2, 0] = 80  # a single dominant candidate
        for dtype in (mx.float32, mx.bfloat16):
            x = mx.array(raw).astype(dtype)
            for count in (1, 16, min(64, vocab)):
                key = f"c{len(cases)}"
                ix, val = topk.topk_rows(x, count)
                arrays.update({key + ".x": x, key + ".indices": ix, key + ".values": val})
                cases.append(dict(key=key, op="topk", k=count))
            for seed, temp, count, prob, min_p in ((0, 0., 20, .95, 0), (1234, 1., 20, .95, 0),
                                                 (2**64-1, .7, 0, .8, 0), (5678, 2., 2048, 1., 0),
                                                 (1234, 1., 20, .95, .2), (5678, 2., 0, 1., .5),
                                                 (2**64-1, .7, 2048, .8, 1.), (9, 1., 0, 1., 1e-12),
                                                 (10, 0., 0, 1., 1.)):
                for mapped in (False, True):
                    key = f"c{len(cases)}"
                    ids = mx.arange(vocab, dtype=mx.uint32) * 3 + 7 if mapped else None
                    positions = [1, 513, 2049, 262144]
                    settings = Sampling(seed, temperature=temp, top_k=count, top_p=prob, min_p=min_p)
                    expected = gpu_sampling.sample(x, settings if temp else None, positions, ids)
                    # Freeze each oracle while its inputs/settings are current;
                    # saving the whole lazy batch produced non-reproducible draws.
                    mx.eval(expected)
                    arrays.update({key + ".x": x, key + ".expected": expected})
                    if mapped:
                        arrays[key + ".ids"] = ids
                    cases.append(dict(key=key, op="sample", seed=seed, temperature=temp,
                                      k=count, p=prob, min_p=min_p, mapped=mapped, positions=positions))
                    key = f"c{len(cases)}"
                    values = np.array(x.astype(mx.float32))
                    mapping = np.array(ids) if mapped else np.arange(vocab, dtype=np.uint32)
                    cpu = [int(mapping[np.argmax(row)]) if not temp else choose(row, mapping, pos, settings)
                           for row, pos in zip(values, positions)]
                    arrays.update({key + ".x": x, key + ".expected": mx.array(cpu, dtype=mx.uint32)})
                    if mapped:
                        arrays[key + ".ids"] = ids
                    cases.append(dict(key=key, op="cpu_sample", seed=seed, temperature=temp,
                                      k=count, p=prob, min_p=min_p, mapped=mapped, positions=positions))
            for mapped in (False, True):
                key = f"c{len(cases)}"
                mixed = mx.concatenate([x, x], axis=0)
                ids = mx.arange(vocab, dtype=mx.uint32) * 3 + 7 if mapped else None
                positions = [1, 513, 2049, 262144, 1007, 1007, 9, 43]
                settings = [
                    Sampling(0, temperature=.7, top_k=17, top_p=.8, min_p=.03),
                    None,
                    Sampling(2**64-1, temperature=2., top_k=0, top_p=1., min_p=.5),
                    Sampling(2**63+9876, temperature=1.2, top_k=2048, top_p=.95, min_p=1e-12),
                    None,
                    Sampling(5678, temperature=.35, top_k=7, top_p=.87, min_p=1.),
                    Sampling(1234, temperature=1., top_k=1, top_p=1., min_p=0.),
                    Sampling(1234, temperature=1.1, top_k=0, top_p=.96, min_p=.2),
                ]
                expected = gpu_sampling.sample_rows(mixed, settings, positions, ids)
                mx.eval(expected)
                separate = mx.concatenate([
                    gpu_sampling.sample(mixed[row:row + 1], setting, [position], ids)
                    for row, (setting, position) in enumerate(zip(settings, positions))
                ])
                mx.eval(separate)
                np.testing.assert_array_equal(np.array(expected), np.array(separate))
                arrays.update({key + ".x": mixed, key + ".expected": expected})
                if mapped:
                    arrays[key + ".ids"] = ids
                cases.append(dict(key=key, op="sample_rows", mapped=mapped, positions=positions,
                                  settings=[dict(metal=True, seed=setting.seed, temperature=setting.temperature,
                                                 top_k=setting.top_k, top_p=setting.top_p, min_p=setting.min_p)
                                            if setting is not None else dict(metal=True, temperature=0.)
                                            for setting in settings]))
    mx.save_safetensors(str(args.output / "arrays.safetensors"), arrays)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} CPU/Metal sampling/top-k cases in {args.output}")


if __name__ == "__main__":
    from native_runtime import fixture_storage
    fixture_storage(main)()
