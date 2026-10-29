"""Propose fixed-byte rank-local cache allocations from measured layer service curves.

This produces experimental line deltas, never a default policy or a speed claim.
Each point must include actual variable device bytes and raw CUDA event samples.
Timing includes miss transfers, reuse and split execution; copy duration alone is
not an additive cost when computation overlaps it. Validate proposals with the
uninstrumented served client before enabling them.
"""

import argparse
import json
from pathlib import Path
import statistics


MAX_STATES = 250_000


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def propose(profile):
    if profile.get("schema") != 1:
        raise ValueError("unsupported profile schema")
    identity = profile["identity"]
    required = ("build_sha256", "checkpoint_sha256", "workload_sha256", "settings_sha256")
    for key in required:
        text = identity[key]
        if not isinstance(text, str) or len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError(f"invalid {key}")
    rank = integer(profile["rank"], "rank")
    world = integer(profile["world"], "world", 1)
    if world not in (1, 2) or rank >= world:
        raise ValueError("invalid rank/world")
    uniform = integer(profile["uniform_lines"], "uniform_lines", 10)
    if uniform > 512 // world:
        raise ValueError("uniform allocation exceeds local experts")
    layers = profile["layers"]
    if not layers or len(layers) > 48:
        raise ValueError("invalid decoder pool count")
    minimum_samples = integer(profile.get("minimum_samples", 5), "minimum_samples", 3)
    budget = 0
    baseline_cost = 0
    baseline_lines = uniform * len(layers)
    curves = []
    for index, layer in enumerate(layers):
        if layer["layer"] != index:
            raise ValueError("decoder layers must be complete and ordered")
        points = []
        seen = set()
        baseline = None
        for point in layer["points"]:
            if point.get("identity") != identity:
                raise ValueError("mixed measurement identity")
            if point.get("outputs_equal") is not True:
                raise ValueError("unverified measurement outputs")
            lines = integer(point["lines"], "lines", 10)
            if lines > 512 // world or lines in seen:
                raise ValueError("invalid or duplicate capacity")
            seen.add(lines)
            device_bytes = integer(point["variable_device_bytes"], "variable_device_bytes", 1)
            samples = point["service_ns"]
            if len(samples) < minimum_samples:
                raise ValueError("insufficient raw timing samples")
            for value in samples:
                integer(value, "service_ns", 1)
            cost = statistics.median_low(samples)
            points.append((lines, device_bytes, cost))
            if lines == uniform:
                baseline = (device_bytes, cost)
        if baseline is None:
            raise ValueError("missing measured uniform baseline")
        budget += baseline[0]
        baseline_cost += baseline[1]
        curves.append(sorted(points))
    # Multiple-choice exact budget search; both line count and full variable bytes
    # must match because the native admission check enforces both invariants.
    states = {(0, 0): (0, 0, ())}
    for index, curve in enumerate(curves):
        remaining = len(curves) - index - 1
        next_states = {}
        for (used_lines, used_bytes), (cost, displacement, chosen) in states.items():
            for lines, device_bytes, measured_cost in curve:
                key = (used_lines + lines, used_bytes + device_bytes)
                if key[0] + remaining * 10 > baseline_lines or key[1] > budget:
                    continue
                candidate = (cost + measured_cost, displacement + abs(lines - uniform), chosen + (lines,))
                previous = next_states.get(key)
                if previous is None or candidate < previous:
                    next_states[key] = candidate
                    if len(next_states) > MAX_STATES:
                        raise ValueError("measurement grid exceeds bounded allocation search")
        states = next_states
    cost, _, chosen = states[(baseline_lines, budget)]
    deltas = ",".join(f"{layer}:{lines - uniform:+d}" for layer, lines in enumerate(chosen) if lines != uniform)
    return {
        "schema": 1,
        "identity": identity,
        "rank": rank,
        "world": world,
        "experimental_only": True,
        "variable_device_bytes": budget,
        "baseline_variable_device_bytes": budget,
        "baseline_surrogate_ns": baseline_cost,
        "candidate_surrogate_ns": cost,
        "surrogate": "sum of per-layer median last-invocation GPU service times; not served latency",
        "lines": chosen,
        "environment": {f"TF_FLASHNEXT_EXL3_LINE_DELTAS_R{rank}": deltas} if deltas else {},
        "requires": "native exact-byte admission, unseen-workload full-output equality and uninstrumented served ABAB",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = propose(json.loads(args.profile.read_text()))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
