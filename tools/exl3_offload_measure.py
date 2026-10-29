"""Normalize diagnostic layer timings only after full served-output verification."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def vectors(report, expected_prompt_tokens=None):
    result = {}
    cells = report.get("cells", [])
    if not cells:
        raise ValueError("empty benchmark")
    for cell in cells:
        key = (cell["prompt"], cell["temperature"], cell["streams"])
        if key in result or cell.get("failed") != 0 or cell.get("unmeasured") != 0:
            raise ValueError("invalid benchmark cell")
        if cell.get("timing_valid") is not True:
            raise ValueError("invalid served timing")
        for check in ("alone", "serial"):
            if check in cell and (cell[check].get("unequal") != 0 or cell[check].get("failed") != 0 or cell[check].get("equal", 0) == 0):
                raise ValueError("failed solo/plain comparison")
        evidence = []
        for rep in cell["reps"]:
            if rep.get("errors") or rep.get("failed") != 0 or rep.get("unmeasured") != 0:
                raise ValueError("failed benchmark repetition")
            rows = rep["evidence"]
            if len(rows) != cell["streams"]:
                raise ValueError("missing stream evidence")
            for row in rows:
                if row.get("complete") is not True or row["tokens"] != report["tokens"]:
                    raise ValueError("incomplete output")
                if expected_prompt_tokens is not None and (row.get("prompt_tokens") != expected_prompt_tokens or row.get("cached_tokens") != 0):
                    raise ValueError("input length or cold-cache evidence differs from required workload")
                pair = tuple(row[field] for field in ("token_sha256", "output_sha256"))
                if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in pair):
                    raise ValueError("missing full output hashes")
                evidence.append((row["tokens"], *pair))
        if not evidence:
            raise ValueError("missing output evidence")
        result[key] = evidence
    return result


def measure(log_text, report, reference, rank, phase="decode", minimum_samples=5, expected_layers=48, expected_prompt_tokens=None):
    if vectors(report, expected_prompt_tokens) != vectors(reference, expected_prompt_tokens):
        raise ValueError("candidate differs from reference")
    if "EXL3-CACHE-INVALID" in log_text or "CUDA_ERROR" in log_text:
        raise ValueError("invalid runtime telemetry")
    layers = {}
    for line in log_text.splitlines():
        if "EXL3-CACHE v=1" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields.get("cache") != "expert-layer" or fields.get("kind") != "decoder":
            continue
        if int(fields["rank"]) != rank or fields["phase"] != phase:
            continue
        if fields.get("timing_scope") != "last_device_invocation":
            continue
        index = int(fields["layer"])
        shape = tuple(int(fields[key]) for key in ("lines", "device_bytes", "line_bytes"))
        item = layers.setdefault(index, dict(shape=shape, service_ns=[], copy_ns=[], exposed_copy_wait_ns=[]))
        if item["shape"] != shape:
            raise ValueError("pool geometry changed during measurement")
        for key in ("service_ns", "copy_ns", "exposed_copy_wait_ns"):
            item[key].append(int(fields["last_device_" + key]))
    if not layers or sorted(layers) != list(range(expected_layers)):
        raise ValueError("missing decoder layer samples")
    for item in layers.values():
        if len(item["service_ns"]) < minimum_samples or min(item["service_ns"]) <= 0:
            raise ValueError("insufficient event samples")
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--expected-prompt-tokens", type=int)
    parser.add_argument("--phase", choices=("decode", "verify"), default="decode")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = measure(args.log.read_text(), json.loads(args.report.read_text()), json.loads(args.reference.read_text()), args.rank, args.phase, expected_prompt_tokens=args.expected_prompt_tokens)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result = {"rank": args.rank, "phase": args.phase, "timing_scope": "last_device_invocation", "layers": data,
              "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
