"""GLP control vectors in Zig against weightless's container reader: the same layers, numbers and digest.

The reference reader is read from a checkout the operator names (--reference); nothing of it is copied here.
GLP and weightless are Matt Suiche's (@msuiche, https://github.com/msuiche/weightless); the spec this tool follows
is docs/spec/GLP.md at pin b181a07d3980fc48abe39d95ddb044459dd0b868. The Zig reader and writer began as #1835's.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Where the reference's container reader sits in a checkout at the pin, and where an older layout kept it.
REFERENCE_PATHS = (
    Path("plugins") / "vllm" / "weightless_steer" / "container.py",
    Path("vllm-plugin") / "weightless_steer" / "container.py",
)

# The reference's bound on a unit direction's squared norm, which this reader's project check matches.
UNIT_TOLERANCE = 1e-3


def load_reference(reference: Path):
    """weightless's container module, read from the checkout; torch is stubbed when it is absent."""
    for relative in REFERENCE_PATHS:
        path = reference / relative
        if path.is_file():
            break
    else:
        tried = ", ".join(str(reference / p) for p in REFERENCE_PATHS)
        raise SystemExit(f"no container.py at {tried}: point --reference at a weightless checkout")
    try:
        import torch  # noqa: F401
    except ImportError:
        # The module imports torch at the top; the parse below needs numpy alone.
        sys.modules["torch"] = types.ModuleType("torch")
    spec = importlib.util.spec_from_file_location("weightless_container", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direction_ids(tensors: dict) -> list[int]:
    """The layer ids the file's tensor names carry, ascending."""
    ids = []
    for name in tensors:
        if not name.startswith("direction."):
            continue
        ids.append(int(name.split(".")[1]))
    return sorted(ids)


def payload_digest(tensors: dict) -> str:
    """SHA-256 over the tensor payloads in the order the file holds them."""
    h = hashlib.sha256()
    for values in tensors.values():
        h.update(values.astype("<f4").tobytes())
    return h.hexdigest()


HUB = "https://huggingface.co/"


def base_of(meta: dict) -> tuple[str, str]:
    """The base repo and commit pin from the spec's general.base_model.0.* keys, as the Zig reader takes them.

    The repo comes from repo_url (the hub's prefix dropped), else organization and name, else a bare name; the pin is
    version. A producer's own glp.* keys name no base, so none is consulted.
    """
    url = meta.get("general.base_model.0.repo_url") or ""
    org = meta.get("general.base_model.0.organization") or ""
    name = meta.get("general.base_model.0.name") or ""
    repo = url[len(HUB):] if url.startswith(HUB) else url
    if not repo:
        repo = f"{org}/{name}" if org and name else name
    return repo, meta.get("general.base_model.0.version") or ""


def compare(meta: dict, tensors: dict, ours: dict, blob: bytes, problems: list[str]) -> None:
    """The reference's view against ours: the ids, the metadata and every direction's bytes."""
    their_ids = direction_ids(tensors)
    if their_ids != ours["layers"]:
        problems.append(f"layer ids differ: reference {their_ids[:4]}..{their_ids[-1:]} against ours {ours['layers'][:4]}..{ours['layers'][-1:]}")
        return
    for key, field in (("glp.mode", "mode"), ("glp.hook_point", "hook_point"), ("glp.method", "method"),
                       ("glp.contrast", "contrast")):
        if key in meta and meta[key] != ours[field]:
            problems.append(f"{key} is {meta[key]!r} for the reference and {ours[field]!r} for us")
    their_repo, their_pin = base_of(meta)
    if their_repo != ours["base_model"]:
        problems.append(f"the base repo is {their_repo!r} for the reference and {ours['base_model']!r} for us")
    if their_pin != ours["base_revision"]:
        problems.append(f"the base commit pin is {their_pin!r} for the reference and {ours['base_revision']!r} for us")
    if "glp.spec_version" in meta and int(float(str(meta["glp.spec_version"]))) != int(ours["spec_version"]):
        problems.append(f"glp.spec_version is {meta['glp.spec_version']} for the reference and {ours['spec_version']} for us")
    if "glp.alpha_default" in meta and float(meta["glp.alpha_default"]) != float(ours["alpha"]):
        problems.append(f"glp.alpha_default is {meta['glp.alpha_default']} for the reference and {ours['alpha']} for us")
    if "glp.layer_ids_zero_based" in meta:
        declared = sorted(int(part) for part in str(meta["glp.layer_ids_zero_based"]).split(","))
        if declared != their_ids:
            problems.append(f"the file names layers {declared[:4]}..{declared[-1:]} and holds {their_ids[:4]}..{their_ids[-1:]}")

    # A project direction this reader accepted is a unit vector by the reference's own arithmetic too.
    if ours["mode"] == "project":
        for layer in their_ids:
            values = tensors[f"direction.{layer}"].astype("float64")
            squared = float((values * values).sum())
            if abs(squared - 1.0) > UNIT_TOLERANCE:
                problems.append(f"direction.{layer} has squared norm {squared:.6f} by the reference's arithmetic, yet this reader accepted it")
                break

    step = len(tensors[next(iter(tensors))].astype("<f4").tobytes()) if tensors else 0
    for i, layer in enumerate(ours["layers"]):
        their_bytes = tensors[f"direction.{layer}"].astype("<f4").tobytes()
        our_bytes = blob[i * step:(i + 1) * step]
        if their_bytes != our_bytes:
            differing = sum(1 for a, b in zip(their_bytes, our_bytes) if a != b)
            problems.append(f"direction.{layer} differs in {differing} of {len(their_bytes)} bytes")
            break

    claimed = meta.get("glp.content_sha256")
    if claimed:
        if payload_digest(tensors) != claimed:
            problems.append("the reference's digest over the payloads does not match the file's claim")
        if ours["sha256"] != claimed:
            problems.append(f"our digest {ours['sha256'][:16]} does not match the file's claim {claimed[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True, help="a weightless checkout")
    ap.add_argument("--vector", required=True, action="append", help="a control vector file to compare")
    ap.add_argument("--exe", default=str(ROOT / "zig-out" / "bin" / "tf-glp-check"), help="tf-glp-check")
    ap.add_argument("--work", default="/tmp/tf-glp-parity", help="folder for the receipts")
    args = ap.parse_args()

    container = load_reference(Path(args.reference))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    checked = 0

    for vector in args.vector:
        path = Path(vector)
        if not path.is_file():
            problems.append(f"{path} is not there")
            continue
        meta, tensors = container.read_gguf_cvec(str(path))

        receipt = work / "ours.json"
        blob_path = work / "ours.bin"
        read = subprocess.run([args.exe, "read", str(path), str(receipt), str(blob_path)], capture_output=True, text=True)
        if read.returncode != 0:
            # The reference took the file and this reader did not: a difference the pair has to name.
            problems.append(f"{path.name}: the reference reads it, this reader refuses it ({read.stdout.strip()})")
            print(f"{path.name}: the reference reads it, this reader refuses it")
            continue
        ours = json.loads(receipt.read_text())
        blob = blob_path.read_bytes()

        compare(meta, tensors, ours, blob, problems)
        if ours["file_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            problems.append(f"{path.name}: the reader's file SHA-256 is not the SHA-256 of the file's bytes")

        # The file read in Zig and written out again, then read by the reference.
        rewritten = work / "rewritten.gguf"
        subprocess.run([args.exe, "rewrite", str(path), str(rewritten)], check=True)
        back_meta, back_tensors = container.read_gguf_cvec(str(rewritten))
        if direction_ids(back_tensors) != direction_ids(tensors):
            problems.append(f"{path.name}: the rewrite holds different layers")
        for layer in direction_ids(tensors):
            if back_tensors[f"direction.{layer}"].astype("<f4").tobytes() != tensors[f"direction.{layer}"].astype("<f4").tobytes():
                problems.append(f"{path.name}: direction.{layer} did not survive the rewrite")
                break
        for key in ("glp.mode", "glp.hook_point", "glp.spec_version"):
            if back_meta.get(key) != meta.get(key):
                problems.append(f"{path.name}: the rewrite changed {key} to {back_meta.get(key)!r}")
        if base_of(back_meta) != base_of(meta):
            problems.append(f"{path.name}: the rewrite changed the base model or its commit pin")
        if abs(float(back_meta.get("glp.alpha_default", 0)) - float(meta.get("glp.alpha_default", 0))) > 1e-6:
            problems.append(f"{path.name}: the rewrite changed the strength")

        print(f"{path.name}: {len(tensors)} layers, {ours['width']} wide, {ours['mode']}, "
              f"alpha {ours['alpha']}, digest {ours['sha256'][:12]}, file sha256 {ours['file_sha256'][:12]}: reference and Zig agree")
        checked += 1

    for problem in problems:
        print(f"differs: {problem}")
    print(f"glp parity: {checked - len(problems)}/{checked} files agree" if problems else f"glp parity: {checked}/{checked} files agree")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
