"""Measure original LaneEngine or native CLI in sequential fresh processes.

This uses the original production loaders and engine, without correctness-oracle
substitutions. Reports retain actual draft availability and token IDs. Cold process
does not imply cold filesystem or Metal compiler caches. Run one benchmark at a time.
"""
import argparse
import copy
from dataclasses import asdict
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import uuid


MODELS = {
    "qwen": "Qwen3.8-27B-MLX-4bit",
    "nemotron": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
    "flash": "Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "bonsai": "Ternary-Bonsai-2-27B-mlx-2bit",
    "gemma": "gemma-4-26b-a4b-it-4bit",
}
SYNTHETIC_MODELS = {"glm": "GLM-5.3-Flash-MLX-4bit-MTP", "deepseek": "DeepSeek-V4-Flash-4bit"}
FAMILY_PACKAGES = dict(qwen="qwen3_5", nemotron="nemotron_h", flash="qwen4_exp",
                       bonsai="bonsai", gemma="gemma4", glm="glm5_next", deepseek="deepseek_v4")
DRAFTERS = dict(qwen="Qwen3.8-27B-DFlash2", bonsai="Qwen3.8-27B-DFlash2",
                gemma="gemma-4-26B-A4B-it-DFlash")
PROMPT = "Write a short Python function that computes the Fibonacci sequence."
RUNTIME_ENV = ("TF_LANE_TILE", "TF_FLASH_MTP", "TF_FLASH_DRAFT_VOCAB", "TF_FLASH_QUEUED",
               "TF_NEMOTRON_FOLD_SHARED", "TF_NEMOTRON_LANE_QMM", "TF_NEMOTRON_ROWS",
               "TF_NEMOTRON_ROW_EXPERTS")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def local_drafter(family, model_root, cache_root=None):
    name = DRAFTERS[family]
    cache_root = Path.home() / ".models" if cache_root is None else cache_root
    destination = model_root / name
    for directory in (destination, cache_root / "z-lab" / name):
        if (directory / "config.json").is_file() and any(directory.glob("*.safetensors")):
            if directory != destination:
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError(f"Incomplete drafter destination {destination}; refusing to overwrite it")
                model_root.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(directory.resolve())
            return destination
    raise RuntimeError(f"Cached {name} checkpoint required; no download is attempted")


def checkpoint_identity(directory):
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".txt"):
            stat = path.stat()
            with path.open("rb") as handle:
                if path.suffix == ".safetensors":
                    length = struct.unpack("<Q", handle.read(8))[0]
                    if length > 64 * 1024**2:
                        raise ValueError(f"Invalid safetensors header: {path}")
                    data = handle.read(length)
                    scope = "header"
                else:
                    data, scope = handle.read(), "file"
            files.append(dict(name=path.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              sha256=hashlib.sha256(data).hexdigest(), hash_scope=scope))
    return {"path": str(directory.resolve()), "files": files,
            "weight_identity": "local file size/mtime and safetensors header; weight payloads are not hashed"}


class GoldenCapture:
    """Observe the production protocol; array evaluation makes this unsuitable for timing."""

    methods = ("prefill", "hidden", "hidden_pass", "head", "hidden_rows", "keep_rows", "keep_rows_streams",
               "absorb_draft_context", "speculate", "settle", "unspeculate", "draft", "draft_streams")

    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True)
        self.events = []
        self.arrays = []
        self.originals = []

    def value(self, value):
        import mlx.core as mx
        import numpy as np
        if isinstance(value, mx.array):
            mx.eval(value)
            dtype = str(value.dtype)
            # NumPy cannot represent MLX BF16; every BF16 value is exact in FP32.
            array = np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
            name = f"array-{len(self.arrays):06}.npy"
            np.save(self.directory / name, array, allow_pickle=False)
            entry = dict(file=name, shape=list(value.shape), mlx_dtype=dtype, storage_dtype=str(array.dtype),
                         sha256=hashlib.sha256((self.directory / name).read_bytes()).hexdigest())
            self.arrays.append(entry)
            return entry
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.value(v) for k, v in value.items()}
        return {"type": f"{type(value).__module__}.{type(value).__name__}"}

    def cache(self, cache):
        out = []
        for item in cache:
            scalars = {k: v for k, v in vars(item).items() if v is None or isinstance(v, (bool, int, float, str))}
            out.append(dict(type=f"{type(item).__module__}.{type(item).__name__}", metadata=scalars,
                            state=self.value(item.state)))
        return out

    def install(self, model, engine):
        for obj, names in ((model, self.methods), (engine, ("_draw", "_conclude", "_plan_window"))):
            for name in names:
                original = getattr(obj, name, None)
                if not callable(original):
                    continue
                self.originals.append((obj, name, name in vars(obj), original))

                def traced(*args, _original=original, _name=name, **kwargs):
                    event = dict(method=_name, inputs=self.value(args), keywords=self.value(kwargs))
                    self.events.append(event)
                    result = _original(*args, **kwargs)
                    event["output"] = self.value(result)
                    return result

                setattr(obj, name, traced)

    def close(self):
        for obj, name, owned, original in self.originals:
            if owned:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        write_json(self.directory / "trace.json", dict(events=self.events, arrays=self.arrays,
                                                      timing_valid=False, bf16_storage="lossless float32"))


def golden_cases(tokenizer, synthetic=False):
    tokens = list(range(1, 5)) if synthetic else tokenizer.encode(PROMPT, add_special_tokens=False)
    pattern = (1, 38, 75, 112) if synthetic else (1000, 1037, 1074, 1111)
    cases = [dict(name="text-greedy", tokens=tokens, temperature=0.0, drafts=False),
             dict(name="text-sampled", tokens=tokens, temperature=0.7, drafts=False),
             dict(name="draft-greedy", tokens=tokens, temperature=0.0, drafts=True),
             dict(name="draft-sampled", tokens=tokens, temperature=0.7, drafts=True)]
    for size in (16, 17, 2049, 4097):
        cases.append(dict(name=f"rows-{size}", tokens=[pattern[i % 4] for i in range(size)],
                          temperature=0.0, drafts=False))
    return cases


def run_golden_case(model, options, case, args, capture=None):
    import mlx.core as mx
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    engine = LaneEngine(model, **options)
    stream = LaneStream("golden", case["tokens"], args.max_tokens, drafts=case["drafts"],
                        sampling=Sampling(args.seed, case["temperature"], args.top_k, args.top_p)
                        if case["temperature"] else None)
    # Empty EOS set deliberately tests a fixed output budget, including budget-cut rollback.
    snapshots = []
    if capture:
        capture.install(model, engine)
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    try:
        engine.add_stream(stream)
        first_seconds = time.perf_counter() - started
        cache = engine._live[0][1] if engine._live else []
        if capture:
            snapshots.append(dict(stage="prefill", cache=capture.cache(cache), cache_len=stream.cache_len,
                                  pending=stream.pending[:], emitted=stream.emitted[:]))
        decode_started = time.perf_counter()
        while engine.active_count:
            landed = engine.step()
            if capture:
                snapshots.append(dict(stage="round", landed=landed, cache=capture.cache(cache),
                                      cache_len=stream.cache_len, pending=stream.pending[:],
                                      emitted=stream.emitted[:], drafted=stream.drafted, accepted=stream.accepted))
        mx.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        result = dict(tokens=stream.emitted, prompt_tokens=case["tokens"], finish_reason=stream.finish_reason,
                      token_sha256=hashlib.sha256(struct.pack(f"<{len(stream.emitted)}I", *stream.emitted)).hexdigest(),
                      first_token_seconds=first_seconds, decode_seconds=decode_seconds,
                      total_seconds=time.perf_counter() - started, timing_valid=capture is None,
                      peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory(),
                      rounds=stream.rounds, drafted=stream.drafted, accepted=stream.accepted,
                      prefill_widths=stream.prefill_widths, prefill_raised=stream.prefill_raised,
                      round_stats=[asdict(stat) for stat in engine.round_stats], snapshots=snapshots)
        if len(stream.emitted) != args.max_tokens or stream.finish_reason != "length":
            raise RuntimeError(f"Incomplete output budget: {case['name']}")
        return result
    finally:
        if capture:
            capture.close()
        engine.reset()
        engine.release_rounds()


def synthetic_checkpoint(family, directory, source=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests import glm5_fakes, dsv4_fakes
    fake = glm5_fakes if family == "glm" else dsv4_fakes
    original_text, original_dims = fake.TEXT, fake.D
    config = copy.deepcopy(original_text)
    if source is not None:
        actual = json.loads((source / "config.json").read_text())
        config = copy.deepcopy(actual.get("text_config") or actual)
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        if family == "glm":
            config.update(num_hidden_layers=2, layer_types=["linear_attention", "deepseek_sparse_attention"],
                          mlp_layer_types=["sparse", "sparse"], first_k_dense_replace=0,
                          indexer_types=["full", "full"])
        else:
            config.update(num_hidden_layers=3, compress_ratios=[0, 4, 128, 0], num_hash_layers=1)
    config.update(vocab_size=256, pad_token_id=0, eos_token_id=1)
    fake.TEXT, fake.D = config, config["hidden_size"]
    try:
        fake.write_checkpoint(directory)
        if family == "deepseek":
            fake.write_mtp(directory / "drafter")
    finally:
        fake.TEXT, fake.D = original_text, original_dims
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0"))
    tokenizer.save(str(directory / "tokenizer.json"))
    write_json(directory / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast",
                                                   "unk_token": "t0", "eos_token": "t1"})
    return dict(seed=0, mtp_seed=1 if family == "deepseek" else 0, config=config,
                shape="production" if source is not None else "reduced",
                source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest() if source else None,
                scope="random weights, reduced layers/vocabulary; not full-model performance")


def golden_worker(args):
    from native_runtime import require_mlx
    versions = require_mlx()
    import importlib
    package = importlib.import_module(f"tensorfold.families.{FAMILY_PACKAGES[args.family]}")
    for key, value in getattr(package, "MLX_ENV", {}).items():
        os.environ.setdefault(key, value)
    import mlx.core as mx
    from tensorfold.families import kernel_version, detect
    synthetic = args.family in SYNTHETIC_MODELS
    directory = args.output / "checkpoint" if synthetic else args.model_root / MODELS[args.family]
    source = args.model_root / SYNTHETIC_MODELS[args.family] if synthetic and args.synthetic_shape == "production" else None
    specification = synthetic_checkpoint(args.family, directory, source) if synthetic else None
    load_options = dict(mtp_drafts=3)
    if args.family in DRAFTERS:
        load_options["drafter"] = str(local_drafter(args.family, args.model_root))
    if args.family == "flash":
        load_options["ple_on_ssd"] = True
    if args.family == "deepseek":
        load_options["drafter"] = str(directory / "drafter")
    load_started = time.perf_counter()
    model, tokenizer = package.load(directory, **load_options)
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    declared_options = package.engine_settings(model)
    # This fixture fixes the chunk grid, exposing 16-row and 2048-row boundaries across families.
    options = {k: v for k, v in declared_options.items() if k != "prefill_steps"}
    manifest = dict(schema=1, family=args.family, backend="mlx-metal", synthetic=synthetic,
                    synthetic_specification=specification, checkpoint=checkpoint_identity(directory),
                    drafter=checkpoint_identity(Path(load_options["drafter"])) if load_options.get("drafter") else None,
                    dependencies=versions, python=platform.python_version(), platform=platform.platform(),
                    device=mx.device_info(), physical_memory_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    kernel_version=kernel_version(detect(directory), model), load_options=load_options,
                    load_seconds=load_seconds, engine_options=options, declared_engine_options=declared_options,
                    prefill_grid=2048, eos_policy="disabled for fixed output budget", context_copy=False,
                    seed=args.seed, top_k=args.top_k, top_p=args.top_p,
                    environment={k: v for k, v in os.environ.items()
                                 if k.startswith(("TF_", "TENSORFOLD_", "MLX_"))},
                    calibration={k: getattr(model, k, None) for k in
                                 ("exact_width", "window_costs", "shared_costs", "check_report", "mtp_step_ms")},
                    draft_available=getattr(model, "mtp", None) is not None,
                    scope="production lane protocol, text only; fixed prefill grid; no HTTP admission or prefix reuse",
                    cases=[], complete=False)
    write_json(args.output / "manifest.json", manifest)
    expected = {}
    cases = golden_cases(tokenizer, synthetic)
    for case in cases:
        capture = GoldenCapture(args.output / case["name"])
        result = run_golden_case(model, options, case, args, capture)
        write_json(capture.directory / "result.json", result)
        baseline = case["name"].replace("draft-", "text-")
        if case["drafts"] and result["tokens"] != expected[baseline]:
            raise RuntimeError(f"Production drafts changed serial output: {case['name']}")
        expected[case["name"]] = result["tokens"]
        manifest["cases"].append(dict(**case, result=f"{case['name']}/result.json",
                                      trace=f"{case['name']}/trace.json", measurements=[]))
        write_json(args.output / "manifest.json", manifest)
        print(f"CAPTURE {args.family}/{case['name']}: {len(capture.arrays)} arrays, "
              f"{result['rounds']} rounds, {result['accepted']}/{result['drafted']} drafts", flush=True)
    # All numerical/draft checks precede measurement. No capture hooks remain during these runs.
    for case, entry in zip(cases, manifest["cases"]):
        for repetition in range(args.repetitions + 1):
            result = run_golden_case(model, options, case, args)
            if result["tokens"] != expected[case["name"]]:
                raise RuntimeError(f"Untraced execution disagrees with fixture: {case['name']}")
            if repetition:
                entry["measurements"].append(result)
                write_json(args.output / "manifest.json", manifest)
        print(f"MEASURE {args.family}/{case['name']}: {len(entry['measurements'])} warm runs", flush=True)
    manifest["complete"] = True
    write_json(args.output / "manifest.json", manifest)


def golden_suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "suite.json"
    if path.exists():
        raise SystemExit("Golden output already exists; choose a new directory to preserve the fixture")
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    suite = dict(schema=1, run_id=str(uuid.uuid4()), physical_memory_bytes=ram, complete=False,
                 cuda=dict(status="unmeasured", reason="no CUDA hardware; Metal results do not verify CUDA"), models=[])
    write_json(path, suite)
    selected = [args.family] if args.family else [*MODELS, *SYNTHETIC_MODELS]
    failed = False
    for family in selected:
        synthetic = family in SYNTHETIC_MODELS
        checkpoint = args.model_root / (SYNTHETIC_MODELS if synthetic else MODELS)[family]
        shards = list(checkpoint.glob("*.safetensors"))
        weight_bytes = sum(p.stat().st_size for p in shards)
        entry = dict(family=family, original_checkpoint=str(checkpoint), weight_bytes=weight_bytes,
                     synthetic=synthetic, status="pending")
        suite["models"].append(entry)
        if not synthetic and (not shards or not (checkpoint / "config.json").is_file()):
            entry.update(status="failed", reason="local checkpoint missing or incomplete; no download attempted")
            failed = True
        elif not synthetic and weight_bytes >= ram:
            entry.update(status="failed", reason="weights exceed physical RAM; add explicit synthetic coverage")
            failed = True
        else:
            out = args.output / family
            out.mkdir()
            entry.update(status="running", manifest=f"{family}/manifest.json")
            if synthetic:
                entry["reason"] = "full checkpoint does not fit this machine; reduced synthetic measurements only"
            write_json(path, suite)
            command = [sys.executable, str(Path(__file__).resolve()), "--golden-worker", "--family", family,
                       "--model-root", str(args.model_root), "--output", str(out), "--max-tokens", str(args.max_tokens),
                       "--repetitions", str(args.repetitions), "--seed", str(args.seed),
                       "--top-k", str(args.top_k), "--top-p", str(args.top_p),
                       "--synthetic-shape", args.synthetic_shape]
            with (out / "worker.log").open("w") as log:
                # Fresh sequential processes release each model before the next load; no inference deadline.
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            entry["status"] = "ok" if process.returncode == 0 else "failed"
            entry["exit_code"] = process.returncode
            failed |= process.returncode != 0
            print(f"GOLDEN {family}: {entry['status']}; {out / 'worker.log'}", flush=True)
        write_json(path, suite)
    suite["complete"] = not failed
    write_json(path, suite)
    if failed:
        raise SystemExit("Golden suite incomplete; retained per-family failures in suite.json")


def verify_golden(directory):
    import numpy as np
    suite = json.loads((directory / "suite.json").read_text())
    if not suite["complete"] or not suite["models"]:
        raise ValueError("Golden suite is incomplete")
    arrays = 0
    for model in suite["models"]:
        if model["status"] != "ok":
            raise ValueError(f"Failed family: {model['family']}")
        manifest_path = directory / model["manifest"]
        manifest = json.loads(manifest_path.read_text())
        if not manifest["complete"] or manifest["synthetic"] != model["synthetic"]:
            raise ValueError("Incomplete or mislabeled family fixture")
        expected = {}
        for case in manifest["cases"]:
            result = json.loads((manifest_path.parent / case["result"]).read_text())
            trace_path = manifest_path.parent / case["trace"]
            trace = json.loads(trace_path.read_text())
            if trace["timing_valid"] or result["timing_valid"] or not result["snapshots"]:
                raise ValueError("Capture must have cache snapshots and invalid timing")
            if not any(s["cache"] for s in result["snapshots"]):
                raise ValueError("No working cache captured")
            methods = {event["method"] for event in trace["events"]}
            if not {"head", "_draw"} <= methods or not methods.intersection(("hidden", "prefill", "hidden_pass")):
                raise ValueError("Missing production forward/head/draw observations")
            for entry in trace["arrays"]:
                path = trace_path.parent / entry["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Array hash mismatch: {path}")
                value = np.load(path, allow_pickle=False)
                if list(value.shape) != entry["shape"] or str(value.dtype) != entry["storage_dtype"]:
                    raise ValueError(f"Array shape/dtype mismatch: {path}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite fixture array: {path}")
                arrays += 1
            tokens = result["tokens"]
            digest = hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()
            if digest != result["token_sha256"] or result["prompt_tokens"] != case["tokens"]:
                raise ValueError("Token identity mismatch")
            if not case["measurements"] or any(not r["timing_valid"] or r["tokens"] != tokens
                                                for r in case["measurements"]):
                raise ValueError("Untraced measurements disagree with capture")
            baseline = case["name"].replace("draft-", "text-")
            if case["drafts"] and tokens != expected[baseline]:
                raise ValueError("Drafted output disagrees with serial output")
            expected[case["name"]] = tokens
    print(f"PASS golden fixture: {len(suite['models'])} families, {arrays} hashed finite arrays", flush=True)


def python_worker(args):
    from native_runtime import require_mlx
    require_mlx()
    started = time.perf_counter()
    import mlx.core as mx
    from tensorfold import __version__
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream

    calibration = []

    def time_method(cls, name):
        original = getattr(cls, name)

        @wraps(original)
        def measured(*positional, **keywords):
            before = time.perf_counter()
            try:
                return original(*positional, **keywords)
            finally:
                calibration.append({"method": f"{cls.__name__}.{name}",
                                    "seconds": time.perf_counter() - before})

        setattr(cls, name, measured)

    model_dir = args.model_root / MODELS[args.family]
    load_started = time.perf_counter()
    if args.family == "qwen":
        from tensorfold.families import qwen3_5
        from tensorfold.families.qwen3_5.family import Qwen35Family
        time_method(Qwen35Family, "check_windows")
        drafter = str(args.model_root / "Qwen3.8-27B-DFlash2") if args.drafts else ""
        model, tokenizer = qwen3_5.load(model_dir, drafter=drafter)
        engine_options = qwen3_5.engine_settings(model)
    elif args.family == "nemotron":
        from tensorfold.families.nemotron_h import model as implementation
        time_method(implementation.NemotronH, "check_windows")
        time_method(implementation.NemotronH, "_time_mtp_step")
        model, tokenizer = implementation.load(model_dir, mtp_drafts=args.drafts)
        engine_options = {"max_rows": 16, "max_draft": 15}
    else:
        from tensorfold.families.qwen4_exp import runtime as implementation
        time_method(implementation.FlashNext, "check_windows")
        time_method(implementation.FlashNext, "_time_mtp_step")
        model, tokenizer = implementation.load(model_dir, drafts=args.drafts)
        engine_options = {"max_rows": 16, "max_draft": 15}
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    sampling = Sampling(args.seed, args.temperature, args.top_k, args.top_p) if args.temperature else None
    prompt = ([1000 + (i % 4) * 37 for i in range(args.prompt_tokens)] if args.prompt_tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = [tokenizer.eos_token_id]
    engine = LaneEngine(model, **engine_options)
    stream = LaneStream("benchmark", prompt, args.max_tokens,
                        eos_ids=frozenset(int(t) for t in eos if t is not None),
                        sampling=sampling, drafts=bool(args.drafts))
    startup_seconds = time.perf_counter() - started
    before = time.perf_counter()
    engine.add_stream(stream)
    # Preserve the production overlap: add_stream may queue the next serial pass.
    # An extra synchronization here would change the engine's scheduling.
    prefill_seconds = time.perf_counter() - before
    before = time.perf_counter()
    engine.run()
    mx.synchronize()
    decode_seconds = time.perf_counter() - before
    result = {
        "prompt_tokens": prompt, "tokens": stream.emitted,
        "text": tokenizer.decode(stream.emitted),
        "token_sha256": hashlib.sha256(struct.pack(f"<{len(stream.emitted)}I", *stream.emitted)).hexdigest(),
        "seed": args.seed, "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "metal_sampling": args.family != "qwen",
        "context_copy": False, "startup_seconds": startup_seconds, "load_seconds": load_seconds,
        "calibration_seconds": sum(item["seconds"] for item in calibration),
        "calibration_calls": calibration, "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "engine_seconds": time.perf_counter() - started,
        "peak_mlx_bytes": mx.get_peak_memory(), "active_mlx_bytes": mx.get_active_memory(),
        "rounds": stream.rounds, "accepted_drafts": stream.accepted,
        "drafted": stream.drafted, "finish_reason": stream.finish_reason,
        "mtp_enabled": args.family != "qwen" and getattr(model, "mtp", None) is not None,
        "dflash_enabled": args.family == "qwen" and getattr(model, "head_drafts", None) is not None,
        "exact_width": getattr(model, "exact_width", None),
        "engine_options": engine_options, "engine_summary": engine.summary(),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "python_version": platform.python_version(),
        "tensorfold_version": __version__,
        "installed_tensorfold_version": importlib.metadata.version("tensorfold"),
        "prefill_method": "production family prefill",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("python", "zig"), default="python")
    parser.add_argument("--family", choices=(*MODELS, *SYNTHETIC_MODELS))
    parser.add_argument("--golden", action="store_true", help="Capture and measure all local Python families, plus synthetic GLM/DeepSeek")
    parser.add_argument("--golden-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-golden", type=Path, help="Verify a retained golden suite's arrays, provenance labels and token comparisons")
    parser.add_argument("--synthetic-shape", choices=("production", "reduced"), default="production",
                        help="Synthetic GLM/DeepSeek retain checkpoint widths/experts with reduced layers/vocabulary; reduced is a quick smoke")
    parser.add_argument("--drafts", type=int, default=0, choices=(0, 3, 15))
    parser.add_argument("--resident-ple", action="store_true")
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/tensorfold"))
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="Use a repeating four-token prompt of this length (0 uses --prompt)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_golden:
        return verify_golden(args.verify_golden)
    if args.output is None:
        parser.error("--output is required")
    if args.golden or args.golden_worker:
        if args.engine != "python" or args.resident_ple or args.max_tokens < 2 or args.repetitions < 1:
            parser.error("golden fixtures require Python, at least two tokens and positive repetitions")
        if args.golden_worker:
            if not args.family:
                parser.error("golden worker requires --family")
            return golden_worker(args)
        return golden_suite(args)
    if args.family not in ("qwen", "nemotron", "flash"):
        parser.error("benchmark requires --family qwen|nemotron|flash; use --golden for other families")
    if args.resident_ple and (args.engine != "zig" or args.family != "flash"):
        parser.error("--resident-ple applies only to native Flash")
    if args.family == "qwen" and args.drafts not in (0, 15):
        parser.error("Qwen uses the original 15-node DFlash2 configuration")
    if args.repetitions < 1 or args.max_tokens < 2:
        parser.error("positive repetitions and at least two output tokens required")
    if not 0 <= args.prompt_tokens <= 32768:
        parser.error("--prompt-tokens must be 0..32768")
    if args.worker:
        return python_worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for repetition in range(args.repetitions):
        report = args.output / f"run-{repetition}.json"
        log = args.output / f"run-{repetition}.log"
        shared = ["--prompt", args.prompt, "--max-tokens", str(args.max_tokens),
                  "--seed", str(args.seed), "--temperature", str(args.temperature),
                  "--top-k", str(args.top_k), "--top-p", str(args.top_p)]
        if args.engine == "python":
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--engine", "python",
                       "--family", args.family, "--drafts", str(args.drafts),
                       "--model-root", str(args.model_root), "--output", str(report),
                       "--prompt-tokens", str(args.prompt_tokens), *shared]
        else:
            command = [str(args.binary.resolve()), "run", str(args.model_root / MODELS[args.family]),
                       *shared, "--no-copy", "--warmup", "--report", str(report)]
            if args.prompt_tokens:
                command += ["--tokens", ",".join(str(1000 + (i % 4) * 37) for i in range(args.prompt_tokens))]
            # The production Qwen LaneEngine uses exact_sampling (CPU f64).
            # FamilyRounds uses the fp32 GPU sampler for Nemotron and Flash.
            if args.family != "qwen":
                command.append("--metal-sampling")
            if args.family == "qwen" and args.drafts:
                command += ["--drafter", str(args.model_root / "Qwen3.8-27B-DFlash2")]
            elif args.family != "qwen":
                command += ["--mtp-drafts", str(args.drafts)] if args.drafts else ["--no-drafts"]
            if args.resident_ple:
                command.append("--resident-ple")
        before = time.perf_counter()
        with log.open("w") as handle:
            try:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "ok" if process.returncode == 0 else f"exit {process.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        elapsed = time.perf_counter() - before
        result = json.loads(report.read_text()) if status == "ok" else {}
        result.update(engine=args.engine, family=args.family, requested_drafts=args.drafts,
                      repetition=repetition, process_seconds=elapsed, status=status, command=command,
                      platform=platform.platform(), environment={k: os.environ[k] for k in RUNTIME_ENV if k in os.environ})
        report.write_text(json.dumps(result, indent=2) + "\n")
        reports.append(result)
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"{args.engine}/{args.family}/{args.drafts} run {repetition}: {status}, {elapsed:.3f}s process", flush=True)
        if status != "ok":
            raise SystemExit(f"Benchmark failed; see {log}")
        if len(result["tokens"]) != args.max_tokens:
            raise SystemExit("Early EOS prevents the requested token-count comparison; retain report and choose another prompt")
        if repetition and result["tokens"] != reports[0]["tokens"]:
            raise SystemExit("Output changed between repetitions")


if __name__ == "__main__":
    main()
