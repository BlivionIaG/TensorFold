"""`tensorfold serve --backend rocm`: the family's ROCm engine behind the torch server the CUDA lane uses."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any


def cache_budget(args: argparse.Namespace) -> tuple[int, int | None]:
    """Prefix slots and a byte budget. Zero slots or zero GiB turns the cache off."""

    keep = 8 if args.checkpoint_slots is None else int(args.checkpoint_slots)
    if keep < 0:
        raise ValueError("--checkpoint-slots must be 0 or more")
    gib = args.prompt_cache_gib
    if gib is not None and float(gib) < 0:
        raise ValueError("--prompt-cache-gib must be 0 or more")
    if keep == 0 or gib == 0:
        return 0, 0
    budget = None if gib is None else int(float(gib) * 1024**3)
    return keep, budget


def serve_rocm(args: argparse.Namespace, family: Any, model_dir: Path, sampling: dict,
               context: int | None = None) -> int:
    """Serve with the family's ROCm engine (``rocm_engine``) behind the same torch server the CUDA lane uses."""

    from tensorfold import hub
    from tensorfold.cuda.server import App, serve

    if args.tp not in (1, 2, 4, 8):
        raise ValueError(f"--tp {args.tp} is not a supported ROCm world size; choose 1, 2, 4 or 8")
    if args.tp > 1 and not args.master:
        raise ValueError("--tp > 1 needs --master: rank 0's address on the link between the machines")
    if args.tp == 1 and args.rank != 0:
        raise ValueError("--rank must be 0 when --tp 1")
    if not 0 <= args.rank < args.tp:
        raise ValueError(f"--rank {args.rank} not in [0, --tp {args.tp})")
    if args.tp > 1 and args.host == "0.0.0.0" and args.rank == 0:
        print("[tensorfold] --tp > 1: rank 0 binds --host 0.0.0.0 by default; other ranks follow over --master",
              flush=True)
    started = time.perf_counter()
    served = args.name or (args.model.rstrip("/").split("/")[-1] if hub.is_repo_id(args.model) else model_dir.name)
    keep, budget = cache_budget(args)
    where = f", rank {args.rank} of {args.tp}" if args.tp > 1 else ""
    print(f"[tensorfold] loading {served}: {family.title} ({family.model_type}) on ROCm{where}", flush=True)
    p2p = getattr(args, "p2p", None)
    engine = family.package.rocm_engine(model_dir,
                                       context=context if context is not None else args.context,
                                       context_explicit=args.context is not None, keep=keep, byte_budget=budget,
                                       tp=int(args.tp), rank=int(args.rank),
                                       master=args.master, master_port=int(args.master_port),
                                       p2p=p2p, no_drafts=bool(getattr(args, "no_drafts", False)))
    if args.tp > 1 and args.rank > 0:
        print(f"[tensorfold] rank {args.rank} of {args.tp} loaded in {time.perf_counter() - started:.1f}s; "
              f"following rank 0 at {args.master}:{args.master_port}", flush=True)
        engine.follow()
        return 0
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k),
                       ("min_p", args.min_p)):
        if value is not None:
            sampling[key] = value
    app = App(engine, model_dir, served, default_thinking=bool(args.thinking), sampling=sampling,
              max_tokens=int(args.max_tokens), context_window=engine.context_window,
              reasoning_effort=args.reasoning_effort, thinking_budget=int(args.thinking_budget))
    shown = "greedy" if float(sampling.get("temperature", 1.0)) <= 0 else ", ".join(
        f"{k} {v}" for k, v in sampling.items())
    effective = app.effective_context_window
    if budget == 0:
        kept = "off"
    elif budget is None:
        kept = f"{keep} slots"
    else:
        kept = f"{keep} slots, {budget / 1024**3:.1f} GiB"
    print(f"[tensorfold] serving {served} at http://{args.host}:{args.port}/v1 on ROCm "
          f"(sampling: {shown}; drafts: {'off' if getattr(args, 'no_drafts', False) else 'on'}; "
          f"context: {'unlimited' if effective is None else effective}; "
          f"prefix cache: {kept}; loaded in {time.perf_counter() - started:.1f}s)", flush=True)
    serve(app, args.host, int(args.port))
    return 0
