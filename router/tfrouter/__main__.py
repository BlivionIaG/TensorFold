import argparse
import asyncio
import json
import os
import signal
import sys

from . import config as cfgmod
from .server import Router


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tfrouter", description="cost-based, cache-aware OpenAI-compatible router for S (Sparks) + M (M5)")
    ap.add_argument("--config", action="append", help="JSON config, repeatable (later files override earlier ones; defaults: tfrouter/config.py; see router.config.json)")
    ap.add_argument("--listen", help="HOST:PORT (overrides listen.*)")
    ap.add_argument("--explain", action="store_true", help="log every routing decision with the predicted costs (JSON line per request)")
    ap.add_argument("--explain-file", help="write the decision log here instead of stderr")
    ap.add_argument("--dry-run", action="store_true", help="contact no backend for requests: answer with the decision JSON")
    ap.add_argument("--shadow", metavar="BACKEND", help="proxy every request to BACKEND, only log what the policy would have chosen")
    ap.add_argument("--mode", choices=("cost", "ll"), help="policy.mode override")
    ap.add_argument("--time-scale", type=float, help="TEST ONLY: router seconds per wall second")
    ap.add_argument("--dump-config", action="store_true", help="print the default config (every parameter with its measured default) and exit")
    ap.add_argument("--check-config", action="store_true", help="validate --config and exit")
    ap.add_argument("--no-access-log", action="store_true")
    a = ap.parse_args(argv)
    if a.dump_config:
        json.dump(cfgmod.dump_defaults(), sys.stdout, indent=2)
        print()
        return 0
    ov = {}
    if a.listen:
        h, _, p = a.listen.rpartition(":")
        ov["listen"] = {"host": h or "127.0.0.1", "port": int(p)}
    if a.explain:
        ov.setdefault("log", {})["explain"] = True
    if a.explain_file:
        ov.setdefault("log", {}).update(explain=True, explain_file=a.explain_file)
    if a.no_access_log:
        ov.setdefault("log", {})["access"] = False
    if a.dry_run:
        ov.setdefault("mode", {})["dry_run"] = True
    if a.shadow:
        ov.setdefault("mode", {})["shadow"] = a.shadow
    if a.mode:
        ov.setdefault("policy", {})["mode"] = a.mode
    if a.time_scale:
        ov["time_scale"] = a.time_scale
    try:
        cfg = cfgmod.load(a.config, ov)
    except (ValueError, OSError) as e:
        print("tfrouter: config error: %s" % e, file=sys.stderr)
        return 2
    if a.check_config:
        print("config ok: backends %s" % ", ".join(cfg["backends"]))
        return 0
    return asyncio.run(serve(cfg))


async def serve(cfg):
    if os.environ.get("TFROUTER_SLOW_CALLBACK_S"):         # diagnostics: log every event-loop callback slower than this many seconds
        asyncio.get_running_loop().set_debug(True)
        asyncio.get_running_loop().slow_callback_duration = float(os.environ["TFROUTER_SLOW_CALLBACK_S"])
    r = Router(cfg)
    port = await r.start()
    sys.stderr.write("tfrouter listening on %s:%d  backends: %s  policy=%s%s%s\n" % (
        cfg["listen"]["host"], port, ", ".join("%s=%s" % (b.name, b.cfg["url"]) for b in r.order), cfg["policy"]["mode"],
        "  DRY-RUN" if r.dry_run else "", "  SHADOW->%s" % r.shadow if r.shadow else ""))
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    def reload_keys():
        for b in r.order:
            b.key.reload(force=True)
        r.client_keys.mt = None
        r.client_keys._reload()
    loop.add_signal_handler(signal.SIGHUP, reload_keys)
    await stop.wait()
    await r.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
