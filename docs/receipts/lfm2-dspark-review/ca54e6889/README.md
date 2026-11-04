# LFM2 DSpark #1394 — M5-only revision and predeclared A/A

Source: [`ca54e6889ed8ed00400bfeb4fa1be30a3a687de5`](https://github.com/BlivionIaG/TensorFold/commit/ca54e6889ed8ed00400bfeb4fa1be30a3a687de5). Pinned upstream main: `00ff0f3b2c7ee67546e7a406f074e6d1d5c02593`.

This protocol is published **before any M5 model execution on this revision**. It responds only to [review6633416133](https://github.com/ashhart/TensorFold/pull/1394#issuecomment-6633416133). It is not a completed qualification or a performance-pass claim.

## Why this source follows the failed prerequisite

The [source-dc4558f5c packet](../dc4558f5c/README.md) retains both chips' original guard failures. Its matrix and A/A never started. Current main's native facts read `runner.model.config.moe()`; four host fixtures had left that model pointer undefined. This revision initializes the dense model fact in those fixtures. No production Qwen/LFM2 code, assertions, test selection, numerical bounds or workload sizes changed. The passive observer's separate API repair is also included; it does not invoke models.

## Code already pushed

- Pre-M5 and unrecognized Metal device names stand aside before LFM2 neural allocation/calibration. Ordinary target forwarding and cheap copies remain; M5+ remains subject to normal funded admission. CLI path validation still applies.
- `metal_main.zig`: 637 → 547 lines, with DSpark registrations owned by the existing family suite. Cases and numerical bounds are unchanged.
- Current main integrated, including its non-LFM2 support. Unrelated CI-dispatch and CONTRIBUTING additions removed; prior published commits preserved. No CI PR or unrelated model work.
- Actual lean comparison: **425 head / 425 pinned-base warnings, zero new**. Formatting passed for this revision before commit.

## Observed preliminary M4 route smoke

Source `3d0f720c4824fbf7fb6b34f18f5f5789c8178975`: 24/24 complete 128-token requests matched hashes across plain, requested-BF16 and requested-8-bit arms, including `draft:false` oracles. Both requested arms logged stand-aside; no neural initialization/calibration was logged. Three clean exits, no forced kills. **Routing smoke only; not the final source numerical guards or performance qualification.**

## Frozen M5 procedure

[`protocol/study.json`](protocol/study.json) declares the original 144-cell / 306-oracle matrix once, then separate plain/plain A/A with 96 cells / 204 oracles, then both original functional modes and original served/native long-context checks. Both A/A arms use identical plain argv and retain ordinary copies. Three rotations necessarily give a disclosed 2:1 first-position imbalance. Original seeds, lengths, warmups, fixtures, locks and bounds are retained. No extra conditioning, retries or pressure/admission overrides.

[`protocol/predeclared-binding.json`](protocol/predeclared-binding.json) binds the source archive, protocol, Python 3.13.15 binary, models, original driver and every execution artifact. A/A wrappers are included byte-identically. The private interpreter path is aliased; original hashes remain recorded. The original full-matrix result is preserved before and after A/A, including exit 3. A/A is variability evidence, **not a tolerance or noise-floor waiver**.

## Historical results remain failures

The [complete previous packet](https://github.com/BlivionIaG/TensorFold/blob/86cd67c15770e0e3ae1e4e737608ef1bc899e66a/docs/receipts/lfm2-dspark-review/5fdbfb874/README.md) retains the source-5f matrices, correctness, source-4ef CI and diagnostic OOMs. No result is relabelled as this source. The user's <1% observed-loss budget is distinct from the maintainer's every-cell-at-least-plain gate. The M4 maximum-context independent diagnostic OOM is not claimed fixed by excluding neural serving on M4.
