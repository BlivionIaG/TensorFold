# M5 original contracts — pushed source `2acdb56a2`

**Correctness passed; this is not a full performance qualification or review-ready claim.**
Production source is [`2acdb56a26d71361412136f5f36bf3bfdd12d563`](https://github.com/BlivionIaG/TensorFold/commit/2acdb56a26d71361412136f5f36bf3bfdd12d563), including the scheduled-cohort pricing fix. This packet changes evidence only; it does not change the PR's code head.

## Exact source and device

- Apple M5 Ultra, 256 GiB. One model workflow, persistent shared lock inode `3719605`; original pressure and competing-process checks retained.
- Exact source archive SHA256: `50d6f5c91a62197fe53f6e026c394d85c75e861adda45c64683aca46a5ae2f81`.
- Serving native binary SHA256: `ca3f3cc0d7423a6331419aa002a435f5d72e335a0532f1c588697024283d0716`.
- Original target/drafter revisions, all model-file hashes and seven frozen driver hashes are in [binding.json](binding.json). The actual native build uses Zig 0.17.0, `-Dcpu=apple_m1 -Dversion=1.2.0 -j1`.
- [Original six-command guard record](original-guards.json) and [complete three-stage controller outcome](verification-summary.json). Every command/stage exited 0. No numerical bound, launch/resource constraint, test or functional scenario was removed.

## Original native/numerical scope

[Build output](build.log): **82/82 steps; 767/771 tests passed, four skipped**. Original attention, BF16 and norm/tap GPU contracts passed, followed by both actual 64-context, nine-query native scenarios. Prefix bytes/first pick/confidence invariants, 8/16 logical cohorts, context accounting and planted failure controls remain checked.

| Head weights | FP32-reference picks | Worst confidence difference | Largest FP32 margin over a differing GPU pick |
|---|---:|---:|---:|
| BF16 | 9/9 | 0.00080 | 0 |
| 8-bit | **8/9** | 0.00225 | 0.075 |

Activations remain **BF16** in both modes; 8-bit changes weights only. The independently implemented FP32 host reference uses the same drafter checkpoint and BF16 target taps, not an FP32 target server. Native reporting exit 0 is not relabelled as 9/9 for the quantized head. Original logs: [BF16](native-bits0-depth9.log), [8-bit](native-bits8-depth9.log).

## Actual cache, sampled and cancellation behavior

The original `qualification.py::functional` ran unchanged at context 16,384, parallel 16, prompt-cache capacity 1 GiB, with the original Metal debug setting. The wrapper only checks source/environment/pressure/ownership and selects head weight mode; it does not replace the functional scenarios.

Both modes passed cache resume against a fresh server. Each resumed request reported 3,517 cached prompt tokens. All original sampled solo/drafts-off/concurrent token hashes matched at widths 1/4/8/16. Full original outputs: [BF16](functional-0.json), [8-bit](functional-8.json); completed source/binary status: [BF16](functional-0.observed.json), [8-bit](functional-8.observed.json).

| Head weights | Slots | Stress seconds | Completed | Cancelled | Wrong replies | Errors |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 6 | 60 | 451 | 247 | 0 | 0 |
| BF16 | 16 | 90 | 937 | 507 | 0 | 0 |
| 8-bit | 6 | 60 | 463 | 254 | 0 | 0 |
| 8-bit | 16 | 90 | 915 | 493 | 0 | 0 |

All four actual servers exited **0 without forced termination**: [BF16 cached](functional-0-cache.shutdown.json), [BF16 fresh](functional-0-fresh.shutdown.json), [8-bit cached](functional-8-cache.shutdown.json), [8-bit fresh](functional-8-fresh.shutdown.json). No pressure override or retry-for-luck was used.

## What remains open

- These are consumer correctness and lifecycle results, not pooled throughput samples.
- The last complete same-head matrices remain bound to **CF `cf2376236`**: M4 12/32 strict losses, M5 16/32. [All 64 comparisons and original verdicts](../../cf2376236/qualification/README.md). They are not relabelled to this newer source.
- Correctly declining every known four-stream context did not remove the observed target-work divergence. The [pushed diagnosis](https://github.com/BlivionIaG/TensorFold/blob/2acdb56a26d71361412136f5f36bf3bfdd12d563/docs/receipts/lfm2-dspark-review/diagnosis.md#scheduled-cohort-follow-up-one-enrollment-defect-fixed-performance-still-open) records that remaining investigation and M4 profiling/memory-pressure limits.
- GitHub reported this exact head `CONFLICTING/DIRTY` with no CI checks against pinned base `6a5048697`. No re-review, force push or merge is implied.

[SHA256.json](SHA256.json) binds all 17 original packet files. Private server logs, credentials, raw system traces and unrelated process metadata are not published.
