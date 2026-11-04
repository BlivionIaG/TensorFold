# Review readiness — qualified runtime `f7eb5f0bc`, docs tip `51dce7451`

## Current decision

**The pushed code and its11 upstream PR-attached CI checks pass. This is not a complete review/merge clearance.** Two independent items remain: the fixed-source M4 performance/uncertainty workflow has not delivered its complete result, and actual upstream main moved after the successful fixed-base integration, creating two proven new conflicts. The maintainer’s literal no-negative-median policy is also distinct from the owner’s accepted observed<1% budget; neither is silently replaced.

## What was actually wrong and what changed

1. **Cost calibration ran outside the actual serving working set.** Target shape prices measured before the ordinary serving hold and drafter buffers were resident could change admission and cheap-copy work. The correction defers timing until those allocations are resident, flushes the temporary timing working set, then prepares tap destinations. This is a startup/accounting correction—not faster arithmetic, lower activation precision, a looser admission margin or a changed benchmark.
2. **Logical proposal length was allowed to change the fixed target query geometry.** The source-bound correction keeps the configured query block9, returns only the valid proposal prefix and preserves zero-row/rewind and8/16 physical-cohort contracts. Logical width and actual padded GPU work are not interchangeable.
3. **Copy-window planning state could be consumed more than once.** The corrected reservation owns a window’s planning decision, including empty/backoff outcomes, rather than advancing borrowed state again. The existing diagnosis and causal receipts distinguish changed work from elapsed-time variation.
4. **The integration exceeded the repository’s600-line production-module limit.** Commit `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce` extracts coherent startup scratch and behavioral-test companions, restores the main tap-only test module and migrates every fixture caller. No assertions, numerical tolerances, runtime math, hot-path allocation/copy behavior or CI gate was weakened. Local same-base lean comparison:424 findings against424 on pinned main,0 new.
5. **A final desktop whitespace gate was overlooked by a subset-only CI check.** Atf7 the actual desktop security/runtime tests and typechecks passed, but the final whole-PR gate rejected one extra blank EOF line in the diagnosis Markdown. Commit `51dce745143d72dd52231c44a31989632107af4a` removes only that line. All11 actual upstream PR-attached checks subsequently succeeded. `current-pr-ci.json` records the complete check list; `docs-tip-runtime-identity.json` proves identical native/build source objects and the explicit build version.

Earlier causal and same-work evidence is [source-labelled in the production diagnosis](https://github.com/BlivionIaG/TensorFold/blob/afe8bda62562519b4fd84ee9a956392a21f26a60/docs/receipts/lfm2-dspark-review/diagnosis.md) and [corrected-serving-calibration packet](https://github.com/BlivionIaG/TensorFold/blob/d6dd8d420983676c81b2af06e4a85a25f4ef5cfe/docs/receipts/lfm2-dspark-review/79837c4e2/serving-calibration/README.md). Historical results retain their actual source/native labels; none is retagged asf7.

## Completed current-runtime verification

- Integrated local host/security/golden/fuzz command:675/675 build steps,10,405 passing tests and68 skips. Actual macOS developer tools:226/226 build steps. These are observed local runs,not invented upstream CI counts.
- M4 real GPU/model safety:82/82 steps,767 tests passed/8 skipped; GPU contract16/16 kernels and24/24 planted controls,zero missed/limit problems. Both BF16 and8-bit heads match9/9 FP32-reference picks; worst confidence differences0.00085/0.00152.
- [Complete M5 Ultra qualification](../qualification/m5/README.md):144/144 cells,306/306 exact target-token oracles,0 exactness/scenario failures and9 clean non-forced native exits. All32 point-estimate comparisons fit the owner’s observed<1% budget; worst loss−0.328889768%. The original zero-tolerance gate remains **FAIL/exit3**,19 negative medians. Saved-data replay reproduces the original summary byte-for-byte and returns3.
- M5 guards:82/82 steps,771 tests passed/4 skipped;17/17 kernels and24/24 planted controls. BF16 FP32-reference picks9/9;8-bit8/9,largest reported different-pick margin0.075,confidence differences0.0008/0.00225. The unmodified original native guard passes. Do not call8-bit head picks FP32-exact.

Both head-weight modes retain BF16 activations. The FP32 comparison is the head calculation on BF16 target features,not an FP32 target server. Serving target-token equality is a separate contract and passed every completed M5 oracle.

## Measurement interpretation—not a waiver

A negative point estimate is not,by itself,a demonstrated regression or an estimate of measurement error. Conversely,visible spread does not prove every loss is noise. The historical M4 diagnostic showed reference drift and same-input/same-work GPU elapsed variation; it did **not** identify a universal error floor,prove downclocking as the cause,or replace a frozen qualification verdict.

The [M4 controlled method](../controlled-study/README.md) was published before this collection:fixed six rounds,bracketing plain A/A runs,mandatory same-cell A/A validity and every candidate’s one-sided95% lower bound within the owner’s1% margin. It includes all measured and warmup cells,exactness checks and full actual clock acquisition. No adaptive sample count,slow-run discard,head relabelling or rerun for a better sign. No completed controlled result is asserted here.

Previous operator captures place most target GPU elapsed in expert gate/up/down work,not sub-millisecond CPU planning/copy work. Global GPU busy,DVFM and command-buffer interval coverage are **not** SIMD occupancy or bandwidth measurements. There is no controlled current-head MLX-LM comparison here,and no evidence-backed claim that this branch beats MLX-LM or that a particular hardware resource is the remaining bottleneck.

## Actual moved-main integration blocker

The successful normal fixed-base merge was `c2a4dba92436d8db672879d836d4e4fce85c8620` against `2c8422a0c49d04a1867cffec081a1a42b65c857f`. The current51d head’s11 checks are green. A later actual Git ref query reports upstream main `a897e165bb271f8cc7a6530bbfc5235abf270341`. The REST PR response still lists the old2c base but now says `mergeable=false` / `dirty`; the actual Git ref,not that stale base field,was used for diagnosis.

Read-only `git merge-tree --write-tree --messages a897e165bb271f8cc7a6530bbfc5235abf270341 51dce745143d72dd52231c44a31989632107af4a` exits1,with conflicts in:

- `zig/src/native/metal.zig`
- `zig/tests/contract/metal_main.zig`

The resulting diagnostic tree is `3419f01bb15c450fac0b2a9d194a0a5bc5812325`. `new-main-ref-observation.json` preserves the exact snapshots and result. No new merge/rebase/main-chasing source change was applied; the ongoing fixed-source qualification and sealed production executable were not disturbed. New integration source would need its own appropriate proof; it cannot silently inherit a different native binary’s result.

No PR merge,auto-merge,force push,empty metadata fix,additional PR,repeat re-review request,pressure or policy waiver was performed.
