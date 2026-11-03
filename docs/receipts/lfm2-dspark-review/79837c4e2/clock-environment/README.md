# Complete M4 clock coverage and reference sensitivity

## Decision

Do not turn negative timing point estimates into confirmed code-regression counts. Keep the proven serving-residency calibration correction; no further runtime change is justified by treating the remaining sampled-solo point difference as an isolated head-mode cost. This investigation identifies **uncontrolled baseline/history/time sensitivity**, not a universal measurement-error floor or a proved hardware cause.

The [complete production qualification](../../afe8bda62/qualification/README.md) remains unchanged: M5's observed medians fit the task owner's less-than-1% budget; M4 cannot reliably resolve that margin. Literal zero-tolerance script verdicts are retained separately. This packet is **diagnostic, not requalification**.

## Actual execution and source binding

- Runtime source: `79837c4e205a18b5eb7236ea393e51dca1264fe2`; its documentation-only production descendant is `afe8bda62562519b4fd84ee9a956392a21f26a60`. Do not relabel this private binary as an AFE build.
- Instrumented native SHA256: `d91a2dd41c3183d6f4eca8bac0aeb5f60daa0c30374f2594be16a3a93500d943`; unchanged private patch `41fd643ed0ba78fddf752257ec8b35f4323cd145bb9a5c7dbc49df19c930dc40`.
- One original-history diagnostic: code then chat, T0/T1, widths1/4/8/16,512-token replies, context4096, parallel16, cache0. All seven original drivers and model revisions retain their hashes. Copy tracing and intrusive target counters are off; existing aggregate target timers remain available.
- All three arms complete104 phases each and exit0 without forced kills. All208 cross-arm phase pairs have identical run seeds, lengths and token hashes, including controls and warmups.
- Its [original scenario/oracle records](diagnostic-original-contract-check.json) contain48 cells and102 oracles with zero exactness/scenario failures, across one ordered diagnostic rotation. These are not substituted for the complete three-rotation production qualification.
- Controller and sampler exit0; the sampler's actual owned child stops cleanly, reader reaches EOF,1745 complete clock records, zero parse errors and zero incomplete tail bytes. Coverage includes the complete 8-bit arm.

[Original capture metadata](observed.json), [original phase observations](probe/observed.json), [controller output](controller.log), [prepared source binding](full-history-prepared-binding.json), [complete analysis](complete-clock-analysis.json) and [manifest](SHA256.json).

## What was wrong with the performance interpretation

The frozen production M4 data already show that **all16 unchanged-plain groups decrease at both rotation transitions**, not just a single unfortunate cell. Their [source-bound trend](plain-rotation-drift.json) is derived from the original saved cells; no model was rerun to obtain it. The largest plain range is32.256410% using `(max/min-1)*100`. This is common drift, not sixteen head-mode regressions.

The new full-history capture exposes a stronger control. The original solo oracle and later one-stream measured request have identical inputs and, in every one of12 arm/scenario pairs, equal activity and target physical-row histograms. For sampled code, all modes at both request points do504 rounds,66 proposed drafts and7 accepted drafts. All three measured requests decline neural context and record zero captured target calls.

| Existing request point | Plain decode GPU ms | BF16 decode GPU ms | BF16 delta | 8-bit decode GPU ms | 8-bit delta |
|---|---:|---:|---:|---:|---:|
| solo | 4733.002992 | 4515.002829 | -4.605959% | 4510.041495 | -4.710783% |
| measured | 4315.448793 | 4490.471254 | +4.055719% | 4548.823625 | +5.407893% |

The plain request's decode GPU time changes **-8.822183%** between these two existing occurrences,151.219 seconds apart. The apparent 8-bit delta reverses sign depending on which existing reference point is used. **Neither reference is substituted into the frozen verdict.** This demonstrates reference/history/time sensitivity; it does not make8.8% a noise allowance, prove equivalence or identify thermal state as the cause. These are not randomized, steady-state A/A trials.

[All12 existing repeat pairs](existing-request-repeat-observations.json) and [sampled-code reference sensitivity](sampled-solo-reference-sensitivity.json).

## The measured residual is target GPU elapsed, not a large CPU wrapper bill

For the sampled-code measured requests:

| Arm | Decode GPU ms | Target CPU encode ms | Plan CPU ms | Copy CPU ms | Enrolled contexts |
|---|---:|---:|---:|---:|---:|
| plain | 4315.448793 | 42.213375 | 0.623667 | 0.433584 | 0 |
| bf16 | 4490.471254 | 41.429291 | 0.546542 | 0.385167 | 0 |
| bits8 | 4548.823625 | 43.108500 | 0.529917 | 0.384292 | 0 |

Activity and both prefill/decode physical-row histograms agree exactly; no neural tap capture is recorded. The plan/copy timings do not account for the observed hundreds-of-milliseconds target GPU difference. GPU command-buffer elapsed is not a proof of per-process shader efficiency: scheduling, state/history and unmeasured memory behavior remain possible contributors. No unsupported causal explanation is assigned.

The [older calibrated operator capture](../../2acdb56a2/copy-gpu-attribution/README.md) remains useful for locating large GPU work, especially expert projections, not proving this current ordinary-timing residual. Its profiled command-buffer stopwatches were invalid and excluded; its [same-case class crosscheck](older-class-crosscheck.json) remains explicitly bound to source2ac. No new encoder-stage profiling capture was launched.

## Complete clocks show changing thermal conditions, not a simple universal downclock explanation

| Recorded thermal pressure | Complete samples | Sample elapsed seconds |
|---|---:|---:|
| Nominal | 846 | 850.463378 |
| Moderate | 826 | 830.269014 |
| Heavy | 73 | 73.386304 |

[Actual transitions](environment-observation.json) include Nominal, Moderate and Heavy. Across the whole diagnostic, global GPU busy time is99.215906% of nominal sample elapsed;95.075837% of busy time is at the maximum reported DVFM state. **Busy time is not SIMD occupancy or bandwidth utilization.** Frequencies are preserved in the raw tool's units rather than silently labelling `freq_hz` as SI Hz.

The sampled-code interior frequency observations do not support a simple lower-GPU-frequency explanation for the loaded arms' larger measured target elapsed. There are only one or two conservative interior samples for these short requests, so this is not a complete per-kernel clock attribution. No DRAM-clock, bandwidth or SIMD-occupancy measurement is claimed. Recorded thermal changes are in this later diagnostic, not retroactively measured during qualification.

The historical [nonpersonal power/display records](historical-power-events.json) show AC/80% snapshots and display changes during the earlier qualification. They are not continuous power-mode proof and do not establish the cause. Unrelated private application assertions are not published. No power setting, admission limit, pressure guard or user process was changed.

## Timing method and reproduction

The clock stream uses NUL separators, not a guaranteed trailing terminator. Reader delivery may wait for the next record; its receipt time must not be treated as sampling time. Source plist dates have one-second resolution. The analyzer conservatively covers both start-date and end-date interpretations with `[date-elapsed_ns,date+elapsed_ns+1second]`; a sample is assigned only if that entire interval lies inside the RPC window. The observed wall/perf offset range is39.297083ms and is included in the bounds. Boundary samples are excluded, not imputed; short requests may have no interior sample.

[Clock samples](clock-samples.jsonl), [lossless raw plist stream](powermetrics.pliststream.gz), [raw binding](raw-clock-binding.json), [collector](run-with-clocks.py), [controller](run-full-history-clock-probe.py) and [offline analyzer](analyze-complete-clocks.py) retain their actual source/data hashes. The initial sampler-only smoke exposed a sudo-parent cleanup bug before loading any model; [failed metadata](collector-smoke/observed.json) and [failed source](failed-sudo-parent-cleanup.py) are preserved. The corrected [actual child-owned smoke](collector-smoke-owned-child/observed.json) exits0, without forced stop. No qualification was repeated for a better result.

To replay the analysis without loading a model:

```sh
gunzip -k powermetrics.pliststream.gz
python3 analyze-complete-clocks.py --capture-root . --output replay.json
```

The published layout is exercised with this [byte-identical offline replay](publication-replay-check.json); original capture files remain byte-identical. `private-logs/` remain excluded by the original driver's publication contract.

## Readiness and limits

The [retained calibration fix](../serving-calibration/README.md) removes the specific captured old pricing/work divergence. It does not promise identical copy allocations under every subsequent measured curve/history; this new full-history capture also contains wider cells with different work. The sampled-solo comparison above is singled out because its actual activity and physical-row histograms agree.

The exact hardware/state cause of within-plain timing drift is not established. This packet is enough to reject **automatic bug counts from negative timing signs**, not enough to declare M4 noninferior at1%. Establishing that requires a controlled baseline/history/thermal design; the existing frozen dataset cannot be repaired by choosing another reference or rounding away losses. M5's satisfied observed-median budget is not upgraded to a statistical confidence claim. GitHub mergeability/CI and outstanding review decisions remain separate readiness blockers. No re-review or merge request is made.
