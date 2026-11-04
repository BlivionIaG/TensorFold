# LFM2 DSpark: review fixes and complete two-device matrix

## Verdict

**Not ready to land.** Both complete matrices are token-exact, but both original performance gates fail. M4 also failed the separate maximum-context native diagnostic with GPU out-of-memory. No failure is waived or called measurement error.

Qualified runtime source: `5fdbfb8749b14bc20bc9021dbe463cd56206d53a`, integrating pinned main `7eae7dc008ca8f280b3557596e7f8d425693b12d`. Scope is only [LFM2 DSpark #1394](https://github.com/ashhart/TensorFold/pull/1394).

| Device | Complete cells | Token oracles equal | Medians below plain | Losses ≥1% | User observed loss<1% | Worst median | Original exit |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4 Pro | 144/144 | 306/306 | 13/32 | 12/32 | 20/32 | −4.4895182331% | 3 (FAIL) |
| M5 Ultra | 144/144 | 306/306 | 10/32 | 0/32 | 32/32 | −0.4451788786% | 3 (FAIL) |

The user's <1% allowance and the maintainer's every-cell-at-least-plain rule are different. M5 satisfies the former on these observed medians, not the latter. M4 satisfies neither. The original summaries' `failures_total=0` concerns token/scenario failures; it does not erase `performance.negative_cells` or exit 3.

## What was wrong, and what changed

1. **Unfunded request enrollment.** Missing serving quotes enrolled a neural context, and unpaid recent histories could buy a recovery probe on hypothetical perfect acceptance. Enrollment now requires a complete quote that pays at the existing conservative rates and unchanged 1.05 margin. Cold paid trials keep their bounded cadence; fully unbound direct diagnostics remain available. Declined serving requests keep cheap-copy speculation without a neural lane or taps. This is a real economic-policy correction, not evidence that all remaining timing losses are fixed.
2. **Main integration needed a semantic union.** The requested main snapshot moved drafter refusal into a centralized check and admitted Llama there. Integration retains upstream Llama and LFM2 DSpark rather than choosing either conflicting branch wholesale. Both independent Metal contract registrations remain. The admission regression covers accepted and refused families.
3. **Review branch carried raw results.** Fourteen JSON files / 96,536 lines were removed from the landing branch. [The relocation ledger](../receipt-relocation.json) binds the original files; eleven were already preserved byte-for-byte, three anonymize personal paths only. Historical source labels, measurements and failures are retained.
4. **The native diagnostic retained avoidable overlapping allocations.** Source `763c8678cc680a89ff0104cde4bb8b59cd9fa766` shortens independent target-cache lifetimes. Follow-up `b57b839119a10eedfd1cb0909c1e660d4a5abffe` also defers the main tap buffer and separates independent stage buffers. The final short BF16/8-bit cases pass, but the unchanged 130,816 case still fails GPU OOM in the full eight-column stage control. Both changes affect diagnostic tests only; the production source and actual M4 binary are unchanged. All failures remain retained; this is not full diagnostic or performance clearance.

## What is still not established

The remaining performance losses are not causally localized by these matrices. M4's largest plain three-sample range is 8.6923071517%, and all 32 paired contrasts change sign across rotations; M5 has 17 sign-changing contrasts and a largest plain range of 1.3662228823%. These are observed variability, not an A/A-derived error floor, confidence bound, or permission to waive losses. No new controlled-study rerun, extra warmup, trimming, pooling across chips, or rerun for a better sign was used.

There is no matched MLX-LM execution breakdown in this packet. A regression against TensorFold's own no-drafter arm does not identify the cause of an MLX-LM gap. Aggregate proposal/accept counters include cheap-copy speculation and do not prove actual neural/operator work, SIMD occupancy or bandwidth saturation. Earlier component localization remains [in the diagnosis](../diagnosis.md); none is relabelled as this source's full performance clearance.

### What the completed work counters narrow down

The [measured-cell counter comparison](aggregate-work-parity.json) compares each drafter arm with its same-rotation plain cell. Aggregate rounds/proposed/accepted agree in 69/96 M4 contrasts and 50/96 M5 contrasts. These include all proposal sources, not a neural-work counter.

For M4's worst median cell (8-bit, greedy code, one stream):

| Rotation | Plain rounds | Drafter rounds | Plain accepted | Drafter accepted | Decode change |
|---|---:|---:|---:|---:|---:|
| 0 | 272 | 272 | 239 | 239 | −4.8660933839% |
| 1 | 272 | 269 | 239 | 242 | +2.7545797180% |
| 2 | 272 | 284 | 239 | 227 | −4.9882630717% |

Rotation2 really does more target rounds for the same512 output tokens; rotation0 loses with identical aggregate counters. Extra target work is therefore part of the observed problem, not a complete explanation. Equal aggregates also do not prove equal neural or operator execution.

Source inspection identifies a pricing limitation to investigate: [request admission](https://github.com/BlivionIaG/TensorFold/blob/5fdbfb8749b14bc20bc9021dbe463cd56206d53a/zig/src/families/lfm2_moe/dspark_admission.zig#L59-L105) compares with `target.get(streams)`, one target row/token per stream; the [live hold planner](https://github.com/BlivionIaG/TensorFold/blob/5fdbfb8749b14bc20bc9021dbe463cd56206d53a/zig/src/core/lanes/hold_price.zig#L141-L178) can compare against a committed copy baseline with multiple expected tokens. **[INFERENCE]** Request-level admission may fund context capture that beats one-row decode but not the actual cheap-copy route. The packet does not establish how much of the observed loss this explains. Correcting that requires a directly observed copy-baseline cost, not hardcoding benchmark winners or pretending aggregate proposal counts are neural acceptance.

## Precision and reference

Both drafter-weight modes retain **BF16 activations**. The independent FP32 host reference consumes the same BF16 target features widened to FP32, the released checkpoint, and original target embedding/head. It is not an FP32 target-server arm. Each Markov reference step is teacher-forced on the previous GPU pick. Numerical gates and memory admission were not relaxed.

| Device | BF16-weight short picks | 8-bit-weight short picks | Different 8-bit pick max margin | Worst BF16 confidence difference | Worst 8-bit confidence difference |
|---|---:|---:|---:|---:|---:|
| M4 Pro | 9/9 | 9/9 | 0 | 0.00085 | 0.00152 |
| M5 Ultra | 9/9 | 8/9 | 0.075 | 0.00080 | 0.00225 |

M5's 8-bit discrepancy is preserved under the original native guard rules, not called universal exactness.

## Original protocol and identities

Each chip ran the unmodified [frozen driver](driver/) once: all original rotations, widths1/4/8/16, code/chat, T0/T1, plain/BF16/8-bit, 512-token replies, warmups and oracles. Plain keeps cheap-copy speculation. The original verdict is replayed offline from original metadata/cells/events and reproduces each summary byte-for-byte. Raw request streams are compressed without changing decoded bytes; [M4 bindings](qualification/m4/raw-bindings.json), [M5 bindings](qualification/m5/raw-bindings.json).

| Identity | M4 Pro | M5 Ultra |
|---|---|---|
| Production binary SHA256 | `ee8f3451f5264f11bc9dea24af7ba5b0b6e00d5004e496d0f8a7d7f18bc92d02` | `d96296edf478a962c416d2339c06cf2e0677b114c33c2726fd7ec434d0337dfa` |
| Guard build | 82/82 steps; 779 passed, 8 skipped | 82/82 steps; 783 passed, 4 skipped |
| Actual LFM2 kernel contract | 16/16 kernels; 24/24 controls red; zero limit problems | 17/17 kernels; 24/24 controls red; zero limit problems |

Source archive SHA256: `5f13d1bd16293d31b911757805d1aa3b3a28934496b40bc4145b80d2327d0307`.
Target revision: `667408d494bde191c2cb9dd4f4549827816b9275`; drafter revision: `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`. Full checkpoint and driver hashes are in the bound guard records.

Formatting passed. Lean comparison: 425 findings on the pinned main and this runtime source, zero new. All 11 CI checks succeeded on source5f ([host/Windows/macOS](https://github.com/ashhart/TensorFold/actions/runs/41810461429), [applications](https://github.com/ashhart/TensorFold/actions/runs/41810461623), [format](https://github.com/ashhart/TensorFold/actions/runs/41810461431)). This does not claim later-head CI.

## All 64 observed comparisons

Medians of the original three samples; percentages relative to the same cell's no-drafter median. Each table has 16 cells and two drafter comparisons per cell. No rounding changes acceptance.

### M4 Pro

| Cell | Plain TPS | BF16 TPS | BF16 Δ% | 8-bit TPS | 8-bit Δ% |
|---|---:|---:|---:|---:|---:|
| code T0 w1 | 169.518161 | 169.835263 | +0.187061 | 161.907612 | -4.489518 |
| code T0 w4 | 283.403984 | 292.461046 | +3.195813 | 292.152841 | +3.087062 |
| code T0 w8 | 313.417165 | 322.778100 | +2.986733 | 322.734480 | +2.972816 |
| code T0 w16 | 349.732125 | 359.473174 | +2.785289 | 359.207059 | +2.709198 |
| code T1 w1 | 114.001293 | 114.927833 | +0.812745 | 115.566768 | +1.373208 |
| code T1 w4 | 198.551836 | 193.997245 | -2.293906 | 205.387896 | +3.442960 |
| code T1 w8 | 238.255909 | 234.784972 | -1.456810 | 246.930988 | +3.641076 |
| code T1 w16 | 291.664868 | 288.652615 | -1.032779 | 302.272210 | +3.636825 |
| chat T0 w1 | 117.476400 | 118.145282 | +0.569376 | 114.950563 | -2.150081 |
| chat T0 w4 | 242.779538 | 243.340596 | +0.231098 | 237.505871 | -2.172204 |
| chat T0 w8 | 278.516749 | 282.276266 | +1.349835 | 274.342325 | -1.498805 |
| chat T0 w16 | 335.011421 | 336.316223 | +0.389480 | 335.064479 | +0.015837 |
| chat T1 w1 | 114.774055 | 113.509740 | -1.101569 | 118.684667 | +3.407226 |
| chat T1 w4 | 207.607873 | 207.029117 | -0.278774 | 210.465207 | +1.376313 |
| chat T1 w8 | 257.524716 | 247.913392 | -3.732195 | 250.786950 | -2.616357 |
| chat T1 w16 | 315.312945 | 301.486679 | -4.384934 | 304.988948 | -3.274206 |

### M5 Ultra

| Cell | Plain TPS | BF16 TPS | BF16 Δ% | 8-bit TPS | 8-bit Δ% |
|---|---:|---:|---:|---:|---:|
| code T0 w1 | 285.327547 | 288.501738 | +1.112473 | 290.037859 | +1.650844 |
| code T0 w4 | 645.572975 | 646.352945 | +0.120818 | 642.769089 | -0.434325 |
| code T0 w8 | 853.342299 | 854.166349 | +0.096567 | 852.936735 | -0.047526 |
| code T0 w16 | 872.409335 | 872.945866 | +0.061500 | 886.425412 | +1.606594 |
| code T1 w1 | 271.808396 | 272.001818 | +0.071161 | 274.336633 | +0.930154 |
| code T1 w4 | 552.168119 | 553.548106 | +0.249922 | 553.620272 | +0.262991 |
| code T1 w8 | 712.168612 | 714.953266 | +0.391010 | 712.488682 | +0.044943 |
| code T1 w16 | 878.615969 | 878.165438 | -0.051277 | 876.781757 | -0.208762 |
| chat T0 w1 | 264.968329 | 265.569516 | +0.226890 | 265.186690 | +0.082410 |
| chat T0 w4 | 625.349844 | 625.463733 | +0.018212 | 625.629179 | +0.044669 |
| chat T0 w8 | 822.340726 | 821.651357 | -0.083830 | 821.778015 | -0.068428 |
| chat T0 w16 | 810.029592 | 808.961481 | -0.131861 | 806.423511 | -0.445179 |
| chat T1 w1 | 259.097887 | 259.470361 | +0.143758 | 259.886158 | +0.304237 |
| chat T1 w4 | 562.059080 | 562.730134 | +0.119392 | 563.388770 | +0.236575 |
| chat T1 w8 | 721.210992 | 722.373845 | +0.161236 | 721.164108 | -0.006501 |
| chat T1 w16 | 885.192343 | 884.372959 | -0.092566 | 885.623621 | +0.048721 |

## Cache, sampling, cancellation and long context

Both modes on both chips passed cached=fresh with 3,517 reused tokens and all 29 sampled hash triples per mode (plain/solo/concurrent). Cancellation stress used the unchanged 6-slot/60s and 16-slot/90s workloads.

| Device | Sampled equal triples | Completed stress requests | Cancellations | Wrong/errors | Clean non-forced server exits |
|---|---:|---:|---:|---:|---:|
| M4 Pro | 58/58 | 820 | 465 | 0/0 | 14 |
| M5 Ultra | 58/58 | 2,776 | 1,506 | 0/0 | 14 |

Served long-context plain=drafted passed 4/4 on each chip at 4,096/16,384/65,536/130,816 input tokens, 64 complete output tokens per reply. These are correctness, not speed qualifications.

M5's native 65,536 and 130,816 references both passed 9/9 FP32 picks, with worst confidence differences 0.00625 and 0.00911. M4's 65,536 native reference passed 9/9, confidence difference 0.01050. M4's original 130,816 native diagnostic failed **before the FP32 comparison**: `LFM2 forward: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)` then `GpuFailed`; exit 1, not a timeout or token mismatch. The served path at that length passed; it does not clear this separate failure.

Original long native references use BF16 weights and BF16 activations. No 8-bit long FP32-reference run is claimed.

The native helper reserves full-prompt scratch, unlike serving's chunked working set. Static formulas give a scratch lower bound of 15.5068 GiB and scratch+two independent target KV caches+original taps lower bound of 22.9937 GiB, before target weights, drafter buffers, transient resources and OS use. These are static lower bounds, **not a measured allocation peak**. Actual M4 physical RAM is 51,539,607,552 bytes and Metal's recommended working set is 40,200,896,512 bytes. No budget, input length or oracle was changed to make the failure disappear.

### Diagnostic-only lifetime follow-up

[Source763's complete record](diagnostic-lifetimes/763c8678cc680a89ff0104cde4bb8b59cd9fa766/observed.json) and [original/published hashes](diagnostic-lifetimes/763c8678cc680a89ff0104cde4bb8b59cd9fa766/bindings.json) preserve the first lifetime correction's outcome:

| Native case | Exit | FP32 picks | Worst confidence difference |
|---|---:|---:|---:|
| 64 rows, BF16 weights | 0 | 9/9 | 0.00085 |
| 64 rows, 8-bit weights | 0 | 9/9 | 0.00152 |
| 130,816 rows, BF16 weights | 1 | Not reached: GPU OOM | Not reached |

The failed long case took 2,097.326 seconds, below its original 3,600-second deadline. Its [failure log](diagnostic-lifetimes/763c8678cc680a89ff0104cde4bb8b59cd9fa766/native-130816-bits0.log) remains byte-identical. The sealed diagnostic SHA-256 is `3a5c97b40cec5818ab8227f60d73078adeeaaf536e27110ba536d16ddab70a2a`; the production binary remains byte-identical to the full-matrix M4 binary `ee8f3451f5264f11bc9dea24af7ba5b0b6e00d5004e496d0f8a7d7f18bc92d02`.

Follow-up diagnostic source `b57b839119a10eedfd1cb0909c1e660d4a5abffe` removes two further overlaps: the original full-prompt tap destination is not allocated until independent controls finish, and table-only stage controls finish before the other stage buffer is allocated. The same row counts, three column layouts, twin/fallback modes, byte comparisons, guards and independent reference remain.

Its [complete source/binary-bound record](diagnostic-lifetimes/b57b839119a10eedfd1cb0909c1e660d4a5abffe/observed.json) preserves another failed maximum-context result: exit 1 after 2,341.274 seconds, within the original 3,600-second deadline. The [phase-labelled failure log](diagnostic-lifetimes/b57b839119a10eedfd1cb0909c1e660d4a5abffe/native-130816-bits0.log) shows the full-length untapped target, five-column controls, three-column controls and eight-column table-only control completing in twin-enabled mode. The first full eight-column staged forward then fails GPU OOM. The drafter FP32 comparison is not reached.

The changed diagnostic's short BF16 and 8-bit cases both exit 0 and match 9/9 FP32 picks; worst confidence differences remain 0.00085 and 0.00152. Its sealed binary SHA-256 is `4ec0ff3e4e7d1dff25ac48a52c38da0746b948739f899a46606688b7d97e6e96`. The actual M4 production binary again matches the full-matrix binary byte-for-byte. [Original/published hashes](diagnostic-lifetimes/b57b839119a10eedfd1cb0909c1e660d4a5abffe/bindings.json) retain each log and any path-only redaction.

At the failing call, the source retains these [three independent full-length buffers](diagnostic-lifetimes/b57b839119a10eedfd1cb0909c1e660d4a5abffe/allocation-overlap.json), in addition to full scratch, target KV, target/drafter weights, other transients and OS use:

| Required control allocation | Bytes | GiB |
|---|---:|---:|
| Eight-column raw capture destination and address table, Metal shared | 4,287,625,216 | 3.9932 |
| Eight-column independent stage and two guard rows, Metal shared | 4,286,644,224 | 3.9922 |
| Independent captured-tap reference, host | 4,286,578,688 | 3.9922 |

These are static allocation sizes, not a measured resident-memory peak or proof of the precise driver allocation that failed. The control is not reduced or skipped and no budget is raised. Each changed source was run once; neither failure was retried unchanged for a pass. The separate served 130,816-token path still passes; this diagnostic OOM does not explain the serving speed losses.

M5's collector disconnected after persisting successful functional8, retaining `BrokenPipeError`. Only the never-started long phase was continued once; no completed model phase or full matrix was rerun. [Completion accounting](qualification/m5/completion-accounting.json) keeps that failure and every actual exit separate from model correctness/performance.

## Readiness and remaining evidence

- The funded-admission fix, requested main integration and receipt relocation are implemented and pushed.
- Both original full performance verdicts remain FAIL; M4 also exceeds the user's observed1% allowance in12comparisons.
- Original M4 maximum-context native OOM is retained. The diagnostic lifetime correction's exact-case result will be attached separately, not used to relabel the failed source5f run.
- No waiver, precision change, threshold increase, resource-admission bypass, fixture shrink, or claim of zero performance failures.
