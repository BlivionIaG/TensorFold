# Completed prespecified M4 controlled study — **INCONCLUSIVE**

**No performance clearance.** All24 fixed launches completed,with384 measured cells,768 fixed-warmup cells,8,352 complete512-token requests,zero cached tokens and every original token oracle equal. Only **2/16** plain A/A cells meet the prespecified stability criterion. Candidate point estimates fit the owner's<1% loss margin in **31/32** comparisons,but only **23/32** one-sided95% lower bounds clear−1%. The worst observed paired median is **−1.012328125%**,8-bit code/T0/1 stream; it is not rounded into a pass. Its lower bound is−3.991890203%,so that cell is not cleared either.

The unchanged analyzer's actual verdict is **INCONCLUSIVE**. The collector and analyzer exit0 mean complete acquisition/analysis,not a semantic performance pass. The outer pipeline deliberately returns the original full-matrix exit3. No sample count changed,no slow run was discarded,no result was retagged,and neither original frozen verdict was replaced.

## Bound source,method and full scope

Runtime source `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce`, sealed production native SHA256 `a7be9659f48c7cdb6a19a63a013bc850c679343a9e18e5dfa81ef220ff702fc0`,source archive SHA256 `3ca340ec39493b7d32ba23902e2087111e54def1184c1665331b4ada1cbe7b12`. The current docs-only descendant `51dce745143d72dd52231c44a31989632107af4a` has identical native/build source objects. The original output metadata retains the source/doc identifiers it was created with; it was not rewritten to call a later commit qualified.

[The original method,binding,collector,analyzer,clock-acquisition correction and retained superseded attempt](https://github.com/BlivionIaG/TensorFold/blob/9ee4a1399bf6e5f60c43b6a28dd843b9a9c7749b/docs/receipts/lfm2-dspark-review/f7eb5f0bc/controlled-study/README.md) were published before this collection. Protocol SHA256 `d4dea2d29b0a4e2c4d2af09222b60c89c27a8090500789e7a660a9e639f31765` is unchanged. Six rounds each run plain_before,BF16/8-bit in alternating order,plain_after. Every launch runs two complete fixed16-cell warmup sweeps and one measured sweep:code/chat,T0/T1,1/4/8/16 streams,512 tokens,parallel16,context4096,cache0 and plain cheap copies enabled.

All24 actual shutdown events show return0 and no forced kill. All1,152 recorded cells match the original observed state. Publication validation checked every one of8,352 raw requests against the34 original source-bound target-token oracle references,512 complete tokens andcached_tokens0. Both head modes keep BF16 activations. This is the production executable,not a diagnostic instrumented binary.

## Prespecified decision—not a noise waiver

For each cell/round,the candidate reference is the geometric mean of bracketing plain TPS. Reported candidate change is the median of six paired ratios; the geometric-mean change and one-sided95% lower bound use mean log ratio,Student-t df5,critical2.015048373333. Every candidate needs observed loss<1% **and** lower bound>−1%.

Independently,the90% interval for plain_after/plain_before must lie entirely within[−1%,+1%] in every one of16 cells. This requirement fails in14 cells,so the global verdict is **INCONCLUSIVE**,not PASS or a confirmed32-cell regression. The normal-log-ratio and independent-round assumptions are limitations,not a universal error floor. All observations—including the large excursion below—remain in the published result.

Two full warmup sweeps condition both the device and the adaptive planner. The change from the original matrix's2/32 inside-margin observations to31/32 here is **not a code speedup**:the native executable is identical and the methods/states differ. Neither better-looking medians nor this uncertainty study erase [the original M4 performance FAIL](../../qualification/m4/README.md),where30/32 comparisons exceed the owner's budget.

## All16 A/A validity checks

| Prompt | T | Streams | Plain A/A geometric Δ | A/A 90% lower | A/A 90% upper | A/A valid |
|---|---:|---:|---:|---:|---:|---|
| chat | 0 | 1 | -1.543413% | -3.476111% | +0.427983% | NO |
| chat | 0 | 4 | -0.775383% | -1.482450% | -0.063241% | NO |
| chat | 0 | 8 | +0.095730% | -0.894227% | +1.095576% | NO |
| chat | 0 | 16 | +0.093931% | -0.340375% | +0.530130% | YES |
| chat | 1 | 1 | -1.148094% | -3.093526% | +0.836393% | NO |
| chat | 1 | 4 | +0.367868% | -0.900215% | +1.652177% | NO |
| chat | 1 | 8 | +0.873091% | -0.928164% | +2.707095% | NO |
| chat | 1 | 16 | +0.413581% | -0.737421% | +1.577929% | NO |
| code | 0 | 1 | -0.254187% | -0.869184% | +0.364625% | YES |
| code | 0 | 4 | +8.018935% | -7.628131% | +26.316492% | NO |
| code | 0 | 8 | +1.092152% | -0.689161% | +2.905417% | NO |
| code | 0 | 16 | +0.380233% | -0.571477% | +1.341054% | NO |
| code | 1 | 1 | -1.720035% | -3.416742% | +0.006478% | NO |
| code | 1 | 4 | +0.195890% | -0.606126% | +1.004377% | NO |
| code | 1 | 8 | +0.637920% | -0.724669% | +2.019210% | NO |
| code | 1 | 16 | +0.578020% | -0.561280% | +1.730373% | NO |

## All32 candidate comparisons and lower bounds

Rounded display never decides the verdict;`analysis.json` preserves the unrounded values and all six paired log ratios. A candidate bound marked YES does not override a failed same-cell A/A check.

| Prompt | T | Streams | Head | Paired median Δ | Geometric mean Δ | One-sided 95% lower | Observed <1% loss | Bound clears −1% |
|---|---:|---:|---|---:|---:|---:|---|---|
| chat | 0 | 1 | bf16 | +0.281915% | +0.820171% | -0.360656% | YES | YES |
| chat | 0 | 1 | bits8 | -0.514948% | -0.622309% | -1.952798% | YES | NO |
| chat | 0 | 4 | bf16 | +0.127929% | +0.478838% | -0.450443% | YES | YES |
| chat | 0 | 4 | bits8 | -0.291894% | -0.301916% | -0.636514% | YES | YES |
| chat | 0 | 8 | bf16 | -0.003194% | +0.464803% | -0.397823% | YES | YES |
| chat | 0 | 8 | bits8 | -0.182245% | -0.035146% | -0.490983% | YES | YES |
| chat | 0 | 16 | bf16 | -0.018859% | +0.191933% | -0.167391% | YES | YES |
| chat | 0 | 16 | bits8 | -0.094468% | -0.002874% | -0.229632% | YES | YES |
| chat | 1 | 1 | bf16 | -0.133226% | +0.452215% | -0.633239% | YES | YES |
| chat | 1 | 1 | bits8 | -0.506926% | -0.719894% | -1.412937% | YES | NO |
| chat | 1 | 4 | bf16 | +0.086895% | -0.390823% | -1.452883% | YES | NO |
| chat | 1 | 4 | bits8 | +0.040314% | +0.259464% | -0.382428% | YES | YES |
| chat | 1 | 8 | bf16 | -0.042491% | -0.265435% | -0.810005% | YES | YES |
| chat | 1 | 8 | bits8 | -0.046931% | +0.406503% | -0.499786% | YES | YES |
| chat | 1 | 16 | bf16 | +0.111418% | +0.161580% | +0.005397% | YES | YES |
| chat | 1 | 16 | bits8 | -0.015470% | +0.267407% | -0.321750% | YES | YES |
| code | 0 | 1 | bf16 | -0.423876% | -0.875729% | -2.246514% | YES | NO |
| code | 0 | 1 | bits8 | -1.012328% | -1.857483% | -3.991890% | NO | NO |
| code | 0 | 4 | bf16 | +0.104500% | +4.061052% | -3.782371% | YES | NO |
| code | 0 | 4 | bits8 | -0.036809% | +3.805872% | -4.075176% | YES | NO |
| code | 0 | 8 | bf16 | -0.233903% | -0.033366% | -0.620998% | YES | YES |
| code | 0 | 8 | bits8 | -0.278989% | +0.175232% | -0.818811% | YES | YES |
| code | 0 | 16 | bf16 | +0.014692% | +0.304007% | -0.388981% | YES | YES |
| code | 0 | 16 | bits8 | -0.045349% | +0.171084% | -0.309861% | YES | YES |
| code | 1 | 1 | bf16 | +0.097088% | +0.321935% | -1.086232% | YES | NO |
| code | 1 | 1 | bits8 | -0.506655% | -0.766524% | -1.581525% | YES | NO |
| code | 1 | 4 | bf16 | +0.034841% | +0.307301% | -0.465228% | YES | YES |
| code | 1 | 4 | bits8 | +0.042466% | +0.140710% | -0.371537% | YES | YES |
| code | 1 | 8 | bf16 | +0.084498% | +0.528682% | -0.420686% | YES | YES |
| code | 1 | 8 | bits8 | -0.027987% | +0.288446% | -0.368769% | YES | YES |
| code | 1 | 16 | bf16 | +0.233100% | +0.654971% | -0.317396% | YES | YES |
| code | 1 | 16 | bits8 | +0.114819% | +0.344388% | -0.182436% | YES | YES |

## Why a single global measurement-error percentage would be wrong

All96 plain A/A observations are retained in`work-clock-diagnosis.json`. Aggregate round/draft/accept counters match in **80/96** pairs and differ in16. Aggregate parity is not proof of identical padded GPU work. The three largest absolute plain A/A changes are shown descriptively,not selected as substitute qualification data:

| Plain A/A cell | Before TPS | After TPS | Change | Aggregate work equal | Active-weighted raw frequency, before → after | Thermal records, before → after |
|---|---:|---:|---:|---|---|---|
| r2 code T0 w4 | 183.488752 | 292.176791 | +59.234170% | NO | 1488.171 → 1577.966 | {'Heavy': 12} → {'Nominal': 7} |
| r2 chat T0 w1 | 126.672823 | 119.559109 | -5.615817% | YES | 1573.815 → 1577.843 | {'Heavy': 5} → {'Nominal': 1, 'Moderate': 3} |
| r2 chat T1 w8 | 246.338122 | 259.843166 | +5.482320% | YES | 1466.000 → 1577.957 | {'Moderate': 17} → {'Moderate': 3, 'Nominal': 13} |

The59.234% baseline excursion coincides with **both different proposal work and Heavy-versus-Nominal thermal state**,so clocks alone cannot explain it. The chat/T0/1-stream case has511 rounds,no drafts in either plain arm,and a5.616% slowdown despite both intervals spending almost all active time near maximum reported DVFM; neither neural-head compute nor a simple universal downclock explanation covers every change. The chat/T1/8-stream example has equal aggregate counters and a lower recorded frequency in the slower interval. These are observations/associations,not a causal hardware attribution.

Clock acquisition retained **15,466 complete actual GPU/thermal records**,zero parse errors,zero incomplete tail bytes,reader EOF,controller/sampler return0 and clean owned sampler shutdown. Every predeclared coverage check passes. Approximate controller-wide GPU busy is **99.120466%**; **95.888058%** of active DVFM time is at the maximum reported state. Thermal record counts:Nominal6,733,Moderate7,769,Heavy964. These are **global** measurements,not process attribution,SIMD occupancy or memory bandwidth. The raw DVFM frequency field's units are preserved,not relabelled SI Hz. Per-cell alignment uses wall/perf anchors and one-second sample timestamps; it cannot resolve individual kernel latency.

## Concrete head-cost/dispatch evidence—not an invented bottleneck

Completed original-run startup logs report these rounded best-after-warm **CPU-encode+GPU** calibration costs,not serving operator times or confidence intervals:

| Device | Head weights | 1-draft clean hold ms, three-launch range | 9-draft clean hold ms, three-launch range |
|---|---|---:|---:|
| M4 | bf16 | 7.93–8.39 | 10.12–10.58 |
| M4 | bits8 | 5.05–5.16 | 6.42–6.43 |
| M5 | bf16 | 2.05–2.35 | 2.45–2.48 |
| M5 | bits8 | 2.26–2.27 | 1.54–1.56 |

The actual GPU guards report tensor units false on M4 and true on M5. Source `f7eb5f0bc` selects optional tensor pipelines through`kernels.zig:75-83,111-113`; otherwise BF16 uses row/MMA fallbacks and8-bit uses scalar/twin-window paths (`dspark_bf16.zig:10-35`,`dspark_block.zig:33-101`). The selected non-tensor quantized window uses **BF16 matrix operands and FP32 accumulators**,not an FP16 activation shortcut (`zig/kernels/metal/core/tile_qmv_rows.metal:13-15,66-103`). The fixed bidirectional query block remains9 even for a short retained prefix.

This establishes that the drafter is materially more expensive on M4 in the recorded startup calibration and that its dispatch differs from M5. It does **not** establish how much of the complete serving loss comes from neural compute,target physical cohorts,adaptive admission or device state. The current original M4 dataset also contains changed proposal work,while plain A/A variation exists with no neural drafter at all. Existing source calibration uses empty-context holds plus separate context-projection grids and runtime repricing; whether those prices adequately predict serving needs actual hold/target attribution,not arbitrary margin or precision changes.

There is no current controlled MLX-LM comparison or measured SIMD/bandwidth bottleneck here. No claim that this PR beats MLX-LM is made. The earlier expert-operator diagnosis keeps its historical source label and cannot become current-run proof.

## Complete evidence and privacy

- `analysis.json`,`study-observed.json`,`clock-observed.json`,`pipeline-observed.json`,`controller.log` and every launch's`cells.jsonl`/`events.jsonl` preserve the original bytes.
- Every launch's`requests.jsonl.gz` losslessly preserves its entire original raw request file,including both fixed warmups. `clock-samples.jsonl.gz` preserves **all15,466 complete parsed GPU/thermal records**,including fully idle zero-frequency records; none was filtered to improve the result.
- `inputs.json` records original/compressed SHA256s,sizes,per-phase request counts and exact oracle checks. Every gzip decode was SHA-checked against its source. `SHA256SUMS` binds the entire public packet.
- Full original system powermetrics pliststream remains private as labelled by the collector,complete and SHA-verified. The original private native logs are retained with per-launch hashes; no personal local paths,names,addresses or credentials are published.
- `work-clock-diagnosis.json` is explicitly posthoc descriptive analysis;`completed-original-calibration-and-dispatch.json` includes selected safe startup-log lines and whole original log hashes. Neither modifies the predeclared decision.

## Readiness consequence

**M4 is not performance-cleared.** The original matrix fails the owner's observed budget;the controlled study is inconclusive under its own published method. [M5's complete inside-budget result](../../qualification/m5/README.md) does not clear M4. The literal maintainer zero-tolerance FAILs remain unchanged.

The pushed51d code has11 successful PR-attached checks. Separately,a read-only exact-ref check proved two conflicts after upstream main moved;the new central drafter admission must preserve both LFM2 and Llama. [Current code/CI/integration evidence and actual fixes](../../review-readiness/README.md) are source-labelled. No further main-chasing merge/rebase,force push,PR merge,auto-merge,extra PR,repeat re-review request or policy waiver was performed.
