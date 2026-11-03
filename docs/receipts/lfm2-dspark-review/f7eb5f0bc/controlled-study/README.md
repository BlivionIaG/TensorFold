# Controlled M4 comparison — exact integrated source

This is the binding and complete collection/decision method, published before controlled performance collection on `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce`. It is not a performance pass. The previous source’s partial attempt is preserved, not relabelled or silently restarted.

## Source and exercised safety

- PR #1394 source: `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce`; normal integration merge `c2a4dba92436d8db672879d836d4e4fce85c8620`; pinned upstream `2c8422a0c49d04a1867cffec081a1a42b65c857f`.
- Committed source archive SHA256: `3ca340ec39493b7d32ba23902e2087111e54def1184c1665331b4ada1cbe7b12`. Runtime object IDs are in `study-binding.json`.
- Actual production M4 native SHA256: `a7be9659f48c7cdb6a19a63a013bc850c679343a9e18e5dfa81ef220ff702fc0`. The executable is an immutable read-only copy, checked before and after every study launch. No private instrumentation.
- The source refactor preserves LFM2 inference arithmetic, BF16 activations, both head formats and the behavioral assertions. The separate original tap-only oracle remains unchanged, avoiding the maintainer’s independent #2055 rounding work.
- Actual local host invocation completed675/675 steps;10405/10473 tests passed,68 skipped. Actual macOS developer-tool compilation completed226/226 steps. The unchanged lean comparison reports424 existing findings versus424 on pinned main,0 new.
- Exact-source M4 guard build completed82/82 steps;767/775 tests passed,8 skipped. The real LFM2 GPU contract completed16 kernels,16 passing,24 planted controls red,0 missed and0 limit problems. Both BF16 and8-bit modes matched9/9 FP32-reference **head picks**. That reference consumes BF16 target features; it is not an FP32 target server. Worst confidence differences were0.00085 and0.00152 respectively. Full guard details/hashes are retained in `m4-guard-proof.json`.
- Original full144-cell qualification is collected separately with the unchanged frozen seven-file driver. Its literal zero-tolerance verdict is never replaced by this analysis.

## Fixed method and decisions

`protocol.json` is byte-identical to the method published before source176 collection. Six fixed rounds; four sequential launches per round: plain-before, the two head modes (alternating order by round), plain-after. Every original code/chat,temperature0/1,width1/4/8/16 cell remains. Each launch performs two complete16-cell warmup sweeps and one complete measured sweep:24 launches,384 measured cells and768 warmup cells. Every request generates512 tokens; parallel16,context4096,cache0; plain cheap copies remain enabled.

Every output must match the original AFE same-prompt/temperature/seed token oracle. The original pressure/process checks, persistent lock/inode, model/driver hashes and clean native exits remain mandatory. No sample-dependent stopping, slower-run discard, width cutoff, precision shortcut, power override or successful-attempt selection.

For each of16 cells in each round, use the geometric mean of its bracketing plain measurements as reference. Each candidate requires both median paired loss<1% and a one-sided95% Student-t lower bound on the mean log ratio greater than `log(0.99)` (6 rounds,df5,critical2.015048373333). Independently, each cell’s90% two-sided Student-t interval for log(plain-after/plain-before) must lie wholly inside `[log(0.99),log(1.01)]`.

All16 A/A validity checks and all32 candidate decisions must pass. Failed A/A means **INCONCLUSIVE**, not a performance pass or proof of a code defect. Valid A/A with a failed candidate bound means **NOT_CLEARED**. Independent launch-round log ratios and the normal/log-ratio model are assumptions, not an empirically universal error floor. The owner’s observed<1% budget and the maintainer’s every-cell>=plain policy remain distinct from statistical uncertainty.

## Clock acquisition correction, before collection

The idle preflight exposed a real collector boundary: powermetrics emitted10 complete records with `freq_hz=0.0,idle_ratio=1.0`; the old collector rejected them. It now retains that actual idle value instead of treating it as corruption. Active-zero,negative,nonfinite and missing-frequency records still fail. All11 original records replay successfully; the new actual idle capture has2 complete records,0 parse errors,controller/sampler exit0 and clean reader EOF. The failed preflight, raw-input hash and replay evidence are retained. No model was loaded in either preflight.

This corrects acquisition, not the performance threshold or the sample count. The new wrapper hash is bound explicitly. Original source798/176 clock evidence keeps its old wrapper/source labels. Global GPU frequency/thermal/idle records are not SIMD occupancy, memory bandwidth or proof of an MLX bottleneck; raw frequency units are preserved without assuming SI Hz.

## Superseded source176 attempt

`superseded-attempt-observed.json` is the complete original aggregate receipt, byte-identical to the saved file. It contains one complete plain launch (32 warmup+16 measured cells) and six BF16 warmup cells before the source cutover. It produced no candidate performance decision. The stop was for actual upstream conflicts/style defects, before inspecting timing results—not a retry for a better sign. Owned native processes and the sampler exited cleanly; controller interruption was-2. Original request/native logs remain privately retained because public receipts must not expose local paths.

## Files and existing evidence

`SHA256SUMS` covers every file in this packet. Collection/analysis code, exact binding, immutable-source IDs, original protocol, idle-boundary evidence and superseded-source receipt are included. Collection and analysis abort on incomplete data; no unmeasured result is supplied here.

- [Frozen two-device AFE qualification and all64 comparisons](https://github.com/BlivionIaG/TensorFold/blob/b9fff9c7acb48959960de6a5e789f8af7ead152d/docs/receipts/lfm2-dspark-review/afe8bda62/qualification/README.md)
- [Original completed clock/reference-drift diagnosis and its limits](https://github.com/BlivionIaG/TensorFold/blob/b9fff9c7acb48959960de6a5e789f8af7ead152d/docs/receipts/lfm2-dspark-review/79837c4e2/clock-environment/README.md)
