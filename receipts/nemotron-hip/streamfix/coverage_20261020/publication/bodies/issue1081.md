# HIP/Nemotron review tracker

## Review status

All 22 PRs are Ready for review. PR1082 was already Ready; the remaining 21 transitions are verified separately before this tracker is published. The linked implementation, evidence and limitations are available for review. Landing decisions remain open; no merge or full CONTRIBUTING acceptance is claimed.

## Current source chain

- [#1082](https://github.com/ashhart/TensorFold/pull/1082) — HIP kernel-contract runner infrastructure. Head `826a0635634dee3793a68e473064d17d5c139945`; Ready for review, unchanged by this body update.

All branches still target main. Dependencies below describe incremental review scope; they do not waive the one-change rule or replace the eventual ordered landing/base arrangement.

- [#1083](https://github.com/ashhart/TensorFold/pull/1083) — fix(hip): preserve the MLX stream nonfinite classes. Head `096e0aeacf68d9250dd4dc33d0741b6ed54ee7c8`; depends on #1082 at `826a0635634dee3793a68e473064d17d5c139945`. [Incremental 11-file diff](https://github.com/BlivionIaG/TensorFold/compare/826a0635634dee3793a68e473064d17d5c139945...096e0aeacf68d9250dd4dc33d0741b6ed54ee7c8). Ready for review; landing/dependency decisions remain open.

- [#1084](https://github.com/ashhart/TensorFold/pull/1084) — fix(qwen3_5): no prompt cache step goes to the other ranks once the group stopped. Head `1d1bcdd35a482271448e0ff8a19eb9013be1e503`; depends on #1083 at `096e0aeacf68d9250dd4dc33d0741b6ed54ee7c8`. [Incremental 1-file diff](https://github.com/BlivionIaG/TensorFold/compare/096e0aeacf68d9250dd4dc33d0741b6ed54ee7c8...1d1bcdd35a482271448e0ff8a19eb9013be1e503). Ready for review; landing/dependency decisions remain open.

- [#1085](https://github.com/ashhart/TensorFold/pull/1085) — refactor(hip): shared HIP serving pieces and lane cost timing in hip_serve; Qwen3.5 on them. Head `63025caf8555fe474d6d066c966c8f1e9b9a7352`; depends on #1084 at `1d1bcdd35a482271448e0ff8a19eb9013be1e503`. [Incremental 16-file diff](https://github.com/BlivionIaG/TensorFold/compare/1d1bcdd35a482271448e0ff8a19eb9013be1e503...63025caf8555fe474d6d066c966c8f1e9b9a7352). Ready for review; landing/dependency decisions remain open.

- [#1086](https://github.com/ashhart/TensorFold/pull/1086) — fix(hip): refuse --kv-cache fp8 by name until HIP has fp8 caches. Head `9c655701055674cb795feec0f59f5adce6bc74dc`; depends on #1085 at `63025caf8555fe474d6d066c966c8f1e9b9a7352`. [Incremental 1-file diff](https://github.com/BlivionIaG/TensorFold/compare/63025caf8555fe474d6d066c966c8f1e9b9a7352...9c655701055674cb795feec0f59f5adce6bc74dc). Ready for review; landing/dependency decisions remain open.

- [#1087](https://github.com/ashhart/TensorFold/pull/1087) — feat(hip): a HIP family can refuse --slide and --mtp-confidence by name before it loads. Head `e4fed24d0d5a0e9a518fb2965fd876afa61f700f`; depends on #1086 at `9c655701055674cb795feec0f59f5adce6bc74dc`. [Incremental 1-file diff](https://github.com/BlivionIaG/TensorFold/compare/9c655701055674cb795feec0f59f5adce6bc74dc...e4fed24d0d5a0e9a518fb2965fd876afa61f700f). Ready for review; landing/dependency decisions remain open.

- [#1088](https://github.com/ashhart/TensorFold/pull/1088) — feat(hip): Nemotron Mamba-2 mixer kernels, launches, ops and tests. Head `1f19dbac8ea123c20c39afdd3e62afbbf9b12797`; depends on #1087 at `e4fed24d0d5a0e9a518fb2965fd876afa61f700f`. [Incremental 14-file diff](https://github.com/BlivionIaG/TensorFold/compare/e4fed24d0d5a0e9a518fb2965fd876afa61f700f...1f19dbac8ea123c20c39afdd3e62afbbf9b12797). Ready for review; landing/dependency decisions remain open.

- [#1089](https://github.com/ashhart/TensorFold/pull/1089) — refactor(hip): the routed MoE's tile choice is public, its rule tested. Head `f3f6c3f2a925b68ff18788063cb8234ea7924f11`; depends on #1088 at `1f19dbac8ea123c20c39afdd3e62afbbf9b12797`. [Incremental 1-file diff](https://github.com/BlivionIaG/TensorFold/compare/1f19dbac8ea123c20c39afdd3e62afbbf9b12797...f3f6c3f2a925b68ff18788063cb8234ea7924f11). Ready for review; landing/dependency decisions remain open.

- [#1090](https://github.com/ashhart/TensorFold/pull/1090) — feat(hip): Nemotron-H MoE layer on the shared expert plan and products, its pick ranked on 512 threads. Head `017b0953f3517cdda3821c9fbbc1abfb0965227b`; depends on #1089 at `f3f6c3f2a925b68ff18788063cb8234ea7924f11`. [Incremental 14-file diff](https://github.com/BlivionIaG/TensorFold/compare/f3f6c3f2a925b68ff18788063cb8234ea7924f11...017b0953f3517cdda3821c9fbbc1abfb0965227b). Ready for review; landing/dependency decisions remain open.

- [#1091](https://github.com/ashhart/TensorFold/pull/1091) — perf(hip): the planned values walk splits a KV head's query heads while its grid is small. Head `76a1f34f05f14f7de9a321b2810f9f77d41fe7ad`; depends on #1090 at `017b0953f3517cdda3821c9fbbc1abfb0965227b`. [Incremental 7-file diff](https://github.com/BlivionIaG/TensorFold/compare/017b0953f3517cdda3821c9fbbc1abfb0965227b...76a1f34f05f14f7de9a321b2810f9f77d41fe7ad). Ready for review; landing/dependency decisions remain open.

- [#1092](https://github.com/ashhart/TensorFold/pull/1092) — feat(nemotron): Nemotron-H on the native HIP engine, one GPU: admission, loader, forward and lane backend. Head `89bc8409933617057a8ad9c5e379b4603604d557`; depends on #1091 at `76a1f34f05f14f7de9a321b2810f9f77d41fe7ad`. [Incremental 23-file diff](https://github.com/BlivionIaG/TensorFold/compare/76a1f34f05f14f7de9a321b2810f9f77d41fe7ad...89bc8409933617057a8ad9c5e379b4603604d557). Ready for review; landing/dependency decisions remain open.

- [#1093](https://github.com/ashhart/TensorFold/pull/1093) — feat(nemotron): tensor parallelism on HIP: followers run rank 0's steps, the cache and drafts agreed across ranks. Head `71163c15ee202bea0d41e8b64f26f8c4c059a042`; depends on #1092 at `89bc8409933617057a8ad9c5e379b4603604d557`. [Incremental 11-file diff](https://github.com/BlivionIaG/TensorFold/compare/89bc8409933617057a8ad9c5e379b4603604d557...71163c15ee202bea0d41e8b64f26f8c4c059a042). Ready for review; landing/dependency decisions remain open.

- [#1094](https://github.com/ashhart/TensorFold/pull/1094) — feat(nemotron): lane rounds, lane costs and graph replay for the Nemotron HIP engine, and the decode-path test. Head `56916dd0db0df19eb1c4ab370c97f11bdfa7b96f`; depends on #1093 at `71163c15ee202bea0d41e8b64f26f8c4c059a042`. [Incremental 16-file diff](https://github.com/BlivionIaG/TensorFold/compare/71163c15ee202bea0d41e8b64f26f8c4c059a042...56916dd0db0df19eb1c4ab370c97f11bdfa7b96f). Ready for review; landing/dependency decisions remain open.

- [#1095](https://github.com/ashhart/TensorFold/pull/1095) — perf(nemotron): HIP prompts in 2048-row chunks, so the routed experts get enough rows a tile. Head `4322276b240fb4a16c62767e24ab08dcd3c5e4f1`; depends on #1094 at `56916dd0db0df19eb1c4ab370c97f11bdfa7b96f`. [Incremental 2-file diff](https://github.com/BlivionIaG/TensorFold/compare/56916dd0db0df19eb1c4ab370c97f11bdfa7b96f...4322276b240fb4a16c62767e24ab08dcd3c5e4f1). Ready for review; landing/dependency decisions remain open.

- [#1096](https://github.com/ashhart/TensorFold/pull/1096) — perf(hip): the residual add and the RMS norm in one launch, the Nemotron MoE sum among its inputs. Head `29409f2d7c58326351da64599b051ccd8f98b867`; depends on #1095 at `4322276b240fb4a16c62767e24ab08dcd3c5e4f1`. [Incremental 12-file diff](https://github.com/BlivionIaG/TensorFold/compare/4322276b240fb4a16c62767e24ab08dcd3c5e4f1...29409f2d7c58326351da64599b051ccd8f98b867). Ready for review; landing/dependency decisions remain open.

- [#1097](https://github.com/ashhart/TensorFold/pull/1097) — perf(hip): a Mamba-2 window in three launches, each row's state kept in its snapshot slot. Head `dfcbfb1d4ec601803f116d454b3cdead6a6818fd`; depends on #1096 at `29409f2d7c58326351da64599b051ccd8f98b867`. [Incremental 10-file diff](https://github.com/BlivionIaG/TensorFold/compare/29409f2d7c58326351da64599b051ccd8f98b867...dfcbfb1d4ec601803f116d454b3cdead6a6818fd). Ready for review; landing/dependency decisions remain open.

- [#1098](https://github.com/ashhart/TensorFold/pull/1098) — perf(hip): Nemotron's relu^2 as the stream tile's epilogue for one-row MoE layers. Head `51d68203c4c2e3450345e31087f8dad711c6d479`; depends on #1097 at `dfcbfb1d4ec601803f116d454b3cdead6a6818fd`. [Incremental 13-file diff](https://github.com/BlivionIaG/TensorFold/compare/dfcbfb1d4ec601803f116d454b3cdead6a6818fd...51d68203c4c2e3450345e31087f8dad711c6d479). Ready for review; landing/dependency decisions remain open.

- [#1099](https://github.com/ashhart/TensorFold/pull/1099) — perf(hip): the planned decode walk reads q and writes its rows in the activation type. Head `edc5ba65c29136897e8d8e6e7ce1ab4b1c8b823b`; depends on #1098 at `51d68203c4c2e3450345e31087f8dad711c6d479`. [Incremental 9-file diff](https://github.com/BlivionIaG/TensorFold/compare/51d68203c4c2e3450345e31087f8dad711c6d479...edc5ba65c29136897e8d8e6e7ce1ab4b1c8b823b). Ready for review; landing/dependency decisions remain open.

- [#1100](https://github.com/ashhart/TensorFold/pull/1100) — perf(hip): the argmax of a wide logits row in parts across the card. Head `8859ff69a3d2023ae19607189b2981987a86fcef`; depends on #1099 at `edc5ba65c29136897e8d8e6e7ce1ab4b1c8b823b`. [Incremental 9-file diff](https://github.com/BlivionIaG/TensorFold/compare/edc5ba65c29136897e8d8e6e7ce1ab4b1c8b823b...8859ff69a3d2023ae19607189b2981987a86fcef). Ready for review; landing/dependency decisions remain open.

- [#1101](https://github.com/ashhart/TensorFold/pull/1101) — perf(nemotron): fewer launches a HIP decode token: q, k and v grouped, the residual and next norm merged, Mamba windows in three launches. Head `56511fa7d7d7ce69adeb3666dafe962b4dc405c4`; depends on #1100 at `8859ff69a3d2023ae19607189b2981987a86fcef`. [Incremental 6-file diff](https://github.com/BlivionIaG/TensorFold/compare/8859ff69a3d2023ae19607189b2981987a86fcef...56511fa7d7d7ce69adeb3666dafe962b4dc405c4). Ready for review; landing/dependency decisions remain open.

- [#1102](https://github.com/ashhart/TensorFold/pull/1102) — perf(hip): the planned decode walk in one launch a query head for spans up to 4096 keys. Head `75db6d0150ef92028f375aa94c35dc3a9e5b8e15`; depends on #1101 at `56511fa7d7d7ce69adeb3666dafe962b4dc405c4`. [Incremental 8-file diff](https://github.com/BlivionIaG/TensorFold/compare/56511fa7d7d7ce69adeb3666dafe962b4dc405c4...75db6d0150ef92028f375aa94c35dc3a9e5b8e15). Ready for review; landing/dependency decisions remain open.

- [#1103](https://github.com/ashhart/TensorFold/pull/1103) — perf(hip): one row's routed fc1 and shared up in one launch, both with the relu^2 epilogue. Head `a44e7af98183721e58f2fb4d7e96ae874dfecd83`; depends on #1102 at `75db6d0150ef92028f375aa94c35dc3a9e5b8e15`. [Incremental 13-file diff](https://github.com/BlivionIaG/TensorFold/compare/75db6d0150ef92028f375aa94c35dc3a9e5b8e15...a44e7af98183721e58f2fb4d7e96ae874dfecd83). Ready for review; landing/dependency decisions remain open.

## Completed data / current checks

[Immutable current 22-head CI audit](https://github.com/BlivionIaG/TensorFold/tree/9413cadef9fc387ff22c93ff60e6daa89682e603/receipts/nemotron-hip/streamfix/coverage_20261020/current_ci_all22) binds all 110 completed checks (87 success/23 failure), tested merges, job URLs and baseline comparisons.

Native requested measurements are complete:48 O/N/O cells864requests648 measured, all 16shape/C1/C8/mode/rig cases,1warmup+3 measured batches. Actual baseline drift and long intervening gaps remain reported; arithmetic bookend means are descriptive, not confidence/drift models or automatic thresholds. Canonical AMD gpu.lock was held. No duplicate native benchmark is pending. Measured source 729/tree 60a8 retains its source label; newfirst PR at 826/top PR at a44 only amend validation input/guard files and have actual5/49-kernel live gates,0 failures, 0 known findings and 0 limit problems,5 numeric controls red+bothmetadata controlsred/current on4chips. First826 fullLinux170steps2472pass9 skip; golden306/319equal13known0 unexpected. No intermediate own-head or new model/performance execution is invented.

Latest completed source-bound CI: all 22Linux/fmt/mobileSUCCESS,21desktopSUCCESS;1085desktop reports the guest-VM EPERM assertion (820/821passed); its tested-merge whole app/workflow inputs equal main and historical PR1092 has the same inputs/symptom. This is baseline-equivalent app evidence, not a proven syscall cause or HIP bug. All22Windows failures match the 52provedbaseline compiler diagnostics; generic run-test-w stderr output is not a runtime-test failure. Current checks remain linked from each PR. Baseline accounting is not a source repair or blanket CI waiver.

## Actual benchmark values

Completion E2E throughput mean of3 measured batches, tokens/s: OLDfirst → candidate → OLDlast; full precision comparisons retained, display6decimals.

- W7800tp2 off code_1024_512 C1: 132.733206 → 129.174842 → 131.410698; baseline drift-0.996366%; candidate/bookend-2.193585%.

- W7800tp2 off code_1024_512 C8: 331.323856 → 327.217216 → 329.772364; baseline drift-0.468271%; candidate/bookend-1.007688%.

- W7800tp2 off prose_16384_1024 C1: 82.185503 → 80.979738 → 82.006843; baseline drift-0.217386%; candidate/bookend-1.359911%.

- W7800tp2 off prose_16384_1024 C8: 135.336859 → 134.654501 → 135.420669; baseline drift+0.061926%; candidate/bookend-0.534990%.

- V620tp1 off code_1024_512 C1: 115.286461 → 113.378325 → 114.250402; baseline drift-0.898683%; candidate/bookend-1.211227%.

- V620tp1 off code_1024_512 C8: 202.387526 → 199.404021 → 201.817382; baseline drift-0.281709%; candidate/bookend-1.335181%.

- V620tp1 off prose_16384_1024 C1: 57.101946 → 56.865976 → 56.959989; baseline drift-0.248601%; candidate/bookend-0.289302%.

- V620tp1 off prose_16384_1024 C8: 75.426964 → 75.050107 → 75.436788; baseline drift+0.013024%; candidate/bookend-0.506111%.

- W7800tp2 on code_1024_512 C1: 222.653829 → 209.928699 → 222.373145; baseline drift-0.126063%; candidate/bookend-5.655741%.

- W7800tp2 on code_1024_512 C8: 402.256466 → 383.122464 → 399.409236; baseline drift-0.707815%; candidate/bookend-4.418397%.

- W7800tp2 on prose_16384_1024 C1: 108.276848 → 106.810361 → 108.918744; baseline drift+0.592829%; candidate/bookend-1.645922%.

- W7800tp2 on prose_16384_1024 C8: 138.221485 → 136.518783 → 138.851070; baseline drift+0.455490%; candidate/bookend-1.456293%.

- V620tp1 on code_1024_512 C1: 177.593603 → 171.544560 → 177.108265; baseline drift-0.273286%; candidate/bookend-3.273946%.

- V620tp1 on code_1024_512 C8: 274.025765 → 270.110048 → 273.776880; baseline drift-0.090826%; candidate/bookend-1.384175%.

- V620tp1 on prose_16384_1024 C1: 68.501116 → 67.768862 → 68.469700; baseline drift-0.045863%; candidate/bookend-1.046275%.

- V620tp1 on prose_16384_1024 C8: 79.704922 → 79.229267 → 79.684522; baseline drift-0.025594%; candidate/bookend-0.584048%.

Every request,3batch values, per-hold means/medians, TTFT/server prefill/decode/deliverymean TPOT/chunk gaps and aggregate denominators are retained. Serverprefill elapsed is not pure GPU compute; mean TPOT/chunk gaps are not a true per-token ITL distribution. Requester accepts measured performance, including recorded regressions; no upstream acceptance waiver follows. Historical32 selected pairs and originalinterruptedattempts remain immutable and are not pooled into this48cell selection.

## Fidelity, state and memory

Five HF/FP32 differences remain unwaived and reported with historical rounded reference margins/exact native margins:2 exact native max ties[83666,117316] with reference117316 and3 unique native winners. Historical rounded HF margins and exact native margins are retained; unrounded HF values, reference runner-up IDs and original runtime versions are unavailable. No tie waiver is claimed. 1131 retainednative teacher rows (754 primary+377 supplementary duplicate-rank rows) remain old/current finitebit/top2/margin unchanged. Executed sampled FP64 fixtures and actual conditional reduction source map are credited.

Bothrig enginek−2restore/last2replay and4streams10 rows equality, V6206coldcontexts through262080/resume-fresh/cancel-drain-reuse equality, and full C1/C8matrix cover their actual state-validity scope. Rollback/replay and cancellation/reuse checks passed. VmHWM records peak RSS; PSS is sampled; full pinned-allocation peaks were not reported. Family measurements are distinguished from infrastructure and kernel-only scopes.

## Explicit user scope and landing decisions

No vLLM/llama.cpp server/controller/model inference was launched. The requester omitted external-engine performance work; no such job is pending and no inference equivalence is inferred from CPU-only derivedFP16 conversion/row checks. Applicable upstream comparator wording remains unwaived before landing; it is not a runner-infrastructure gate.

The21cumulative-main diffs cannot be made one-change-compliant by prose or incrementalcompare links alone. Predecessor landing/base alignment or actual accepted stack arrangement must resolve this before landing. Newfamily/arithmetic gate discussion and5 findings remain disclosed where applicable; issue1081 existing is not proof of agreement. This tracker requests review; the landing decisions above remain open.

## Immutable complete history and primary evidence

[Exact prior tracker with every historical number/link](https://github.com/BlivionIaG/TensorFold/blob/9d1744b5dd2e254a69343454f888099dc3400dc8/receipts/nemotron-hip/streamfix/coverage_20261020/publication/bodies/issue1081.md). Each PR also links its byte-identical priorbody archive; stale present-tense checklists are superseded by these current statements.

- https://github.com/BlivionIaG/TensorFold/blob/0beefc3abdff231804f86ab7369768634745366a/receipts/nemotron-hip/streamfix/coverage_20261020/reference_actual_rows/actual_row_sample_result.json
- https://github.com/BlivionIaG/TensorFold/blob/514e2c8a4131bd84ce09f1a964c9fa703ef07c79/receipts/nemotron-hip/streamfix/coverage_20261020/fp64_conditional_source_review.json
- https://github.com/BlivionIaG/TensorFold/blob/514e2c8a4131bd84ce09f1a964c9fa703ef07c79/receipts/nemotron-hip/streamfix/coverage_20261020/lifecycle729_independent_audit.json
- https://github.com/BlivionIaG/TensorFold/blob/514e2c8a4131bd84ce09f1a964c9fa703ef07c79/receipts/nemotron-hip/streamfix/coverage_20261020/lifecycle729_raw/summary.json
- https://github.com/BlivionIaG/TensorFold/blob/514e2c8a4131bd84ce09f1a964c9fa703ef07c79/receipts/nemotron-hip/streamfix/coverage_20261020/oracle_subset729/result.json
- https://github.com/BlivionIaG/TensorFold/blob/514e2c8a4131bd84ce09f1a964c9fa703ef07c79/receipts/nemotron-hip/streamfix/coverage_20261020/teacherforced729/result.json
- https://github.com/BlivionIaG/TensorFold/blob/b5d4b36ef5e757d6c25879db4c10306678876b45/receipts/nemotron-hip/streamfix/coverage_20261020/reference_compatibility/header_inventory.json
- https://github.com/BlivionIaG/TensorFold/blob/b5d4b36ef5e757d6c25879db4c10306678876b45/receipts/nemotron-hip/streamfix/coverage_20261020/reference_compatibility/previous_release_scope.json
- https://github.com/BlivionIaG/TensorFold/blob/b5d4b36ef5e757d6c25879db4c10306678876b45/receipts/nemotron-hip/streamfix/coverage_20261020/reference_compatibility/source_metadata_binding.json
- https://github.com/BlivionIaG/TensorFold/blob/b5d4b36ef5e757d6c25879db4c10306678876b45/receipts/nemotron-hip/streamfix/coverage_20261020/reference_compatibility/vllm_mlx_loader_source_audit.json
- https://github.com/BlivionIaG/TensorFold/blob/dc595eeb198c66d3327d7f0ce4a3eb6c9a807ec5/zig/tests/hip/contract_limits.zig#L43
- https://github.com/BlivionIaG/TensorFold/blob/df70b95290cda79e4550468544a27b6ce70908f9/receipts/nemotron-hip/streamfix/coverage_20261020/continuation_all4/audits/final_V_ON/selected_all4_reduction.json
- https://github.com/BlivionIaG/TensorFold/commit/3f991646915e24bb1c7c638ec118e27880d5a031
- https://github.com/BlivionIaG/TensorFold/commit/42529bfab171c2f30d7ce3641b78ca3b0ff821fc
- https://github.com/BlivionIaG/TensorFold/commit/519c395da4a913afb2d124ab3f8a8c2c4a2ca1c6
- https://github.com/BlivionIaG/TensorFold/commit/52840409d2e30ea7e55d9d79c10375cb2ffce6c5
- https://github.com/BlivionIaG/TensorFold/commit/5b12f74a27326797a540d094387e7ffa472b55ed
- https://github.com/BlivionIaG/TensorFold/commit/60a8da5243ef313fcab23bdb0d85c790a5cda935
- https://github.com/BlivionIaG/TensorFold/commit/6af43c9db14516a85fbe22351eb0d3e485bd41bb
- https://github.com/BlivionIaG/TensorFold/commit/6e73bf059c5646e683d680ab78ed5326aa9fba87
- https://github.com/BlivionIaG/TensorFold/commit/736e0ac1ed64ce2c1b06edd6300c1119c17e2b4e
- https://github.com/BlivionIaG/TensorFold/commit/74b506e50265ccef559ce66d8d1cd55db55655ae
- https://github.com/BlivionIaG/TensorFold/commit/754786d613dde5539d3a0c534007a1a9733cfaf8
- https://github.com/BlivionIaG/TensorFold/commit/7bf5f195fb06c0651a1a5d74277f24cc70da591c
- https://github.com/BlivionIaG/TensorFold/commit/8a1c6ae90a9212c481607eb6633697543e6cb3ff
- https://github.com/BlivionIaG/TensorFold/commit/9b56916fa71d40f9dcd924956c09a3c5f7e18019
- https://github.com/BlivionIaG/TensorFold/commit/9c206c22bf386348f2f1eb473a31ef8370163b5e
- https://github.com/BlivionIaG/TensorFold/commit/9f7d265d802cf394b4f60a142aa9d5214cdbfdbf
- https://github.com/BlivionIaG/TensorFold/commit/cf2ad661a2b1cc2fd00e9899aa2501ab2a1a58e2
- https://github.com/BlivionIaG/TensorFold/commit/d6134300b3fed9baa29842e20d28ce738fd77574
- https://github.com/BlivionIaG/TensorFold/commit/dc595eeb198c66d3327d7f0ce4a3eb6c9a807ec5
- https://github.com/BlivionIaG/TensorFold/commit/e22da9dab3e8d70d5dbe150be39ed3794d6bffda
- https://github.com/BlivionIaG/TensorFold/commit/f8cd5a4d72805fabe7350746ce9ba91274b54d29
- https://github.com/BlivionIaG/TensorFold/commit/fd5b50111c0e0dd394fbac022ce8e6ff1db3d257
- https://github.com/BlivionIaG/TensorFold/tree/0beefc3abdff231804f86ab7369768634745366a/receipts/nemotron-hip/streamfix/coverage_20261020/cold_http729_checkpoint
- https://github.com/BlivionIaG/TensorFold/tree/19ef422352a55adcca49b894dc6d52d2b6ab3678/receipts/nemotron-hip/streamfix/coverage_20261020/ci_docs_control
- https://github.com/BlivionIaG/TensorFold/tree/606ddd61b274eccc38aa37b99dbc1850d5165b8f/receipts/nemotron-hip/streamfix/coverage_20261020/line81_complete
- https://github.com/BlivionIaG/TensorFold/tree/8265f09ad44742c126b3e8690ef3138adab055e7/receipts/nemotron-hip/streamfix/coverage_20261020/saved_evidence_final
- https://github.com/BlivionIaG/TensorFold/tree/a4faa87a0887f1c1ee43739cfb885f9d8eea38b0/receipts/nemotron-hip/streamfix/coverage_20261020/windows_runtime_classification
- https://github.com/BlivionIaG/TensorFold/tree/a758bf7da35574a0b40cca642560ec73b22457b0/receipts/nemotron-hip/streamfix/contributing_refresh_20261020
- https://github.com/BlivionIaG/TensorFold/tree/a758bf7da35574a0b40cca642560ec73b22457b0/receipts/nemotron-hip/streamfix/contributing_refresh_20261020/historical_bodies/issue1081.md
- https://github.com/BlivionIaG/TensorFold/tree/b5d4b36ef5e757d6c25879db4c10306678876b45/receipts/nemotron-hip/streamfix/coverage_20261020/recovered_matrix729
- https://github.com/BlivionIaG/TensorFold/tree/bc6f6e7ab1b25c75ff6ca2e1917f3d7224710557/receipts/nemotron-hip/streamfix/coverage_20261020/hip_control_fullcpu826
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/decode_recovery729
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/decode_serial729
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/followup729
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/inline729
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/inline_stack_review
- https://github.com/BlivionIaG/TensorFold/tree/c12660143d6de8bb9eaf9491ffde8d3b209229fa/receipts/nemotron-hip/streamfix/ownheads729
- https://github.com/BlivionIaG/TensorFold/tree/c962f63f9a5edcb5f702af581102afa99a485a65/receipts/nemotron-hip/streamfix/coverage_20261020/cold_http729_final
- https://github.com/BlivionIaG/TensorFold/tree/df70b95290cda79e4550468544a27b6ce70908f9/receipts/nemotron-hip/streamfix/coverage_20261020/continuation_all4
- https://github.com/BlivionIaG/TensorFold/tree/eb640c5c60d37ee099238324e7151a45b57889cd/receipts/nemotron-hip/streamfix/coverage_20261020/hip_control_qualification
- https://github.com/BlivionIaG/TensorFold/tree/f8f4dfa2293e1fde7cada4045d5e072ab0daf58e/receipts/nemotron-hip/streamfix/coverage_20261020/matrix_transport_checkpoint
- https://github.com/BlivionIaG/TensorFold/tree/f8f4dfa2293e1fde7cada4045d5e072ab0daf58e/receipts/nemotron-hip/streamfix/coverage_20261020/windows_baseline
- https://github.com/ashhart/TensorFold/actions/runs/39182366686/job/117837632862
- https://github.com/ashhart/TensorFold/actions/runs/39487050150/job/118850059735
- https://github.com/ashhart/TensorFold/blob/d1018b318fb51981c4a18f12e40269a4d3ee9713/CONTRIBUTING.md
- https://github.com/ashhart/TensorFold/issues/491#issuecomment-6096495227
- https://github.com/ashhart/TensorFold/pull/1082
- https://github.com/ashhart/TensorFold/pull/1083
- https://github.com/ashhart/TensorFold/pull/1084
- https://github.com/ashhart/TensorFold/pull/1085
- https://github.com/ashhart/TensorFold/pull/1086
- https://github.com/ashhart/TensorFold/pull/1087
- https://github.com/ashhart/TensorFold/pull/1088
- https://github.com/ashhart/TensorFold/pull/1089
- https://github.com/ashhart/TensorFold/pull/1090
- https://github.com/ashhart/TensorFold/pull/1091
- https://github.com/ashhart/TensorFold/pull/1092
- https://github.com/ashhart/TensorFold/pull/1093
- https://github.com/ashhart/TensorFold/pull/1094
- https://github.com/ashhart/TensorFold/pull/1095
- https://github.com/ashhart/TensorFold/pull/1096
- https://github.com/ashhart/TensorFold/pull/1097
- https://github.com/ashhart/TensorFold/pull/1098
- https://github.com/ashhart/TensorFold/pull/1099
- https://github.com/ashhart/TensorFold/pull/1100
- https://github.com/ashhart/TensorFold/pull/1101
- https://github.com/ashhart/TensorFold/pull/1102
- https://github.com/ashhart/TensorFold/pull/1103
- https://github.com/ashhart/TensorFold/pull/490#issuecomment-6096494868
