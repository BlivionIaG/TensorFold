# PR1082 independent source review

Exactsource dc595eeb198c66d3327d7f0ce4a3eb6c9a807ec5 versus testedbase76aafcf2. Scope13files: existingHIPcontract wiring, metadata parser/resource registry/row fixtures/plantedcontrols/docs; no new production family/backend or arithmetic kernel. Shared HIP object compilation is factored with the same flags. The fixture calls existing RMS kernels, uses independent common float64 row reference and guarded buffers, and preserves all five planted numerical controls. Host tests exercise parser/resource functions; no unrelated model/performance requirement is inferred for this infrastructure-only slice.

## Actionable P2: per-chip planted limit control can turn green undetected

`zig/tests/hip/contract_limits.zig:43–45`: table() accepts `records.check(control)>0` across allfourchips, then prints “red on every chip”. `objects():124,142` similarly accumulates total violations and rejects only when the total is zero. A control missing its intended violation on onechip stillpasses if anyotherchip remainsred. `hip_records.zig:103` unit control check has the same aggregate weakness.

ConcreteCPUcounterexample using exact shipped limits/registry modules: copy the boundedcontrol and change only gfx1030 usage.bound256→512 (the recorded launch is512threads). gfx1030 nowhas0violations; gfx1100/1151/1201eachhave1. Actualshipped records.check returns3; both shipped aggregateacceptance predicates succeed. A per-chip all-red predicate rejects. This demonstrates the false per-chip guarantee without GPU work, modifying no product source or measured resource record. Log prints each actualchip result.

Narrowfix proposal: test every control against each chip individually in table/registry tests; count a missedplant for everyzero-violation chip. In objects() inspect each chip's actual parsed usage inside the loop and add oneproblem if thatcontrol haszero violations, ratherthan testing the allchip sum. Keep normal kernel limit counting unchanged. Add a regression using this one-green/three-red mutation, plus the reciprocal one-red/three-green case; existing all-red controls must stillpass. No GPU math, kernels or limits relaxation is needed. Separate metadata-currentness observation: objects() currently only rejects stale records for !control at126; clarify whether the final“every record current” promise includes controls and enforce it if so. Do not silently relabel historical proofs.

This is an infrastructure readiness finding; keep PR1082 draft until corrected/reviewed. Existing captured actualfourchip logs may independently show allcontrols red, so this doesnot itself invalidate their measured values or performance/native production proofs. The codechange can preserve all production HIP input blobs. Any stack propagation requires an explicit sourcebridge, fresh appropriate host/parser/limits proof and scope-labelled numerical proof reuse—not relabelling old runtime sourceIDs.

## Reproducer

`<ZIG> test control_counterexample.zig --cache-dir cache --global-cache-dir global`

CPUonly, exactsource copiedbyGitshow. Actualexit0,5/5tests, because the test deliberately asserts that the oldaggregate accepts while the proposed per-chip condition rejects. It is a counterexample proof, not a claim of a shippedfix or newGPUrun. No other actionable defect found in bounded parser/build/rig/rows review; malformed/untrusted-object hardening was not a security audit.
