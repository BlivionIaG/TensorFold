# Contribution review refresh — 2026-10-20

This is a documentation/evidence refresh for upstream draft PRs1082–1103 and issue1081. It does not change the proposed source heads or qualify the drafts for landing. The earlier c126 source-bound evidence and every historical artifact remain available in the parent tree.

CONTRIBUTING.current.md and requirement_source.json pin the reviewed rules. maintainer_requirements.json captures the actual architectural/splitting instructions; none is Nemotron gate approval. requirements_evidence_matrix.json maps each proposal to verified, source-specific evidence and remaining requirements. The independent incremental source audit covers all22 heads/30 commits. Historical bodies are byte-exact pre-refresh public snapshots; their old statuses are not current qualification.

All22 current Windows checks fail; desktop also fails on1085/1092/1096/1098 and Linux zig on1098. Actual CI metadata, failed-job logs, normalized compiler-message sets and source attribution are retained under ci/. No inherited/flaky/root-cause finding is inferred from duplicate diagnostics. The upstream main snapshot exposes successful desktop/mobile checks but no exact Windows/Linux baseline checks. Successful current jobs are integrated host/build evidence, not GPU/model integration qualification.

The original runner excerpt preserves the original source-bound five-kernel proof, not a new run. Other original data stays at c126: own-head23/43, equivalent full-tree49 at729833/review60a8, finite comparisons, host/model followups, performance regression, source/static/identity and privacy records.

public_manifest.json maps original downloaded bytes to the published bytes. CI home/build/temp paths and runner names are replaced with neutral placeholders; non-noreply/non-documentation email addresses are removed. These privacy transformations change neither diagnostic content, source IDs, test counts nor timings. The CI download manifest still describes the ORIGINAL downloaded hashes; public_manifest.json is authoritative for published transformed copies. Original public job URLs remain available. No model weights or private successor qualification are included.

The proposed source remains a cumulative main-based stack after the first infrastructure PR. Incremental comparison links are review aids, not waiver of the one-change-per-PR rule. Gate agreement, current CI, affected runtime/integration checks, cold/context/lifecycle/reference/memory/GPU-lock receipts and baseline performance remain open where identified. All PRs stay drafts.
