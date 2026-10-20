# Existing Windows CI control with main-equivalent compile inputs

This is an existing upstream CI run, not a new workflow trigger or rerun. A bounded search of50 recent PRs found PR1128, which changes only THIRD_PARTY_NOTICES.md. Its actual Windows job checks out merge fe94c2d9bbf625470bf07d53bfff0284171d4753, whose parents are untouched main d1018b318fb51981c4a18f12e40269a4d3ee9713 and documentation head13f79fbeaf197429dae17f0cb8734563100643dc.

The retained Git commit/root-tree API evidence compares every root entry, including type/mode/SHA: all22 entries other than THIRD_PARTY_NOTICES.md are identical. Consequently entire zig/, .github/, tools/, app/, build.zig, .zig-version and other source/build inputs are identical to main. References to the changed notice file are release packaging/dist, not a changed Zig compile input. The actual job runs zig build test with Zig0.17.0 on windows-2025-vs2026 and fails. Its80 raw compiler-error lines normalize to exactly the same52 location/message pairs in every one of the22 HIP draft CI archives: zero extra or missing diagnostics.

Actual job: https://github.com/ashhart/TensorFold/actions/runs/39487050150/job/118850059735
Control PR: https://github.com/ashhart/TensorFold/pull/1128

This provides main-equivalent compiler-failure evidence, stronger than a file-equality assertion or a forced component test root. It is not a main-branch workflow execution: the tested tree is explicitly the existing notice-only PR merge. Exact Windows compiler binary SHA is unavailable; version, setup-action identities, runner image and actual command remain in the unmodified job log. It does not classify test.exe runtime failure causes, desktop/Linux failures, GPU/model qualification, or make current CI green. No readiness waiver or source correction is implied.

Raw public job logs are byte-identical. Public Git-commit API metadata omits personal author/committer fields; run-metadata author emails are replaced with a literal privacy placeholder; sha/tree/parents/message remain exact. Every original/public hash is mapped below. Other bounded-search metadata is retained to make candidate selection auditable. No model or native binary/weights are included.
