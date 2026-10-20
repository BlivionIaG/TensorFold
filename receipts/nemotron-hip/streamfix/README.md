# Nemotron HIP draft review evidence

DRAFT / NOT READY TO LAND. This update contains measured candidate729833615595df66c9f9a438ce9a85061843f010 and review top60a8da5243ef313fcab23bdb0d85c790a5cda935. Their complete Git trees are identical; this is an explicit source bridge, not an own-head runtime run of60a8. New per-proposal heads and compare ranges are in inline_stack_review/.

inline729 contains strict49/49 contracts on both GPU types with zero failures, zero known findings, all five planted controls red, production O3 finite-byte comparisons and actual four-chip resources, plus fresh macOS checks. ownheads729 separately proves foundation9f7d23/23 and relu-squaref8cd43/43. followup729 contains actual candidate Linux, shared HIP kernels, Qwen, Flash Next and Nemotron comparisons. Read individual manifests for source identities, scope, actual exits and raw/public hash bindings. Failed infrastructure attempts are preserved.

decode_recovery729 retains all21 completed concurrent raw holds (16selected, five excluded) and the port collision history. Both rigs shared one host concurrently; those measurements do not satisfy the single-model-per-testmachine condition. decode_serial729 is the separate serial16-hold comparison with actual host preflights, unchanged admission behavior and recorded cleanup. The reviewed scripts exposed only an existing CPU build lock; existing GPU lock qualification remains pending. Neither narrow comparison establishes full performance or CONTRIBUTING qualification.

ci_pre_inline captures all22 historical draft CI results at their exact old heads and GitHub tested merge refs: desktop/mobile/zig/fmt passed, zig-windows failed. Equal compiler diagnostics and unchanged files do not prove a base failure or establish its cause. New-source CI is separate. Historical statements that no integration tree was built are superseded by this actual old-source merge-ref CI evidence; upstream HIP/model runtime integration remains unqualified.

Original heads.json and older package directories retain historical1596156e/cc18314d identities and the measured31–51% decode regression. ab_complete includes all32 historical holds and drift cautions; both rigs also ran concurrently on one host. historical_draft_bodies preserves all22 published old bodies exactly. No historical measurement is relabelled.

Unrun intermediate exact heads, upstream HIP/model runtime integration, Windows CI and broader context/lifecycle/memory/comparator requirements remain open. Draft publication does not waive landing requirements. Every claim must retain its actual source and measurement scope.

## Contribution evidence refresh

See [the 2026-10-20 source/CI/requirement audit](contributing_refresh_20261020/README.md) and byte-exact pre-refresh draft bodies. This documentation refresh keeps the source heads and all existing proof artifacts unchanged; qualification gaps remain explicit.
