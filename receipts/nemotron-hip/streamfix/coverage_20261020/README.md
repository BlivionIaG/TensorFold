# Current729 validation and recovered coverage

Measured source729833615595df66c9f9a438ce9a85061843f010 has the identical complete Git tree as review top60a8da5243ef313fcab23bdb0d85c790a5cda935. Baseline8649e9858b38e9ef3dcd8d579879c6512a363050. Private successors are excluded.

## Retained-logit derivation

CPU-only rescoring of24 retained FP32 arrays /12 baseline-current pairs /1131 compared rows (754 primary rows plus377 supplementary W7800 TP-rank1 rows). Every float bit and top1/top2 choice is unchanged; all finite. Per-row token IDs/top-two logits/margins and per-column max/mean/RMSE vectors are retained. Seven exact top-one ties occur in the rank-included compared rows; deterministic lower-index argmax is documented, with no tolerance or waiver. The16 primary outputs match hashes recorded in original summaries;8 supplementary rank1 arrays are newly hashed and source/input/runtime bound, not falsely described as previously recorded primary digests. These are same-history regression results, not independent HF/MLX/FP64 reference inference.

Run `python3 teacherforced729/derive.py RAW_FOLLOWUP_DIRECTORY EMPTY_OUTPUT_DIRECTORY` with NumPy. Original large logits remain retained locally; their exact hashes, sizes, shapes, checkpoint/input/tool/rank-log/runtime bindings are recorded. The source-bound inference recipe and original summary receipts are in the previous immutable followup729 package.

## Actual current-source window and stream gates

Existing tf-nemotron-hip-test binaries were rehashed against the current729 build manifest. On W7800 TP2 (HIP0,1) and V620 TP1 (HIP2), run existing `window MODEL chat_full.ids.bin UNUSED --steps 4` with TF_NEMOTRON_PAGE_SHIFT=1, then `rounds MODEL chat_full.ids.bin UNUSED --steps 4 --streams 4`; TP2 adds actual per-rank `--tp 2 --rank R --master 127.0.0.1 --port P`. See retained exact argv for unambiguous commands. Actual59-token input, checkpoint revision, loaded executables/libraries, default admission environment and existing canonical GPU lock are bound in the receipts.

Four tests / six actual rank exits passed. Window vs serial rows, snapshotted vs plain window, last-two-row rollback, physical page interchange and four mixed-width/position streams vs solo compare equal. Source uses f32 value comparisons; the tool's phrase “same bits” does not separately test signed-zero payload identity. This is engine state coverage, not HTTP cancellation/cache-resume or actual MTP proposal/partial-acceptance qualification. All owned rank processes were reaped and process groups verified absent. Own rigs were serial; no physical-host isolation claim.

## Historical coverage and limits

historical_evidence_gap_map.json identifies already-published cold/context/concurrency/memory/fidelity data by original source and relevant PR. Structural host-source comparison does not turn old kernel timing, memory peaks or FP32 oracle scores into current729 measurements. Nonzero historical reference mismatches remain reported.

Public copies replace private home/temp paths and the SSH host alias. public_manifest.json binds original and published bytes. Hashes within redacted records still identify ORIGINAL artifacts; use the public manifest for actual public file hashes. Existing inference outputs are not modified. No all-gates, upstream approval or landing-readiness claim is made. The requester accepts current measured performance including the4.53% case; that acceptance is not an upstream waiver.

## Exact historical reference-score subset bridge

oracle_subset729 binds entire native-array and token-history SHA256 equality for four current729 outputs to the historical82e68433 HF-CPU FP32/dequantized-MLX comparison. It reuses512 recorded reference-score rows: V620 decode223/224 and story prompt32/32 top1 agreements; W7800 decode221/224 and story prompt31/32, total507/512. All five reference top-choice mismatches and recorded margins/errors are retained. This is an exact native-output/input bridge to historical rounded CSV scores, not a new oracle inference run, universal equality, an approved tie-policy waiver or FP64 projection proof. Raw historical reference arrays, reference runner version records and reference top2 IDs are unavailable; unmatched chat/code rows are excluded.

## Conditional reduction-order review and existing tile tests

fp64_conditional_source_review.json maps every incremental fork/upstream PR, its projection classes and source-bound existing tests. Fusion changes retain their explicit accumulation chains/folds. Prefill batching (fork#38/upstream#1095) does not edit reduction code but can select another existing kp/matrix tile. Existing current729 HIP suites passed103/103 on each architecture and execute the relevant affine tests without a name filter: all registry-pickable compatible prefill tiles twice plus the engine pick on identical synthetic rows, byte comparison of full output buffers, dense/routed/ragged cases across widths/groups/row tiers, including served2048/8192-row cases. A separate affine test checks compatible tiles against sampled16x16 independently dequantized FP64 outputs and output-digest equality. This is executed fixture coverage, beyond a comment asserting equivalence; it is not a universal proof for every operand. No additional duplicate#38 run is justified without identifying an uncovered class.

The new Mamba/combine kernels also have recorded independent FP64 contract fixtures. Actual49-kernel logs retain normalized reference-bound errors; replacement checks are byte comparisons. Neither zero replacement differences, digest equality nor normalized bound ratios are relabelled as newly exported full per-column absolute/aggregate FP64 error vectors. Source review and synthetic fixture passes do not waive model-reference fidelity, kernel registration or other applicable gates.
