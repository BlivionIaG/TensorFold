# Complete Linux CPU qualification of amended first head

Actual source 826a0635634dee3793a68e473064d17d5c139945, tree bbb17bdf4db45f1084bb6e35442d4b5dde32039e. Existing trusted Zig 0.17.0 and canonical build wrapper executed `zig build test test-golden test-kernel-contract fuzz-cortheon --summary all` (wrapper adds the existing pinned hipcc option). The source was clean before and after. This is the complete Linux `test` suite, not the earlier CUDA-host subset.

Actual exit zero: 170/170 steps; 2472/2481 tests passed, nine skipped. Golden results: 306/319 equal, 13 known, zero unexpected differences. Raw log and wrapper exit line are preserved. No opt-in GPU run target was used. Owned CPU process group was gone and both canonical GPU/build locks were available at final verification.

Private home/temp roots are normalized; the manifest distinguishes original and public SHA256/bytes. This does not claim a new model/performance run or replace architecture-specific hardware receipts.
