# M5 Ultra — complete integrated-source qualification

**Owner’s observed<1% budget:32/32 pass. Original zero-tolerance verdict:FAIL/exit3, retained.** This is source `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce`, not relabelled AFE or earlier evidence. Current PR tip `51dce745143d72dd52231c44a31989632107af4a` changes only one Markdown EOF line; runtime/build source object IDs are identical. The performance executable was an immutable production copy, without private instrumentation.

## Scope and observed result

Original unchanged full matrix: code/chat,temperature0/1,width1/4/8/16,512 tokens,parallel16,context4096,cache0,three balanced arm rotations. Plain cheap-copy speculation stays enabled. Both head modes use BF16 activations. Every arm uses the same native binary.

- **144/144 cells,306/306 exact target-token oracles;0 exactness/scenario failures;9 clean non-forced native exits.**
- All32 observed median comparisons are within the owner’s<1% loss budget. Worst: **-0.328889768%**,8-bit chat/T1/4 streams. Decisions use unrounded values.
- The maintainer’s literal rule still flags **19 negative comparisons**, so the original summary and original offline replay both return **FAIL/exit3**. No tolerance was added to that gate, no negative cell removed, no model rerun for a better sign.
-16 of32 paired comparisons change sign over the original three rotations. Largest three-sample plain range is2.453770433% (code/T0/8 streams,`(max/min-1)*100`). This is observed variation, **not** a universal measurement-error allowance or proof that every negative is noise. This full matrix is not the separately prespecified six-round M4 A/A study.

Native SHA256: `0ada11bcd2ac70eee71afd0fc0cfaf418983c2c29975c47979e8444a54ef6f43`. Committed source archive SHA256: `3ca340ec39493b7d32ba23902e2087111e54def1184c1665331b4ada1cbe7b12`. Exact driver/model/compiler hashes and runtime source IDs are in `inputs.json`; observed OS/hardware/Python are in original `metadata.json`.

## All original median comparisons

| Cell | Plain TPS | BF16 TPS | BF16 Δ% | 8-bit TPS | 8-bit Δ% |
|---|---:|---:|---:|---:|---:|
| code T0 w1 | 285.809635 | 288.613485 | +0.981020 | 290.204223 | +1.537593 |
| code T0 w4 | 637.880904 | 646.825098 | +1.402173 | 645.928409 | +1.261600 |
| code T0 w8 | 854.676865 | 854.333098 | -0.040222 | 853.774474 | -0.105583 |
| code T0 w16 | 872.757496 | 873.556876 | +0.091592 | 912.260031 | +4.526175 |
| code T1 w1 | 271.726102 | 271.438471 | -0.105853 | 273.351401 | +0.598139 |
| code T1 w4 | 552.865180 | 552.389772 | -0.085990 | 551.868464 | -0.180282 |
| code T1 w8 | 713.827546 | 714.910224 | +0.151672 | 714.106040 | +0.039014 |
| code T1 w16 | 878.533817 | 878.015331 | -0.059017 | 879.703864 | +0.133182 |
| chat T0 w1 | 265.395802 | 264.909182 | -0.183356 | 265.185279 | -0.079324 |
| chat T0 w4 | 625.801746 | 625.268822 | -0.085159 | 624.927451 | -0.139708 |
| chat T0 w8 | 821.844532 | 822.141840 | +0.036176 | 821.424615 | -0.051095 |
| chat T0 w16 | 808.455373 | 810.827864 | +0.293460 | 892.879094 | +10.442595 |
| chat T1 w1 | 259.762932 | 259.128347 | -0.244294 | 259.366490 | -0.152617 |
| chat T1 w4 | 563.277872 | 562.469272 | -0.143553 | 561.425309 | -0.328890 |
| chat T1 w8 | 722.631077 | 721.082908 | -0.214241 | 720.463280 | -0.299987 |
| chat T1 w16 | 885.151985 | 884.832273 | -0.036119 | 884.632536 | -0.058685 |

TPS is the original per-cell aggregate metric; each delta is the ratio of candidate/plain medians from the three balanced rotations. No pooling across devices,source revisions or cells.

## Exercised guards and FP32-reference scope

Exact-source guard build:82/82 steps succeeded;771/775 tests passed,4 skipped. Real M5 LFM2 GPU contract:17/17 kernels passed;24/24 planted controls detected,0 missed,0 known findings,0 limit problems. Prefix/capture,zero-row rewind,zero-grant context,8/16-lane and allocation-failure behavior remain under the original guards.

The FP32 reference is the **head computation on BF16 target features**, not an FP32 target server. The original native test exited0 for both modes; the8-bit head is not claimed bit-exact to FP32:

| Head weights | FP32-reference picks matched | Largest reported differing-pick margin | Worst confidence difference |
|---|---:|---:|---:|
| BF16 | 9/9 | 0 | 0.0008 |
| 8-bit | 8/9 | 0.075 | 0.00225 |

No reference threshold or finite arithmetic was changed for these results. Target-token equality in the full serving matrix is a different property and passed all306 oracles.

## Replay, raw evidence and privacy

`summary.json`,`metadata.json`,`cells.jsonl` and`events.jsonl` are original byte-identical files. `requests.jsonl.gz` decompresses to all original request bytes; its uncompressed SHA is in `verdict-replay-proof.json`. All artifacts were checked for private paths,addresses and credentials before publication. Private native logs remain retained but are not published; their receipt hashes remain available.

The original seven-file driver was exercised on copied saved inputs only. It recreated the original summary **byte-for-byte** and returned3 again. This is an offline replay, not a qualification rerun. `SHA256SUMS` covers the whole packet.

## Meaning and limits

The user’s observed1% criterion passes on this M5 run; the maintainer’s every-cell>=plain policy does not. These are reported separately. Speculative totals include all proposal sources and do not by themselves prove neural-head contribution. These measurements compare native modes, not MLX-LM; they do not establish SIMD occupancy,bandwidth saturation or an MLX bottleneck.

The retained calibration correction prices the target after normal serving residency and head buffers are admitted, before capture destinations are prepared ([native startup](https://github.com/BlivionIaG/TensorFold/blob/f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce/zig/src/native/lfm2_moe.zig#L109-L136),[deferred measure](https://github.com/BlivionIaG/TensorFold/blob/f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce/zig/src/families/lfm2_moe/backend.zig#L57-L76)). Correcting that context and passing correctness do not turn the failed literal gate into a pass.

[Published fixed M4 comparison method and retained superseded attempt](https://github.com/BlivionIaG/TensorFold/blob/9ee4a1399bf6e5f60c43b6a28dd843b9a9c7749b/docs/receipts/lfm2-dspark-review/f7eb5f0bc/controlled-study/README.md) · [Historical clock/reference-drift diagnosis](https://github.com/BlivionIaG/TensorFold/blob/b9fff9c7acb48959960de6a5e789f8af7ead152d/docs/receipts/lfm2-dspark-review/79837c4e2/clock-environment/README.md)
