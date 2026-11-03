# M4 Pro — exact qualification, timing interpretation inconclusive

Source `afe8bda62562519b4fd84ee9a956392a21f26a60`; native binary `c03ab9600d36c32dd1d80b7d94bf9dae15d7256fa8f41df7f23b488c19b43a2c` in all arms. **144/144 cells and306/306 token oracles pass exactness**, zero scenario failures, nine clean non-forced exits. Original own-head 82-step scope and all six original numerical/native commands pass before this complete unchanged matrix.

## Do not interpret timing flags as16 confirmed regressions

The raw zero-tolerance script flags16 negative median point estimates,12 at least1% below plain; worst **-3.269046023111555%** at sampled-chat/eight-stream/8-bit. But15/16 flagged comparisons change sign across rotations, and the unchanged plain baseline's largest three-run range is **32.25640969309376%**. This dataset is **inconclusive for a1% non-regression decision**. It cannot certify the owner's budget or establish16 real code regressions. No loss is hidden, and nothing is rerun for a more favorable timing sign.

For plain sampled-chat/16 streams, the original records are:

| Rotation | Plain TPS | Rounds | Proposed | Accepted |
|---|---:|---:|---:|---:|
| 0 | 315.822245 | 8176 | 0 | 0 |
| 1 | 290.219123 | 8176 | 0 | 0 |
| 2 | 238.795417 | 8175 | 8 | 1 |

One almost-identical-work baseline is nonstationary; there is no retrospective clock/control record for the full matrix to assign its cause. Observed range is descriptive, not a statistically established measurement-error bound. The worst negative median's rotation-paired differences are +0.068%,−3.269%,+0.329%—two positive and one negative. Taking a ratio of medians does not remove run-position drift.

[All per-cell paired differences and plain sample ranges](measurement-variation.json) are retained. Three rotations and no unchanged-binary A/A control do not provide a reliable confidence bound. We do not declare all differences noise or change the original strict verdict. The next investigation is a separate complete-history global-clock capture, not another qualification attempt or performance override.

## All32 comparisons, raw point estimates

The following are the untouched original median calculations. They are shown for transparency, not promoted to stable performance conclusions.

| Prompt | T | Streams | Plain TPS | BF16 TPS | BF16 Δ | 8-bit TPS | 8-bit Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 1 | 162.644836 | 169.326728 | +4.108272% | 161.981031 | -0.408132% |
| code | 0 | 4 | 280.422162 | 292.220043 | +4.207186% | 283.753543 | +1.187988% |
| code | 0 | 8 | 313.096190 | 322.687667 | +3.063428% | 314.113096 | +0.324790% |
| code | 0 | 16 | 350.022045 | 359.244471 | +2.634813% | 349.078235 | -0.269643% |
| code | 1 | 1 | 114.791766 | 115.592929 | +0.697927% | 112.191899 | -2.264855% |
| code | 1 | 4 | 199.695860 | 199.421767 | -0.137255% | 198.225695 | -0.736202% |
| code | 1 | 8 | 240.224832 | 235.589992 | -1.929376% | 237.802183 | -1.008493% |
| code | 1 | 16 | 295.727495 | 287.876627 | -2.654765% | 290.869652 | -1.642676% |
| chat | 0 | 1 | 117.540345 | 117.968590 | +0.364339% | 116.051363 | -1.266784% |
| chat | 0 | 4 | 244.664867 | 245.425385 | +0.310841% | 240.872726 | -1.549933% |
| chat | 0 | 8 | 281.758310 | 272.931303 | -3.132829% | 277.310261 | -1.578676% |
| chat | 0 | 16 | 340.437404 | 330.456033 | -2.931926% | 334.709738 | -1.682443% |
| chat | 1 | 1 | 109.293881 | 115.167259 | +5.373931% | 114.758276 | +4.999726% |
| chat | 1 | 4 | 202.005371 | 207.118240 | +2.531056% | 209.444878 | +3.682826% |
| chat | 1 | 8 | 245.002211 | 248.380593 | +1.378919% | 236.992976 | -3.269046% |
| chat | 1 | 16 | 290.219123 | 305.001772 | +5.093616% | 295.211121 | +1.720079% |

## Raw evidence and unchanged replay

[Metadata](metadata.json), [cells/oracles](cells.jsonl), [events](events.jsonl), [literal original summary](summary.json), [controller output](controller.log), [source/binary binding](controller-observed.json) and [pipeline status](pipeline-observed.json) are unchanged. [Requests](requests.jsonl.gz) decompress byte-for-byte to the original37MB-class JSONL; [binding](raw-request-binding.json) records actual byte counts/hashes rather than inferring them. Original own-head build, three kernel-contract logs and two native-reference logs are retained under `original-contracts/`.

The same [frozen verdict](../frozen-driver/bench_lfm2_dspark_verdict.py) reproduces this summary byte-for-byte. [Both-device replay identity](../verdict-replay-check.json) and [complete manifest](../SHA256.json) bind the evidence. The driver, numerical contracts, original matrix, cheap-copy baseline and resource guards are not altered.

Both head-weight modes retain BF16 activations. FP32 head-reference picks match9/9 in both modes on M4; this does not turn timing flags into correctness failures. The code correction itself is documented in the separate [runtime798 causal report](../../../79837c4e2/serving-calibration/README.md), not substituted for this final-head matrix.
