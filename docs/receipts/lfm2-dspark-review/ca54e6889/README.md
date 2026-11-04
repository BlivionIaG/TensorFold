# LFM2 DSpark #1394 — M5-only revision and predeclared A/A

Source: [`ca54e6889ed8ed00400bfeb4fa1be30a3a687de5`](https://github.com/BlivionIaG/TensorFold/commit/ca54e6889ed8ed00400bfeb4fa1be30a3a687de5). Pinned upstream main: `00ff0f3b2c7ee67546e7a406f074e6d1d5c02593`.

The protocol was [published at `634a3031d`](https://github.com/BlivionIaG/TensorFold/tree/634a3031d082eba23746b073429c944af09e0d32/docs/receipts/lfm2-dspark-review/ca54e6889) **before this revision's M5 launch**. This report responds only to [review6633416133](https://github.com/ashhart/TensorFold/pull/1394#issuecomment-6633416133). M4 verification, both devices' original numerical guards and the M5 full matrix are complete. Every M5 contrast meets the user's loss budget, but the original stricter gate still fails. Separate A/A and the remaining M5 correctness phases continue; no whole-workflow completion is claimed yet.

## Why this source follows the failed prerequisite

The [source-dc4558f5c packet](../dc4558f5c/README.md) retains both chips' original guard failures. Its matrix and A/A never started. Current main's native facts read `runner.model.config.moe()`; four host fixtures had left that model pointer undefined. This revision initializes the dense model fact in those fixtures. No production Qwen/LFM2 code, assertions, test selection, numerical bounds or workload sizes changed. The passive observer's separate API repair is also included; it does not invoke models.

## Code already pushed

- Pre-M5 and unrecognized Metal device names stand aside before LFM2 neural allocation/calibration. Ordinary target forwarding and cheap copies remain; M5+ remains subject to normal funded admission. CLI path validation still applies.
- `metal_main.zig`: 637 → 547 lines, with DSpark registrations owned by the existing family suite. Cases and numerical bounds are unchanged.
- Current main integrated, including its non-LFM2 support. Unrelated CI-dispatch and CONTRIBUTING additions removed; prior published commits preserved. No CI PR or unrelated model work.
- Actual lean comparison: **425 head / 425 pinned-base warnings, zero new**. Formatting passed for this revision before commit.
- Current-source upstream CI is **11/11 successful jobs** across [Zig format](https://github.com/ashhart/TensorFold/actions/runs/41874396542), [No GPU](https://github.com/ashhart/TensorFold/actions/runs/41874396528) and [App/mobile](https://github.com/ashhart/TensorFold/actions/runs/41874396523). [Head-bound raw proof](ci/upstream.json); no cancellations or reruns.

## Current-source numerical guards

| Chip | Build steps | Fresh test summary | LFM2 positive kernels | Negative controls |
|---|---:|---|---:|---:|
| M4 Pro | 82/82 | 122/125 passed, 3 skipped; remaining unchanged groups cached | 16/16 | 24/24 red as planted |
| M5 Ultra | 82/82 | 785/789 passed, 4 skipped | 17/17 | 24/24 red as planted |

M4's build summary counts freshly executed test steps; unchanged groups were normal Zig-cache hits.
No test-selection or cache flags changed. Raw [M4](m4/guards/build.log) and
[M5](m5-guard/m5-guards/build.log) logs disclose every step. The shared fixture SEGV is fixed on both chips.

The native head diagnostic uses BF16 activations in both weight modes, compared with an independent FP32
head reference on captured BF16 target features. This is **not an FP32 target-server arm**.

| Chip | Drafter weights | FP32 pick agreement | Largest different-pick margin | Maximum absolute confidence difference |
|---|---|---:|---:|---:|
| M4 Pro | Released BF16 | 9/9 | 0 | 0.00085 |
| M4 Pro | 8-bit | 9/9 | 0 | 0.00152 |
| M5 Ultra | Released BF16 | 9/9 | 0 | 0.00080 |
| M5 Ultra | 8-bit | 8/9 | 0.075 | 0.00225 |

These are nonzero numerical differences, not a “zero error” claim. M5's 8-bit case has one proposal-pick
mismatch and passes the unchanged reference criteria; final serving-token exactness is a separate check.
Both devices retain fixed query width 9, the 8/16 logical-cohort cases, all 60 context quotes, zero-row rewind
and all five nonfinite-producer controls. [M4 receipts](m4/guards/observed.json);
[M5 receipts](m5-guard/m5-guards/observed.json).

## Current-source M4 stand-aside lifecycle

Both requested weight modes take the canonical no-neural serving route. Every cache/fresh server logs
stand-aside, none logs neural initialization/calibration, and all four shut down cleanly without force.

| Requested drafter mode | Reused cache tokens | Sampled oracle/solo/concurrent triples equal | Stress completed | Cancelled | Wrong / errors | Clean servers |
|---|---:|---:|---:|---:|---|---:|
| BF16 | 3517 | 29/29 | 425 | 241 | 0 / 0 | 2/2 |
| 8-bit | 3517 | 29/29 | 425 | 241 | 0 / 0 | 2/2 |

Each mode retains the original 6-stream/60-second and 16-stream/90-second stress fixtures. Resumed and
fresh token hashes match. Normal cheap-copy eligibility remains; the recorded resumed requests have
zero drafted proposals, so these data do not claim copy or neural activity in every request.
Independent head diagnostics above remain enabled on M4; they are not neural serving.
[Complete source/binary/lifecycle proof](m4/pipeline-observed.json) checks source files before and after.

This is not a new active-drafter M4 performance matrix: M4 is now ineligible. No new M4 long invocation
was made; prior long receipts and the independent maximum-context diagnostic OOM retain their original
source bindings. The earlier routing-only smoke is retained in the immutable predeclaration.

## Completed original M5 matrix

The original frozen verdict was replayed from the raw records and exactly reproduced the stored summary.
Source and native binary identities match the completed guard receipt.

| Check | Result |
|---|---|
| Measured cells | 144/144 complete and valid |
| Oracle seed pairs | 306/306 valid |
| Exactness / scenario failures | 0 / 0 |
| User's observed loss budget, less than 1% | **32/32 within budget**, none losing at least 1% |
| Maintainer's every-cell-at-least-plain rule | **FAIL: 11/32 negative contrasts** |
| Worst median delta | **−0.4829273348%**, BF16, greedy code, four streams |
| Original exit | **3**, preserved before A/A |
| Original full-matrix servers | 9/9 clean exits, no forced kill |

Median aggregate decode tokens/s across the original three rotations; deltas are relative to each plain median:

| Prompt | T | Streams | Plain t/s | BF16 t/s | BF16 delta | 8-bit t/s | 8-bit delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 1 | 285.441 | 289.272 | +1.34218% | 289.993 | +1.59468% |
| code | 0 | 4 | 641.352 | 638.255 | −0.482927% | 645.995 | +0.723975% |
| code | 0 | 8 | 853.441 | 854.530 | +0.127536% | 853.523 | +0.00961988% |
| code | 0 | 16 | 873.740 | 872.941 | −0.0914648% | 887.631 | +1.58978% |
| code | 1 | 1 | 271.721 | 272.037 | +0.116139% | 274.045 | +0.855061% |
| code | 1 | 4 | 550.835 | 552.574 | +0.315656% | 550.585 | −0.0454163% |
| code | 1 | 8 | 713.359 | 714.059 | +0.0980673% | 714.963 | +0.224819% |
| code | 1 | 16 | 879.739 | 877.623 | −0.240449% | 877.651 | −0.237362% |
| chat | 0 | 1 | 264.912 | 265.219 | +0.11586% | 265.560 | +0.244467% |
| chat | 0 | 4 | 624.653 | 624.317 | −0.0536707% | 624.876 | +0.0357964% |
| chat | 0 | 8 | 821.379 | 820.754 | −0.0761216% | 821.523 | +0.0175538% |
| chat | 0 | 16 | 811.253 | 811.495 | +0.0298579% | 813.005 | +0.215991% |
| chat | 1 | 1 | 259.296 | 259.699 | +0.155542% | 260.414 | +0.431135% |
| chat | 1 | 4 | 560.857 | 562.742 | +0.336128% | 559.997 | −0.153416% |
| chat | 1 | 8 | 720.257 | 720.850 | +0.0824292% | 720.255 | −0.000254007% |
| chat | 1 | 16 | 885.140 | 884.359 | −0.0882655% | 884.136 | −0.113501% |

The frozen summary's `failures_total` counts exactness/scenario/startup failures, **not performance
negatives**. Its zero does not mean zero performance failures or zero floating-point error. Even the
−0.000254007% contrast above remains negative under the unchanged original rule. A/A will not rewrite it.

Across paired repetitions, 44/96 comparisons have identical aggregate rounds/drafted/accepted counts;
21/32 contrasts change sign across the three repetitions. The worst-median case has different aggregate
work in all three pairs, including two extra target rounds in repetition 1. These counts include ordinary
copies and are **not neural-execution counters**. Neither count parity nor sign changes establish a
measurement-error floor or attribute all losses. [Complete work observations](m5-full/aggregate-work-parity.json).

[Metadata](m5-full/metadata.json), [cells](m5-full/cells.jsonl), [events](m5-full/events.jsonl),
[original summary](m5-full/summary.json), [original phase result](m5-full/full-matrix.controller.log),
[hash/redaction ledger](m5-full/original-published-sha256.json). The
[lossless request stream](m5-full/requests.lossless.jsonl.gz) decompresses byte-identically to the original
request JSONL; private server logs stay private. The [frozen driver](protocol/frozen-driver/) is included
byte-identically. Independent historical model-input/backbone/dispatch findings and their limits remain
in [the code diagnosis](https://github.com/BlivionIaG/TensorFold/blob/ca54e6889ed8ed00400bfeb4fa1be30a3a687de5/docs/receipts/lfm2-dspark-review/diagnosis.md).

## Frozen M5 procedure

[`protocol/study.json`](protocol/study.json) declares the original 144-cell / 306-oracle matrix once, then separate plain/plain A/A with 96 cells / 204 oracles, then both original functional modes and original served/native long-context checks. Both A/A arms use identical plain argv and retain ordinary copies. Three rotations necessarily give a disclosed 2:1 first-position imbalance. Original seeds, lengths, warmups, fixtures, locks and bounds are retained. No extra conditioning, retries or pressure/admission overrides.

[`protocol/predeclared-binding.json`](protocol/predeclared-binding.json) binds the source archive, protocol, Python 3.13.15 binary, models, original driver and every execution artifact. A/A wrappers are included byte-identically. The private interpreter path is aliased; original hashes remain recorded. The original full-matrix result is preserved before and after A/A, including exit 3. A/A is variability evidence, **not a tolerance or noise-floor waiver**.

## Historical results remain failures

The [complete previous packet](https://github.com/BlivionIaG/TensorFold/blob/86cd67c15770e0e3ae1e4e737608ef1bc899e66a/docs/receipts/lfm2-dspark-review/5fdbfb874/README.md) retains the source-5f matrices, correctness, source-4ef CI and diagnostic OOMs. No result is relabelled as this source. The user's <1% observed-loss budget is distinct from the maintainer's every-cell-at-least-plain gate. The M4 maximum-context independent diagnostic OOM is not claimed fixed by excluding neural serving on M4.
