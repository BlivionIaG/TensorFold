# Qwen3.8-Flash-Next EXL3, TP=2 on two RTX 5060 Ti: results ledger

One row per measured state, newest last. Same setup every row unless noted: one user, graphs, 192 output tokens, temperature 0, second request of two,
prompts short (40 tok), zig (16,022 tok), docs (12,780 tok). Every row reports TTFT s / prefill tok/s / decode tok/s / mean ITL ms; `n/m` = not measured then.
Append a row for every change that is meant to move a number, with the commit, and keep the plain baseline row beside it.

| state | commit | short decode | zig TTFT / prefill / decode / ITL | docs TTFT / prefill / decode / ITL | notes |
|---|---|---|---|---|---|
| plain, 72 lines | before embed-on-host | 46.3 | 10.4 / n/m / 34.5 / 29.0 | 10.3 / n/m / 35.7 / 28.0 | start of the lines sweep |
| plain, 80 lines | embed-on-host | 47.8 | n/m / n/m / 35.5 / 28.2 | n/m / n/m / 36.8 / 27.2 | |
| plain, 88 lines | embed-on-host (e.g. 349434944) | 48.9 | 10.64 / 1,506 / 36.2 / 27.7 | 10.35 / 1,234 / 38.2 / 26.2 | baseline for the rows below; Strata 75-104 decode |
| drafts on, no guard, 80 lines | before f84f8566e | 37-40 | n/m / n/m / 101.8 / n/m | n/m / n/m / 40.3 / n/m | every plain round still ran the head's chain |
| + tokens/ms guard | a8adcb7c2 | 44.3 | n/m / n/m / 102.0 / n/m | n/m / n/m / 40.6 / n/m | |
| + window capture on the 12th sighting | a7329cf77 | 50.0 | 11.25 / 1,424 / 106.0 / 9.4 | 10.94 / 1,168 / 59.5 / 16.8 | drafts cost ~5-6% of TTFT/prefill; bursty streaming (zig chunk gap p99 310 ms) |
| rebased onto upstream/main (52f5d9567) | e1b112aea | 49.95 | 11.25 / 1,424 / 106.0 / 9.4 | 10.94 / 1,168 / 59.3 / 16.8 | no change within noise |

Not in the table (measured, rejected, in the final report): lookahead expert prefetch (-11%), lazy head absorb (noise), layer co-occurrence prefetch (covers 3-8% of misses),
DeltaNet split through prefill (TTFT +18%), attention split without paging (-1 to -2%). Opt-in: `TF_FLASHNEXT_GDN_SPLIT=1` (+5% plain decode, neutral with drafts).
