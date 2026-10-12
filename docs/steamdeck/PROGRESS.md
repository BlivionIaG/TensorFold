# Steam Deck speed: progress log

Branch `steamdeck-optimized` (from steamdeck-opt2). Deck: gfx1033, 8 CUs, slowPPT and fastPPT caps both 15 W
unless an entry says otherwise. One entry per experiment, failed ones included. Exactness means byte-identical
logits (`tf-qwen35-test logits`) or identical served reply hashes, as the entry says.

Models: EigenLabs `Qwen3.5-9B-MLX-4bit-mtp`, Ornith-1.5-9B MLX 4-bit plus the MTP head built here
(`~/models/ornith/Ornith-1.5-9B-MLX-4bit-mtp`, see its PROVENANCE.txt), Qwen3.8-27B oQ2e-mtp.

## 1. fp16 stream-tile unpack for 2-, 4- and 8-bit codes (c7e4749d): kept

- Change: code pairs stay in place in their word under an exponent that makes their low bit the units; one
  mask, one or, one hsub2 per pair.
- Caps: slow 15 W, fast 15 W (rerun after fastPPT was found at 6 W).
- Exactness: kernel tests 63/63; logits identical, 9B at 9/15/47/64/256/2046 tokens, 27B prefill 15/64 and decode 15.
- Stream tile alone, GB/s of weight bytes, 3 alternating runs each, median old -> new:
  2-bit n34816 k5120 cb8 38.8 -> 34.5, cb4 42.1 -> 45.9; 2-bit n5120 k17408 31.6 -> 35.8;
  4-bit n34816 k5120 64.9 -> 66.2; 4-bit n5120 k17408 53.7 -> 54.0; 8-bit 76.2 -> 77.1.
- 27B oQ2e served decode, w220 prompt, 128 tokens, 2 runs each: drafts default 5.72/5.72 -> 6.00 tok/s,
  drafts off 5.61/5.61 -> 5.88 tok/s (+4.9 %); reply hashes identical.
- Kept: net decode gain on the 27B, exact.

## 2. The same unpack for 3-, 5- and 6-bit (funnel-shift pairs): reverted

- Exact (kernel tests, 27B logits identical) but no gain: 3-bit 25.8 -> 26.0, 5-bit 32.0 -> 31.9 / 31.6 -> 31.5,
  6-bit 35.5 -> 35.4 GB/s. Measured at slow 15 W / fast 6 W, old and new under the same caps. Not committed.

## 3. Host-embedding race: open

- With the embedding on the host (b8af08fd), drafted replies differ from serial unless the MTP head batch syncs the
  stream before it runs. The sync is in b8af08fd as a stopgap; the missing ordering is not yet named.
- Evidence so far (9B, served, drafted vs serial, 995-char reply): no head sync differs 5/5; sync after the host
  gather and before the head launch same 1/1; head graphs off and no sync differs 2/2; head embedding copied with a
  blocking hipMemcpyHtoD instead of the stream copy same 1/2.
