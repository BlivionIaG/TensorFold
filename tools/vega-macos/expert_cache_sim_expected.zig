//! Expected outputs of the offline tests, produced with the Python original (tools/vega-macos/expert_cache_sim.py, since removed) on the same synthetic traces:
//! `uv run --with numpy python expert_cache_sim.py traces --cap 1400 --jobs 2` and the same with `--long none --only LRU`; the two timing lines are left out.

/// A real `np.savez_compressed(ids=uint8 (2, 40, 8) of (l*8 + j + t*5) % 256, kind=uint8 [0, 1])`, as hex.
pub const tiny_npz_hex =
    \\504b03042d00000008000000210044be287cffffffffffffffff070014006964732e6e70790100100000030000000000
    \\0060010000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a05e536aa8aea3a09e965f54529498179f5f94
    \\920a12774bcc294e058a17672416a402f91a463a0a26063a0a169a3a0ab50a64012e06462666165636760e4e2e6e1e5e
    \\3e7e0141216111513171094929691959397905452565155535750d4d2d6d1d5d3d7d03432363135333730b4b2b6b1b5b
    \\3b7b07472767175737770f4f2f6f1f5f3fff80c0a0e090d0b0f088c8a8e898d8b8f884c4a4e494d4b4f48cccacec9cdc
    \\bcfc82c2a2e292d2b2f28acaaaea9adabafa86c6a6e696d6b6f68eceaeee9edebefe0913274d9e3275daf4193367cd9e
    \\3377defc050b172d5eb274d9f2152b57ad5eb376ddfa0d1b376ddeb275dbf61d3b77eddeb377dffe03070f1d3e72f4d8
    \\f113274f9d3e73f6dcf90b172f5dbe72f5daf51b376fddbe73f7defd070f1f3d7ef2f4d9f3172f5fbd7ef3f6ddfb0f1f
    \\3f7dfef2f5dbf71f3f7ffdfef3f7df7f4afd3f1a7494f91f00504b03042d000000080000002100996277dfffffffffff
    \\ffffff080014006b696e642e6e707901001000820000000000000046000000000000009bec17ea1b10c9c850c650ad9e
    \\925a9c5ca46ea5a05e536aa8aea3a09e965f54529498179f5f94920a12774bcc294e058a17672416a402f91a463a9a3a
    \\0ab50a14002e064600504b01022d032d00000008000000210044be287c60010000000300000700000000000000000000
    \\008001000000006964732e6e7079504b01022d032d000000080000002100996277df4600000082000000080000000000
    \\0000000000008001990100006b696e642e6e7079504b050600000000020002006b000000190200000000
;

/// Whole report, cap 1400, short + long + lock4 datasets, every configuration.
pub const pipeline_full =
    \\# expert cache replay, 1400 lines (2.31 GiB), cold start per prompt
    \\short: 12 prompts, 558 decode tokens; long: 2 generations, 230 decode tokens; lock4: 3 groups of 4 short prompts in lockstep
    \\
    \\
    \\## 1. cold-start replay, every policy admit-on-miss, one global pool unless stated
    \\hit% = decode expert accesses served from VRAM; miss/tok = link reads per decode token (320 accesses/token); lock4 miss is per sequence-token;
    \\adm/evi per token on the short set (admissions that evict count in both).
    \\
    \\| policy | short hit% | short miss/tok | long hit% | long miss/tok | lock4 hit% | lock4 miss/seq-tok | adm/tok | evict/tok | short Δ vs LRU (pts) |
    \\|---|---|---|---|---|---|---|---|---|---|
    \\| LRU global (baseline) | 59.6 | 129.30 | 62.1 | 121.24 | 52.3 | 125.26 | 129.30 | 99.20 | +0.00 |
    \\| LRU per layer, equal split | 62.6 | 119.78 | 65.5 | 110.56 | 65.3 | 91.06 | 119.78 | 89.67 | +2.98 |
    \\| SLRU protected 25% | 60.5 | 126.33 | 63.1 | 118.00 | 56.7 | 113.88 | 126.33 | 96.22 | +0.93 |
    \\| SLRU protected 50% | 61.4 | 123.66 | 63.8 | 115.97 | 62.6 | 98.23 | 123.66 | 93.56 | +1.76 |
    \\| SLRU protected 60% | 61.0 | 124.96 | 63.3 | 117.43 | 50.4 | 130.33 | 124.96 | 94.85 | +1.36 |
    \\| SLRU protected 70% | 61.3 | 123.95 | 63.9 | 115.41 | 50.9 | 129.04 | 123.95 | 93.85 | +1.67 |
    \\| SLRU protected 75% | 61.1 | 124.47 | 63.8 | 115.83 | 50.9 | 128.97 | 124.47 | 94.37 | +1.51 |
    \\| SLRU protected 80% | 61.9 | 121.84 | 64.7 | 113.07 | 50.8 | 129.23 | 121.84 | 91.73 | +2.33 |
    \\| SLRU protected 90% | 61.9 | 121.97 | 64.6 | 113.16 | 49.6 | 132.56 | 121.97 | 91.86 | +2.29 |
    \\| 2Q A1in 10% ghost 50% | 60.6 | 126.08 | 63.6 | 116.48 | 58.3 | 109.61 | 126.08 | 95.97 | +1.01 |
    \\| 2Q A1in 25% ghost 50% | 60.0 | 128.15 | 62.8 | 119.07 | 57.5 | 111.80 | 128.15 | 98.05 | +0.36 |
    \\| 2Q A1in 50% ghost 50% | 58.6 | 132.32 | 61.2 | 124.16 | 51.9 | 126.28 | 132.32 | 102.21 | -0.94 |
    \\| TinyLFU W=2.5k door=n | 62.4 | 120.18 | 65.7 | 109.84 | 64.7 | 92.77 | 67.33 | 37.22 | +2.85 |
    \\| TinyLFU W=2.5k door=y | 61.7 | 122.66 | 64.7 | 112.91 | 61.7 | 100.53 | 84.34 | 54.23 | +2.07 |
    \\| TinyLFU W=5k door=n | 63.3 | 117.49 | 66.2 | 108.20 | 65.8 | 89.82 | 51.55 | 21.44 | +3.69 |
    \\| TinyLFU W=5k door=y | 63.0 | 118.51 | 65.9 | 109.13 | 65.8 | 89.97 | 54.81 | 24.70 | +3.37 |
    \\| TinyLFU W=10k door=n | 62.9 | 118.65 | 65.4 | 110.66 | 66.4 | 88.22 | 44.87 | 14.77 | +3.33 |
    \\| TinyLFU W=10k door=y | 62.9 | 118.75 | 65.5 | 110.45 | 66.5 | 88.14 | 44.98 | 14.88 | +3.30 |
    \\| TinyLFU W=40k door=n | 62.7 | 119.31 | 63.6 | 116.38 | 66.5 | 87.91 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=40k door=y | 62.7 | 119.31 | 63.6 | 116.38 | 66.5 | 87.90 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=160k door=n | 62.7 | 119.31 | 63.6 | 116.38 | 66.4 | 88.23 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=160k door=y | 62.7 | 119.31 | 63.6 | 116.38 | 66.4 | 88.23 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=640k door=n | 62.7 | 119.31 | 63.6 | 116.38 | 66.4 | 88.23 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=640k door=y | 62.7 | 119.31 | 63.6 | 116.38 | 66.4 | 88.23 | 41.76 | 11.65 | +3.12 |
    \\| TinyLFU W=40k door=n admit if >= | 62.5 | 120.07 | 64.1 | 114.82 | 65.6 | 90.50 | 54.34 | 24.23 | +2.89 |
    \\| TinyLFU W=40k door=y admit if >= | 62.5 | 120.07 | 64.1 | 114.80 | 65.5 | 90.58 | 54.34 | 24.23 | +2.89 |
    \\| W-TinyLFU window 1% W=160k door=y | 62.2 | 120.91 | 62.9 | 118.71 | 65.8 | 89.75 | 120.91 | 90.81 | +2.62 |
    \\| W-TinyLFU window 5% W=160k door=y | 60.2 | 127.24 | 61.0 | 124.70 | 63.4 | 96.13 | 127.24 | 97.13 | +0.65 |
    \\| W-TinyLFU window 20% W=160k door=y | 61.5 | 123.22 | 61.9 | 122.08 | 55.8 | 116.18 | 123.22 | 93.11 | +1.90 |
    \\| LFU aged W=1.25k (history kept) | 59.8 | 128.66 | 62.3 | 120.49 | 52.7 | 124.25 | 128.66 | 98.55 | +0.20 |
    \\| LFU aged W=2.5k (history kept) | 61.2 | 124.28 | 64.0 | 115.10 | 56.5 | 114.38 | 124.28 | 94.17 | +1.57 |
    \\| LFU aged W=5k (history kept) | 62.2 | 120.82 | 65.2 | 111.22 | 60.1 | 104.78 | 120.82 | 90.72 | +2.65 |
    \\| LFU aged W=10k (history kept) | 62.4 | 120.46 | 64.9 | 112.23 | 61.5 | 101.11 | 120.46 | 90.35 | +2.76 |
    \\| LFU aged W=20k (history kept) | 62.3 | 120.68 | 64.4 | 114.02 | 62.2 | 99.27 | 120.68 | 90.57 | +2.70 |
    \\| LFU aged W=40k (history kept) | 62.3 | 120.68 | 63.6 | 116.38 | 62.8 | 97.66 | 120.68 | 90.57 | +2.70 |
    \\| LFU aged W=160k (history kept) | 62.3 | 120.68 | 63.6 | 116.41 | 62.8 | 97.78 | 120.68 | 90.57 | +2.70 |
    \\| LFU aged W=640k (history kept) | 62.3 | 120.68 | 63.6 | 116.41 | 62.8 | 97.78 | 120.68 | 90.57 | +2.70 |
    \\| LFU aged W=40k (count restarts on evict) | 58.6 | 132.60 | 53.9 | 147.52 | 49.6 | 132.47 | 132.60 | 102.49 | -1.03 |
    \\| LFU aged W=160k (count restarts on evict) | 58.6 | 132.60 | 53.9 | 147.52 | 49.5 | 132.70 | 132.60 | 102.49 | -1.03 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=5k | 59.6 | 129.24 | 62.1 | 121.14 | 52.4 | 125.20 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=5k | 59.7 | 129.09 | 62.2 | 121.05 | 52.4 | 125.08 | 129.09 | 98.98 | +0.07 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=5k | 59.8 | 128.78 | 62.3 | 120.72 | 52.5 | 124.76 | 128.78 | 98.67 | +0.16 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=5k | 60.0 | 127.92 | 62.5 | 119.89 | 52.8 | 124.04 | 127.92 | 97.81 | +0.43 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=5k | 60.6 | 126.24 | 63.2 | 117.76 | 53.5 | 122.17 | 126.24 | 96.13 | +0.96 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=10k | 59.6 | 129.24 | 62.1 | 121.22 | 52.4 | 125.17 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=10k | 59.6 | 129.16 | 62.2 | 121.10 | 52.5 | 124.95 | 129.16 | 99.05 | +0.05 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=10k | 59.7 | 128.83 | 62.3 | 120.71 | 52.7 | 124.40 | 128.83 | 98.72 | +0.15 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=10k | 60.0 | 127.98 | 62.5 | 119.85 | 53.0 | 123.40 | 127.98 | 97.87 | +0.41 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=10k | 60.6 | 126.20 | 63.2 | 117.80 | 53.8 | 121.33 | 126.20 | 96.09 | +0.97 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=40k | 59.6 | 129.24 | 62.1 | 121.20 | 52.4 | 125.13 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=40k | 59.6 | 129.19 | 62.1 | 121.13 | 52.5 | 124.87 | 129.19 | 99.08 | +0.04 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=40k | 59.7 | 128.86 | 62.2 | 121.00 | 52.7 | 124.23 | 128.86 | 98.75 | +0.14 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=40k | 60.0 | 128.04 | 62.5 | 120.07 | 53.1 | 123.13 | 128.04 | 97.93 | +0.40 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=40k | 60.6 | 126.21 | 63.1 | 118.01 | 54.1 | 120.74 | 126.21 | 96.10 | +0.97 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=160k | 59.6 | 129.24 | 62.1 | 121.20 | 52.4 | 125.13 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=160k | 59.6 | 129.19 | 62.1 | 121.13 | 52.5 | 124.86 | 129.19 | 99.08 | +0.04 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=160k | 59.7 | 128.86 | 62.2 | 121.01 | 52.7 | 124.20 | 128.86 | 98.75 | +0.14 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=160k | 60.0 | 128.04 | 62.5 | 120.06 | 53.2 | 123.06 | 128.04 | 97.93 | +0.40 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=160k | 60.6 | 126.21 | 63.1 | 118.01 | 54.1 | 120.72 | 126.21 | 96.10 | +0.97 |
    \\| ARC global | 62.0 | 121.57 | 65.3 | 110.98 | 53.2 | 122.91 | 121.57 | 91.46 | +2.42 |
    \\
    \\## 2. evictions / admissions per token: LRU baseline and the two best other candidates (short+long hit rate)
    \\
    \\| policy | short adm/tok | short evict/tok | short miss/tok | long adm/tok | long evict/tok | long miss/tok |
    \\|---|---|---|---|---|---|---|
    \\| LRU global (baseline) | 129.30 | 99.20 | 129.30 | 121.24 | 109.07 | 121.24 |
    \\| TinyLFU W=5k door=n | 51.55 | 21.44 | 117.49 | 34.63 | 22.45 | 108.20 |
    \\| TinyLFU W=5k door=y | 54.81 | 24.70 | 118.51 | 37.94 | 25.77 | 109.13 |
    \\
    \\per-prompt paired comparison against LRU (miss/token difference, negative = fewer link reads than LRU):
    \\
    \\| policy | set | prompts better / equal(<0.05) / worse | mean diff miss/tok | std error |
    \\|---|---|---|---|---|
    \\| TinyLFU W=5k door=n | short | 12 / 0 / 0 | -11.56 | 0.52 |
    \\| TinyLFU W=5k door=n | long | 2 / 0 / 0 | -13.07 | 0.27 |
    \\| TinyLFU W=5k door=y | short | 12 / 0 / 0 | -10.63 | 0.44 |
    \\| TinyLFU W=5k door=y | long | 2 / 0 / 0 | -12.18 | 0.58 |
    \\
    \\## 3. cold vs seeded vs prefill-warm LRU: misses per decode token by decode position (320 accesses/token)
    \\
    \\| set | start | 0-8 | 8-32 | 32-128 | 128-512 | 512+ | all tokens |
    \\|---|---|---|---|---|---|---|---|
    \\| short | LRU cold start | 192.1 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 129.3 |
    \\| short | LRU, seeded with other categories' hottest keys | 106.7 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 114.6 |
    \\| short | LRU, warm from own prefill top counts | 114.4 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 115.9 |
    \\| short | LRU, own prefill top counts + others' hottest fill | 110.5 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 115.3 |
    \\| short | LRU, prefill replayed token by token through LRU | 117.0 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 116.4 |
    \\| long | LRU cold start | 192.9 (n=16) | 115.8 (n=48) | 115.9 (n=164) | 117.5 (n=2) | - | 121.2 |
    \\| long | LRU, seeded with other categories' hottest keys | 106.2 (n=16) | 115.8 (n=48) | 115.9 (n=164) | 117.5 (n=2) | - | 115.2 |
    \\| long | LRU, warm from own prefill top counts | 117.9 (n=16) | 115.8 (n=48) | 115.9 (n=164) | 117.5 (n=2) | - | 116.0 |
    \\| long | LRU, own prefill top counts + others' hottest fill | 113.1 (n=16) | 115.8 (n=48) | 115.9 (n=164) | 117.5 (n=2) | - | 115.7 |
    \\| long | LRU, prefill replayed token by token through LRU | 120.8 (n=16) | 115.8 (n=48) | 115.9 (n=164) | 117.5 (n=2) | - | 116.2 |
    \\
    \\## 4. what the warm starts save (LRU cold minus warm, link reads per prompt, summed over decode tokens, per bucket)
    \\
    \\| set | start | 0-8 | 8-32 | 32-128 | 128-512 | 512+ | total reads saved per prompt | prompts |
    \\|---|---|---|---|---|---|---|---|---|
    \\| short | LRU, seeded with other categories' hottest keys | +683 | -0 | +0 | - | - | +683 | 12 |
    \\| short | LRU, warm from own prefill top counts | +622 | +0 | +0 | - | - | +622 | 12 |
    \\| short | LRU, own prefill top counts + others' hottest fill | +653 | +0 | +0 | - | - | +653 | 12 |
    \\| short | LRU, prefill replayed token by token through LRU | +601 | +0 | +0 | - | - | +601 | 12 |
    \\| long | LRU, seeded with other categories' hottest keys | +693 | +0 | +0 | +0 | - | +694 | 2 |
    \\| long | LRU, warm from own prefill top counts | +600 | +0 | +0 | +0 | - | +600 | 2 |
    \\| long | LRU, own prefill top counts + others' hottest fill | +638 | +0 | +0 | +0 | - | +638 | 2 |
    \\| long | LRU, prefill replayed token by token through LRU | +577 | +0 | +0 | +0 | - | +577 | 2 |
;

/// `--long none --only LRU`.
pub const pipeline_lru_short_only =
    \\# expert cache replay, 1400 lines (2.31 GiB), cold start per prompt
    \\short: 12 prompts, 558 decode tokens; lock4: 3 groups of 4 short prompts in lockstep
    \\
    \\
    \\## 1. cold-start replay, every policy admit-on-miss, one global pool unless stated
    \\hit% = decode expert accesses served from VRAM; miss/tok = link reads per decode token (320 accesses/token); lock4 miss is per sequence-token;
    \\adm/evi per token on the short set (admissions that evict count in both).
    \\
    \\| policy | short hit% | short miss/tok | lock4 hit% | lock4 miss/seq-tok | adm/tok | evict/tok | short Δ vs LRU (pts) |
    \\|---|---|---|---|---|---|---|---|
    \\| LRU global (baseline) | 59.6 | 129.30 | 52.3 | 125.26 | 129.30 | 99.20 | +0.00 |
    \\| LRU per layer, equal split | 62.6 | 119.78 | 65.3 | 91.06 | 119.78 | 89.67 | +2.98 |
    \\| SLRU protected 25% | 60.5 | 126.33 | 56.7 | 113.88 | 126.33 | 96.22 | +0.93 |
    \\| SLRU protected 50% | 61.4 | 123.66 | 62.6 | 98.23 | 123.66 | 93.56 | +1.76 |
    \\| SLRU protected 60% | 61.0 | 124.96 | 50.4 | 130.33 | 124.96 | 94.85 | +1.36 |
    \\| SLRU protected 70% | 61.3 | 123.95 | 50.9 | 129.04 | 123.95 | 93.85 | +1.67 |
    \\| SLRU protected 75% | 61.1 | 124.47 | 50.9 | 128.97 | 124.47 | 94.37 | +1.51 |
    \\| SLRU protected 80% | 61.9 | 121.84 | 50.8 | 129.23 | 121.84 | 91.73 | +2.33 |
    \\| SLRU protected 90% | 61.9 | 121.97 | 49.6 | 132.56 | 121.97 | 91.86 | +2.29 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=5k | 59.6 | 129.24 | 52.4 | 125.20 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=5k | 59.7 | 129.09 | 52.4 | 125.08 | 129.09 | 98.98 | +0.07 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=5k | 59.8 | 128.78 | 52.5 | 124.76 | 128.78 | 98.67 | +0.16 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=5k | 60.0 | 127.92 | 52.8 | 124.04 | 127.92 | 97.81 | +0.43 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=5k | 60.6 | 126.24 | 53.5 | 122.17 | 126.24 | 96.13 | +0.96 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=10k | 59.6 | 129.24 | 52.4 | 125.17 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=10k | 59.6 | 129.16 | 52.5 | 124.95 | 129.16 | 99.05 | +0.05 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=10k | 59.7 | 128.83 | 52.7 | 124.40 | 128.83 | 98.72 | +0.15 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=10k | 60.0 | 127.98 | 53.0 | 123.40 | 127.98 | 97.87 | +0.41 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=10k | 60.6 | 126.20 | 53.8 | 121.33 | 126.20 | 96.09 | +0.97 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=40k | 59.6 | 129.24 | 52.4 | 125.13 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=40k | 59.6 | 129.19 | 52.5 | 124.87 | 129.19 | 99.08 | +0.04 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=40k | 59.7 | 128.86 | 52.7 | 124.23 | 128.86 | 98.75 | +0.14 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=40k | 60.0 | 128.04 | 53.1 | 123.13 | 128.04 | 97.93 | +0.40 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=40k | 60.6 | 126.21 | 54.1 | 120.74 | 126.21 | 96.10 | +0.97 |
    \\| LRU, evict lowest-freq of K=2 oldest, W=160k | 59.6 | 129.24 | 52.4 | 125.13 | 129.24 | 99.13 | +0.02 |
    \\| LRU, evict lowest-freq of K=4 oldest, W=160k | 59.6 | 129.19 | 52.5 | 124.86 | 129.19 | 99.08 | +0.04 |
    \\| LRU, evict lowest-freq of K=8 oldest, W=160k | 59.7 | 128.86 | 52.7 | 124.20 | 128.86 | 98.75 | +0.14 |
    \\| LRU, evict lowest-freq of K=16 oldest, W=160k | 60.0 | 128.04 | 53.2 | 123.06 | 128.04 | 97.93 | +0.40 |
    \\| LRU, evict lowest-freq of K=32 oldest, W=160k | 60.6 | 126.21 | 54.1 | 120.72 | 126.21 | 96.10 | +0.97 |
    \\
    \\## 2. evictions / admissions per token: LRU baseline and the two best other candidates (short+long hit rate)
    \\
    \\| policy | short adm/tok | short evict/tok | short miss/tok |
    \\|---|---|---|---|
    \\| LRU global (baseline) | 129.30 | 99.20 | 129.30 |
    \\| LRU per layer, equal split | 119.78 | 89.67 | 119.78 |
    \\| SLRU protected 80% | 121.84 | 91.73 | 121.84 |
    \\
    \\per-prompt paired comparison against LRU (miss/token difference, negative = fewer link reads than LRU):
    \\
    \\| policy | set | prompts better / equal(<0.05) / worse | mean diff miss/tok | std error |
    \\|---|---|---|---|---|
    \\| LRU per layer, equal split | short | 12 / 0 / 0 | -9.43 | 0.18 |
    \\| SLRU protected 80% | short | 12 / 0 / 0 | -7.33 | 0.43 |
    \\
    \\## 3. cold vs seeded vs prefill-warm LRU: misses per decode token by decode position (320 accesses/token)
    \\
    \\| set | start | 0-8 | 8-32 | 32-128 | 128-512 | 512+ | all tokens |
    \\|---|---|---|---|---|---|---|---|
    \\| short | LRU cold start | 192.1 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 129.3 |
    \\| short | LRU, seeded with other categories' hottest keys | 106.7 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 114.6 |
    \\| short | LRU, warm from own prefill top counts | 114.4 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 115.9 |
    \\| short | LRU, own prefill top counts + others' hottest fill | 110.5 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 115.3 |
    \\| short | LRU, prefill replayed token by token through LRU | 117.0 (n=96) | 115.9 (n=286) | 116.8 (n=176) | - | - | 116.4 |
    \\
    \\## 4. what the warm starts save (LRU cold minus warm, link reads per prompt, summed over decode tokens, per bucket)
    \\
    \\| set | start | 0-8 | 8-32 | 32-128 | 128-512 | 512+ | total reads saved per prompt | prompts |
    \\|---|---|---|---|---|---|---|---|---|
    \\| short | LRU, seeded with other categories' hottest keys | +683 | -0 | +0 | - | - | +683 | 12 |
    \\| short | LRU, warm from own prefill top counts | +622 | +0 | +0 | - | - | +622 | 12 |
    \\| short | LRU, own prefill top counts + others' hottest fill | +653 | +0 | +0 | - | - | +653 | 12 |
    \\| short | LRU, prefill replayed token by token through LRU | +601 | +0 | +0 | - | - | +601 | 12 |
;
