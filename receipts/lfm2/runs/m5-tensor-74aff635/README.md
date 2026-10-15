# Part 7: LFM2 prompt calls on the M5 tensor units (74aff635)

`lfm2/7-lfm2-tensor` at 74aff635 (69fa1cbd plus the merge of the 4-bit twin fix). Zig 0.17.0, `-Dcpu=apple_m1`; the M5 adds `-Dmetal-std=metal4.0`. Checkpoint `LFM2.5-24B-A2B-MLX-4bit` at `667408d4`.

| Directory | Machine | What |
| --- | --- | --- |
| `m5/` | Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0 | `tile_check.txt`, `exact_4bit.txt` (68 of 68), `fp64.txt` (each projection class and the prompt attention: tensor units, tiles and decode twin against float64 on the same rows), the two-turn resume check (`two_turn_cache.json`, `fresh_on.json`, `fresh_off.json`; `resume8.sh`, `fresh2.py`), served `decode.json` and `conc.json` (4 streams against each alone). `gate8.sh` and `after8.sh` ran it. |
| `m4/` | Apple M4 Pro (20-core GPU), 48 GB, macOS 26.5.1 | `tile_check.txt` (tensor section skipped), `exact_4bit.txt` (68 of 68). |

Resume check: `two_turn.py` on a cache-on server (turn 2 resumes 7,987 cached tokens and repeats with 8,051); `fresh2.py` sends the same turn-2 messages cold to a new cache-on server, repeats them (8,051 cached), and sends turn 1 and turn 2 cold to a `--prompt-cache-gib 0` server. All five turn-2 replies are equal, and turn 1 is equal with the cache on and off.
