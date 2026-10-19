# Nemotron 3.5 Lightning on HIP: raw receipts

Receipts for the Nemotron-on-HIP pull requests of this fork (#35 to #46): the family as a whole (`family/`), #38's 2,048-row prompt chunks (`prefill-chunk/`), #44 and #46's decode fusions (`fusions/`) and the fp32-reference fidelity (`fidelity/`). `summary.md` has every run.

## Environment

- Fedora 43, Linux 7.1.7, HIP 7.14 (hipcc 7.14.60850), Zig 0.17.0, Python 3.14 for the clients. AMD EPYC 7K62 (96 threads), 503 GiB RAM.
- GPUs: 2 × Radeon PRO W7800 48 GB (gfx1100, 45 GiB visible to HIP) and 4 × Radeon PRO V620 32 GB (gfx1030, 30 GiB). One model process per GPU set at a time; other GPU sets of the host ran other work during some holds.
- RCCL for tp > 1: the RDNA build of RCCL 2.30.4 (`librccl.so.1.0`, sha256 `47044220951dccc5081a65417485da0669e8e6e8ebe7e431413706136183140a`); every rank of every multi-rank hold maps it (`hold.json` `librccl_mapped`).
- Checkpoint: `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` at revision `d9d758fb83953437f7263256b0d96157e2a348b8`, with its MTP head (`mtp-4bit.safetensors`); every file's sha256 checked against the revision's list before the runs.
- Server: `tensorfold-native serve <checkpoint> --name m --port <port>` on loopback, with the arguments in each `hold.json` (`server_args`); tp > 1 adds `--tp N --rank r --master 127.0.0.1 --master-port <port>`. Everything else default (drafts on unless `--no-drafts`).
- Rigs (`HIP_VISIBLE_DEVICES`): v620-tp1 = one V620; v620-tp2 = two V620; w7800-tp2 = the two W7800; v620-tp4 = the four V620.

## Binaries

Built from clean checkouts of each head with `zig build native-hip tf-nemotron-hip-test -Dhip-arch=<arch> -Dhipcc=<rocm 7.14>/bin/hipcc -Doptimize=ReleaseFast` (one build at a time). Every receipt JSON carries `source_commit` and `binary_sha256` (of `tensorfold-native`); `hold.json` also has `test_tool_sha256`.

| tree | pull request head | arch | `tensorfold-native` sha256 | `tf-nemotron-hip-test` sha256 |
|---|---|---|---|---|
| pr37 | #37 `fc555ad4f1c4e92306c457e5a3a67180034e788f` | gfx1030 | `f396b11de3c4b95e6bb2c544316d1ba641aef7d6ee61993ea1b687836cf36080` | `e77758372dbac6338bcbcbadbf762f01d2671e0cf55289cf7c13ccd6a49fe380` |
| pr37 | #37 `fc555ad4f1c4e92306c457e5a3a67180034e788f` | gfx1100 | `4585afcf7d6050442f7ab31ef3306d7039be80afbb05539788542c12a303b68c` | `c6959073d3ce392852af351ad83260d4cbdee0f8d59df24092f8a8a3593aac23` |
| pr38 | #38 `44c37c5f52b91806de323e72f22fb95a683e007c` | gfx1030 | `4202931c50611156a2ba558d37527459d79749da951eb7687e3e6bffbd0ffc48` | `82b570928e14610ccffa0ae1e13e5139bbf1be33a4c190315f9b227395694b61` |
| pr38 | #38 `44c37c5f52b91806de323e72f22fb95a683e007c` | gfx1100 | `15c45dfdb30f2b4b0e07c988fe73d9cee8a72d46424e97346022a51e107d8583` | `05a984390251b73bb8d38159ed52764786c003b12c3c984f3a3b8f1567d477a6` |
| pr44 | #44 `b86e111b24d192e2bfdcdfb426c217c3b7be713a` | gfx1030 | `88408c5ff6c93720aaa77556a501696e9860769b0721325a11653ffcc592d234` | `64ee8f8f52f046450c01e120ff48518620000a0020a285bbacae654de92ce731` |
| pr44 | #44 `b86e111b24d192e2bfdcdfb426c217c3b7be713a` | gfx1100 | `17b02b5e1d23015ca7d7d90e62427e4e03e6c799882e77723198a3e6c401458d` | `a34dfd6fe46ac1c24e3200ae71d1906af1c68d511394842f0b58320adbc2f5f2` |
| pr46 | #46 `82e68433d0d8647e3ca5a0d48f82f5ffc01397ed` | gfx1030 | `36b84ba7ee88d758fd196637eec5029b799a77001aa1ce0464dbb8622c812a3f` | `3223bcffec3f60bfee76720a85625a6f0105ff445f2d57639ee675eb4029024a` |
| pr46 | #46 `82e68433d0d8647e3ca5a0d48f82f5ffc01397ed` | gfx1100 | `286f49435358e478735d91d78e5abd485d825ec6cf2162ae63e8669b857e9f27` | `dd5f5627bbe75ef18263abbf1cb63323eee5eaa88af708f46381b729adb3722c` |

Later heads of these pull requests add HIP kernel-contract test and registry files only; the measured engine and kernel code is the code of these commits.

## Inputs

- `ids/prompt<N>.json`: raw token arrays of N ids. TensorFold 76aafcf2: every tracked *.md file in path order, each after a line "# <path>"; text sha256 `2fa56795f67259ae0542b09e8b35e4c10d22cdf263a72dc405149c0f53351f81`, tokenized with the checkpoint's `tokenizer.json` (no special tokens), and cut so that each array is a prefix of the next (`tools/mkids.py`, `ids/ids_manifest.json`). `ids_sha256` is the sha256 of the file, which equals `bench_cache.py`'s `fixture_sha256`.

| N | `ids_sha256` |
|---:|---|
| 2,048 | `6cd03c840e970559a354e5ddd8dfcf46248212cf6803d1056d9e931b702cddcb` |
| 8,192 | `921f521cd5803e29161e56c98b818c03dddf8dcbd1de6f076d57f9013825ade1` |
| 32,768 | `ef34ff41a3cb4510d50cf72636c5f145cc3842423fb6e158a89d52b736bb389f` |
| 65,536 | `2737dee209b232f542b0e327e5a15d4fc346cdc87479aa05170170eae65a3750` |
| 131,072 | `8dd239f59d732fbe847f062bd203a46f00ec5ff05a5c376749cc28f03a53af5d` |
| 262,080 | `a525bfedba4dc9edf4d3ecd00eed34c9cd6d0798d214e827d624ee378661f960` |

- `ids/requests/cold_<N>.json`: the exact request body of each cold request (`POST /v1/completions`, as `bench_cache.py` sends it); `plain_<N>.json` the same with `"draft": false`.
- Decode prompts: `bench_openai.py`'s and `bench_concurrent.py`'s own public prompts, `fibonacci-raw` / `code` (completion: "Write a short Python function that computes the Fibonacci sequence and explain it.") and `gpu-chat-no-think` / `chat` (chat, thinking off: "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.").

## Commands

Each hold is one fresh server; `tools/hold.sh` runs it and `tools/drive.sh` runs `tools/plan.txt` in order on a rig. In a hold:

- prefill: per size, `python3 tools/bench_cache.py URL m --prompt-file ids/prompt<N>.json --require-cold --tokens 16 --repeats 1 --streams 1 --expected-prompt-tokens <N> --output cold_<N>.json` (three cold requests: first, repeated, concurrent; the server runs `--prompt-cache-gib 0`, so each one starts from fresh KV), then `tools/plain.py` (one 16-token request with `"draft": false`, `plain_<N>.json`).
- decode: `python3 tools/bench_openai.py URL m --tokens 256 --reps 3 --temperatures 0 --label <tree>_<mode> --output decode.json`.
- concurrent: `python3 tools/bench_concurrent.py URL m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output concurrent.json`, and the same with `--levels 1,4,8` on a `--parallel 8` server.
- memory (`family/*/f2_pr46_long`): `tools/memmon.sh` samples every 5 s `VmHWM` and `VmRSS` of each rank's process and `rocm-smi --showmeminfo vram` of the rig's GPUs; `memory_summary.json` has the peaks (`tools/mem.py`).

The clients are the tools of TensorFold v1.1.0 (`tools/` of these heads is unchanged from v1.1.0).

## Files

- `<topic>/<rig>/driver.log`: the run log of the rig, every hold and request in order, with times.
- `<topic>/<rig>/<hold>_<pr>_<mode>/`: one hold. `hold.json` (binary, server arguments, RCCL mapped, server start, health, end and stop times, VRAM before start, after health, after the work and after stop), `server.log` (rank 0; `server_r<k>.log` the other ranks; each request's `done` line has its `cached=` count), and the results: `cold_<N>.json`, `plain_<N>.json`, `decode.json`, `concurrent.json`, `memory_samples.jsonl`, `memory_summary.json`.

Every hold directory and its binary:

| directory | rig | `source_commit` | `binary_sha256` | mode |
|---|---|---|---|---|
| `family/v620-tp1/f1_pr46_conc/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc` |
| `family/v620-tp1/f1_pr46_conc8/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc8` |
| `family/v620-tp1/f1_pr46_decode/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `family/v620-tp1/f2_pr46_long/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `long:131072 262080` |
| `family/v620-tp2/f1_pr46_conc/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc` |
| `family/v620-tp2/f1_pr46_conc8/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc8` |
| `family/v620-tp2/f1_pr46_decode/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `family/v620-tp2/f2_pr46_long/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `long:131072 262080` |
| `family/v620-tp4/f1_pr46_conc/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc` |
| `family/v620-tp4/f1_pr46_conc8/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `conc8` |
| `family/v620-tp4/f1_pr46_decode/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `family/v620-tp4/f2_pr46_long/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `long:131072 262080` |
| `family/w7800-tp2/f1_pr46_conc/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `conc` |
| `family/w7800-tp2/f1_pr46_conc8/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `conc8` |
| `family/w7800-tp2/f1_pr46_decode/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `decode` |
| `family/w7800-tp2/f2_pr46_long/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `long:131072 262080` |
| `fusions/v620-tp1/h1_pr38_decode/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp1/h1_pr38_decodeoff/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp1/h1_pr38_prefill/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp1/h2_pr44_decode/` | v620-tp1 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp1/h2_pr44_decodeoff/` | v620-tp1 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp1/h3_pr46_decode/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp1/h3_pr46_decodeoff/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp1/h3_pr46_prefill/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp1/h4_pr46_decode/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp1/h4_pr46_decodeoff/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp1/h4_pr46_prefill/` | v620-tp1 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp1/h5_pr44_decode/` | v620-tp1 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp1/h5_pr44_decodeoff/` | v620-tp1 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp1/h6_pr38_decode/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp1/h6_pr38_decodeoff/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp1/h6_pr38_prefill/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp2/h1_pr38_decode/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp2/h1_pr38_decodeoff/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp2/h1_pr38_prefill/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp2/h2_pr44_decode/` | v620-tp2 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp2/h2_pr44_decodeoff/` | v620-tp2 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp2/h3_pr46_decode/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp2/h3_pr46_decodeoff/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp2/h3_pr46_prefill/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp2/h4_pr46_decode/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp2/h4_pr46_decodeoff/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp2/h4_pr46_prefill/` | v620-tp2 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp2/h5_pr44_decode/` | v620-tp2 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp2/h5_pr44_decodeoff/` | v620-tp2 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp2/h6_pr38_decode/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp2/h6_pr38_decodeoff/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp2/h6_pr38_prefill/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp4/h1_pr38_decode/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp4/h1_pr38_decodeoff/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp4/h1_pr38_prefill/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp4/h2_pr44_decode/` | v620-tp4 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp4/h2_pr44_decodeoff/` | v620-tp4 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp4/h3_pr46_decode/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp4/h3_pr46_decodeoff/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp4/h3_pr46_prefill/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp4/h4_pr46_decode/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decode` |
| `fusions/v620-tp4/h4_pr46_decodeoff/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `decodeoff` |
| `fusions/v620-tp4/h4_pr46_prefill/` | v620-tp4 | `82e68433d0d8` | `36b84ba7ee88d758…` | `prefill:2048 8192 32768 65536` |
| `fusions/v620-tp4/h5_pr44_decode/` | v620-tp4 | `b86e111b24d1` | `88408c5ff6c93720…` | `decode` |
| `fusions/v620-tp4/h5_pr44_decodeoff/` | v620-tp4 | `b86e111b24d1` | `88408c5ff6c93720…` | `decodeoff` |
| `fusions/v620-tp4/h6_pr38_decode/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `decode` |
| `fusions/v620-tp4/h6_pr38_decodeoff/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `decodeoff` |
| `fusions/v620-tp4/h6_pr38_prefill/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536` |
| `fusions/w7800-tp2/h1_pr38_decode/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `decode` |
| `fusions/w7800-tp2/h1_pr38_decodeoff/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `decodeoff` |
| `fusions/w7800-tp2/h1_pr38_prefill/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `prefill:2048 8192 32768 65536` |
| `fusions/w7800-tp2/h2_pr44_decode/` | w7800-tp2 | `b86e111b24d1` | `17b02b5e1d23015c…` | `decode` |
| `fusions/w7800-tp2/h2_pr44_decodeoff/` | w7800-tp2 | `b86e111b24d1` | `17b02b5e1d23015c…` | `decodeoff` |
| `fusions/w7800-tp2/h3_pr46_decode/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `decode` |
| `fusions/w7800-tp2/h3_pr46_decodeoff/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `decodeoff` |
| `fusions/w7800-tp2/h3_pr46_prefill/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `prefill:2048 8192 32768 65536` |
| `fusions/w7800-tp2/h4_pr46_decode/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `decode` |
| `fusions/w7800-tp2/h4_pr46_decodeoff/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `decodeoff` |
| `fusions/w7800-tp2/h4_pr46_prefill/` | w7800-tp2 | `82e68433d0d8` | `286f49435358e478…` | `prefill:2048 8192 32768 65536` |
| `fusions/w7800-tp2/h5_pr44_decode/` | w7800-tp2 | `b86e111b24d1` | `17b02b5e1d23015c…` | `decode` |
| `fusions/w7800-tp2/h5_pr44_decodeoff/` | w7800-tp2 | `b86e111b24d1` | `17b02b5e1d23015c…` | `decodeoff` |
| `fusions/w7800-tp2/h6_pr38_decode/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `decode` |
| `fusions/w7800-tp2/h6_pr38_decodeoff/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `decodeoff` |
| `fusions/w7800-tp2/h6_pr38_prefill/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `prefill:2048 8192 32768 65536` |
| `prefill-chunk/v620-tp1/h1_pr37_prefill/` | v620-tp1 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp1/h2_pr38_prefill/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp1/h3_pr38_prefill/` | v620-tp1 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp1/h4_pr37_prefill/` | v620-tp1 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp2/h1_pr37_prefill/` | v620-tp2 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp2/h2_pr38_prefill/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp2/h3_pr38_prefill/` | v620-tp2 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp2/h4_pr37_prefill/` | v620-tp2 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp4/h1_pr37_prefill/` | v620-tp4 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp4/h2_pr38_prefill/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp4/h3_pr38_prefill/` | v620-tp4 | `44c37c5f52b9` | `4202931c50611156…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/v620-tp4/h4_pr37_prefill/` | v620-tp4 | `fc555ad4f1c4` | `f396b11de3c4b95e…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/w7800-tp2/h1_pr37_prefill/` | w7800-tp2 | `fc555ad4f1c4` | `4585afcf7d605044…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/w7800-tp2/h2_pr38_prefill/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/w7800-tp2/h3_pr38_prefill/` | w7800-tp2 | `44c37c5f52b9` | `15c45dfdb30f2b4b…` | `prefill:2048 8192 32768 65536 131072 262080` |
| `prefill-chunk/w7800-tp2/h4_pr37_prefill/` | w7800-tp2 | `fc555ad4f1c4` | `4585afcf7d605044…` | `prefill:2048 8192 32768 65536 131072 262080` |

## Fidelity

`fidelity/`: the teacher-forced top-1 scores against an fp32 reference (HF `NemotronHForCausalLM` on the CPU in fp32, the MLX 4-bit weights dequantized exactly; `nem_ref.py`, `ref_long.sh`), pooled over 480 rows of three sequences (`sequences/*.ids.bin`, u32 ids: story_full 32 rows, chat_long 224, code_long 224). `rows_<rig>_<path>.csv` has every row (both tops, both margins, the row's max and mean |error|); `fidelity.json` the summary, every top-token difference and the sha256 of each native logits file. `score2.py` and `miss.py` are the earlier scoring scripts, and `score2_out.txt` and `miss_out.txt` their output on these files. The logits of 1 × V620 tp 1, 2 × V620 tp 2, 4 × V620 tp 4 and 2 × W7800 tp 2 were written by #46's `tf-nemotron-hip-test` (82e68433) and are byte-identical to the earlier 1.0.12 runs; 1 × W7800 tp 1 is the 1.0.12 run (no W7800 tp 1 rig in this set). The logits files themselves (250 MB a rig and path) and the reference (`.npy`) are not included.
