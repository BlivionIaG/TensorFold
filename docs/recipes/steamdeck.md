# Qwen3.5 on a Steam Deck (gfx1033), fork branch `steamdeck`

The HIP engine serves Qwen3.5-9B MLX 4-bit with its MTP head on an LCD Steam Deck (Van Gogh APU, gfx1033, RDNA2).

## Build (on any Linux host with HIP)

```sh
zig build native-hip tf-qwen35-test -Dhip-arch=gfx1033 -Dhipcc=<rocm>/bin/hipcc -Doptimize=ReleaseFast --prefix out1033
zig build grammar -Dxgrammar=<xgrammar v0.2.8 checkout> --prefix out1033   # structured output
```

Copy `out1033` and the HIP runtime libraries (`libamdhip64`, `libhsa-runtime64`, `libamd_comgr`, `librocprofiler-register`, `librocm_kpack`, `rocm_sysdeps`, `llvm/lib/libLLVM`, `llvm/lib/libclang-cpp`) into the Deck's home directory; nothing goes to the read-only root.

## GPU memory

HIP on the Deck allocates from the GTT, not the BIOS carve-out (raising the carve-out lowers what the engine counts).
Raise the GTT from 8 to 12 GiB with `ttm.pages_limit=3145728 ttm.page_pool_size=3145728 amdgpu.gttsize=12288` in `GRUB_CMDLINE_LINUX_DEFAULT` (`/etc/default/grub`), then `sudo update-grub` and a reboot.

## Run

```sh
export LD_LIBRARY_PATH=$HOME/tensorfold/rocm/lib:$HOME/tensorfold/rocm/lib/llvm/lib:$HOME/tensorfold/out1033/lib
export HSA_ENABLE_SDMA=0   # SDMA host-to-device copies hang on gfx1033 with HIP 7.14
tensorfold-native serve <model> --parallel 1 --context 8192 --prompt-cache-gib 0.5
TENSORFOLD_MEMORY_RESERVE_GIB=2.5 tf-qwen35-test check <model>   # 14 streams need the smaller reserve
```

Graphs stay on: the MTP head runs eagerly on integrated GPUs, where its first graph launch faults.

## Measured (Qwen3.5-9B MLX 4-bit MTP, 12 GiB GTT, 15 W)

| prompt | drafts | TTFT | prefill | decode | ITL |
| --- | --- | --- | --- | --- | --- |
| 270 tokens | on | 2.7 s | 99 tok/s | 14.3 tok/s | 70 ms |
| 270 tokens | off | 2.8 s | 96 tok/s | 13.3 tok/s | 75 ms |
| 1,883 tokens | on | 18.2 s | 104 tok/s | 14.8 tok/s | 68 ms |
| 1,883 tokens | off | 18.1 s | 104 tok/s | 12.8 tok/s | 78 ms |

`tf-qwen35-test check`: 46 passed, 0 failed (drafted == serial included). Structured output: JSON schema, JSON object, regex and choice valid, drafted == serial.
