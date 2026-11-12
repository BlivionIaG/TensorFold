# Qwen3.5 / Qwen3.6 MoE on HIP: routed experts in host memory

The native HIP engine can serve an MLX affine MoE checkpoint (`qwen3_5_moe`) whose routed experts do not fit the device.
The experts stay in host memory, packed one blob per expert; the device keeps the dense weights, the shared experts and a cache of the hot experts.
It is a capacity feature: a model that fits the device runs as before, and a host with the memory runs the same checkpoint faster on its own.

## What runs where

- **Host tier:** every routed expert of every layer, in pageable memory.
- **Window:** a pinned, device-mapped host buffer (`TF_EXPERT_WINDOW_MIB`, default 1536). The host copies the experts a layer misses into it and writes the layer's pointer-table row there.
- **Cache lines:** device memory, one line per expert, chosen by a segmented LRU (`TF_EXPERT_PROTECT` percent protected, default 75; 0 is a plain LRU).
- **Per layer, in the stream:** router, pick and plan, `tf_expert_trace` (the picks go to the window), a stream write of a request word, a stream wait for the host's answer, then the expert products.
  A host worker thread serves the request: policy, copies, table row, answer.
- **Pointer-table tiles:** a routed product reads an expert from its cache line when the table names one and from its window slot otherwise, and a miss is stored into the line the host chose.
  They are a separate code object (`affine_tiles_ptr`, built with `-DTF_EXPERT_PTRS=1`) that only a projection with a table launches, so no other tile changes.
- **Prompt spans:** the experts a span names are copied from the window into a one-layer scratch on a copy stream before the layer's GEMMs.

## Turning it on

- **Automatically:** a checkpoint whose weights do not fit the memory budget is offloaded when its experts can be.
- **Explicitly:** `TF_EXPERT_OFFLOAD=1` always offloads; `TF_EXPERT_OFFLOAD=0` never does (the checkpoint then loads whole or is refused).
- **Cache size:** `--ssd-experts GIB` sets the device bytes of the expert cache and implies `TF_EXPERT_OFFLOAD=1`. Without it the cache takes what the plan leaves.
- **Admissions a round:** `TF_EXPERT_COPIES` bounds how many experts the cache admits a round (default 4096, every miss).

The switches are listed in [environment validation](../environment.md).

## Requirements and limits

- **Chip:** the pointer-table tiles are built where the caps table sets `expert_ptrs`, today gfx900 (Vega 10). Without it an offload is refused, never run slowly.
- **Runtime capability:** the stream write and wait need the HIP runtime's `hipStreamWriteValue32` and `hipStreamWaitValue32`, and mapped host memory the device reads in place.
  A runtime without them refuses the offload with `NoValueOps`. A runtime that bounds how long a stream value wait may last must be configured with no bound: a forward queues every layer's wait at once.
- **Eager rounds:** stream value waits cannot be captured, so an offloaded model's rounds run without graphs.
- **One rank:** tensor parallelism, a bf16 chat checkpoint and the MTP draft head are not offloaded. Experts must be MLX affine and fused (`gate_up`) in the checkpoint.
- **Rows:** a forward stages at most 2048 rows; a larger `--parallel` is refused at sizing.
- **Prompt cache:** with the offload on, the prompt cache is only what `--prompt-cache-gib` names.

## Verifying

- Host: `zig build test-expert-offload-host` (policy, window layout, stage, worker and a request/answer handshake against a fake device) and `zig build hip-host-test -Dhip=true`.
- Device (a card whose runtime has the capability): greedy tokens with the cache on and with `TF_EXPERT_COPIES=0` must be identical, and equal the reference implementation's up to bf16 near-ties.
- Throughput depends on the link between host and device: decode is limited by the bytes the misses read, so tokens per second scale with one over the miss rate.
