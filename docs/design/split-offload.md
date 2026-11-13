# Split offload: one model across a host's GPU and an external GPU

The split core (`zig/src/split`) lets a model's layers run partly on the host's own GPU (Metal) and partly on an external GPU on the same host, one token at a time. It has no GPU imports: everything here runs on the CPU, so `zig build test-split` needs no device.

## The device

`device.ExternalDevice` is a vtable over what a stream device (HIP, CUDA) provides and nothing vendor specific:

- one in-order stream: `enqueue` (a recorded launch plan), `enqueue_wait` (stall until a 32-bit word in a host window equals a value), `enqueue_write` (store a word after all earlier work), `sync`;
- windows: pinned host memory the device reads and writes and the host maps (`alloc_window`), its address and length multiples of `Caps.window_align` (16 KiB, the size Metal wraps);
- device memory: `alloc`, `free`, `upload`, `download`; `timestamp_ns`.

`mock.Mock` is a CPU implementation (one thread runs the stream; windows are anonymous pages) with optional write and wait delays. It is the reference for what a real adapter must provide, and the only device in this series' tests.

## The slot protocol

`wire.Slot` is a 64-byte header (sequence word, kind, layer, rows, bytes, status) and a payload. The sequence word is written last with release ordering; a consumer waits for it with acquire ordering and an abort status is an error, never data. Sequence numbers start at 1; 0 is an empty slot.

Two levels protect the hidden row the host hands to the device:

- level A: the sequence word is stored after the producer has completed;
- level B: the word may be stored while the data is still landing, so every 16-byte group carries a 4-byte stamp equal to the step's sequence number (`packStamped`, `readStamped`); a group counts only when its stamp is seen before and after its copy, and a stale or future stamp is never accepted.

## The pipeline

`pipeline.Pipeline` runs two stages over one window of two slots (host to device, device to host). The device stream waits on its input slot, runs its stage (`ExternalStage`) and writes its output slot's word; the host waits on that, runs its stage (`Peer`) and publishes the next input. `Order` says which side owns the embedding and the first layers (`external_first`, `metal_first`) or that the device runs everything (`external_only`). Every wait is bounded; a timeout or a lost device releases the other side with an abort and is a named error, and `reset` drains the stream for the next request.

## The toy model

`toy.zig` is a small dense model in exact integer arithmetic (an embedding, twelve layers with per-layer recurrent state, a head, a greedy draw). Its tests split it at several layers in both orders, at both levels (level B with the row landing late and out of order), and with the device alone, and require the tokens of the unsplit run; a stuck device and a pulled cable end in `Timeout` and `DeviceLost`, and the next run is clean.

## The planner, the probe and `tf-offload`

`geometry.zig` reads a Hugging Face `config.json` (Qwen3.5 dense and MoE, MLX affine layout) into per-layer resident bytes, bytes read per decoded token, routed-expert bytes, KV bytes per context token, recurrent-state bytes and launches. `planner.zig` combines it with the Mac (working set, decode bandwidth), the external device (VRAM, peak bandwidth, a decode efficiency, launch cost) and the link (hop times) into every placement the model admits (`all_metal`, `resident`, `layer_split`, `layer_split_pipelined`, `expert_split`, `reverse_expert_split`) with a predicted decode rate and memory use, or the quantity that is missing when nothing fits. `choose` is `auto`: the fastest fitting placement the request can use; all-Metal is a candidate, so a placement predicted slower than all-Metal is never chosen while all-Metal fits. Only the placements the pipeline can run are executable (`all_metal`, `resident`, the dense layer splits); the rest is reported for information.

`probe.zig` measures the link through `ExternalDevice` alone (window round trips, bulk copies each way) and takes the Mac's half through `probe.Host` (working set, a streaming read, the wake of an event-gated command buffer, a window wrap). Its result is the receipt (`receipt.zig`, schema `tensorfold-offload-probe/1`): every number carries a source label (`documented`, `measured`, `probed`, `assumed`), and the planner reads link constants from the receipt and nowhere else. The planner's output is `placement.json` (`placement.zig`, schema `tensorfold-split/1`): the owner of each layer, the embedding and the head, who prefills, the fallback, the timeouts and the prediction. Both files are read strictly (`strict.zig`): every field required, unknown keys refused, errors name the path.

`tf-offload` has three commands and, in this series' first form, opens only the mock device and a CPU stand-in for the Mac (`cpu_host.zig`), so its receipts say nothing about a Mac or a card:

```
zig build tf-offload
tf-offload probe --device mock --out receipt.json
tf-offload plan --config CONFIG.JSON --receipt receipt.json --streams 2 --out placement.json
tf-offload demo --order metal_first --level b --split 7
```

`plan` loads no weights and runs no model. `demo` splits the toy model between the mock device and the CPU and exits 1 if its tokens differ from the unsplit run.

## The Metal half

`metal.HostWindow` (`zig/src/metal/window.zig`) wraps the window's pinned, 16 KiB aligned memory as a zero-copy shared Metal buffer; `createShm` makes an unlinked POSIX shared mapping for tests. A resident Metal kernel does not reliably see CPU or DMA stores and GPU stores reach the CPU only when the command buffer completes, so `metal.Relay` makes each step one command buffer, committed ahead and parked on a shared event: the host stores its data, opens the gate and waits for the done signal. `split/metal_toy.zig` is the toy model's host stage over a `Relay` and a `HostWindow`; `zig build test-split-metal` splits the toy model at several layers with Metal on either side of the mock device and requires the unsplit run's tokens, and `zig build test-metal-relay` runs the window and relay tests.

`split/metal_probe.zig` is the probe's Mac side on Metal: the GPU's name, working set and OS build, a streaming read by a kernel over a 512 MiB buffer, the wake of a parked command buffer (`metal.wake_probe`) and the cost of wrapping a 64 MiB window. On a Mac `zig build tf-offload` builds `tf-offload` with that host (`offload_metal_main.zig`), so `tf-offload probe --device mock` reports this Mac and `tf-offload demo` splits the toy model between the mock device and Metal; elsewhere it keeps the CPU stand-in host.

## Not here

The HIP device and the model families' layer ranges are separate changes.
