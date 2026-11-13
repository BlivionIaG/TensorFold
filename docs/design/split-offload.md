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

## Not here

The planner, the probe and `tf-offload`, the Metal window and relay, the HIP device and the model families' layer ranges are separate changes.
