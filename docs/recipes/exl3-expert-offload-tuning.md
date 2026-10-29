# Experimental EXL3 expert offload tuning

These controls diagnose Flash Next CUDA expert-weight residency. KV host tiering is separate. Defaults preserve the existing cache and transfer policies; none of the experiments is a speed recommendation without a matched serving receipt.

## Collect costs before allocating lines

Enable `TF_FLASHNEXT_CACHE_TELEMETRY=1`, `TF_FLASHNEXT_EXL3_LAYER_TELEMETRY=1` and `TF_FLASHNEXT_EXL3_OFFLOAD_TIMING=1` together for a diagnostic run. The existing telemetry boundary synchronizes consumers. Do not report these runs as uninstrumented serving performance.

Each decoder layer reports its pool's device-buffer bytes, including its share of device planner state when present. Other engine buffers and MTP stay separately accounted. This is requested buffer storage, not physical VRAM usage or CUDA driver/event overhead. Timing allocates six CUDA events per pool and no additional expert device buffers.

`last_device_service_ns` measures the most recent device-planned layer invocation, including planning, transfers, expert execution and table restoration. `last_device_copy_ns` measures the mapped-host copy kernel, not a copy-engine DMA or PCIe byte counter. With overlap enabled, `last_device_exposed_copy_wait_ns` measures the interval from resident-expert completion through the copy join. Event overhead is included. These values are gauges, not cumulative counters or whole-request totals. Repeated layer invocations inside one reported interval retain only the last timing. Host-planned prefill does not acquire device timings.

Host counters separately report extra split passes, busy drains and their host wait time, staging payload bytes/CPU time, staging-slot waits, and release/gate dependency counts. A dependency count is not a stall duration. A zero host counter does not establish a zero-cost device path.

Use `tools/exl3_offload_measure.py` to normalize raw layer event samples after checking every complete served reply against a reference with full token/output SHA256 values. It refuses missing decoder layers, changing pool geometry, runtime telemetry errors, incomplete replies and truncated hashes. Keep calibration and validation prompts separate.

`tools/exl3_offload_cost.py` consumes a schema-1 JSON profile with an identity containing full `build_sha256`, `checkpoint_sha256`, `workload_sha256` and `settings_sha256`, plus `rank`, `world`, `uniform_lines` and ordered decoder `layers`. Every layer's measured `points` contains that same `identity`, `outputs_equal: true`, `lines`, `variable_device_bytes` and raw `service_ns` samples. Every curve must include the measured uniform baseline; five samples per point are required by default. Keep all nonallocation settings fixed in the identity. Do not interpolate unmeasured capacities or describe the sum of median layer service times as served latency.

The bounded allocation search conserves both the total line count and measured pool/planner device bytes exactly. Its output is an experimental rank-local line-delta environment setting, not an automatically applied policy. Native admission independently checks line costs, planner bytes and the startup reservation. MTP allocation is unchanged. Validate the proposed geometry using unseen prompts and uninstrumented baseline/candidate ABAB runs before enabling it.

## Compare existing transfer and eviction paths

Set controls before constructing the engine. They remain constant for its lifetime, including graph capture/replay:

- `TF_FLASHNEXT_EXL3_COPY_OVERLAP`: `1` by default; `0` serializes the existing device copy/compute path for comparison.
- `TF_FLASHNEXT_EXL3_COPY_BLOCKS`: optional integer 1 through 1024, overriding the existing copy-kernel grid. Measure bandwidth contention against resident-expert computation.
- `TF_FLASHNEXT_EXL3_RECENT_EVICTION`: `0` by default; `1` selects the existing device planner's bounded recent-demand policy. It does not change the host planner's policy.
- `TF_FLASHNEXT_EXL3_STAGING_LINES`: `0` by default copies directly from pinned expert storage on the host-planned path. Integers 1 through 8 select a bounded pinned staging ring per pool. This adds pinned host storage while retaining the pinned source bank. Device-planned mapped copies still read the source bank directly, so this is not a replacement of that path.
- `TF_FLASHNEXT_EXL3_LINE_DELTAS_R0` and `_R1`: rank-specific forms of the existing `TF_FLASHNEXT_EXL3_LINE_DELTAS`; the selected rank's setting takes precedence over the generic setting. Each must independently conserve bytes and total lines. Uniform pool sizing remains available with `TF_FLASHNEXT_EXL3_UNIFORM_LINES=1`.

No additional speculative prefetch is enabled. Existing byte-budget and reader-release dependencies remain in force. Do not change expert weights, routing, accumulation order or graph-owned addresses to improve a result.

Compare code and prose at widths 1/2/4 with the same parallel capacity, context, draft mode, pool bytes and cache state. Record TTFT, prefill/decode throughput, aggregate throughput, mean/p95 ITL, full output hashes, observed device/pinned allocations and device-wide PCIe samples. Keep counters, event diagnostics and uninstrumented performance in separate receipts. Reject a candidate that fails actual graph replay, solo/plain equality, TP equality, cancellation or restored-cache checks.
