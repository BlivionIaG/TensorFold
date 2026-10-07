//! Qwen3.5 / Qwen3.6 text checkpoints for the native HIP engine: configuration, host index and upload.

pub const config = @import("model/config.zig");
pub const table = @import("model/table.zig");
pub const host = @import("model/host.zig");
pub const checkpoint = @import("model/checkpoint.zig");
pub const weights = @import("model/weights.zig");
pub const Config = config.Config;
pub const Spec = config.Spec;
pub const Checkpoint = checkpoint.Checkpoint;
pub const Model = weights.Model;
pub const view = @import("model/view.zig");
pub const state = @import("forward/state.zig");
pub const pages = @import("forward/pages.zig");
pub const forward = @import("forward/forward.zig");
pub const window = @import("forward/window.zig");
pub const plan = @import("forward/plan.zig");
pub const moe = @import("forward/moe.zig");
pub const sample = @import("engine/sample.zig");
pub const draw = @import("engine/draw.zig");
pub const bridge = @import("model/bridge.zig");
pub const engine = @import("engine/engine.zig");
pub const memory = @import("engine/memory.zig");
pub const prefix = @import("engine/prefix.zig");
pub const radix = @import("engine/radix.zig");
pub const hip_lanes = @import("engine/hip_lanes.zig");
pub const mtp = @import("engine/mtp.zig");
pub const worker = @import("engine/worker.zig");
pub const slicing = @import("model/slicing.zig");
pub const reduce = @import("forward/reduce.zig");

test {
    _ = config;
    _ = table;
    _ = @import("model/shard.zig");
    _ = @import("model/projection.zig");
    _ = @import("model/checkpoint_test.zig");
    _ = @import("model/real_test.zig");
    _ = view;
    _ = moe;
    _ = sample;
    _ = prefix;
    _ = radix;
    _ = pages;
    _ = plan;
    _ = slicing;
    // the engine and its workers compile on a host with no GPU
    @import("std").testing.refAllDecls(hip_lanes.Hip);
    @import("std").testing.refAllDecls(worker.Worker);
    @import("std").testing.refAllDecls(engine.Engine);
}
