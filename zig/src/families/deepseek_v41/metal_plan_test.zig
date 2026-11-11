//! The Metal memory plan at --context 1048576: four full lanes beside the weights admit on a 256 GB M5 Ultra.
const std = @import("std");
const prefix = @import("kv_prefix.zig");
const plan = @import("metal_plan.zig");

const GB: u64 = 1_000_000_000;

// metal_kv_state.zig's pool layout and plan terms, guarded against drift below
const layout: prefix.Layout = .{ .ratios = &.{ 2, 2, 2, 1 }, .strides = &.{ 288, 68 } };
const mutable: u64 = 40 * 128 * 528 + 7 * 2 * 512 * 4 + 4 * (8 * 2 * 512 * 4);
const descriptors: u64 = 3 << 20;
const per_position: f64 = 890.0;

test "metal_kv_state.zig still has the layout and plan terms this plan counts" {
    const src = @embedFile("metal_kv_state.zig");
    inline for (.{
        "pub const main_stride = 288;",
        "pub const index_stride = 68;",
        "pub const ring_stride = 528;",
        "pub const window = 128;",
        "pub const ratios = [_]u32{ 2, 2, 2, 1 };",
        "pub const max_context = 1048576;",
        "return p.packedBytes() + 2 * p.mutableBytes();",
        "return @as(usize, p.layers) * window * ring_stride + 7 * 2 * 512 * 4 + owners.len * captureBytes();",
    }) |needle| try std.testing.expect(std.mem.indexOf(u8, src, needle) != null);
}

/// The M5's one-lane load at 1M, as logged: weights 210.58 GB, total 218.52 GB, headroom 12.04 GB.
fn at(lanes: u32, capacity: u32) plan.Plan {
    const lane = plan.laneBytes(layout, capacity, mutable, descriptors);
    const one = plan.laneBytes(layout, 1 << 20, mutable, descriptors);
    const weights: u64 = 210_580_000_000;
    const shared: u64 = 218_520_000_000 - weights - one; // workspaces and rotary at the 1M one-lane load
    return plan.Plan.of(.{ .weights = weights, .workspace = shared - 270_000_000, .rotary = 270_000_000, .lanes = lanes, .lane = lane, .lane_per_position = per_position, .capacity = capacity, .limit = 240 * GB, .available = 230_560_000_000 });
}

test "one lane at 1M is the logged load; four lanes fit with 9 GB to spare" {
    const l = plan.laneBytes(layout, 1 << 20, mutable, descriptors);
    try std.testing.expect(l > 935 * 1_000_000 and l < 945 * 1_000_000); // 0.93 GB packed + two 2.86 MB blocks
    const one = at(1, 1 << 20);
    try std.testing.expectApproxEqAbs(@as(f64, 218.52), @as(f64, @floatFromInt(one.needed)) / 1e9, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 12.04), @as(f64, @floatFromInt(one.headroom())) / 1e9, 0.01);
    const four = at(4, 1 << 20);
    try std.testing.expect(four.fits());
    try std.testing.expectApproxEqAbs(@as(f64, 9.2), @as(f64, @floatFromInt(four.headroom())) / 1e9, 0.05);
    try std.testing.expect(four.maxLanes() >= 12);
}

test "the ceiling is the tighter of the load limit and Metal's admission, and a refusal names what fits" {
    var p = at(4, 1 << 20);
    p.in.limit = 220 * GB; // a GLM_LOAD_LIMIT_GB under the 4-lane load
    p = plan.Plan.of(p.in);
    try std.testing.expect(!p.fits());
    try std.testing.expectEqual(@as(u64, 0), p.headroom());
    try std.testing.expectEqual(@as(u32, 2), p.maxLanes()); // 217.58 GB fixed + 0.94 per lane under 220
    var buf: [512]u8 = undefined;
    const why = p.refusal(&buf, 2 * mutable + descriptors);
    try std.testing.expect(std.mem.indexOf(u8, why, "4 lanes at context 1048576 need 221.35 GB resident") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "--parallel 2 fits at this --context") != null);
    const ctx = p.maxContext(4, 2 * mutable + descriptors);
    try std.testing.expect(ctx > 600_000 and ctx < 700_000); // 4 lanes share 2.42 GB: 0.6 GB each, 890 B a position
    const line = at(4, 1 << 20).line(&buf);
    try std.testing.expect(std.mem.indexOf(u8, line, "+ 4 lanes x 0.94 GB (context 1048576, 890 B a position) = 221.35 GB") != null);
}

test "a lane costs 890 bytes a position, so 4 x 1M is 3.7 GB of lanes and 4 x 262k is 0.9 GB" {
    const a = plan.laneBytes(layout, 1 << 20, mutable, descriptors);
    const b = plan.laneBytes(layout, 1 << 18, mutable, descriptors);
    try std.testing.expectApproxEqAbs(@as(f64, per_position), @as(f64, @floatFromInt(a - b)) / @as(f64, (1 << 20) - (1 << 18)), 40.0);
    try std.testing.expect(at(4, 1 << 20).lanes_bytes > 3_700_000_000 and at(4, 1 << 18).lanes_bytes < 1_000_000_000);
}
