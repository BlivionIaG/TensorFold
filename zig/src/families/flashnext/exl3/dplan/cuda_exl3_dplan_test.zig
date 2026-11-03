//! Host tests of the EXL3 device planner: the demand trace, its ownership checkpoints and the planner's state words.

const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const layout = @import("exl3_dplan_layout.zig");
const dplan = @import("cuda_exl3_dplan.zig");
const DevPlan = dplan.DevPlan;
const PoolState = dplan.PoolState;
const demand_trace = dplan.demand_trace;
const stateWords = dplan.stateWords;
const experts_n = layout.experts_n;
const recent_window = layout.recent_window;
const recent_words_per_row = experts_n / 32;
const misc_words = layout.misc_words;
const recentOffset = layout.recentOffset;
const miscOffset = layout.miscOffset;

test "exl3: demand trace diagnostic APIs reject disabled capture" {
    var dp: DevPlan = undefined;
    dp.trace_mem = null;
    try std.testing.expectEqual(@as(usize, 0), dp.traceDeviceBytes());
    try std.testing.expectError(error.TraceDisabled, dp.readTraceAfterConsumers(0, &.{}));
    try std.testing.expectError(error.TraceDisabled, dp.downloadTraceAfterConsumers(std.testing.allocator, 0));
    try dp.traceOwnership(undefined, true);
    try dp.traceOwnership(undefined, false);
}

test "exl3: ownership trace checkpoints use ragged pool state and serialized compute stream" {
    const Probe = struct {
        var calls: usize = 0;
        var arguments: [4][4]u64 = undefined;
        var streams: [4]cuda.abi.Stream = undefined;

        fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, stream: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            inline for (.{ u64, c_int, u64, c_int }, 0..) |Type, index| {
                const value: *const Type = @ptrCast(@alignCast(parameters.?[index].?));
                arguments[calls][index] = @intCast(value.*);
            }
            streams[calls] = stream;
            calls += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Probe.launch;
    var entries = [_]PoolState{ try PoolState.init(10, 0), try PoolState.init(31, stateWords(10)) };
    var device: DevPlan = undefined;
    device.pool_states = &entries;
    device.state.ptr = 0x10000;
    device.trace_mem = .{ .d = &driver, .ptr = 0x100000, .len = 2 * demand_trace.pool_bytes, .borrowed = true };
    device.trace_checkpoint_fn = .{ .d = &driver, .handle = @ptrFromInt(1) };
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(2) };
    Probe.calls = 0;
    try device.traceOwnership(stream, true);
    try device.traceOwnership(stream, false);
    for (0..4) |call| {
        const index = call % 2;
        try std.testing.expectEqual([4]u64{ device.state.ptr + entries[index].byteOffset(), entries[index].lines, 0x100000 + index * demand_trace.pool_bytes, @intFromBool(call < 2) }, Probe.arguments[call]);
        try std.testing.expectEqual(stream.handle, Probe.streams[call]);
    }
    device.trace_mem = null;
    device.trace_checkpoint_fn = null;
    try device.traceOwnership(stream, true);
    try std.testing.expectEqual(@as(usize, 4), Probe.calls);
}

test "exl3: recent-demand planner state is separate from lifetime profile counts" {
    try std.testing.expectEqual(@as(usize, 3 * experts_n), recentOffset(10) - 20);
    try std.testing.expectEqual(@as(usize, recent_window * recent_words_per_row), miscOffset(10) - recentOffset(10));
    try std.testing.expectEqual(@as(usize, misc_words), stateWords(10) - miscOffset(10));
    try std.testing.expectEqual(stateWords(10), layout.activeOffset(10)); // an elastic pool's extra word comes after
}

test "exl3: setActive refuses a pool that is not elastic and counts outside the minimum and the capacity" {
    var resident: pool_.Pool = undefined;
    resident.opt = .{ .layer = 0, .lines = 20 };
    resident.active = 20;
    var pools = [_]*pool_.Pool{&resident};
    var device: DevPlan = undefined;
    device.pools = &pools;
    // every refusal comes before the planner pulls the residency or touches a stream
    try std.testing.expectError(error.Invalid, device.setActive(0, 12));
    resident.opt.elastic = true;
    try std.testing.expectError(error.Invalid, device.setActive(0, pool_.min_lines - 1));
    try std.testing.expectError(error.Invalid, device.setActive(0, 21));
    try std.testing.expectError(error.Invalid, device.setActive(1, 12));
    try std.testing.expectError(error.Invalid, resident.setActive(pool_.min_lines - 1));
    try std.testing.expectError(error.Invalid, resident.setActive(21));
    try std.testing.expectEqual(@as(usize, 20), device.active(0));
    try resident.setActive(20); // already there: nothing to do, nothing touched
}

test "exl3: demand trace planner arguments stay fixed and joined reads isolate pools" {
    const Probe = struct {
        var trace_address: u64 = 0;
        var launches: usize = 0;

        fn launch(function: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            launches += 1;
            if (@intFromPtr(function.?) == 1) {
                const address: *const u64 = @ptrCast(@alignCast(parameters.?[25].?));
                trace_address = address.*;
            }
            return 0;
        }

        fn download(destination: ?*anyopaque, source: u64, length: usize) callconv(.c) c_int {
            const target: [*]u8 = @ptrCast(destination.?);
            const origin: [*]const u8 = @ptrFromInt(source);
            @memcpy(target[0..length], origin[0..length]);
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Probe.launch;
    driver.api.cuMemcpyDtoH_v2 = Probe.download;
    const words = try std.testing.allocator.alloc(u32, 2 * demand_trace.pool_words);
    defer std.testing.allocator.free(words);
    for (0..2) |index| @memcpy(words[index * demand_trace.pool_words ..][0..demand_trace.header_words], &try demand_trace.header(index, index, 10, 16));
    var resident: pool_.Pool = undefined;
    resident.lines.tables.ptr = 4096;
    resident.lines.mem.ptr = 8192;
    resident.lines.tlay = cuda.exl3_pool.Tables.init(513);
    resident.lines.layout = cuda.exl3_pool.Layout.init(.{ 256, 256, 256 });
    var pools = [_]*pool_.Pool{ &resident, &resident };
    var states = [_]PoolState{ try PoolState.init(10, 0), try PoolState.init(10, stateWords(10)) };
    var device: DevPlan = undefined;
    device.pools = &pools;
    device.pool_states = &states;
    device.max_rows = 16;
    device.state.ptr = 16384;
    device.statics.ptr = 32768;
    device.work.ptr = 65536;
    device.plan_fn = .{ .d = &driver, .handle = @ptrFromInt(1) };
    device.copy_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    device.copy_blocks = 1;
    device.overlap = false;
    device.trace_mem = .{ .d = &driver, .ptr = @intFromPtr(words.ptr), .len = words.len * 4, .borrowed = true };
    const out = try std.testing.allocator.alloc(u32, demand_trace.pool_words);
    defer std.testing.allocator.free(out);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    Probe.launches = 0;
    for (0..2) |index| {
        for (0..2) |_| {
            try device.plan(stream, index, 1024, 1);
            try std.testing.expectEqual(@intFromPtr(words.ptr) + index * demand_trace.pool_bytes, Probe.trace_address);
        }
        const view = try device.readTraceAfterConsumers(index, out);
        try std.testing.expectEqual(@as(usize, 0), view.recorded());
        try std.testing.expectEqual(@as(u32, @intCast(index)), view.words[2]);
    }
    try std.testing.expectEqual(@as(usize, 8), Probe.launches);
    try std.testing.expectEqual(words.len * 4, device.traceDeviceBytes());
    try std.testing.expectError(error.Invalid, device.readTraceAfterConsumers(2, out));
    try std.testing.expectError(error.Invalid, device.readTraceAfterConsumers(0, out[1..]));
    device.trace_mem = null;
    try device.plan(stream, 0, 1024, 1);
    try std.testing.expectEqual(@as(u64, 0), Probe.trace_address);
    try std.testing.expectEqual(@as(usize, 0), device.traceDeviceBytes());
    try std.testing.expectError(error.TraceDisabled, device.readTraceAfterConsumers(0, out));
}

test "exl3: two engines in one process keep separate planner state" {
    const Layer = @import("../../cuda_prompt_load.zig").Layer;
    const driver: *const cuda.Driver = undefined; // never reached: no layer here has a pool
    var layers: [1]Layer = undefined;
    layers[0].exl3 = null;
    const plan_b: *DevPlan = @ptrFromInt(0x2000);
    const pool_b: *pool_.Pool = @ptrFromInt(0x3000);
    var a: dplan.State = .{};
    var b: dplan.State = .{ .plan = plan_b, .active = plan_b, .mtp_pool = pool_b };
    // engine a has no pools: it plans nothing, and engine b keeps the planner it made
    try std.testing.expectEqual(@as(?*DevPlan, null), try dplan.forLayers(std.testing.allocator, driver, &a, &layers));
    try std.testing.expectEqual(@as(?*DevPlan, plan_b), try dplan.forLayers(std.testing.allocator, driver, &b, &layers));
    try std.testing.expectEqual(@as(?*DevPlan, null), dplan.current(&a));
    try std.testing.expectEqual(@as(?*DevPlan, plan_b), dplan.current(&b));
    // a round of a enters and leaves its plan; b's active plan and head pool are untouched
    a.active = @ptrFromInt(0x1000);
    a.mtp_pool = @ptrFromInt(0x4000);
    try std.testing.expect(b.active.? == plan_b and b.mtp_pool.? == pool_b);
    a.active = null;
    try dplan.toHost(&a); // nothing planned: no device work, and b is not asked either
    dplan.release(&a);
    try std.testing.expectEqual(@as(?*pool_.Pool, null), a.mtp_pool);
    try std.testing.expect(b.active.? == plan_b and b.mtp_pool.? == pool_b and b.plan.? == plan_b);
}

const PlanProbe = struct {
    var args: [26]u64 = undefined;
    var launches: usize = 0;

    fn launch(function: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        if (@intFromPtr(function.?) != 1) return 0;
        launches += 1;
        for (0..26) |index| {
            const narrow = index == 1 or index == 3 or index == 4; // n, lines and the flags are c_int
            args[index] = if (narrow) @as(u64, @intCast(@as(*const c_int, @ptrCast(@alignCast(parameters.?[index].?))).*)) else @as(*const u64, @ptrCast(@alignCast(parameters.?[index].?))).*;
        }
        return 0;
    }
};

/// The planner's 26 launch arguments for a 20-line pool before and after it holds `active_after` lines (a fake launch).
fn planArgs(elastic: bool, active_after: usize) ![2][26]u64 {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = PlanProbe.launch;
    var resident: pool_.Pool = undefined;
    resident.opt = .{ .layer = 0, .lines = 20, .elastic = elastic };
    resident.active = 20;
    resident.lines.tables.ptr = 4096;
    resident.lines.mem = .{ .d = &driver, .ptr = 1 << 40, .len = 20 * 256 * 3, .borrowed = true };
    resident.lines.tlay = cuda.exl3_pool.Tables.init(513);
    resident.lines.layout = cuda.exl3_pool.Layout.init(.{ 256, 256, 256 });
    var pools = [_]*pool_.Pool{&resident};
    var states = [_]PoolState{try PoolState.initWith(20, 0, elastic)};
    var device: DevPlan = undefined;
    device.recent_eviction = false;
    device.pools = &pools;
    device.pool_states = &states;
    device.max_rows = 16;
    device.state.ptr = 16384;
    device.statics.ptr = 32768;
    device.work.ptr = 65536;
    device.plan_fn = .{ .d = &driver, .handle = @ptrFromInt(1) };
    device.copy_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    device.copy_blocks = 1;
    device.overlap = false;
    device.trace_mem = null;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    PlanProbe.launches = 0;
    try device.plan(stream, 0, 1024, 1);
    const before = PlanProbe.args;
    // what a shrink leaves behind: the count and the bytes behind the lines, never an address
    resident.active = active_after;
    resident.lines.mem.len = active_after * 256 * 3;
    try device.plan(stream, 0, 1024, 1);
    try std.testing.expectEqual(@as(usize, 2), PlanProbe.launches);
    try std.testing.expectEqual(active_after, device.active(0));
    return .{ before, PlanProbe.args };
}

test "exl3: no planner kernel argument depends on the active count, so a captured graph survives a resize" {
    const pair = try planArgs(true, 12);
    try std.testing.expectEqualSlices(u64, &pair[0], &pair[1]);
    try std.testing.expectEqual(@as(u64, 20), pair[1][3]); // the capacity, as at init
    try std.testing.expectEqual(@as(u64, 1 << 40), pair[1][14]); // the lines' base address
    try std.testing.expectEqual(@as(u64, 2), pair[1][4]); // the state ends with the active word
}

test "exl3: a pool that is not elastic launches the planner with exactly the arguments it had before" {
    const plain = try planArgs(false, 20);
    const elastic = try planArgs(true, 20);
    try std.testing.expectEqual(@as(u64, 0), plain[0][4]); // no flag bit
    // the one difference an elastic pool makes is bit 1 of the flags: the same 26 arguments otherwise
    for (plain[0], elastic[0], 0..) |a, b, index| if (index != 4) try std.testing.expectEqual(a, b);
    try std.testing.expectEqual(@as(usize, layout.stateWords(20)), (try PoolState.init(20, 0)).words);
    try std.testing.expectEqual(@as(usize, layout.stateWords(20) + 1), (try PoolState.initWith(20, 0, true)).words);
}
