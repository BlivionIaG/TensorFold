//! Wake latency of a parked command buffer, for `offload probe`: the time from the host opening a relay gate to the
//! buffer's done signal reaching the host, plus the clock helpers the probe CLI times its own calls with.
const std = @import("std");
const relay_mod = @import("relay.zig");
const device_mod = @import("device.zig");
const command = @import("command.zig");
const library = @import("library.zig");
const types = @import("types.zig");

const Device = device_mod.Device;
const Relay = relay_mod.Relay;

pub const Error = relay_mod.Error || library.Error || error{Timeout};

/// How long the gated buffer gets to park on its event before the clock starts (as the offline probe waited).
const park_ns: u64 = 300_000;
const warmup = 8;
const timeout_ms = 5000;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void nop(device uint* p [[buffer(0)]], uint t [[thread_position_in_grid]]) { if (t == 0) p[0] += 1; }
;

/// Monotonic nanoseconds (CLOCK_UPTIME_RAW, the mach_absolute_time clock Metal's GPU timestamps use).
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Sleep at least `ns` nanoseconds.
pub fn sleepNs(ns: u64) void {
    const req: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&req, null);
}

const Nop = struct {
    pipeline: library.Pipeline,
    scratch: device_mod.Buffer,

    fn encode(ctx: *anyopaque, cb: command.CommandBuffer) void {
        const self: *Nop = @ptrCast(@alignCast(ctx));
        const enc = cb.compute(.serial);
        enc.setPipeline(self.pipeline);
        enc.setBuffer(self.scratch, 0, 0);
        enc.dispatchThreads(types.Size.of(1, 1, 1), types.Size.of(1, 1, 1));
        enc.end();
    }
};

/// Fill `out[0..iters]` with the nanoseconds from `Relay.open` to `Relay.waitDone` of a parked one-thread kernel,
/// after a few unrecorded warm-up steps. Each step parks for 300 us first, so the number is the wake of an
/// already-waiting buffer, not its submit. error.Timeout when a step does not finish within 5 s.
pub fn wake(device: Device, iters: usize, out: []u64) Error!void {
    std.debug.assert(out.len >= iters);
    const lib = try library.Library.fromSource(device, source, .{});
    defer lib.deinit();
    var job: Nop = .{
        .pipeline = try library.Pipeline.init(device, lib, "nop", false),
        .scratch = try device.buffer(std.heap.pageSize(), types.ResourceOptions.shared),
    };
    defer job.pipeline.deinit();
    defer job.scratch.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var relay = try Relay.init(device, queue);
    defer relay.deinit();
    for (0..warmup + iters) |i| {
        const ticket = try relay.submit(&job, Nop.encode);
        sleepNs(park_ns);
        const start = nowNs();
        relay.open(ticket);
        if (!relay.waitDone(ticket, timeout_ms)) return error.Timeout;
        const elapsed = nowNs() - start;
        if (i >= warmup) out[i - warmup] = elapsed;
    }
}

test "wake latency of a parked buffer stays far below 5 ms at p50" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var times: [500]u64 = undefined;
    try wake(device, times.len, &times);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    // other builds load this machine, so only a generous bound is asserted; the probe CLI reports the real numbers
    std.debug.print("\nrelay open -> waitDone over {d} steps: p10 {d:.1} us, p50 {d:.1} us, p90 {d:.1} us, p99 {d:.1} us, max {d:.1} us\n", .{
        times.len,
        @as(f64, @floatFromInt(times[times.len / 10])) / 1e3,
        @as(f64, @floatFromInt(times[times.len / 2])) / 1e3,
        @as(f64, @floatFromInt(times[times.len * 9 / 10])) / 1e3,
        @as(f64, @floatFromInt(times[times.len * 99 / 100])) / 1e3,
        @as(f64, @floatFromInt(times[times.len - 1])) / 1e3,
    });
    try std.testing.expect(times[times.len / 2] > 0);
    try std.testing.expect(times[times.len / 2] < 5_000_000);
}

test "wake fills exactly the requested prefix" {
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var out: [8]u64 = @splat(0);
    try wake(device, 5, &out);
    for (out[0..5]) |ns| try std.testing.expect(ns > 0);
    for (out[5..]) |ns| try std.testing.expectEqual(@as(u64, 0), ns);
}

test "nowNs advances and sleepNs waits at least as long as asked" {
    const a = nowNs();
    sleepNs(2_000_000);
    try std.testing.expect(nowNs() - a >= 2_000_000);
}
