//! Random packed products and device helpers the kernel groups share: seeded data, device compares and timing.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const Gpu = check.Gpu;

pub const Rng = struct {
    state: u64,

    pub fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    /// A value in [-1, 1) with a few mantissa bits, so activation sums stay far from overflow.
    pub fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }
};

pub fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

pub fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

pub fn fill(gpa: std.mem.Allocator, comptime T: type, n: usize, rng: *Rng, make: *const fn (*Rng) T) ![]T {
    const host = try gpa.alloc(T, n);
    for (host) |*v| v.* = make(rng);
    return host;
}

pub fn toDevice(gpu: Gpu, host: anytype) !hip.DeviceBuffer {
    return hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}

pub fn makeWord(r: *Rng) u32 {
    return @truncate(r.next() >> 16);
}

pub fn makeScale(r: *Rng) u16 {
    return bf16Bits((r.unit() + 1.0) / 16.0);
}

pub fn makeBias(r: *Rng) u16 {
    return bf16Bits(r.unit() / 2.0);
}

/// A full-mantissa activation in [-1, 1) at one of six scales: sums of products then round, in any order.
pub fn fine(r: *Rng) f32 {
    const v = r.next();
    const m: i32 = @intCast(v >> 41 & 0x3fffff);
    return @as(f32, @floatFromInt(m - 0x200000)) / 2097152.0 * std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(v % 6)));
}

pub fn makeX16(r: *Rng) u16 {
    return f16Bits(fine(r));
}

pub fn makeXB(r: *Rng) u16 {
    return bf16Bits(fine(r));
}

/// Words of two device buffers that differ, compared through 32 MiB windows.
pub fn sameOnDevice(gpu: Gpu, a: hip.DeviceBuffer, b: hip.DeviceBuffer, len: usize) !usize {
    const win = 32 << 20;
    const ha = try gpu.gpa.alloc(u8, win);
    defer gpu.gpa.free(ha);
    const hb = try gpu.gpa.alloc(u8, win);
    defer gpu.gpa.free(hb);
    var diff: usize = 0;
    var at: usize = 0;
    while (at < len) : (at += win) {
        const n = @min(win, len - at);
        try a.download(at, ha[0..n]);
        try b.download(at, hb[0..n]);
        var i: usize = 0;
        while (i + 4 <= n) : (i += 4) {
            if (!std.mem.eql(u8, ha[i..][0..4], hb[i..][0..4])) diff += 1;
        }
    }
    return diff;
}

pub const Timing = struct { best: f32, median: f32 };

pub fn time(gpu: Gpu, stream: hip.Stream, reps: usize, ctx: anytype, comptime go: fn (@TypeOf(ctx)) hip.Error!void) !Timing {
    try go(ctx);
    try stream.synchronize();
    var start = try hip.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try hip.Event.init(gpu.d, true);
    defer stop.deinit();
    const ms = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(ms);
    for (ms) |*t| {
        try start.record(stream);
        try go(ctx);
        try stop.record(stream);
        try stop.synchronize();
        t.* = try hip.Event.elapsedMs(start, stop);
    }
    const best = std.mem.min(f64, ms);
    return .{ .best = @floatCast(best), .median = @floatCast(check.median(ms)) };
}
