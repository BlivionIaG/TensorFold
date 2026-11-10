//! The row a layer range takes in and hands on: f32 words, plain or in stamped 16-byte groups (three words and a
//! sequence stamp, as `split.wire` lays a stamped row out). Kernels q27_row_store and q27_row_load, copy.metal.
const std = @import("std");
const mtl = @import("metal");
const source = @import("qwen_runtime_sources");
const Ref = @import("projection.zig").Ref;

const Store = extern struct { rows: u32, width: u32, stamp: u32 };
const Load = extern struct { rows: u32, width: u32 };

pub const Rows = struct {
    store: mtl.Pipeline,
    load: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Rows {
        const lib = try mtl.Library.fromSource(device, source.copy, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const store = try mtl.Pipeline.init(device, lib, "q27_row_store", false);
        errdefer store.deinit();
        return .{ .store = store, .load = try mtl.Pipeline.init(device, lib, "q27_row_load", false) };
    }

    pub fn deinit(r: Rows) void {
        r.store.deinit();
        r.load.deinit();
    }

    /// Bytes `rows` rows of `width` words take: plain, or `stamped` in groups of three words.
    pub fn bytes(width: usize, rows: usize, stamped: bool) usize {
        return rows * if (stamped) (width + 2) / 3 * 16 else width * 4;
    }

    /// `out` (f32 words, in stamped groups when `stamp` is not 0) takes bf16 `hidden + pending`, the sum rounded to bf16.
    pub fn encodeStore(r: Rows, e: mtl.ComputeEncoder, hidden: Ref, pending: Ref, out: Ref, stamp: u32, width: u32, rows: u32) void {
        e.setPipeline(r.store);
        e.setBuffer(hidden.buffer, hidden.offset, 0);
        e.setBuffer(pending.buffer, pending.offset, 1);
        e.setValue(Store{ .rows = rows, .width = width, .stamp = stamp }, 2);
        e.setBuffer(out.buffer, out.offset, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// bf16 rows of `out` from the plain f32 words in `in`.
    pub fn encodeLoad(r: Rows, e: mtl.ComputeEncoder, in: Ref, out: Ref, width: u32, rows: u32) void {
        e.setPipeline(r.load);
        e.setBuffer(in.buffer, in.offset, 0);
        e.setValue(Load{ .rows = rows, .width = width }, 1);
        e.setBuffer(out.buffer, out.offset, 2);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }
};

/// Store then load `rows` rows of `width` on this Mac's GPU, and compare with the bf16 sums computed on the CPU.
fn roundTrip(width: u32, rows: u32, stamp: u32) !void {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const kernels = try Rows.init(device);
    defer kernels.deinit();
    const opts = mtl.ResourceOptions.shared;
    const n = @as(usize, width) * rows;
    const hidden = try device.buffer(n * 2, opts);
    defer hidden.deinit();
    const pending = try device.buffer(n * 2, opts);
    defer pending.deinit();
    const wire = try device.buffer(Rows.bytes(width, rows, stamp != 0), opts);
    defer wire.deinit();
    const back = try device.buffer(n * 2, opts);
    defer back.deinit();
    var rng = std.Random.DefaultPrng.init(0x5eed);
    const h = hidden.slice(u16, n);
    const p = pending.slice(u16, n);
    // moderate normal values: the GPU flushes denormals and a NaN's payload is not ours to preserve
    for (h, p) |*x, *y| for ([_]*u16{ x, y }) |v| {
        const r: u16 = @truncate(rng.random().int(u32) >> 8);
        v.* = (r & 0x807f) | (@as(u16, 0x3a + (r >> 8) % 12) << 7);
    };
    @memset(wire.contents()[0..wire.length()], 0xee);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    kernels.encodeStore(e, .{ .buffer = hidden }, .{ .buffer = pending }, .{ .buffer = wire }, stamp, width, rows);
    e.end();
    if (stamp == 0) {
        const e2 = cb.compute(.serial);
        kernels.encodeLoad(e2, .{ .buffer = wire }, .{ .buffer = back }, width, rows);
        e2.end();
    }
    cb.commit();
    cb.wait();
    const words = wire.slice(u32, wire.length() / 4);
    for (0..rows) |row| for (0..width) |c| {
        const sum = bf16(f32f(h[row * width + c]) + f32f(p[row * width + c]));
        const at = if (stamp == 0) row * width + c else (row * ((width + 2) / 3) + c / 3) * 4 + c % 3;
        try std.testing.expectEqual(@as(u32, sum) << 16, words[at]);
        if (stamp == 0) try std.testing.expectEqual(sum, back.slice(u16, n)[row * width + c]);
    };
    if (stamp != 0) for (0..rows * ((width + 2) / 3)) |g| try std.testing.expectEqual(stamp, words[g * 4 + 3]);
}

fn f32f(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Round to nearest even to bf16 bits, as Metal's bfloat conversion (finite values only).
fn bf16(v: f32) u16 {
    const x: u32 = @bitCast(v);
    return @truncate((x + 0x7fff + ((x >> 16) & 1)) >> 16);
}

test "the hand-off row is the bf16 sum widened, plain and in stamped groups, at widths that are not multiples of three" {
    try roundTrip(2560, 1, 0);
    try roundTrip(2560, 1, 9);
    try roundTrip(100, 3, 77);
    try roundTrip(100, 3, 0);
    try roundTrip(5120, 2, 0xfffffffe);
}
