//! The producer's read of the GLP fold site: the post-fold mean over stream copies, summed per block.
//! A capture run loads no plan, so the undosed stream is what it sums; the served path never installs one.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// The kernels' `TfGlpCaptureArgs` record, field for field.
pub const Args = extern struct { dim: u32, rows: u32, last_only: u32, pad: u32 = 0 };

pub const threads: u32 = 256;

pub const Error = error{ BadCaptureShape, BadCaptureLayer };

/// One fp32 row a backbone block, accumulated over the prompts of a set; the host resets it per prompt.
pub const Capture = struct {
    allocator: std.mem.Allocator,
    layers: u32,
    hidden: u32,
    /// every row of a chunk (false), or the last alone (true), where the model decides what to say next
    last_only: bool,
    /// the sums of every block's row, [layers, hidden] fp32, one shared buffer the host reads after a forward
    buf: mtl.Buffer,
    pipeline: mtl.Pipeline,

    /// Capture state for `layers` blocks of `hidden`: every row of a chunk, or the last one alone.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, layers: u32, hidden: u32, last_only: bool) !*Capture {
        if (layers == 0 or hidden == 0) return error.BadCaptureShape;
        const p = try a.create(Capture);
        errdefer a.destroy(p);
        const buf = try device.buffer(@as(usize, layers) * hidden * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buf.deinit();
        const lib = try mtl.Library.fromSource(device, sources.dsv41.glp, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_dsv41_glp_capture", false);
        @memset(buf.slice(f32, @as(usize, layers) * hidden), 0);
        p.* = .{ .allocator = a, .layers = layers, .hidden = hidden, .last_only = last_only, .buf = buf, .pipeline = pipe };
        return p;
    }

    pub fn deinit(c: *Capture) void {
        c.pipeline.deinit();
        c.buf.deinit();
        c.allocator.destroy(c);
    }

    /// The sums join the model's residency set (the caller commits the set).
    pub fn registerResidency(c: *const Capture, set: mtl.ResidencySet) void {
        set.add(c.buf);
    }

    /// Zeroes every block's row, so one prompt's sums stand alone.
    pub fn reset(c: *Capture) void {
        @memset(c.buf.slice(f32, @as(usize, c.layers) * c.hidden), 0);
    }

    /// Adds block `layer`'s post-fold stream `x` [rows][4][hidden] bf16 at `off` to that block's row.
    pub fn record(c: *Capture, e: mtl.ComputeEncoder, layer: u32, x: mtl.Buffer, off: usize, rows: u32) void {
        if (layer >= c.layers or rows == 0) return;
        e.setPipeline(c.pipeline);
        e.setBuffer(x, off, 0);
        e.setBuffer(c.buf, @as(usize, layer) * c.hidden * 4, 1);
        e.setValue(Args{ .dim = c.hidden, .rows = rows, .last_only = @intFromBool(c.last_only) }, 2);
        e.dispatchGroups(.{ .width = 1, .height = 1 }, .{ .width = threads, .height = 1 });
    }

    /// Every block's row as the host reads it, [layers, hidden] fp32: only valid once the queue has completed.
    pub fn sums(c: *const Capture) []const f32 {
        return c.buf.slice(f32, @as(usize, c.layers) * c.hidden);
    }

    /// One block's row of sums.
    pub fn row(c: *const Capture, layer: u32) []const f32 {
        return c.sums()[@as(usize, layer) * c.hidden ..][0..c.hidden];
    }
};

const testing = std.testing;

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn real(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

test "the capture sums the fold's mean over the four copies, every row or the last alone" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const a = testing.allocator;
    const layers: u32 = 3;
    const hidden: u32 = 64;
    const rows: u32 = 5;
    // A synthetic four-copy stream: copy c of row r column i holds (r + c + i) / 8, rounded to bf16.
    const xs = try a.alloc(u16, @as(usize, rows) * 4 * hidden);
    defer a.free(xs);
    for (0..rows) |r| {
        for (0..4) |cp| {
            for (0..hidden) |i| xs[(r * 4 + cp) * hidden + i] = bf16(@as(f32, @floatFromInt(r + cp + i)) / 8);
        }
    }
    const x = try device.buffer(xs.len * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memcpy(x.slice(u8, xs.len * 2), std.mem.sliceAsBytes(xs));

    // What the kernel must produce, in float64, over the bf16 values it reads.
    const per_row = try a.alloc(f64, @as(usize, rows) * hidden);
    defer a.free(per_row);
    for (0..rows) |r| {
        for (0..hidden) |i| {
            var s: f64 = 0;
            for (0..4) |cp| s += real(xs[(r * 4 + cp) * hidden + i]) * 0.25;
            per_row[r * hidden + i] = s;
        }
    }

    for ([_]bool{ false, true }) |last_only| {
        const c = try Capture.init(a, device, layers, hidden, last_only);
        defer c.deinit();
        try testing.expectEqualSlices(f32, &[_]f32{0}, c.row(0)[0..1]);
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..layers) |l| c.record(enc, @intCast(l), x, 0, rows);
        enc.end();
        cb.commit();
        cb.wait();
        try testing.expect(cb.failure() == null);
        for (0..layers) |l| {
            const got = c.row(@intCast(l));
            for (0..hidden) |i| {
                var want: f64 = 0;
                if (last_only) {
                    want = per_row[(rows - 1) * hidden + i];
                } else {
                    for (0..rows) |r| want += per_row[r * hidden + i];
                }
                try testing.expectApproxEqAbs(want, got[i], 1e-2 * @max(1, @abs(want)));
            }
        }
        // A reset drops the whole table.
        c.reset();
        try testing.expectEqualSlices(f32, &[_]f32{0}, c.row(layers - 1)[0..1]);
    }
    // A block past the table is left alone rather than misaddressed, and the stream is never written.
    const c = try Capture.init(a, device, layers, hidden, false);
    defer c.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    c.record(enc, layers, x, 0, rows);
    enc.end();
    cb.commit();
    cb.wait();
    for (c.sums()) |v| try testing.expectEqual(@as(f32, 0), v);
    try testing.expectEqualSlices(u16, xs, x.slice(u16, xs.len));
}
