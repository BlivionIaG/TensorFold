//! Device expert-line pool for one layer of EXL3 offload.
//! A line holds one routed expert: gate, up and down, each as strips-order trellis, suh and svh.
//! The nine regions are separate device buffers so `Exl3.hit` addresses them as three `Packed` bases
//! (line `i` is `base + i * stride`). The host fills a pinned staging line, then `enqueue` copies the nine
//! regions on a dedicated copy stream and records the line's event. A line is usable once its event is done.
//! Which expert sits in which line is the policy's business (`exl3_offload.zig` in flashnext).

const std = @import("std");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const exl3 = @import("exl3.zig");

const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const DeviceBuffer = memory.DeviceBuffer;
const HostBuffer = memory.HostBuffer;
const Stream = stream_.Stream;
const Event = stream_.Event;

/// One matrix of the expert: K x N with `halves` (trellis int16 words per tile = halves * 8).
pub const Mat = struct { k: usize, n: usize, halves: u8 };

pub const Spec = struct { gate: Mat, up: Mat, down: Mat };

const fp16_bytes: usize = 2;

pub const Region = struct {
    trellis_off: usize,
    trellis_bytes: usize,
    suh_off: usize,
    suh_bytes: usize,
    svh_off: usize,
    svh_bytes: usize,
};

/// Byte layout of one staged line: gate (trellis, suh, svh), then up, then down.
pub const Layout = struct {
    regions: [3]Region,
    line_bytes: usize,

    pub fn init(spec: Spec) error{Exl3Shape}!Layout {
        const mats = [3]Mat{ spec.gate, spec.up, spec.down };
        var l: Layout = .{ .regions = undefined, .line_bytes = 0 };
        for (mats, &l.regions) |m, *r| {
            const span = try exl3.tileSpan(m.k, m.n, m.halves);
            r.trellis_off = l.line_bytes;
            r.trellis_bytes = span * fp16_bytes;
            r.suh_off = r.trellis_off + r.trellis_bytes;
            r.suh_bytes = m.k * fp16_bytes;
            r.svh_off = r.suh_off + r.suh_bytes;
            r.svh_bytes = m.n * fp16_bytes;
            l.line_bytes = r.svh_off + r.svh_bytes;
        }
        return l;
    }
};

/// Device buffers for one matrix: `lines` rows of each region.
const Bufs = struct {
    trellis: DeviceBuffer,
    suh: DeviceBuffer,
    svh: DeviceBuffer,

    fn free(b: *Bufs) void {
        b.trellis.free();
        b.suh.free();
        b.svh.free();
    }
};

pub const Pool = struct {
    d: *const Driver,
    spec: Spec,
    layout: Layout,
    lines: usize,
    bufs: [3]Bufs,
    staging: HostBuffer,
    copy_stream: Stream,
    events: []Event,
    /// True from `enqueue` until `wait` returns for that line.
    pending: []bool,
    gpa: std.mem.Allocator,
    copied_bytes: u64 = 0,
    copies: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const Driver, spec: Spec, lines: usize) !Pool {
        if (lines == 0) return error.Exl3Shape;
        const layout = try Layout.init(spec);
        const mats = [3]Mat{ spec.gate, spec.up, spec.down };
        var bufs: [3]Bufs = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (mats, &bufs) |m, *b| {
            const span = try exl3.tileSpan(m.k, m.n, m.halves);
            b.trellis = try DeviceBuffer.alloc(d, lines * span * fp16_bytes);
            errdefer b.trellis.free();
            b.suh = try DeviceBuffer.alloc(d, lines * m.k * fp16_bytes);
            errdefer b.suh.free();
            b.svh = try DeviceBuffer.alloc(d, lines * m.n * fp16_bytes);
            made += 1;
        }
        var staging = try HostBuffer.alloc(d, lines * layout.line_bytes);
        errdefer staging.free();
        var cs = try Stream.init(d, true);
        errdefer cs.deinit();
        const events = try gpa.alloc(Event, lines);
        errdefer gpa.free(events);
        var ev_made: usize = 0;
        errdefer for (events[0..ev_made]) |*e| e.deinit();
        for (events) |*e| {
            e.* = try Event.init(d, false);
            ev_made += 1;
        }
        const pending = try gpa.alloc(bool, lines);
        @memset(pending, false);
        return .{ .d = d, .spec = spec, .layout = layout, .lines = lines, .bufs = bufs, .staging = staging, .copy_stream = cs, .events = events, .pending = pending, .gpa = gpa };
    }

    pub fn deinit(p: *Pool) void {
        for (p.events) |*e| e.deinit();
        p.gpa.free(p.events);
        p.gpa.free(p.pending);
        p.copy_stream.deinit();
        p.staging.free();
        for (&p.bufs) |*b| b.free();
        p.* = undefined;
    }

    /// Device bytes held by the line buffers.
    pub fn deviceBytes(p: Pool) usize {
        return p.lines * p.layout.line_bytes;
    }

    /// The pinned bytes of one line, laid out per `layout`. Fill all nine regions, then `enqueue`.
    /// Do not touch a line whose copy is pending.
    pub fn stagingLine(p: *Pool, line: usize) []u8 {
        return p.staging.bytes[line * p.layout.line_bytes ..][0..p.layout.line_bytes];
    }

    /// The three packed bases for `Exl3.hit`. Index 0 gate, 1 up, 2 down.
    pub fn packed_(p: Pool, which: usize) !exl3.Packed {
        const m = switch (which) {
            0 => p.spec.gate,
            1 => p.spec.up,
            2 => p.spec.down,
            else => return error.Exl3Shape,
        };
        const b = p.bufs[which];
        return .{
            .trellis = b.trellis.ptr,
            .suh = b.suh.ptr,
            .svh = b.svh.ptr,
            .trellis_stride = try exl3.tileSpan(m.k, m.n, m.halves),
            .suh_stride = m.k,
            .svh_stride = m.n,
            .halves = m.halves,
            .codebook = 2,
        };
    }

    /// Queues the nine region copies of a staged line on the copy stream, after `after` when given
    /// (the compute stream's last use of the line being replaced), and records the line's event.
    pub fn enqueue(p: *Pool, line: usize, after: ?Event) !void {
        if (line >= p.lines) return error.Exl3Shape;
        if (p.pending[line]) return error.Busy;
        if (after) |e| try p.copy_stream.wait(e);
        const base = p.stagingLine(line);
        const mats = [3]Mat{ p.spec.gate, p.spec.up, p.spec.down };
        for (mats, p.bufs, p.layout.regions) |m, b, r| {
            const span = r.trellis_bytes / fp16_bytes;
            try b.trellis.uploadAsync(line * span * fp16_bytes, base[r.trellis_off..][0..r.trellis_bytes], p.copy_stream.handle);
            try b.suh.uploadAsync(line * m.k * fp16_bytes, base[r.suh_off..][0..r.suh_bytes], p.copy_stream.handle);
            try b.svh.uploadAsync(line * m.n * fp16_bytes, base[r.svh_off..][0..r.svh_bytes], p.copy_stream.handle);
        }
        try p.events[line].record(p.copy_stream);
        p.pending[line] = true;
        p.copied_bytes += p.layout.line_bytes;
        p.copies += 1;
    }

    /// True when the line has no copy in flight or its copy finished. Does not block.
    pub fn ready(p: *Pool, line: usize) !bool {
        if (!p.pending[line]) return true;
        return p.events[line].done();
    }

    /// Blocks until the line's copy finished. Afterwards the staging bytes may be rewritten.
    pub fn wait(p: *Pool, line: usize) !void {
        if (!p.pending[line]) return;
        try p.events[line].synchronize();
        p.pending[line] = false;
    }

    /// Makes `compute` wait for the line's copy on the device without blocking the host.
    pub fn gate(p: *Pool, compute: Stream, line: usize) !void {
        if (p.pending[line]) try compute.wait(p.events[line]);
    }
};

test "layout sums the nine regions of a line" {
    const spec = Spec{
        .gate = .{ .k = 2560, .n = 640, .halves = 4 },
        .up = .{ .k = 2560, .n = 640, .halves = 5 },
        .down = .{ .k = 640, .n = 2560, .halves = 4 },
    };
    const l = try Layout.init(spec);
    const gate_t = (2560 / 16) * (640 / 16) * 4 * 8 * 2;
    try std.testing.expectEqual(@as(usize, gate_t), l.regions[0].trellis_bytes);
    try std.testing.expectEqual(@as(usize, 2560 * 2), l.regions[0].suh_bytes);
    try std.testing.expectEqual(@as(usize, 640 * 2), l.regions[0].svh_bytes);
    try std.testing.expectEqual(l.regions[0].svh_off + l.regions[0].svh_bytes, l.regions[1].trellis_off);
    try std.testing.expectEqual(l.regions[2].svh_off + l.regions[2].svh_bytes, l.line_bytes);
    try std.testing.expectError(error.Exl3Shape, Layout.init(.{ .gate = .{ .k = 2560, .n = 600, .halves = 4 }, .up = spec.up, .down = spec.down }));
}
