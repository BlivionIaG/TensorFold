//! How a handoff's page rows split into DATA frames, and how a frame's rows are gathered from a cache tensor.
const std = @import("std");
const row_stage = @import("row_stage.zig");
const Allocator = std.mem.Allocator;

/// One DATA frame: rows [row_start, row_start + rows) of the export's layer at `position`.
pub const Frame = struct { position: u32, row_start: u64, rows: u64 };

/// A layer's rows of `row_bytes` each, or (protocol 2, recurrent) parts of `sizes[i]` bytes, a frame each.
pub const LayerRows = struct { rows: u64, row_bytes: u64, sizes: []const u64 = &.{} };

/// export.py's plan_frames: each layer's rows in order, as many whole rows a frame as `capacity` holds.
pub fn plan(gpa: Allocator, layers: []const LayerRows, capacity: u64) ![]Frame {
    var out: std.ArrayList(Frame) = .empty;
    errdefer out.deinit(gpa);
    for (layers, 0..) |l, position| {
        if (l.sizes.len > 0) {
            for (l.sizes, 0..) |n, part| {
                if (n > capacity) return error.PartTooLarge;
                try out.append(gpa, .{ .position = @intCast(position), .row_start = part, .rows = 1 });
            }
            continue;
        }
        const per_frame = capacity / l.row_bytes;
        if (per_frame < 1) return error.RowTooLarge;
        var start: u64 = 0;
        while (start < l.rows) : (start += per_frame) {
            try out.append(gpa, .{ .position = @intCast(position), .row_start = start, .rows = @min(per_frame, l.rows - start) });
        }
    }
    return out.toOwnedSlice(gpa);
}

/// What a producer's frame waits on: the rows (or latch) it names, the producer's failure, or its stall.
fn awaited(st: *const row_stage.Stage, end: u64, wait_ns: u64) !void {
    switch (st.wait(end, wait_ns, null)) {
        .ready => {},
        .failed => return error.ProducerFailed,
        .timeout, .aborted => return error.ProducerStalled,
    }
}

/// A cache tensor in host-visible memory (a unified-memory Metal buffer, or mapped host memory) and its exported rows.
pub const Source = struct {
    bytes: []const u8,
    shape: []const u64,
    block_dim: usize,
    elem: u64,
    rows: []const u64,
    parts: []const []const u8 = &.{}, // a recurrent layer's state, part by part: gather copies one whole
    /// rows still being produced (the layer-split boundary stream, row-major in `rows * row_bytes`)
    stage: ?*row_stage.Stage = null,
    /// state still being produced: gather waits until this latch (one row) is published
    latch: ?*row_stage.Stage = null,
    /// how long gather waits for either before the producer is called stalled
    wait_ns: u64 = 120 * std.time.ns_per_s,

    /// Bytes of one block row: the dimensions after the block one.
    fn inner(s: Source) u64 {
        var n = s.elem;
        for (s.shape[s.block_dim + 1 ..]) |d| n *= d;
        return n;
    }

    fn outer(s: Source) u64 {
        var n: u64 = 1;
        for (s.shape[0..s.block_dim]) |d| n *= d;
        return n;
    }

    /// Bytes `gather` writes for `count` rows.
    pub fn frameBytes(s: Source, count: u64) u64 {
        return s.outer() * count * s.inner();
    }

    /// index_select(block_dim, rows[start..start+count]).contiguous(), as the producer's fill copies it.
    pub fn gather(s: Source, start: u64, count: u64, out: []u8) !u64 {
        if (s.latch) |l| try awaited(l, 1, s.wait_ns);
        if (s.stage) |st| {
            if (count == 0 or start + count > st.rows) return error.RowOutOfRange;
            if (count * st.row_bytes > out.len) return error.FrameTooLarge;
            try awaited(st, start + count, s.wait_ns);
            @memcpy(out[0..@intCast(count * st.row_bytes)], st.slice(start, count));
            return count * st.row_bytes;
        }
        if (s.parts.len > 0) {
            if (count != 1 or start >= s.parts.len) return error.RowOutOfRange;
            const p = s.parts[@intCast(start)];
            if (p.len > out.len) return error.FrameTooLarge;
            @memcpy(out[0..p.len], p);
            return p.len;
        }
        const inner_bytes = s.inner();
        const outer_n = s.outer();
        const blocks = s.shape[s.block_dim];
        const need = outer_n * count * inner_bytes;
        if (need > out.len or start + count > s.rows.len) return error.FrameTooLarge;
        var at: usize = 0;
        for (0..outer_n) |o| {
            for (s.rows[start..][0..count]) |r| {
                if (r >= blocks) return error.RowOutOfRange;
                const from: usize = @intCast((o * blocks + r) * inner_bytes);
                @memcpy(out[at..][0..@intCast(inner_bytes)], s.bytes[from..][0..@intCast(inner_bytes)]);
                at += @intCast(inner_bytes);
            }
        }
        return need;
    }
};

test "frames split layers to fit the reply half, as plan_frames does" {
    const gpa = std.testing.allocator;
    const got = try plan(gpa, &.{ .{ .rows = 5, .row_bytes = 100 }, .{ .rows = 2, .row_bytes = 100 } }, 250);
    defer gpa.free(got);
    const want = [_]Frame{ .{ .position = 0, .row_start = 0, .rows = 2 }, .{ .position = 0, .row_start = 2, .rows = 2 }, .{ .position = 0, .row_start = 4, .rows = 1 }, .{ .position = 1, .row_start = 0, .rows = 2 } };
    try std.testing.expectEqualSlices(Frame, &want, got);
    try std.testing.expectError(error.RowTooLarge, plan(gpa, &.{.{ .rows = 1, .row_bytes = 300 }}, 250));
}

test "gather selects block rows along any dimension, C-contiguous" {
    var data: [2 * 4 * 3]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @intCast(i);
    const s: Source = .{ .bytes = &data, .shape = &.{ 2, 4, 3 }, .block_dim = 1, .elem = 1, .rows = &.{ 3, 1 } };
    var out: [12]u8 = undefined;
    try std.testing.expectEqual(@as(u64, 12), try s.gather(0, 2, &out));
    try std.testing.expectEqualSlices(u8, &.{ 9, 10, 11, 3, 4, 5, 21, 22, 23, 15, 16, 17 }, &out);
    try std.testing.expectEqual(@as(u64, 6), try s.gather(1, 1, &out));
    try std.testing.expectEqualSlices(u8, &.{ 3, 4, 5, 15, 16, 17 }, out[0..6]);
}

test "a recurrent layer's parts go a frame each, whole, and one past the reply half is refused" {
    const gpa = std.testing.allocator;
    const got = try plan(gpa, &.{ .{ .rows = 2, .row_bytes = 0, .sizes = &.{ 200, 40 } }, .{ .rows = 3, .row_bytes = 100 } }, 250);
    defer gpa.free(got);
    const want = [_]Frame{ .{ .position = 0, .row_start = 0, .rows = 1 }, .{ .position = 0, .row_start = 1, .rows = 1 }, .{ .position = 1, .row_start = 0, .rows = 2 }, .{ .position = 1, .row_start = 2, .rows = 1 } };
    try std.testing.expectEqualSlices(Frame, &want, got);
    try std.testing.expectError(error.PartTooLarge, plan(gpa, &.{.{ .rows = 1, .row_bytes = 0, .sizes = &.{300} }}, 250));
    const a = [_]u8{ 1, 2, 3 };
    const b = [_]u8{ 9, 8 };
    const s: Source = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = &.{ &a, &b } };
    var out: [4]u8 = undefined;
    try std.testing.expectEqual(@as(u64, 2), try s.gather(1, 1, &out));
    try std.testing.expectEqualSlices(u8, &b, out[0..2]);
    try std.testing.expectError(error.RowOutOfRange, s.gather(0, 2, &out));
}

test "a staged source gathers rows as they are published, and a failed or silent producer fails the frame" {
    var st = try row_stage.Stage.init(std.testing.allocator, 4, 2);
    defer st.deinit();
    const s: Source = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .stage = &st, .wait_ns = 2_000_000 };
    var out: [8]u8 = undefined;
    try std.testing.expectError(error.ProducerStalled, s.gather(0, 1, &out));
    try st.put(0, &.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(u64, 4), try s.gather(0, 2, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, out[0..4]);
    try std.testing.expectError(error.ProducerStalled, s.gather(2, 1, &out));
    try std.testing.expectError(error.RowOutOfRange, s.gather(3, 2, &out));
    try std.testing.expectError(error.FrameTooLarge, s.gather(0, 5, out[0..2]));
    st.fail();
    try std.testing.expectError(error.ProducerFailed, s.gather(2, 1, &out));
}

test "a latched parts source waits for its state and then copies a part whole" {
    var latch = try row_stage.Stage.init(std.testing.allocator, 1, 0);
    defer latch.deinit();
    var a = [_]u8{ 0, 0 };
    const s: Source = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = &.{&a}, .latch = &latch, .wait_ns = 2_000_000 };
    var out: [4]u8 = undefined;
    try std.testing.expectError(error.ProducerStalled, s.gather(0, 1, &out));
    a = .{ 7, 9 };
    try latch.publish(1);
    try std.testing.expectEqual(@as(u64, 2), try s.gather(0, 1, &out));
    try std.testing.expectEqualSlices(u8, &.{ 7, 9 }, out[0..2]);
}
