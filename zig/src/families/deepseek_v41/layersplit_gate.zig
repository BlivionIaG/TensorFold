//! Layer-split gate G3: enter the forward at layer `k` from another engine's boundary rows, or stop and dump them.
const std = @import("std");
const mtl = @import("metal");
const View = @import("view.zig").View;

pub const hidden_row: usize = 4 * 5120 * 2; // the four residual planes, bf16
pub const pre_row: usize = 4 * 4; // the coefficient set, f32

pub const Gate = struct {
    /// enter at `k` (layers before it and the embedding are skipped) reading `<dir>/<prefix>L<k-1>.{x.bf16,pre.f32}`
    entry_dir: ?[]const u8 = null,
    entry_prefix: []const u8 = "",
    /// enter at `k` from the rows staged in this buffer (layer-split prefill): `rows` of hidden_row, then of pre_row
    entry_buf: ?mtl.Buffer = null,
    /// stop before layer `stop_k` (post applied) and write `<dir>/M.L<stop_k-1>.*` (an own file set)
    stop_dir: ?[]const u8 = null,
    stop_k: usize = 0,
    k: usize = 0,

    pub fn off(g: *Gate) void {
        g.* = .{};
    }

    fn path(buf: []u8, dir: []const u8, prefix: []const u8, layer: usize, ext: []const u8) ![*:0]const u8 {
        const s = try std.fmt.bufPrint(buf, "{s}/{s}L{d}.{s}\x00", .{ dir, prefix, layer, ext });
        return @ptrCast(s.ptr);
    }

    fn read(p: [*:0]const u8, dst: []u8, at: usize) !void {
        const fd = std.c.open(p, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.GateFile;
        defer _ = std.c.close(fd);
        if (std.c.pread(fd, dst.ptr, dst.len, @intCast(at)) != dst.len) return error.GateFileShort;
    }

    fn write(p: [*:0]const u8, src: []const u8, at: usize) !void {
        const fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.GateFile;
        defer _ = std.c.close(fd);
        if (std.c.pwrite(fd, src.ptr, src.len, @intCast(at)) != src.len) return error.GateFileShort;
    }

    /// Whether the forward enters at layer `k` (from files or from memory) instead of from the embedding.
    pub fn entered(g: *const Gate) bool {
        return g.entry_dir != null or g.entry_buf != null;
    }

    /// The rows [p0, p0 + rows) of the entry files into the stream and pre views.
    pub fn loadEntry(g: *const Gate, stream: View, pre: View, p0: usize, rows: usize) !void {
        const x = stream.buffer.contents()[stream.offset..][0 .. rows * hidden_row];
        const q = pre.buffer.contents()[pre.offset..][0 .. rows * pre_row];
        var b: [512]u8 = undefined;
        try read(try path(&b, g.entry_dir.?, g.entry_prefix, g.k - 1, "x.bf16"), x, p0 * hidden_row);
        try read(try path(&b, g.entry_dir.?, g.entry_prefix, g.k - 1, "pre.f32"), q, p0 * pre_row);
    }

    /// Memory rows ([row][stream, then pre]) split into `x` ([row][stream]) and `q` ([row][pre]).
    pub fn deinterleave(src: []const u8, x: []u8, q: []u8, rows: usize) !void {
        if (src.len != rows * (hidden_row + pre_row) or x.len != rows * hidden_row or q.len != rows * pre_row) return error.GateRows;
        for (0..rows) |r| {
            const row = src[r * (hidden_row + pre_row) ..];
            @memcpy(x[r * hidden_row ..][0..hidden_row], row[0..hidden_row]);
            @memcpy(q[r * pre_row ..][0..pre_row], row[hidden_row..][0..pre_row]);
        }
    }

    /// The same two views' rows into the `M.` file set.
    pub fn dumpStop(g: *const Gate, stream: View, pre: View, p0: usize, rows: usize) !void {
        var b: [512]u8 = undefined;
        const x = stream.buffer.contents()[stream.offset..][0 .. rows * hidden_row];
        try write(try path(&b, g.stop_dir.?, "M.", g.stop_k - 1, "x.bf16"), x, p0 * hidden_row);
        const q = pre.buffer.contents()[pre.offset..][0 .. rows * pre_row];
        try write(try path(&b, g.stop_dir.?, "M.", g.stop_k - 1, "pre.f32"), q, p0 * pre_row);
    }
};

test "gate: boundary rows in memory split into the stream and the pre, and a row count they do not match is refused" {
    const rows = 5;
    var src: [rows * (hidden_row + pre_row)]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @truncate(i * 7 + i / 251);
    var x: [rows * hidden_row]u8 = undefined;
    var q: [rows * pre_row]u8 = undefined;
    try Gate.deinterleave(&src, &x, &q, rows);
    for (0..rows) |r| {
        const row = src[r * (hidden_row + pre_row) ..];
        try std.testing.expectEqualSlices(u8, row[0..hidden_row], x[r * hidden_row ..][0..hidden_row]);
        try std.testing.expectEqualSlices(u8, row[hidden_row..][0..pre_row], q[r * pre_row ..][0..pre_row]);
    }
    try std.testing.expectError(error.GateRows, Gate.deinterleave(&src, x[0 .. (rows - 1) * hidden_row], q[0 .. (rows - 1) * pre_row], rows - 1));
    var g: Gate = .{ .k = 14 };
    try std.testing.expect(!g.entered());
    g.entry_dir = "somewhere";
    try std.testing.expect(g.entered());
    g.off();
    try std.testing.expect(!g.entered());
}
