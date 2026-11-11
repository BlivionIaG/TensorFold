//! The hipprof shim's pure parts: the per-launch-shape table, its text dump, the kernel argument decoding.
const std = @import("std");

extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;
extern "c" fn qsort(base: *anyopaque, count: usize, size: usize, cmp: *const fn (*const anyopaque, *const anyopaque) callconv(.c) c_int) void;

pub const name_cap = 200;
pub const max_rows = 1024;

/// One launch shape: kernel name, grid, block, shared bytes and (affine kernels) m, n, k and sides.
pub const Row = struct {
    name: [name_cap]u8,
    g: [3]u32,
    b: [3]u32,
    sh: u32,
    m: i32,
    n: i32,
    k: i32,
    sides: i32,
    count: i64,
    total: f64,
    min: f64,
    rep_total: f64,
    gap_total: f64,
    gap_count: i64,
    rep_count: i64,
};

pub const Shape = struct { g: [3]u32, b: [3]u32, sh: u32, m: i32, n: i32, k: i32, sides: i32 };

/// The first `name_cap - 1` bytes of `name`, the rest zero, as strncpy leaves them.
pub fn copyName(dst: *[name_cap]u8, name: []const u8) void {
    @memset(dst, 0);
    const len = @min(name.len, name_cap - 1);
    @memcpy(dst[0..len], name[0..len]);
}

pub const Table = struct {
    rows: [max_rows]Row = undefined,
    n: usize = 0,

    /// The row of this exact shape, added when new; a full table folds new shapes into its last row.
    pub fn rowFor(self: *Table, name: []const u8, s: Shape) *Row {
        for (self.rows[0..self.n]) |*r| {
            if (std.mem.eql(u8, std.mem.sliceTo(&r.name, 0), name) and std.mem.eql(u32, &r.g, &s.g) and std.mem.eql(u32, &r.b, &s.b) and
                r.sh == s.sh and r.m == s.m and r.n == s.n and r.k == s.k and r.sides == s.sides) return r;
        }
        if (self.n == max_rows) return &self.rows[max_rows - 1];
        const r = &self.rows[self.n];
        self.n += 1;
        r.* = std.mem.zeroes(Row);
        copyName(&r.name, name);
        r.g = s.g;
        r.b = s.b;
        r.sh = s.sh;
        r.m = s.m;
        r.n = s.n;
        r.k = s.k;
        r.sides = s.sides;
        r.min = 1e18;
        return r;
    }

    /// One timed launch of `row`: `us` of wall clock.
    pub fn addTimed(row: *Row, us: f64) void {
        row.count += 1;
        row.total += us;
        if (us < row.min) row.min = us;
    }
};

pub const header = "# name\tcount\tgrid\tblock\tshared\tm\tn\tk\tsides\tavg_us\tmin_us\ttotal_ms\trep_avg_us\tweight_bytes\tgap_us\n";

/// Descending by total plus replayed time, as qsort calls it.
pub fn cmpRows(a: *const anyopaque, b: *const anyopaque) callconv(.c) c_int {
    const ra: *const Row = @ptrCast(@alignCast(a));
    const rb: *const Row = @ptrCast(@alignCast(b));
    const x = ra.total + ra.rep_total;
    const y = rb.total + rb.rep_total;
    return if (x < y) 1 else if (x > y) -1 else 0;
}

/// Sorts `rows` in place with the C library's qsort, so ties land where the C shim put them.
pub fn sortRows(rows: []Row) void {
    if (rows.len > 1) qsort(rows.ptr, rows.len, @sizeOf(Row), cmpRows);
}

/// Weight bytes a launch reads: 4-bit codes plus a 2-byte scale and bias a group of 64.
pub fn weightBytes(r: *const Row) f64 {
    return if (r.k > 0) @as(f64, @floatFromInt(r.n)) * @as(f64, @floatFromInt(r.k)) * 0.5625 else 0;
}

/// The row's line of the table, with its newline, formatted by the C library; null when `buf` is too small.
pub fn formatRow(buf: []u8, r: *const Row) ?[]const u8 {
    const count: f64 = @floatFromInt(r.count);
    const rep_count: f64 = @floatFromInt(r.rep_count);
    const gap_count: f64 = @floatFromInt(r.gap_count);
    const avg = if (r.count != 0) r.total / count else 0;
    const min = if (r.count != 0) r.min else 0;
    const rep_avg = if (r.rep_count != 0) r.rep_total / rep_count else 0;
    const gap = if (r.gap_count != 0) r.gap_total / gap_count else 0;
    const len = snprintf(buf.ptr, buf.len, "%s\t%lld\t%u,%u,%u\t%u,%u,%u\t%u\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.3f\t%.2f\t%.0f\t%.2f\n", @as([*:0]const u8, @ptrCast(&r.name)), @as(c_longlong, r.count), r.g[0], r.g[1], r.g[2], r.b[0], r.b[1], r.b[2], r.sh, r.m, r.n, r.k, r.sides, avg, min, r.total / 1000, rep_avg, weightBytes(r), gap);
    if (len < 0 or len >= buf.len) return null;
    return buf[0..@intCast(len)];
}

pub const Dims = struct { m: i32 = 0, n: i32 = 0, k: i32 = 0, sides: i32 = 0 };

const affine_m = 56;
const affine_n = 60;
const affine_k = 64;
const sides_n = 128;
const sides_count = 164;

fn readI32(p: [*]const u8, off: usize) i32 {
    return std.mem.readInt(i32, p[off..][0..4], .little);
}

/// m, n, k of an affine kernel launch from its parameter block (the first parameter is quant/mlx.hpp's Affine);
/// a stream tile's second parameter, its StreamSides, replaces n with the sum of its sides' widths and sets the sides.
pub fn kernelDims(name: []const u8, params: ?[*]const ?[*]const u8) Dims {
    var d: Dims = .{};
    if (std.mem.indexOf(u8, name, "affine_") == null) return d;
    const list = params orelse return d;
    const aff = list[0] orelse return d;
    d.m = readI32(aff, affine_m);
    d.n = readI32(aff, affine_n);
    d.k = readI32(aff, affine_k);
    if (std.mem.indexOf(u8, name, "stream") != null) {
        if (list[1]) |sd| {
            const cnt = readI32(sd, sides_count);
            if (cnt > 0 and cnt <= 4) {
                d.n = 0;
                for (0..@intCast(cnt)) |x| d.n += readI32(sd, sides_n + 4 * x);
                d.sides = cnt;
            }
        }
    }
    return d;
}

pub const XCheck = struct { non_finite: i32, rough_max: f32 };

/// HIPPROF_XCHK's scan of an fp16 activation block: the non-finite count and a rough maximum.
pub fn xcheck(x: []const u16) XCheck {
    var bad: i32 = 0;
    var mx: f32 = 0;
    for (x) |v| {
        const ex: u32 = (v >> 10) & 31;
        if (ex == 31) {
            bad += 1;
        } else {
            const scale: i32 = if (ex != 0) @as(i32, 1) << @intCast(if (ex > 15) ex - 15 else 0) else 0;
            const f = @as(f32, @floatFromInt(v & 0x3ff)) / 1024.0 * @as(f32, @floatFromInt(scale)) + 0;
            if (f > mx) mx = f;
        }
    }
    return .{ .non_finite = bad, .rough_max = mx };
}

fn shape(g0: u32) Shape {
    return .{ .g = .{ g0, 1, 1 }, .b = .{ 64, 1, 1 }, .sh = 0, .m = 1, .n = 4096, .k = 2560, .sides = 0 };
}

test "rowFor merges equal shapes and separates different ones" {
    var t: Table = .{};
    const a = t.rowFor("affine_x", shape(8));
    const b = t.rowFor("affine_x", shape(8));
    const c = t.rowFor("affine_x", shape(9));
    const d = t.rowFor("other", shape(8));
    try std.testing.expect(a == b);
    try std.testing.expect(a != c and a != d and c != d);
    try std.testing.expectEqual(@as(usize, 3), t.n);
    try std.testing.expectEqual(@as(f64, 1e18), a.min);
}

test "a full table folds new shapes into the last row" {
    var t: Table = .{};
    for (0..max_rows) |i| _ = t.rowFor("k", shape(@intCast(i)));
    try std.testing.expectEqual(@as(usize, max_rows), t.n);
    const over = t.rowFor("k", shape(5000));
    try std.testing.expect(over == &t.rows[max_rows - 1]);
    try std.testing.expectEqual(@as(usize, max_rows), t.n);
}

test "names are cut at 199 bytes" {
    var t: Table = .{};
    const long: [300]u8 = @splat('a');
    const r = t.rowFor(long[0..199], shape(1));
    try std.testing.expectEqual(@as(usize, 199), std.mem.sliceTo(&r.name, 0).len);
    var buf: [name_cap]u8 = undefined;
    copyName(&buf, &long);
    try std.testing.expectEqual(@as(usize, 199), std.mem.sliceTo(&buf, 0).len);
    try std.testing.expectEqual(@as(u8, 0), buf[199]);
}

test "the table line is the one prof_report reads" {
    var t: Table = .{};
    const r = t.rowFor("_Z6affine", .{ .g = .{ 32, 1, 1 }, .b = .{ 256, 1, 1 }, .sh = 4096, .m = 1, .n = 8192, .k = 2560, .sides = 2 });
    Table.addTimed(r, 30.0);
    Table.addTimed(r, 10.0);
    r.rep_count = 1;
    r.rep_total = 7.5;
    r.gap_count = 2;
    r.gap_total = 3.0;
    var buf: [512]u8 = undefined;
    const line = formatRow(&buf, r).?;
    try std.testing.expectEqualStrings("_Z6affine\t2\t32,1,1\t256,1,1\t4096\t1\t8192\t2560\t2\t20.00\t10.00\t0.040\t7.50\t11796480\t1.50\n", line);
    var empty = std.mem.zeroes(Row);
    empty.min = 1e18;
    const zero = formatRow(&buf, &empty).?;
    try std.testing.expectEqualStrings("\t0\t0,0,0\t0,0,0\t0\t0\t0\t0\t0\t0.00\t0.00\t0.000\t0.00\t0\t0.00\n", zero);
    try std.testing.expect(formatRow(buf[0..10], r) == null);
}

test "rows sort by total plus replayed time, largest first" {
    var rows: [3]Row = undefined;
    for (&rows, [_]f64{ 5, 50, 20 }) |*r, total| {
        r.* = std.mem.zeroes(Row);
        r.total = total;
    }
    rows[0].rep_total = 100;
    sortRows(&rows);
    try std.testing.expectEqual(@as(f64, 5), rows[0].total);
    try std.testing.expectEqual(@as(f64, 50), rows[1].total);
    try std.testing.expectEqual(@as(f64, 20), rows[2].total);
}

fn putI32(buf: []u8, off: usize, v: i32) void {
    std.mem.writeInt(i32, buf[off..][0..4], v, .little);
}

test "affine and stream-tile parameters decode like the C shim" {
    var aff: [80]u8 = @splat(0);
    putI32(&aff, affine_m, 1);
    putI32(&aff, affine_n, 4096);
    putI32(&aff, affine_k, 2560);
    var sd: [192]u8 = @splat(0);
    putI32(&sd, sides_count, 2);
    putI32(&sd, sides_n, 1024);
    putI32(&sd, sides_n + 4, 512);
    const params = [_]?[*]const u8{ &aff, &sd };
    const plain = kernelDims("affine_dot2_row", &params);
    try std.testing.expectEqual(Dims{ .m = 1, .n = 4096, .k = 2560, .sides = 0 }, plain);
    const stream = kernelDims("affine_dot2_stream", &params);
    try std.testing.expectEqual(Dims{ .m = 1, .n = 1536, .k = 2560, .sides = 2 }, stream);
    putI32(&sd, sides_count, 5);
    try std.testing.expectEqual(@as(i32, 4096), kernelDims("affine_dot2_stream", &params).n);
    try std.testing.expectEqual(Dims{}, kernelDims("rmsnorm", &params));
    try std.testing.expectEqual(Dims{}, kernelDims("affine_x", null));
}

test "xcheck counts non-finite halves and keeps a rough maximum" {
    const half_one: u16 = 15 << 10;
    const x = [_]u16{ half_one, 0x7c00, 0x7e00, 0, (16 << 10) | 0x200 };
    const r = xcheck(&x);
    try std.testing.expectEqual(@as(i32, 2), r.non_finite);
    try std.testing.expectEqual(@as(f32, 0.5 * 2), r.rough_max);
}
