//! Offline tests of tf-expert-cache-sim: the npz reader, the two random generators and the number formats against CPython / NumPy values, every
//! policy on tiny hand-checked call sequences (the expected lists were also run through the Python original), and the whole report on synthetic
//! traces against the output the Python original printed for them.
const std = @import("std");
const expected = @import("expert_cache_sim_expected.zig");
const sim = @import("expert_cache_sim.zig");
const l = @import("expert_cache_sim_list.zig");
const rng = @import("expert_cache_sim_rng.zig");
const fmt = @import("expert_cache_sim_fmt.zig");
const npz = @import("npz.zig");
const npy = @import("npy.zig");
const pol = @import("expert_cache_sim_policy.zig");
const freq = @import("expert_cache_sim_freq.zig");
const hs = @import("expert_cache_sim_handshake.zig");

const tst = std.testing;

// ------------------------------------------------------------------ number formats and NumPy sums

fn fixedText(buf: []u8, x: f64, prec: u8, plus: bool) []const u8 {
    return fmt.fixedBuf(buf, x, prec, plus);
}

test "fixed-point text is the exact value rounded half to even, as CPython prints it" {
    var buf: [64]u8 = undefined;
    const cases = [_]struct { x: f64, f2: []const u8, f1: []const u8, p0: []const u8 }{
        .{ .x = 2.675, .f2 = "2.67", .f1 = "2.7", .p0 = "+3" }, // 2.675 is a hair below its decimal text
        .{ .x = 0.125, .f2 = "0.12", .f1 = "0.1", .p0 = "+0" }, // exact ties go to the even digit
        .{ .x = 0.375, .f2 = "0.38", .f1 = "0.4", .p0 = "+0" },
        .{ .x = 1.005, .f2 = "1.00", .f1 = "1.0", .p0 = "+1" },
        .{ .x = 9.995, .f2 = "9.99", .f1 = "10.0", .p0 = "+10" },
        .{ .x = 0.5, .f2 = "0.50", .f1 = "0.5", .p0 = "+0" },
        .{ .x = 1.5, .f2 = "1.50", .f1 = "1.5", .p0 = "+2" },
        .{ .x = 2.5, .f2 = "2.50", .f1 = "2.5", .p0 = "+2" },
        .{ .x = 3.5, .f2 = "3.50", .f1 = "3.5", .p0 = "+4" },
        .{ .x = -0.4, .f2 = "-0.40", .f1 = "-0.4", .p0 = "-0" }, // a negative value keeps its sign even when it rounds to zero
        .{ .x = -0.0, .f2 = "-0.00", .f1 = "-0.0", .p0 = "-0" },
        .{ .x = 1e-300, .f2 = "0.00", .f1 = "0.0", .p0 = "+0" },
        .{ .x = 83.25, .f2 = "83.25", .f1 = "83.2", .p0 = "+83" },
        .{ .x = 83.35, .f2 = "83.35", .f1 = "83.3", .p0 = "+83" },
        .{ .x = 99.95, .f2 = "99.95", .f1 = "100.0", .p0 = "+100" },
        .{ .x = 0.045, .f2 = "0.04", .f1 = "0.0", .p0 = "+0" },
        .{ .x = 5e-324, .f2 = "0.00", .f1 = "0.0", .p0 = "+0" },
        .{ .x = 123456.785, .f2 = "123456.79", .f1 = "123456.8", .p0 = "+123457" },
    };
    for (cases) |c| {
        try tst.expectEqualStrings(c.f2, fixedText(&buf, c.x, 2, false));
        try tst.expectEqualStrings(c.f1, fixedText(&buf, c.x, 1, false));
        try tst.expectEqualStrings(c.p0, fixedText(&buf, c.x, 0, true));
        try tst.expectEqualStrings(if (std.math.signbit(c.x)) c.f2 else try std.fmt.bufPrint(buf[32..], "+{s}", .{c.f2}), fixedText(&buf, c.x, 2, true));
    }
    try tst.expectEqualStrings("nan", fixedText(&buf, std.math.nan(f64), 2, false));
    try tst.expectEqualStrings("+nan", fixedText(&buf, std.math.nan(f64), 2, true));
    try tst.expectEqualStrings("inf", fixedText(&buf, std.math.inf(f64), 1, false));
}

fn testSeries(n: usize, out: []f64) []f64 {
    for (out[0..n], 0..) |*x, i| x.* = @as(f64, @floatFromInt((i * 37) % 101)) / 13.0 - 3.1;
    return out[0..n];
}

test "mean and std(ddof=1) reproduce NumPy's pairwise sums" {
    var buf: [300]f64 = undefined;
    var scratch: [300]f64 = undefined;
    // np.array([((i*37)%101)/13.0-3.1 for i in range(n)]): .mean(), .std(ddof=1)
    const cases = [_]struct { n: usize, mean: f64, std: f64 }{
        .{ .n = 5, .mean = -0.5153846153846154, .std = 2.2784038843793764 },
        .{ .n = 35, .mean = 0.667032967032967, .std = 2.298888615568253 },
        .{ .n = 129, .mean = 0.7228980322003578, .std = 2.260895804390412 },
        .{ .n = 300, .mean = 0.7379487179487179, .std = 2.2470401292624516 },
    };
    for (cases) |c| {
        const a = testSeries(c.n, &buf);
        try tst.expectEqual(c.mean, fmt.mean(a));
        try tst.expectEqual(c.std, fmt.stdDdof1(a, &scratch));
    }
}

// ------------------------------------------------------------------ MT19937

test "NumPy RandomState(seed).random_sample" {
    var v: [3]f64 = undefined;
    rng.numpyRandomSample(0, &v);
    try tst.expectEqual(@as(f64, 0.5488135039273248), v[0]);
    try tst.expectEqual(@as(f64, 0.7151893663724195), v[1]);
    try tst.expectEqual(@as(f64, 0.6027633760716439), v[2]);
    rng.numpyRandomSample(1234, &v);
    try tst.expectEqual(@as(f64, 0.1915194503788923), v[0]);
    try tst.expectEqual(@as(f64, 0.6221087710398319), v[1]);
    try tst.expectEqual(@as(f64, 0.4377277390071145), v[2]);
    var w: [2]f64 = undefined;
    rng.numpyRandomSample(4294967295, &w);
    try tst.expectEqual(@as(f64, 0.0976320289940138), w[0]);
    try tst.expectEqual(@as(f64, 0.9123828453026218), w[1]);
}

test "the seed a prompt name gives its prefill tie-break draws" {
    try tst.expectEqual(@as(u32, 1302), try @import("expert_cache_sim_data.zig").stableSeed("chat0"));
    var v: [2]f64 = undefined;
    rng.numpyRandomSample(1302, &v);
    try tst.expectEqual(@as(f64, 0.8461488016371451), v[0]);
    try tst.expectEqual(@as(f64, 0.7898289635254122), v[1]);
}

test "Python random.Random(0): random() and shuffle()" {
    var mt: rng.Mt19937 = .initByArray(&.{0});
    try tst.expectEqual(@as(f64, 0.8444218515250481), mt.random());
    try tst.expectEqual(@as(f64, 0.7579544029403025), mt.random());
    try tst.expectEqual(@as(f64, 0.420571580830845), mt.random());
    var a10: [10]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    rng.pythonShuffle(u8, 0, &a10);
    try tst.expectEqualSlices(u8, &.{ 7, 8, 1, 5, 3, 4, 2, 0, 9, 6 }, &a10);
    var a5: [5]u8 = .{ 0, 1, 2, 3, 4 };
    rng.pythonShuffle(u8, 0, &a5);
    try tst.expectEqualSlices(u8, &.{ 2, 1, 0, 4, 3 }, &a5);
    var a34: [34]u8 = undefined;
    for (&a34, 0..) |*x, i| x.* = @intCast(i);
    rng.pythonShuffle(u8, 0, &a34);
    try tst.expectEqualSlices(u8, &.{ 7, 0, 27, 20, 10, 5, 21, 13, 1, 17, 31, 22, 23, 33, 14, 30, 3, 19, 32, 4, 29, 6, 18, 11, 28, 9, 25, 12, 15, 16, 8, 2, 26, 24 }, &a34);
}

// ------------------------------------------------------------------ npz

fn putInt(out: *std.ArrayList(u8), a: std.mem.Allocator, comptime T: type, v: T) !void {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    try out.appendSlice(a, &b);
}

const Member = struct { name: []const u8, data: []const u8 };

/// A zip of stored members; with `zip64` the central directory holds 0xffffffff sizes and offsets and the zip64 extra field with the real ones.
fn buildZip(a: std.mem.Allocator, members: []const Member, zip64: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const offsets = try a.alloc(u32, members.len);
    defer a.free(offsets);
    for (members, offsets) |m, *off| {
        off.* = @intCast(out.items.len);
        try out.appendSlice(a, "PK\x03\x04");
        try putInt(&out, a, u16, 20);
        try putInt(&out, a, u16, 0); // flags
        try putInt(&out, a, u16, 0); // stored
        try putInt(&out, a, u32, 0); // time, date
        try putInt(&out, a, u32, std.hash.Crc32.hash(m.data));
        try putInt(&out, a, u32, @intCast(m.data.len));
        try putInt(&out, a, u32, @intCast(m.data.len));
        try putInt(&out, a, u16, @intCast(m.name.len));
        try putInt(&out, a, u16, 0);
        try out.appendSlice(a, m.name);
        try out.appendSlice(a, m.data);
    }
    const cd = out.items.len;
    for (members, offsets) |m, off| {
        try out.appendSlice(a, "PK\x01\x02");
        try putInt(&out, a, u16, 45);
        try putInt(&out, a, u16, 45);
        try putInt(&out, a, u16, 0);
        try putInt(&out, a, u16, 0);
        try putInt(&out, a, u32, 0);
        try putInt(&out, a, u32, std.hash.Crc32.hash(m.data));
        const big: u32 = 0xffffffff;
        try putInt(&out, a, u32, if (zip64) big else @intCast(m.data.len));
        try putInt(&out, a, u32, if (zip64) big else @intCast(m.data.len));
        try putInt(&out, a, u16, @intCast(m.name.len));
        try putInt(&out, a, u16, if (zip64) 28 else 0);
        try putInt(&out, a, u16, 0); // comment
        try putInt(&out, a, u16, 0); // disk
        try putInt(&out, a, u16, 0);
        try putInt(&out, a, u32, 0);
        try putInt(&out, a, u32, if (zip64) big else off);
        try out.appendSlice(a, m.name);
        if (zip64) {
            try putInt(&out, a, u16, 1);
            try putInt(&out, a, u16, 24);
            try putInt(&out, a, u64, m.data.len); // size
            try putInt(&out, a, u64, m.data.len); // compressed size
            try putInt(&out, a, u64, off);
        }
    }
    const cd_len = out.items.len - cd;
    try out.appendSlice(a, "PK\x05\x06");
    try putInt(&out, a, u32, 0);
    try putInt(&out, a, u16, @intCast(members.len));
    try putInt(&out, a, u16, @intCast(members.len));
    try putInt(&out, a, u32, @intCast(cd_len));
    try putInt(&out, a, u32, @intCast(cd));
    try putInt(&out, a, u16, 0);
    return out.toOwnedSlice(a);
}

fn hexBytes(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var clean: std.ArrayList(u8) = .empty;
    defer clean.deinit(a);
    for (text) |c| if (c != '\n') try clean.append(a, c);
    const out = try a.alloc(u8, clean.items.len / 2);
    errdefer a.free(out);
    _ = try std.fmt.hexToBytes(out, clean.items);
    return out;
}

test "npz: an archive written by np.savez_compressed (deflated members, zip64 local headers)" {
    const a = tst.allocator;
    const raw = try hexBytes(a, expected.tiny_npz_hex);
    defer a.free(raw);
    var z = try npz.open(a, raw);
    defer z.deinit(a);
    try tst.expectEqual(@as(usize, 2), z.entries.len);
    try tst.expectEqual(npz.Method.deflate, z.find("ids").?.method);
    const ids_bytes = try z.array(a, "ids");
    defer a.free(ids_bytes);
    const ids = try npy.parse(ids_bytes);
    try tst.expectEqual(npy.Dtype.u1, ids.dtype);
    try tst.expectEqual(@as(usize, 3), ids.ndim);
    try tst.expectEqual(@as(usize, 2), ids.shape[0]);
    try tst.expectEqual(@as(usize, 40), ids.shape[1]);
    try tst.expectEqual(@as(usize, 8), ids.shape[2]);
    for (0..2) |t| for (0..40) |layer| for (0..8) |j| {
        try tst.expectEqual(@as(i64, @intCast((layer * 8 + j + t * 5) % 256)), ids.atInt((t * 40 + layer) * 8 + j));
    };
    const kind_bytes = try z.array(a, "kind.npy");
    defer a.free(kind_bytes);
    const kind = try npy.parse(kind_bytes);
    try tst.expectEqual(@as(usize, 2), kind.len());
    try tst.expectEqual(@as(i64, 0), kind.atInt(0));
    try tst.expectEqual(@as(i64, 1), kind.atInt(1));
    try tst.expectError(error.NoMember, z.array(a, "w"));
}

test "npz: stored members, zip64 sizes and offsets in the central directory, damage" {
    const a = tst.allocator;
    const first = try npy.encode(a, "<i4", &.{3}, std.mem.sliceAsBytes(&[_]i32{ 7, -8, 9 }));
    defer a.free(first);
    const second = try npy.encode(a, "|b1", &.{2}, &.{ 1, 0 });
    defer a.free(second);
    for ([_]bool{ false, true }) |zip64| {
        const bytes = try buildZip(a, &.{ .{ .name = "x.npy", .data = first }, .{ .name = "y.npy", .data = second } }, zip64);
        defer a.free(bytes);
        var z = try npz.open(a, bytes);
        defer z.deinit(a);
        const x = try z.array(a, "x");
        defer a.free(x);
        try tst.expectEqual(@as(i64, -8), (try npy.parse(x)).atInt(1));
        const y = try z.array(a, "y");
        defer a.free(y);
        try tst.expectEqual(@as(i64, 1), (try npy.parse(y)).atInt(0));
    }
    const bytes = try buildZip(a, &.{.{ .name = "x.npy", .data = first }}, false);
    defer a.free(bytes);
    // a flipped data byte fails the CRC; cutting the archive loses the end record; text is not a zip
    const damaged = try a.dupe(u8, bytes);
    defer a.free(damaged);
    damaged[30 + "x.npy".len + 20] ^= 0xff;
    var z = try npz.open(a, damaged);
    defer z.deinit(a);
    try tst.expectError(error.BadMember, z.array(a, "x"));
    try tst.expectError(error.NotZip, npz.open(a, bytes[0 .. bytes.len - 25]));
    try tst.expectError(error.NotZip, npz.open(a, "this is not a zip archive at all"));
}

// ------------------------------------------------------------------ policies

fn expectList(od: *const l.OList, want: []const u16) !void {
    var got: [32]u16 = undefined;
    var n: usize = 0;
    var k = od.first();
    while (k) |key| : (k = od.after(key)) {
        got[n] = key;
        n += 1;
    }
    try tst.expectEqualSlices(u16, want, got[0..n]);
    try tst.expectEqual(want.len, od.count);
}

const Step = struct { layer: usize = 0, keys: []const u16, miss: u32, adm: u64, evi: u64 };

fn expectStep(p: anytype, s: Step) !void {
    try tst.expectEqual(s.miss, p.step(s.layer, s.keys));
    try tst.expectEqual(s.adm, p.adm);
    try tst.expectEqual(s.evi, p.evi);
}

test "LRU: hits move to the end, a miss evicts the oldest line the call did not request" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.Lru = try .init(arena.allocator(), 3);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 }, // 1 2
        .{ .keys = &.{ 3, 1 }, .miss = 1, .adm = 3, .evi = 0 }, // 1 hit -> 2 1; 3 fills the cache: 2 1 3
        .{ .keys = &.{ 4, 2 }, .miss = 1, .adm = 4, .evi = 1 }, // 2 hit -> 1 3 2; 4 evicts the oldest, 1: 3 2 4
        .{ .keys = &.{ 3, 2 }, .miss = 0, .adm = 4, .evi = 1 }, // all hits: 4 3 2
        .{ .keys = &.{ 6, 7 }, .miss = 2, .adm = 6, .evi = 3 }, // 4 and 3 go: 2 6 7
        .{ .keys = &.{ 2, 9, 8 }, .miss = 2, .adm = 8, .evi = 5 }, // 2 is touched first, so 6 and 7 go: 2 9 8
    };
    const want = [_][]const u16{ &.{ 1, 2 }, &.{ 2, 1, 3 }, &.{ 3, 2, 4 }, &.{ 4, 3, 2 }, &.{ 2, 6, 7 }, &.{ 2, 9, 8 } };
    for (steps, want) |s, w| {
        try expectStep(&p, s);
        try expectList(&p.od, w);
    }
}

test "LRU: preload stops at the cap and keeps the given order" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.Lru = try .init(arena.allocator(), 3);
    p.preload(&.{ 5, 6, 7, 8 });
    try expectList(&p.od, &.{ 5, 6, 7 });
}

test "per-layer LRU: CAP//40 lines per layer, the remainder to the first layers" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.PerLayer = try .init(arena.allocator(), 42);
    try tst.expectEqual(@as(usize, 2), p.caps[0]);
    try tst.expectEqual(@as(usize, 2), p.caps[1]);
    try tst.expectEqual(@as(usize, 1), p.caps[2]);
    try tst.expectEqual(@as(usize, 1), p.caps[39]);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{3}, .miss = 1, .adm = 3, .evi = 1 },
        .{ .keys = &.{ 1, 4 }, .miss = 2, .adm = 5, .evi = 3 },
        .{ .layer = 1, .keys = &.{ 257, 258 }, .miss = 2, .adm = 7, .evi = 3 },
        .{ .layer = 1, .keys = &.{259}, .miss = 1, .adm = 8, .evi = 4 },
        .{ .layer = 2, .keys = &.{513}, .miss = 1, .adm = 9, .evi = 4 },
        .{ .layer = 2, .keys = &.{514}, .miss = 1, .adm = 10, .evi = 5 },
        .{ .keys = &.{4}, .miss = 0, .adm = 10, .evi = 5 },
    };
    const want0 = [_][]const u16{ &.{ 1, 2 }, &.{ 2, 3 }, &.{ 1, 4 }, &.{ 1, 4 }, &.{ 1, 4 }, &.{ 1, 4 }, &.{ 1, 4 }, &.{ 1, 4 } };
    const want1 = [_][]const u16{ &.{}, &.{}, &.{}, &.{ 257, 258 }, &.{ 258, 259 }, &.{ 258, 259 }, &.{ 258, 259 }, &.{ 258, 259 } };
    const want2 = [_][]const u16{ &.{}, &.{}, &.{}, &.{}, &.{}, &.{513}, &.{514}, &.{514} };
    for (steps, want0, want1, want2) |s, w0, w1, w2| {
        try expectStep(&p, s);
        try expectList(&p.ods[0], w0);
        try expectList(&p.ods[1], w1);
        try expectList(&p.ods[2], w2);
    }
}

test "SLRU: a probation hit is promoted, the protected overflow is demoted to probation's end" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.Slru = try .init(arena.allocator(), 5, 0.2); // 1 protected line
    try tst.expectEqual(@as(usize, 1), p.pcap);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2, 3, 4, 5 }, .miss = 5, .adm = 5, .evi = 0 },
        .{ .keys = &.{ 1, 2, 6 }, .miss = 1, .adm = 6, .evi = 1 }, // 1 promoted, 2 promoted and 1 demoted; 6 evicts 3
        .{ .keys = &.{ 4, 5, 1, 7 }, .miss = 1, .adm = 7, .evi = 2 },
        .{ .keys = &.{ 2, 10 }, .miss = 1, .adm = 8, .evi = 3 },
        .{ .keys = &.{ 11, 12 }, .miss = 2, .adm = 10, .evi = 5 },
    };
    const prob = [_][]const u16{ &.{ 1, 2, 3, 4, 5 }, &.{ 4, 5, 1, 6 }, &.{ 2, 4, 5, 7 }, &.{ 5, 7, 1, 10 }, &.{ 1, 10, 11, 12 } };
    const prot = [_][]const u16{ &.{}, &.{2}, &.{1}, &.{2}, &.{2} };
    for (steps, prob, prot) |s, pb, pt| {
        try expectStep(&p, s);
        try expectList(&p.prob, pb);
        try expectList(&p.prot, pt);
    }
}

test "2Q: a ghost hit goes to Am; a hit in the A1in FIFO does not move, and the call's own lines are never the victim" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.TwoQ = try .init(arena.allocator(), 4, 0.25, 0.5);
    try tst.expectEqual(@as(usize, 1), p.kin);
    try tst.expectEqual(@as(usize, 2), p.gcap);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 3, 4 }, .miss = 2, .adm = 4, .evi = 0 },
        .{ .keys = &.{5}, .miss = 1, .adm = 5, .evi = 1 }, // 1 leaves A1in for the ghost list
        .{ .keys = &.{1}, .miss = 1, .adm = 6, .evi = 2 }, // ghost hit: into Am, 2 evicted to the ghosts
        .{ .keys = &.{ 6, 7 }, .miss = 2, .adm = 8, .evi = 4 },
        .{ .keys = &.{ 2, 3 }, .miss = 2, .adm = 10, .evi = 6 },
        .{ .keys = &.{3}, .miss = 0, .adm = 10, .evi = 6 },
    };
    const a1 = [_][]const u16{ &.{ 1, 2 }, &.{ 1, 2, 3, 4 }, &.{ 2, 3, 4, 5 }, &.{ 3, 4, 5 }, &.{ 5, 6, 7 }, &.{ 7, 2, 3 }, &.{ 7, 2, 3 } };
    const am = [_][]const u16{ &.{}, &.{}, &.{}, &.{1}, &.{1}, &.{1}, &.{1} };
    const out = [_][]const u16{ &.{}, &.{}, &.{1}, &.{2}, &.{ 3, 4 }, &.{ 5, 6 }, &.{ 5, 6 } };
    for (steps, a1, am, out) |s, x, y, z| {
        try expectStep(&p, s);
        try expectList(&p.a1, x);
        try expectList(&p.am, y);
        try expectList(&p.out, z);
    }
    // A1in (1 2 3), kin 1: 1 is a hit that stays the oldest, so 9 evicts 2, not 1
    var q: pol.TwoQ = try .init(arena.allocator(), 3, 0.5, 0.5);
    try expectStep(&q, .{ .keys = &.{ 1, 2, 3 }, .miss = 3, .adm = 3, .evi = 0 });
    try expectStep(&q, .{ .keys = &.{ 1, 9 }, .miss = 1, .adm = 4, .evi = 1 });
    try expectList(&q.a1, &.{ 1, 3, 9 });
    try expectList(&q.out, &.{2});
    // 2 was a ghost, but evicting 1 for it overflows the one-entry ghost list and drops 2: it comes back through A1in
    try expectStep(&q, .{ .keys = &.{ 3, 2 }, .miss = 1, .adm = 5, .evi = 2 });
    try expectList(&q.a1, &.{ 3, 9, 2 });
    try expectList(&q.out, &.{1});
}

fn expectCounts(c: []const u32, want: []const u32) !void {
    try tst.expectEqualSlices(u32, want, c);
}

test "TinyLFU: a miss is refused unless it beats the victim's frequency (strictly, or >= with ge)" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: freq.TinyLfu = try .init(arena.allocator(), 2, 100, false, false);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 1, 2 }, .miss = 0, .adm = 2, .evi = 0 },
        .{ .keys = &.{3}, .miss = 1, .adm = 2, .evi = 0 }, // 3 seen once, victim 1 seen twice: refused
        .{ .keys = &.{3}, .miss = 1, .adm = 2, .evi = 0 }, // 2 against 2: refused
        .{ .keys = &.{3}, .miss = 1, .adm = 3, .evi = 1 }, // 3 against 2: admitted, 1 evicted
        .{ .keys = &.{ 4, 1 }, .miss = 2, .adm = 4, .evi = 2 },
    };
    const od = [_][]const u16{ &.{ 1, 2 }, &.{ 1, 2 }, &.{ 1, 2 }, &.{ 1, 2 }, &.{ 2, 3 }, &.{ 3, 1 } };
    const counts = [_][]const u32{ &.{ 1, 1, 0, 0 }, &.{ 2, 2, 0, 0 }, &.{ 2, 2, 1, 0 }, &.{ 2, 2, 2, 0 }, &.{ 2, 2, 3, 0 }, &.{ 3, 2, 3, 1 } };
    for (steps, od, counts) |s, o, c| {
        try expectStep(&p, s);
        try expectList(&p.od, o);
        try expectCounts(p.f.c[1..5], c);
    }
    var ge: freq.TinyLfu = try .init(arena.allocator(), 2, 100, false, true);
    const gsteps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{3}, .miss = 1, .adm = 3, .evi = 1 }, // 1 against 1: admitted with >=
        .{ .keys = &.{3}, .miss = 0, .adm = 3, .evi = 1 },
        .{ .keys = &.{ 4, 1 }, .miss = 2, .adm = 5, .evi = 3 },
    };
    const god = [_][]const u16{ &.{ 1, 2 }, &.{ 2, 3 }, &.{ 2, 3 }, &.{ 4, 1 } };
    for (gsteps, god) |s, o| {
        try expectStep(&ge, s);
        try expectList(&ge.od, o);
    }
}

test "TinyLFU with a doorkeeper: a first sighting only sets a bit; the counters halve and the door clears every W accesses" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: freq.TinyLfu = try .init(arena.allocator(), 2, 6, true, false);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 1, 2 }, .miss = 0, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 3, 4 }, .miss = 2, .adm = 2, .evi = 0 }, // 6 accesses: everything halves, the door empties
        .{ .keys = &.{ 3, 4 }, .miss = 2, .adm = 4, .evi = 2 },
        .{ .keys = &.{ 3, 4 }, .miss = 0, .adm = 4, .evi = 2 },
        .{ .keys = &.{5}, .miss = 1, .adm = 4, .evi = 2 },
    };
    const od = [_][]const u16{ &.{ 1, 2 }, &.{ 1, 2 }, &.{ 1, 2 }, &.{ 3, 4 }, &.{ 3, 4 }, &.{ 3, 4 } };
    const counts = [_][]const u32{ &.{ 0, 0, 0, 0, 0 }, &.{ 1, 1, 0, 0, 0 }, &.{ 0, 0, 0, 0, 0 }, &.{ 0, 0, 0, 0, 0 }, &.{ 0, 0, 1, 1, 0 }, &.{ 0, 0, 1, 1, 0 } };
    const door = [_][]const u16{ &.{ 1, 2 }, &.{ 1, 2 }, &.{}, &.{ 3, 4 }, &.{ 3, 4 }, &.{ 3, 4, 5 } };
    for (steps, od, counts, door) |s, o, c, d| {
        try expectStep(&p, s);
        try expectList(&p.od, o);
        try expectCounts(p.f.c[1..6], c);
        for (1..6) |k| try tst.expectEqual(std.mem.indexOfScalar(u16, d, @intCast(k)) != null, p.f.door.?[k] == p.f.epoch);
    }
}

test "W-TinyLFU: the window's overflow candidate must beat main's oldest line" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: freq.WTinyLfu = try .init(arena.allocator(), 4, 0.25, 100, false);
    try tst.expectEqual(@as(usize, 1), p.wcap);
    try tst.expectEqual(@as(usize, 3), p.mcap);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 3, 4 }, .miss = 2, .adm = 4, .evi = 0 },
        .{ .keys = &.{1}, .miss = 0, .adm = 4, .evi = 0 },
        .{ .keys = &.{ 5, 6 }, .miss = 2, .adm = 6, .evi = 2 }, // candidates 4 and 5 lose to 2 (seen as often): both leave
        .{ .keys = &.{7}, .miss = 1, .adm = 7, .evi = 3 },
        .{ .keys = &.{ 1, 7 }, .miss = 0, .adm = 7, .evi = 3 },
    };
    const win = [_][]const u16{ &.{2}, &.{4}, &.{4}, &.{6}, &.{7}, &.{7} };
    const main = [_][]const u16{ &.{1}, &.{ 1, 2, 3 }, &.{ 2, 3, 1 }, &.{ 2, 3, 1 }, &.{ 2, 3, 1 }, &.{ 2, 3, 1 } };
    for (steps, win, main) |s, w, m| {
        try expectStep(&p, s);
        try expectList(&p.win, w);
        try expectList(&p.main, m);
    }
}

fn expectLfu(p: *const freq.Lfu, rec: []const u16, counts: []const u32) !void {
    try expectList(&p.rec, rec);
    try expectCounts(p.c[1 .. 1 + counts.len], counts);
}

test "LFU: lowest count first, oldest within a count, aged every W accesses; history kept or restarted at eviction" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{ 1, 3 }, .miss = 1, .adm = 3, .evi = 0 },
        .{ .keys = &.{ 1, 4 }, .miss = 1, .adm = 4, .evi = 1 }, // 6 accesses: counts halve
        .{ .keys = &.{ 5, 1 }, .miss = 1, .adm = 5, .evi = 2 },
        .{ .keys = &.{ 2, 5 }, .miss = 1, .adm = 6, .evi = 3 },
        .{ .keys = &.{6}, .miss = 1, .adm = 7, .evi = 4 },
        .{ .keys = &.{ 7, 6 }, .miss = 1, .adm = 8, .evi = 5 }, // 6 accesses again
    };
    const rec = [_][]const u16{ &.{ 1, 2 }, &.{ 2, 1, 3 }, &.{ 3, 1, 4 }, &.{ 4, 1, 5 }, &.{ 1, 5, 2 }, &.{ 1, 5, 6 }, &.{ 5, 6, 7 } };
    const kept = [_][]const u32{ &.{ 1, 1, 0, 0, 0, 0, 0 }, &.{ 2, 1, 1, 0, 0, 0, 0 }, &.{ 1, 0, 0, 0, 0, 0, 0 }, &.{ 2, 0, 0, 0, 1, 0, 0 }, &.{ 2, 1, 0, 0, 2, 0, 0 }, &.{ 2, 1, 0, 0, 2, 1, 0 }, &.{ 1, 0, 0, 0, 1, 1, 0 } };
    const restarted = [_][]const u32{ &.{ 1, 1, 0, 0, 0, 0, 0 }, &.{ 2, 1, 1, 0, 0, 0, 0 }, &.{ 1, 0, 0, 0, 0, 0, 0 }, &.{ 2, 0, 0, 0, 1, 0, 0 }, &.{ 2, 1, 0, 0, 2, 0, 0 }, &.{ 2, 0, 0, 0, 2, 1, 0 }, &.{ 0, 0, 0, 0, 1, 1, 0 } };
    var hist: freq.Lfu = try .init(arena.allocator(), 3, 6, true);
    var restart: freq.Lfu = try .init(arena.allocator(), 3, 6, false);
    for (steps, rec, kept, restarted) |s, r, h, x| {
        try expectStep(&hist, s);
        try expectLfu(&hist, r, h);
        try expectStep(&restart, s);
        try expectLfu(&restart, r, x);
    }
    // the lowest-count line (2, count 2) is requested by the call, so the victim is the next count's line (1)
    var q: freq.Lfu = try .init(arena.allocator(), 2, 1000, true);
    try expectStep(&q, .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 });
    try expectStep(&q, .{ .keys = &.{1}, .miss = 0, .adm = 2, .evi = 0 });
    try expectStep(&q, .{ .keys = &.{1}, .miss = 0, .adm = 2, .evi = 0 });
    try expectStep(&q, .{ .keys = &.{ 2, 5 }, .miss = 1, .adm = 3, .evi = 1 });
    try expectLfu(&q, &.{ 2, 5 }, &.{ 3, 2, 0, 0, 1 });
    try expectStep(&q, .{ .keys = &.{ 5, 7 }, .miss = 1, .adm = 4, .evi = 2 });
    try expectLfu(&q, &.{ 5, 7 }, &.{ 3, 2, 0, 0, 2, 0, 1 });
}

test "LRU with frequency: the lowest count among the K oldest evictable lines, the older on a tie" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: freq.LruFreq = try .init(arena.allocator(), 3, 2, 100);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2, 3 }, .miss = 3, .adm = 3, .evi = 0 },
        .{ .keys = &.{1}, .miss = 0, .adm = 3, .evi = 0 },
        .{ .keys = &.{1}, .miss = 0, .adm = 3, .evi = 0 },
        .{ .keys = &.{4}, .miss = 1, .adm = 4, .evi = 1 }, // of the two oldest (2, 3), equal counts: the older, 2
        .{ .keys = &.{ 5, 1 }, .miss = 1, .adm = 5, .evi = 2 },
        .{ .keys = &.{ 6, 2 }, .miss = 2, .adm = 7, .evi = 4 },
    };
    const od = [_][]const u16{ &.{ 1, 2, 3 }, &.{ 2, 3, 1 }, &.{ 2, 3, 1 }, &.{ 3, 1, 4 }, &.{ 4, 1, 5 }, &.{ 1, 6, 2 } };
    const counts = [_][]const u32{ &.{ 1, 1, 1, 0, 0, 0 }, &.{ 2, 1, 1, 0, 0, 0 }, &.{ 3, 1, 1, 0, 0, 0 }, &.{ 3, 1, 1, 1, 0, 0 }, &.{ 4, 1, 1, 1, 1, 0 }, &.{ 4, 2, 1, 1, 1, 1 } };
    for (steps, od, counts) |s, o, c| {
        try expectStep(&p, s);
        try expectList(&p.od, o);
        try expectCounts(p.f.c[1..7], c);
    }
    // K = 3: the oldest evictable lines are 2 and 3 (count 2 each, 1 is the call's own line): the older, 2, goes; then 3
    var q: freq.LruFreq = try .init(arena.allocator(), 3, 3, 1000);
    for ([_]Step{
        .{ .keys = &.{ 1, 2, 3 }, .miss = 3, .adm = 3, .evi = 0 },
        .{ .keys = &.{2}, .miss = 0, .adm = 3, .evi = 0 },
        .{ .keys = &.{3}, .miss = 0, .adm = 3, .evi = 0 },
        .{ .keys = &.{ 1, 4, 5 }, .miss = 2, .adm = 5, .evi = 2 },
    }) |s| try expectStep(&q, s);
    try expectList(&q.od, &.{ 1, 4, 5 });
}

test "ARC: ghost hits adapt the target, victims skip the call's lines, evictions go to the ghost lists" {
    var arena: std.heap.ArenaAllocator = .init(tst.allocator);
    defer arena.deinit();
    var p: pol.Arc = try .init(arena.allocator(), 3);
    const steps = [_]Step{
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 2, .evi = 0 },
        .{ .keys = &.{3}, .miss = 1, .adm = 3, .evi = 0 },
        .{ .keys = &.{1}, .miss = 0, .adm = 3, .evi = 0 }, // 1 moves from T1 to T2
        .{ .keys = &.{4}, .miss = 1, .adm = 4, .evi = 1 },
        .{ .keys = &.{2}, .miss = 1, .adm = 5, .evi = 2 }, // a B1 ghost hit: p grows
        .{ .keys = &.{ 5, 6 }, .miss = 2, .adm = 7, .evi = 4 },
        .{ .keys = &.{3}, .miss = 1, .adm = 8, .evi = 5 },
        .{ .keys = &.{ 7, 8, 9 }, .miss = 3, .adm = 11, .evi = 8 },
        .{ .keys = &.{ 1, 2 }, .miss = 2, .adm = 13, .evi = 10 }, // ghost hits in B2 then B1 move p both ways
    };
    const t1 = [_][]const u16{ &.{ 1, 2 }, &.{ 1, 2, 3 }, &.{ 2, 3 }, &.{ 3, 4 }, &.{4}, &.{ 5, 6 }, &.{ 6, 3 }, &.{ 7, 8, 9 }, &.{9} };
    const t2 = [_][]const u16{ &.{}, &.{}, &.{1}, &.{1}, &.{ 1, 2 }, &.{2}, &.{2}, &.{}, &.{ 1, 2 } };
    const b1 = [_][]const u16{ &.{}, &.{}, &.{}, &.{2}, &.{3}, &.{4}, &.{5}, &.{}, &.{ 7, 8 } };
    const b2 = [_][]const u16{ &.{}, &.{}, &.{}, &.{}, &.{}, &.{1}, &.{1}, &.{ 1, 2 }, &.{} };
    const target = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 1, 0 };
    for (steps, t1, t2, b1, b2, target) |s, a, b, c, d, want_p| {
        try expectStep(&p, s);
        try expectList(&p.t1, a);
        try expectList(&p.t2, b);
        try expectList(&p.b1, c);
        try expectList(&p.b2, d);
        try tst.expectEqual(want_p, p.p);
    }
}

// ------------------------------------------------------------------ the whole report on synthetic traces

/// The synthetic generator the expected reports were made with (a Python twin printed them): per (trace, row, layer) eight distinct experts of a
/// 48-wide window that slides every 20 rows, one in five calls swapping its last pick for a cold expert.
fn mix(x0: u32) u32 {
    var x = if (x0 == 0) 1 else x0;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

fn picks(ts: u32, t: u32, layer: u32) [8]u8 {
    const s = mix(mix(ts *% 2654435761 +% t *% 40503 +% layer *% 9973 +% 7));
    const shift = 3 * (t / 20);
    const off = (s >> 4) % 48;
    const steps = [_]u32{ 1, 5, 7, 11 };
    const step = steps[(s >> 12) % 4];
    var ids: [8]u8 = undefined;
    for (&ids, 0..) |*id, j| id.* = @intCast((layer * 13 + shift + (off + step * @as(u32, @intCast(j))) % 48) % 256);
    if ((s >> 16) % 5 == 0) ids[7] = @intCast((layer * 13 + shift + 48 + (s >> 20) % 150) % 256);
    return ids;
}

const Spec = struct { name: []const u8, cat: []const u8, ts: u32, npre: u32, ndec: u32 };

fn writeDataset(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, dir: []const u8, specs: []const Spec) !void {
    try root.createDir(io, dir, .default_dir);
    var meta: std.ArrayList(u8) = .empty;
    defer meta.deinit(a);
    try meta.appendSlice(a, "[\n");
    for (specs, 0..) |s, i| {
        const rows = s.npre + s.ndec;
        const ids = try a.alloc(u8, rows * 320);
        defer a.free(ids);
        const kind = try a.alloc(u8, rows);
        defer a.free(kind);
        for (0..rows) |t| {
            kind[t] = @intFromBool(t >= s.npre);
            for (0..40) |layer| ids[(t * 40 + layer) * 8 ..][0..8].* = picks(s.ts, @intCast(t), @intCast(layer));
        }
        const ids_npy = try npy.encode(a, "|u1", &.{ rows, 40, 8 }, ids);
        defer a.free(ids_npy);
        const kind_npy = try npy.encode(a, "|u1", &.{rows}, kind);
        defer a.free(kind_npy);
        const zip = try buildZip(a, &.{ .{ .name = "ids.npy", .data = ids_npy }, .{ .name = "kind.npy", .data = kind_npy } }, false);
        defer a.free(zip);
        const path = try std.fmt.allocPrint(a, "{s}/{s}.npz", .{ dir, s.name });
        defer a.free(path);
        try root.writeFile(io, .{ .sub_path = path, .data = zip });
        try meta.print(a, "{s} {{\"name\": \"{s}\", \"cat\": \"{s}\", \"think\": false, \"prompt_tokens\": {d}, \"sample\": \"a \\\"quoted\\\" text\"}}\n", .{ if (i > 0) "," else " ", s.name, s.cat, s.npre });
    }
    try meta.appendSlice(a, "]\n");
    const meta_path = try std.fmt.allocPrint(a, "{s}/meta.json", .{dir});
    defer a.free(meta_path);
    try root.writeFile(io, .{ .sub_path = meta_path, .data = meta.items });
}

fn dropTimingLines(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "(replay took ") or std.mem.startsWith(u8, line, "(total wall time ")) continue;
        try out.appendSlice(a, line);
        try out.append(a, '\n');
    }
    const trimmed = std.mem.trimEnd(u8, out.items, "\n");
    out.shrinkRetainingCapacity(trimmed.len);
    return out.toOwnedSlice(a);
}

fn expectReport(want: []const u8, got: []const u8) !void {
    try tst.expect(std.mem.indexOf(u8, got, "\n(replay took ") != null);
    try tst.expect(std.mem.indexOf(u8, got, "\n(total wall time ") != null);
    const a = tst.allocator;
    const norm = try dropTimingLines(a, got);
    defer a.free(norm);
    try tst.expectEqualStrings(std.mem.trimEnd(u8, want, "\n"), norm);
}

test "the report on synthetic traces is the Python original's, whatever the worker count" {
    const io = tst.io;
    const a = tst.allocator;
    var tmp = tst.tmpDir(.{});
    defer tmp.cleanup();
    var shorts: [12]Spec = undefined;
    var names: [12][3]u8 = undefined;
    var cats: [12][1]u8 = undefined;
    for (&shorts, &names, &cats, 0..) |*s, *n, *c, i| {
        n.* = .{ 'p', '0' + @as(u8, @intCast(i / 10)), '0' + @as(u8, @intCast(i % 10)) };
        c.* = .{'a' + @as(u8, @intCast(i % 3))};
        s.* = .{ .name = n, .cat = c, .ts = @intCast(i + 1), .npre = @intCast(4 + i % 4), .ndec = @intCast(30 + 3 * i) };
    }
    try writeDataset(a, io, tmp.dir, "traces", &shorts);
    try writeDataset(a, io, tmp.dir, "traces_long", &.{
        .{ .name = "g0", .cat = "a", .ts = 100, .npre = 5, .ndec = 100 },
        .{ .name = "g1", .cat = "b", .ts = 101, .npre = 5, .ndec = 130 },
    });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const traces = try std.fs.path.join(arena.allocator(), &.{ root, "traces" });

    // every configuration, the long set found next to the traces, three workers
    var full: std.Io.Writer.Allocating = .init(a);
    defer full.deinit();
    try sim.execute(arena.allocator(), io, &full.writer, .{ .traces = traces, .cap = 1400, .jobs_text = "3", .jobs = 3 });
    try expectReport(expected.pipeline_full, full.written());

    // the short set only, the LRU family, one worker
    var lru: std.Io.Writer.Allocating = .init(a);
    defer lru.deinit();
    try sim.execute(arena.allocator(), io, &lru.writer, .{ .traces = traces, .long = "none", .cap = 1400, .jobs_text = "1", .jobs = 1, .only = "LRU" });
    try expectReport(expected.pipeline_lru_short_only, lru.written());

    // the same selection on five workers prints the same report
    var five: std.Io.Writer.Allocating = .init(a);
    defer five.deinit();
    try sim.execute(arena.allocator(), io, &five.writer, .{ .traces = traces, .long = "none", .cap = 1400, .jobs_text = "5", .jobs = 5, .only = "LRU" });
    try expectReport(expected.pipeline_lru_short_only, five.written());

    // without the LRU baseline the warm-start tables have nothing to subtract from
    var nobase: std.Io.Writer.Allocating = .init(a);
    defer nobase.deinit();
    try tst.expectError(error.NoBaseline, sim.execute(arena.allocator(), io, &nobase.writer, .{ .traces = traces, .long = "none", .cap = 1400, .jobs_text = "2", .jobs = 2, .only = "warm" }));
}

test "the handshake replay: the split pool equals the always-asking pool after every token, and the card passes the layers whose picks are all resident" {
    const io = tst.io;
    const a = tst.allocator;
    var tmp = tst.tmpDir(.{});
    defer tmp.cleanup();
    try writeDataset(a, io, tmp.dir, "traces", &.{
        .{ .name = "a0", .cat = "a", .ts = 1, .npre = 4, .ndec = 150 },
        .{ .name = "a1", .cat = "a", .ts = 2, .npre = 4, .ndec = 120 },
        .{ .name = "b0", .cat = "b", .ts = 3, .npre = 4, .ndec = 90 },
    });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const traces = try std.fs.path.join(arena.allocator(), &.{ root, "traces" });
    const set = try @import("expert_cache_sim_data.zig").load(arena.allocator(), io, traces);
    for ([_]hs.Config{
        .{ .cap = 1400, .warm_tokens = 20 },
        .{ .cap = 600, .protect_pct = 0, .warm_tokens = 20 },
        .{ .cap = 900, .copies = 3, .warm_tokens = 20 },
    }) |cfg| {
        const t = try hs.replay(a, cfg, set);
        try tst.expectEqual(@as(u64, 3), t.prompts);
        try tst.expectEqual(@as(u64, 360), t.tokens);
        try tst.expectEqual(@as(u64, 360 * 40), t.stages);
        try tst.expectEqual(@as(u64, 0), t.mismatches);
        // some layers are passed and some ask, and the per-token histogram accounts for every steady token
        try tst.expect(t.passed > 0 and t.passed < t.stages);
        var by: u64 = 0;
        for (t.by_passed) |n| by += n;
        try tst.expectEqual(t.steady_tokens, by);
        var per: u64 = 0;
        for (t.per_layer) |n| per += n;
        try tst.expectEqual(t.steady_passed, per);
    }
    // the report names the check and prints one row of ten layers a line
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try sim.execute(arena.allocator(), io, &text.writer, .{ .traces = traces, .cap = 1400, .jobs_text = "1", .jobs = 1, .handshake = true });
    try tst.expect(std.mem.indexOf(u8, text.written(), "0 of 360 tokens differ (must be 0)") != null);
    try tst.expect(std.mem.indexOf(u8, text.written(), "  L30-39 ") != null);
}
