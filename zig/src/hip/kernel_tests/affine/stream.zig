//! The stream tile's staged x against each lane's own loads, at every width, group, row count and columns a lane.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const cases = @import("cases.zig");
const prod = @import("product.zig");

/// The bytes every launch below writes, hashed in order, as the stream tile before the stage wrote them.
const pinned_f16: u64 = 0x36ef53355d892c81;
/// gfx900 has no dot2: each fp16 pair is two fused steps, which rounds differently from RDNA2's dot2, so its bytes differ and have their own pin
/// (recorded on a Vega 56 through hipmac, after the staged-x and float64-bound checks of this and the other affine tests passed on it).
const pinned_f16_gfx900: u64 = 0x19fdf3b06c3a5b4f;
/// With TF_G9_MIX=2 (the default) a denormal fp16 x counts as zero in the stream tile: its own pin (recorded on the card).
const pinned_f16_gfx900_flush: u64 = 0xbfd29f54a3b90d6a;
const pinned_bf16: u64 = 0x6b703ab847138aad;

/// Row counts: one row a block (the staged tile of gfx900), two, a partial block of four, and the y blocks above.
const row_counts = [_]usize{ 1, 2, 3, 5, 16 };

/// How a launch writes: fp32, the activation type, silu(gate) * up of a stacked pair, or fp32 of a two-expert plan.
const Out = enum { f32, act, pair, routed };

const Shape = struct { n: usize, k: usize, bits: u8, group: u16 };

/// One product at `m` rows through every width of the tile, staged and not: the differing bytes; the hash of the rest.
fn product(t: *prod.Rig, rng: *data.Rng, s: Shape, out: Out, hash: *u64) !usize {
    const most = row_counts[row_counts.len - 1];
    const words_row = s.k * s.bits / 32;
    const groups = s.k / s.group;
    const hx = try data.fill(gpa, u16, most * s.k, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const experts: usize = if (out == .routed) 2 else 1;
    const hw = try data.fill(gpa, u32, experts * s.n * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * s.n * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * s.n * groups, rng, data.makeBias);
    defer gpa.free(hb);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    // a plan's rows reversed: item 0 (expert 0) takes the first half of the members, item 1 the rest
    var members: [most]c_int = undefined;
    for (&members, 0..) |*mb, i| mb.* = @intCast(most - 1 - i);
    var member_buf = try Buffer.fromHost(t.base, @as([]const c_int, &members));
    defer member_buf.free();
    var plan = try Buffer.alloc(t.base, 6 * @sizeOf(c_int));
    defer plan.free();
    const cols = if (out == .pair) s.n / 2 else s.n;
    const size: usize = if (out == .f32 or out == .routed) 4 else 2;
    const bytes = most * cols * size;
    var outs: [2]Buffer = .{ try Buffer.alloc(t.base, bytes), try Buffer.alloc(t.base, bytes) };
    defer for (&outs) |*o| o.free();
    const got = try gpa.alloc(u8, 2 * bytes);
    defer gpa.free(got);
    var k = t.kernels.*;
    var bad: usize = 0;
    for (row_counts) |m| for ([_]usize{ 8, 4, 2 }) |cb| {
        const block_rows: usize = if (m <= 2) m else 4;
        if (cb * block_rows > 16 or cb * s.bits > 32) continue;
        const arg: affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = 0,
            .m = @intCast(m),
            .n = @intCast(s.n),
            .k = @intCast(s.k),
            .bits = s.bits,
            .group = s.group,
            .fp16 = @intFromBool(t.fp16),
        };
        var a0 = arg;
        var items: c_int = 1;
        if (out == .routed) {
            const first = (m + 1) / 2;
            const item = [6]c_int{ 0, 0, @intCast(first), 1, @intCast(first), @intCast(m - first) };
            try plan.upload(0, std.mem.sliceAsBytes(&item));
            items = if (m > first) 2 else 1;
            a0.m = @intCast(first);
            a0.route = .{ .items = plan.ptr, .members = member_buf.ptr, .x_div = 1 };
        }
        k.tune.cb = @intCast(cb);
        for (outs, [_]bool{ false, true }) |o, staged| {
            k.stage_on = staged;
            k.tune.stage = @intFromBool(staged);
            var a = a0;
            if (size == 4) a.out = o.ptr else a.out16 = o.ptr;
            try o.fill8(0xA5, null);
            const sides: affine.StreamSides = if (out == .pair) .{ .pair_cols = @intCast(cols) } else .{};
            try k.streamGo(&t.base.r, a, sides, @intCast(cols), items, t.base.stream.handle);
        }
        try t.base.stream.synchronize();
        const n_bytes = if (out == .routed) most * cols * size else m * cols * size;
        try outs[0].download(0, got[0..n_bytes]);
        try outs[1].download(0, got[bytes..][0..n_bytes]);
        hash.* = std.hash.Wyhash.hash(hash.*, got[0..n_bytes]);
        if (!std.mem.eql(u8, got[0..n_bytes], got[bytes..][0..n_bytes])) {
            if (bad < 6) std.debug.print("  stream n{d} k{d} b{d} g{d} {s} m{d} cb{d}: staged x writes other bytes\n", .{ s.n, s.k, s.bits, s.group, @tagName(out), m, cb });
            bad += 1;
        }
    };
    return bad;
}

/// Every width and group at a few stages, an odd stage count or 4096, and a long run, through every output.
pub fn sweep(t: *prod.Rig) !u64 {
    var rng: data.Rng = .{ .state = 0x51ED270B27C1F5E3 };
    var hash: u64 = 0;
    var bad: usize = 0;
    for (cases.widths) |bits| for (cases.groups) |group| {
        // two rows staged up to k 7680, one to 15360
        const ks = [_]usize{ 3 * @as(usize, group), if (group == 32) 2080 else 4096, 9216 };
        for (ks, 0..) |k, j| for (std.enums.values(Out)) |out| {
            const n: usize = if (j == 0) 202 else if (j == 1) 1006 else 4100;
            bad += try product(t, &rng, .{ .n = n, .k = k, .bits = bits, .group = group }, out, &hash);
        };
    };
    try base.expect(bad == 0, "stream: {d} launches with staged x differ from the lanes' own loads", .{bad});
    return hash;
}

test "the stream tile writes the same bytes with x staged in LDS as with each lane's loads, and as before the stage" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = prod.Rig.open(b);
    const hash = try sweep(&rig);
    const gcn5 = std.mem.startsWith(u8, @import("../../kernels.zig").arch, "gfx900");
    const flush = gcn5 and @import("../../kernels.zig").g9_mix == 2;
    const pinned = if (!rig.fp16) pinned_bf16 else if (flush) pinned_f16_gfx900_flush else if (gcn5) pinned_f16_gfx900 else pinned_f16;
    try base.expect(hash == pinned, "stream: the bytes hash to 0x{x}, the tile before the stage wrote 0x{x}", .{ hash, pinned });
}

fn wall_ns() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @intCast(@as(i64, ts.sec) * 1_000_000_000 + ts.nsec);
}

// TF_STREAM_BENCH=1: the stream tile's time at the 4B's decode shapes by row count, back to back on a stream (what a graph replay pays).
test "time the stream tile by rows" {
    if (std.c.getenv("TF_STREAM_BENCH") == null) return error.SkipZigTest;
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var t = prod.Rig.open(b);
    var rng: data.Rng = .{ .state = 0x51ED270B27C1F5E3 };
    const shapes = [_][2]usize{ .{ 18432, 2560 }, .{ 2560, 9216 }, .{ 12352, 2560 }, .{ 2560, 4096 }, .{ 248320, 2560 } };
    const rows = [_]usize{ 1, 2, 3, 4, 8, 16 };
    const most = 16;
    for (shapes) |sh| {
        const n = sh[0];
        const k = sh[1];
        const hx = try data.fill(gpa, u16, most * k, &rng, if (t.fp16) data.makeX16 else data.makeXB);
        defer gpa.free(hx);
        const hw = try data.fill(gpa, u32, n * k / 8, &rng, data.makeWord);
        defer gpa.free(hw);
        const hs = try data.fill(gpa, u16, n * k / 64, &rng, data.makeScale);
        defer gpa.free(hs);
        const hb = try data.fill(gpa, u16, n * k / 64, &rng, data.makeBias);
        defer gpa.free(hb);
        var x = try Buffer.fromHost(t.base, hx);
        defer x.free();
        var words = try Buffer.fromHost(t.base, hw);
        defer words.free();
        var scale = try Buffer.fromHost(t.base, hs);
        defer scale.free();
        var bias = try Buffer.fromHost(t.base, hb);
        defer bias.free();
        var out = try Buffer.alloc(t.base, most * n * 4);
        defer out.free();
        for (rows) |m| {
            const a: affine.Arg = .{
                .x = x.ptr,
                .words = words.ptr,
                .scale = .{ .p = scale.ptr, .kind = 1 },
                .bias = .{ .p = bias.ptr, .kind = 1 },
                .out = out.ptr,
                .m = @intCast(m),
                .n = @intCast(n),
                .k = @intCast(k),
                .bits = 4,
                .group = 64,
                .fp16 = @intFromBool(t.fp16),
            };
            for (0..4) |_| try t.kernels.streamGo(&t.base.r, a, .{}, @intCast(n), 1, t.base.stream.handle);
            try t.base.stream.synchronize();
            const t0 = wall_ns();
            for (0..40) |_| try t.kernels.streamGo(&t.base.r, a, .{}, @intCast(n), 1, t.base.stream.handle);
            try t.base.stream.synchronize();
            const us = @as(f64, @floatFromInt(wall_ns() - t0)) / 1000.0 / 40.0;
            const bytes = @as(f64, @floatFromInt(n * k)) * 0.5625;
            std.debug.print("  stream n{d} k{d} m{d}: {d:.1} us  {d:.0} GB/s of weights\n", .{ n, k, m, us, bytes / us / 1000.0 });
        }
    }
}

// Row r of a several-row launch is the launch of row r alone, bit for bit: the tiles' sums go in the order a row alone takes, at every row tile.
test "a row of a several-row stream launch writes the bytes of that row alone" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var t = prod.Rig.open(b);
    var rng: data.Rng = .{ .state = 0x6A09E667F3BCC908 };
    const rows = [_]usize{ 2, 3, 4, 5, 8, 16 };
    const most = rows[rows.len - 1];
    var bad: usize = 0;
    for ([_][2]usize{ .{ 1006, 2560 }, .{ 202, 4096 }, .{ 300, 9216 } }) |sh| {
        const n = sh[0];
        const k = sh[1];
        const hx = try data.fill(gpa, u16, most * k, &rng, if (t.fp16) data.makeX16 else data.makeXB);
        defer gpa.free(hx);
        const hw = try data.fill(gpa, u32, n * k / 8, &rng, data.makeWord);
        defer gpa.free(hw);
        const hs = try data.fill(gpa, u16, n * k / 64, &rng, data.makeScale);
        defer gpa.free(hs);
        const hb = try data.fill(gpa, u16, n * k / 64, &rng, data.makeBias);
        defer gpa.free(hb);
        var x = try Buffer.fromHost(t.base, hx);
        defer x.free();
        var words = try Buffer.fromHost(t.base, hw);
        defer words.free();
        var scale = try Buffer.fromHost(t.base, hs);
        defer scale.free();
        var bias = try Buffer.fromHost(t.base, hb);
        defer bias.free();
        var outs: [2]Buffer = .{ try Buffer.alloc(t.base, most * n * 4), try Buffer.alloc(t.base, most * n * 4) };
        defer for (&outs) |*o| o.free();
        const got = try gpa.alloc(u8, 2 * most * n * 4);
        defer gpa.free(got);
        for (rows) |m| {
            const arg: affine.Arg = .{
                .x = x.ptr,
                .words = words.ptr,
                .scale = .{ .p = scale.ptr, .kind = 1 },
                .bias = .{ .p = bias.ptr, .kind = 1 },
                .out = outs[0].ptr,
                .m = @intCast(m),
                .n = @intCast(n),
                .k = @intCast(k),
                .bits = 4,
                .group = 64,
                .fp16 = @intFromBool(t.fp16),
            };
            try outs[0].fill8(0xA5, null);
            try outs[1].fill8(0xA5, null);
            try t.kernels.streamGo(&t.base.r, arg, .{}, @intCast(n), 1, t.base.stream.handle);
            for (0..m) |r| {
                var one = arg;
                one.m = 1;
                one.x = x.ptr + r * k * 2;
                one.out = outs[1].ptr + r * n * 4;
                try t.kernels.streamGo(&t.base.r, one, .{}, @intCast(n), 1, t.base.stream.handle);
            }
            try t.base.stream.synchronize();
            const len = m * n * 4;
            try outs[0].download(0, got[0..len]);
            try outs[1].download(0, got[len..][0..len]);
            if (!std.mem.eql(u8, got[0..len], got[len..][0..len])) {
                std.debug.print("  stream n{d} k{d} m{d}: a row differs from its launch alone\n", .{ n, k, m });
                bad += 1;
            }
        }
    }
    try base.expect(bad == 0, "stream: {d} several-row launches differ from their rows alone", .{bad});
}
