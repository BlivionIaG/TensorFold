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
const pinned_bf16: u64 = 0x6b703ab847138aad;

/// Row counts: one and two rows a block (the staged tiles), a partial block of four, and the y blocks above.
const row_counts = [_]usize{ 1, 2, 3, 5, 16 };

/// How a launch writes: fp32, the activation type, silu(gate) * up of a stacked pair, or fp32 of a two-expert plan.
const Out = enum { f32, act, pair, routed };

const Shape = struct { n: usize, k: usize, bits: u8, group: u16 };

/// The waves a launch of `cb` columns a lane makes, as streamGo counts them: the target that picks that width.
fn wavesOf(s: Shape, m: usize, cb: usize, out: Out) u64 {
    var lpc_log2: u6 = 0;
    while ((@as(usize, 1) << lpc_log2) < s.k / 32 and lpc_log2 < 5) lpc_log2 += 1;
    const rows: usize = if (m == 1) 1 else if (m == 2) 2 else 4;
    const cols = if (out == .pair) s.n / 2 else s.n;
    const per_block = 4 * (@as(usize, 32) >> lpc_log2) * (if (out == .pair) cb / 2 else cb);
    return (cols + per_block - 1) / per_block * 4 * ((m + rows - 1) / rows);
}

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
        // the widest width meets a target of one wave; a narrower one the waves it makes itself
        k.waves_wanted = if (cb == 8) 1 else wavesOf(s, @intCast(a0.m), cb, out) * @as(u64, @intCast(items));
        for (outs, [_]bool{ false, true }) |o, staged| {
            k.stage_on = staged;
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
    const pinned = if (rig.fp16) pinned_f16 else pinned_bf16;
    try base.expect(hash == pinned, "stream: the bytes hash to 0x{x}, the tile before the stage wrote 0x{x}", .{ hash, pinned });
}
