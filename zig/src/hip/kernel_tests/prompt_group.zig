//! A prompt's products sharing x in one launch on gfx11's matrix cores: every side has its own prefill launch's bits.
const std = @import("std");
const rig = @import("rig.zig");
const affine = @import("../launches/affine.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// One side: its words, scales and biases on the device, and the fp32 outputs of the lone and the grouped launch.
const Side = struct { dev: [3]DeviceBuffer, outs: [2]DeviceBuffer, n: usize };

fn arg(t: *const Rig, x: u64, s: Side, rows: usize, k: usize, bits: usize, group: usize, out: u64) affine.Arg {
    return .{
        .x = x,
        .words = at(s.dev[0]),
        .scale = .{ .p = at(s.dev[1]), .kind = 1 },
        .bias = .{ .p = at(s.dev[2]), .kind = 1 },
        .out = out,
        .m = @intCast(rows),
        .n = @intCast(s.n),
        .k = @intCast(k),
        .bits = @intCast(bits),
        .group = @intCast(group),
        .fp16 = @intFromBool(!t.bf16),
    };
}

fn case(t: *Rig, rng: *rig.Rng, rows: usize, ns: []const usize, k: usize, bits: usize, group: usize) !void {
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var sides: [4]Side = undefined;
    for (ns, sides[0..ns.len]) |n, *s| {
        const words = try gpa.alloc(u32, n * k * bits / 32);
        defer gpa.free(words);
        const tables = try gpa.alloc(u16, 2 * n * k / group);
        defer gpa.free(tables);
        for (words) |*w| w.* = @truncate(rng.next() >> 16);
        for (tables[0 .. tables.len / 2]) |*v| v.* = rig.bf16Bits((rng.unit() + 1.0) / 16.0);
        for (tables[tables.len / 2 ..]) |*v| v.* = rig.bf16Bits(rng.unit() / 2.0);
        s.* = .{ .n = n, .dev = .{ try t.upload(words), try t.upload(tables[0 .. tables.len / 2]), try t.upload(tables[tables.len / 2 ..]) }, .outs = .{ try t.alloc(rows * n * 4), try t.alloc(rows * n * 4) } };
    }
    defer for (sides[0..ns.len]) |*s| {
        for (&s.dev) |*b| b.free();
        for (&s.outs) |*b| b.free();
    };
    const kernels = &t.on.affine;
    var group_sides: [4]affine.Side = undefined;
    for (sides[0..ns.len], group_sides[0..ns.len]) |s, *g| {
        try kernels.run(&t.r, arg(t, at(x), s, rows, k, bits, group, at(s.outs[0])), 3, t.stream.handle, 0, 1, false);
        g.* = .{ .words = at(s.dev[0]), .scale = at(s.dev[1]), .bias = at(s.dev[2]), .n = @intCast(s.n), .out = at(s.outs[1]) };
    }
    const first = arg(t, at(x), sides[0], rows, k, bits, group, 0);
    const e = kernels.groupEntry(first, group_sides[0..ns.len], .prefill) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.endsWith(u8, e.id, "matrix"));
    try e.launch(kernels, .{ .r = &t.r, .s = t.stream.handle, .arg = first, .sides = group_sides[0..ns.len] });
    try t.stream.synchronize();
    for (sides[0..ns.len]) |s| {
        const lone = try rig.download(u8, s.outs[0]);
        defer gpa.free(lone);
        const grouped = try rig.download(u8, s.outs[1]);
        defer gpa.free(grouped);
        try std.testing.expectEqualSlices(u8, lone, grouped);
    }
}

test "a prompt's products sharing x in one matrix-core launch keep each lone launch's bits, at every width" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.on.affine.matrix) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        // qkv, z, a and b of a DeltaNet layer: a side narrower than a tile, and rows ending inside one
        try case(t, &rng, 200, &.{ 1024, 512, 32, 48 }, 2048, bits, group);
    };
    try case(t, &rng, 33, &.{ 4096, 1024, 1024 }, 2048, 4, 64);
    try case(t, &rng, 512, &.{ 12288, 12288 }, 5120, 4, 64);
    try case(t, &rng, 1024, &.{ 2560, 2560 }, 1024, 8, 32);
}
