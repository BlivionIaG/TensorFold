//! Layer ranges and the split stage with no device: the dry-run launches of a ranged decode step against the unsplit
//! step's, the allocations a range saves, and the stage's recorded operations in the pipeline's order. Kernels are
//! logged by name on a stand-in runtime (split_fake.zig) and compute nothing, so these check plumbing, not tokens.

const std = @import("std");
const hip = @import("hip");
const view = @import("model/view.zig");
const fwd = @import("forward/forward.zig");
const win = @import("forward/window.zig");
const state = @import("forward/state.zig");
const pages = @import("forward/pages.zig");
const round = @import("engine/round.zig");
const fake = @import("split_fake.zig");

const T = std.testing;
const gpa = T.allocator;

pub const Rig = struct {
    rt: *hip.Runtime,
    lib: hip.Launcher,
    arena: hip.Arena,

    pub fn init() !Rig {
        if (!fake.available()) return error.SkipZigTest;
        const rt = try gpa.create(hip.Runtime);
        errdefer gpa.destroy(rt);
        rt.* = fake.runtime();
        const lib = try hip.Launcher.load(rt, (hip.Policy{}).choices(), hip.kernels.images);
        fake.clear();
        return .{ .rt = rt, .lib = lib, .arena = hip.Arena.counting() };
    }

    pub fn deinit(r: *Rig) void {
        gpa.destroy(r.rt);
    }

    pub fn ops(r: *Rig) hip.ops.Ops {
        r.arena.reset();
        return .{ .l = &r.lib, .bf16 = false, .stream = fake.fake_stream, .arena = &r.arena };
    }
};

test "split stage: the launcher loads on the recording runtime" {
    var rig = try Rig.init();
    defer rig.deinit();
    try T.expect(rig.lib.fuse);
    try T.expectEqual(@as(usize, 0), fake.launches().len);
}

// Fake window and plan addresses: nonzero, 16-byte aligned, never dereferenced.
pub const window_in: u64 = 1 << 42;
pub const window_out: u64 = 1 << 43;
pub const status_at: u64 = 1 << 44;
pub const tokens_at: u64 = 1 << 45;
pub const argmax_at: u64 = 1 << 46;
pub const plan_at: u64 = 1 << 47;

pub const Seq = []const [*:0]const u8;

pub fn same(a: [*:0]const u8, b: [*:0]const u8) bool {
    return std.mem.eql(u8, std.mem.span(a), std.mem.span(b));
}

pub fn expectSeq(want: []const [*:0]const u8, got: Seq) !void {
    var i: usize = 0;
    while (i < @min(want.len, got.len)) : (i += 1) if (!same(want[i], got[i])) {
        std.debug.print("launch {d}: want {s}, got {s}\n", .{ i, want[i], got[i] });
        return error.TestExpectedEqual;
    };
    try T.expectEqual(want.len, got.len);
}

/// Where each layer's launches end in a decode log (`ends[k]` follows the range's k-th layer).
pub const Marks = struct {
    ends: [32]usize = undefined,
    n: usize = 0,

    pub fn layer(ctx: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
        const m: *Marks = @ptrCast(@alignCast(ctx));
        m.ends[m.n] = fake.launches().len;
        m.n += 1;
    }

    pub fn trace(m: *Marks) fwd.Trace {
        return .{ .ctx = m, .layer = layer };
    }
};

/// One decode round (one row, two slots) of `m` over its range, logged; the caller frees the sequence.
pub fn decode(rig: *Rig, m: *const view.Model, hand: fwd.Handoff, marks: *Marks) !Seq {
    fake.clear();
    marks.n = 0;
    const o = rig.ops();
    var table: [32][2]u64 = undefined;
    try win.kept(o, m, 1, &table);
    const args: hip.plan_ops.Args = .{ .pos = plan_at, .slot = plan_at + 64, .first = plan_at + 128, .count = plan_at + 192, .desc = plan_at + 256, .snaps = plan_at + 512, .pages = @intCast(state.tableWord(32)), .pool = 4 };
    const r: win.Round = .{ .plan = .{ .args = args, .rows = 1, .slots = 2 }, .tokens = tokens_at, .span = 128, .inputs = &table, .positions3 = plan_at + 1024, .rotary = plan_at + 1088, .hand = hand };
    _ = try round.forwardOn(o, m, r, marks.trace(), argmax_at);
    return gpa.dupe([*:0]const u8, fake.launches());
}

pub fn handoff(m: *const view.Model, stamped: bool) fwd.Handoff {
    return .{
        .src = if (m.range.embeds()) 0 else window_in,
        .stamp = if (stamped) .{ .seq = 7, .status = status_at, .retries = 1000 } else null,
        .dst = if (m.headed()) 0 else window_out,
    };
}

const prologue = 2; // embed, then the first layer's input norm

/// The launches `r` (a ranged decode of layers [a, b)) must have, from the unsplit decode `u` and its layer marks:
/// the hand-in and the input norm, the unsplit launches of layers a..b-1 (the last layer's fused tail swapped for the
/// plain residual add before the hand-out), then the head of the unsplit decode or the hand-out.
fn expectSlice(u: Seq, um: Marks, r: Seq, a: usize, b: usize, stamped: bool, residual_name: [*:0]const u8) !void {
    var want: std.ArrayList([*:0]const u8) = .empty;
    defer want.deinit(gpa);
    if (a == 0) {
        try want.append(gpa, u[0]);
    } else {
        try want.append(gpa, if (stamped) "tf_split_consume_stamped" else "tf_split_import_row");
    }
    try want.append(gpa, u[1]);
    const from = if (a == 0) prologue else um.ends[a - 1];
    const to = um.ends[b - 1];
    if (b == 32) {
        try want.appendSlice(gpa, u[from..]);
    } else {
        try want.appendSlice(gpa, u[from .. to - 1]);
        try want.append(gpa, residual_name);
        try want.append(gpa, "tf_split_export_row");
    }
    try expectSeq(want.items, r);
}

fn residualLaunch(rig: *Rig) ![*:0]const u8 {
    fake.clear();
    const o = rig.ops();
    const x: hip.ops.Tensor = .{ .ptr = plan_at, .kind = .f16 };
    var f = try fake.Fixture.init(gpa, .{ .first = 0, .last = 1 }, .f16);
    defer f.deinit();
    try fwd.residual(o, &f.model, x, x, 2560);
    try T.expectEqual(@as(usize, 1), fake.launches().len);
    return fake.launches()[0];
}

test "split stage: a ranged decode step is the unsplit step's slice plus its hand-offs" {
    var rig = try Rig.init();
    defer rig.deinit();
    var whole = try fake.Fixture.init(gpa, .whole, .f16);
    defer whole.deinit();
    var um: Marks = .{};
    const u = try decode(&rig, &whole.model, .{}, &um);
    defer gpa.free(u);
    try T.expectEqual(@as(usize, 32), um.n);
    try T.expect(std.mem.indexOf(u8, std.mem.span(u[0]), "embed") != null);
    try T.expect(std.mem.indexOf(u8, std.mem.span(u[1]), "rms") != null);
    // the unsplit step ends in the head: a product and the argmax
    try T.expect(std.mem.indexOf(u8, std.mem.span(u[u.len - 1]), "argmax") != null);
    const residual_name = try residualLaunch(&rig);
    const cuts = [_]view.Range{ .{ .first = 0, .last = 16 }, .{ .first = 8, .last = 24 }, .{ .first = 16 }, .{ .first = 0, .last = 1 }, .{ .first = 31 }, .{ .first = 12, .last = 13 } };
    for (cuts) |range| {
        for ([_]bool{ false, true }) |stamped| {
            var f = try fake.Fixture.init(gpa, range, .f16);
            defer f.deinit();
            var rm: Marks = .{};
            const r = try decode(&rig, &f.model, handoff(&f.model, stamped and !range.embeds()), &rm);
            defer gpa.free(r);
            const a = range.first;
            const b = range.within(32).last;
            try T.expectEqual(b - a, rm.n);
            try expectSlice(u, um, r, a, b, stamped, residual_name);
            // embedding only when it starts at layer 0, the head only when it ends the model
            for (r, 0..) |name, i| {
                const embed = std.mem.indexOf(u8, std.mem.span(name), "embed") != null;
                try T.expectEqual(i == 0 and a == 0, embed);
            }
            try T.expectEqual(b == 32, std.mem.indexOf(u8, std.mem.span(r[r.len - 1]), "argmax") != null);
        }
    }
}

/// A prompt span of `len` rows from position `pos0` over `m`'s range, logged; `head`: the last stage's projection of
/// the last row and its argmax follow (`Engine.spanStage`).
pub fn span(rig: *Rig, m: *const view.Model, hand: fwd.Handoff, len: usize, pos0: usize, head: bool, marks: *Marks) !Seq {
    var pool = try pages.Pool.init(gpa, rig.rt, m, 4, false);
    defer pool.deinit();
    var caches = try state.Caches.init(gpa, &pool, m, 256);
    defer caches.deinit(gpa);
    _ = try caches.grow(gpa, 256);
    fake.clear();
    marks.n = 0;
    const o = rig.ops();
    const hidden = try fwd.spanWith(o, m, &caches, tokens_at, len, pos0, marks.trace(), hand);
    if (head and m.headed()) {
        const y = try o.project(fwd.at(hidden, (len - 1) * m.spec.hidden), m.head.?, 1, false);
        try o.argmaxRows(y, 1, m.head.?.n, argmax_at);
    }
    return gpa.dupe([*:0]const u8, fake.launches());
}

test "split stage: a ranged prompt span is the unsplit span's slice plus its hand-offs" {
    var rig = try Rig.init();
    defer rig.deinit();
    var whole = try fake.Fixture.init(gpa, .whole, .f16);
    defer whole.deinit();
    var um: Marks = .{};
    const len = 3;
    const u = try span(&rig, &whole.model, .{}, len, 64, false, &um);
    defer gpa.free(u);
    try T.expect(std.mem.indexOf(u8, std.mem.span(u[0]), "embed") != null);
    try T.expect(std.mem.indexOf(u8, std.mem.span(u[u.len - 1]), "rms") != null);
    for ([_]view.Range{ .{ .first = 0, .last = 16 }, .{ .first = 8, .last = 24 }, .{ .first = 16 }, .{ .first = 3, .last = 4 } }) |range| {
        // the rows come in plain (one import for all) or stamped (one consume launch a row)
        for ([2]bool{ false, true }) |stamped| {
            if (stamped and range.first == 0) continue;
            var f = try fake.Fixture.init(gpa, range, .f16);
            defer f.deinit();
            var rm: Marks = .{};
            const r = try span(&rig, &f.model, handoff(&f.model, stamped), len, 64, false, &rm);
            defer gpa.free(r);
            const a = range.first;
            const b = range.within(32).last;
            var want: std.ArrayList([*:0]const u8) = .empty;
            defer want.deinit(gpa);
            if (a == 0) try want.append(gpa, u[0]) else if (stamped) {
                for (0..len) |_| try want.append(gpa, "tf_split_consume_stamped");
            } else try want.append(gpa, "tf_split_import_row");
            try want.appendSlice(gpa, u[if (a == 0) 1 else um.ends[a - 1]..um.ends[b - 1]]);
            if (b == 32) try want.append(gpa, u[u.len - 1]) else try want.append(gpa, "tf_split_export_row");
            try expectSeq(want.items, r);
        }
    }
}
