//! aot-needs for the exact decoder replay (forward_prefill_cone.zig): its programs' Triton variants, every row count.

const std = @import("std");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const calls = @import("block_calls.zig");
const ced = @import("ced.zig");
const knobs = @import("serve_knobs.zig");
const needs = @import("aot_needs.zig");
const Config = @import("config.zig").Config;

/// Whether the served config runs the exact replay (replay mode, on unless off or under TF_DSV41_PF_TBO).
pub fn wanted(sv: needs.Served) !bool {
    if (!sv.replay) return false;
    return (try knobs.replayExact(&knobs.env)) orelse !(try knobs.pfTbo(&knobs.env));
}

const Ctx = struct {
    cfg: *const Config,
    w: *const block.Widths,
    o: block.Options,
    layers: []const u32,
    n: i64,
    part: block_prefill.Part,
    cand: ?i64,
};

fn at(ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return block_prefill.emitCone(a, c.cfg, c.w, c.o, c.layers, c.n, end - c.n, c.part, c.cand) catch |e| switch (e) {
        error.Unsupported => &.{},
        else => e,
    };
}

pub fn programs(gpa: std.mem.Allocator, n: *needs.Needs, cfg: *const Config, w: *const block.Widths, sv: needs.Served, layers: []const u32) !void {
    const dec = layers[try ced.decoderStart(cfg)..];
    const nd: u32 = @intCast(dec.len);
    var gbuf: [32]ced.Group = undefined;
    const gs = ced.groups(cfg, dec, try knobs.replayGroup(&knobs.env), &gbuf);
    // the forward's own options (2,048 rows), then the workspace's 4,096-row ones under TF_DSV41_PF_4K
    const rows_fwd: i64 = (block.Options{}).prefill_rows;
    var fwd = sv.o;
    fwd.pf4k = false;
    fwd.prefill_rows = rows_fwd;
    var k4 = sv.o;
    k4.prefill_rows = knobs.pf4k_rows;
    const variants: []const block.Options = if (sv.o.pf4k) &.{ fwd, k4 } else &.{fwd};
    for (variants) |o| for (gs, 0..) |g, gi| {
        const full: i64 = ced.groupRows(cfg.window, nd, g.first, std.math.maxInt(u64));
        // the workspace runs cones past the forward's rows; a warm resume's replay runs from one row in the forward's
        const lo: i64 = @min(full, if (o.pf4k) rows_fwd + 1 else 1);
        const hi: i64 = @min(full, o.prefill_rows);
        var c: Ctx = .{ .cfg = cfg, .w = w, .o = o, .layers = dec[g.first .. g.first + g.count + @intFromBool(gi + 1 < gs.len)], .n = 0, .part = if (gi + 1 == gs.len) .cone_last else .cone_mid, .cand = if (gi == 0) null else 0 };
        var rows = lo;
        while (rows <= hi) : (rows += 1) {
            c.n = rows;
            for ([_]i64{ rows, @divTrunc(o.limit, 2), o.limit - 1 }) |end| {
                var pa = std.heap.ArenaAllocator.init(gpa);
                defer pa.deinit();
                _ = try n.program(try at(&c, pa.allocator(), @max(end, rows)));
            }
        }
        c.n = hi;
        try needs.scan(n, .{ .ctx = &c, .emit = at }, c.n, o.limit - 1);
    };
}
