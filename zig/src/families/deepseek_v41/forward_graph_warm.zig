//! One slot's decode graphs at boot (TF_DSV41_GRAPH_PRECAPTURE): each planned window's program captured, never run.

const forward = @import("forward.zig");
const fwd_follow = @import("forward_follow.zig");
const fwd_glue = @import("forward_glue.zig");
const fwd_graph = @import("forward_graph.zig");
const graphs = @import("graphs.zig");
const plan_mod = @import("graph_plan.zig");
const pre = @import("graph_precapture.zig");
const pf_glue = @import("forward_prefill_glue.zig");
const Forward = forward.Forward;
const Graphed = fwd_graph.Graphed;

/// Rank 0 tells the followers (op_graph_warm) and every rank captures the same plan in order; a no-op without graphs.
pub fn precapture(f: *Forward, last: u64) !void {
    const g = (try f.graphState()) orelse return;
    if (f.leads()) try f.link.?.send(&.{ fwd_follow.op_graph_warm, @intCast(last) });
    return warmPlan(f, g, last);
}

/// The follower's op_graph_warm: its graphs are on when the leader's are (the one environment).
pub fn follow(f: *Forward, msg: []const i64) !void {
    if (msg.len != 2 or msg[1] < 0) return error.BadPlan;
    const g = (try f.graphState()) orelse return error.BadPlan;
    return warmPlan(f, g, @intCast(msg[1]));
}

/// The prompts' one-time buffers at their final size: their growth would change the fingerprint and drop every graph.
pub fn settleBuffers(f: *Forward) !void {
    if (f.prefill_state != null) try pf_glue.ready(f);
}

fn warmPlan(f: *Forward, g: *Graphed, last: u64) !void {
    try settleBuffers(f);
    var rows: [plan_mod.rows_cap]u32 = undefined;
    const most = plan_mod.slotRows(@intCast(f.cfg.dspark_block), g.rows_max);
    const plan: plan_mod.Plan = .{ .rows = plan_mod.windowRows(&rows, most), .ctx = plan_mod.ctxBuckets(last, @intCast(f.opts.limit), g.settings) };
    var w: Warmer = .{ .f = f, .g = g, .floor = pre.Floor.init(graphs.rule orelse .{}, graphs.meter) };
    _ = try pre.run(&w, plan, "decode windows");
}

const Warmer = struct {
    f: *Forward,
    g: *Graphed,
    floor: pre.Floor,

    pub fn warm(w: *Warmer, s: plan_mod.Shape) !pre.Step {
        const f = w.f;
        const g = w.g;
        const r = f.runner;
        const key: graphs.Key = .{ .rows = s.rows, .ctx = s.ctx };
        const cs = try g.program(f, key);
        if (graphs.audit(cs) != null) return .unavailable;
        r.guard(cs) catch |e| return if (e == error.MissingTritonVariant) .unavailable else e;
        // the host state `graphedWindow` leaves for the body: nothing is staged or run, so all of it is put back
        var ids: [plan_mod.rows_cap]u32 = @splat(0);
        const was = .{ f.ids, f.slot.pending, f.gated, r.glue };
        defer {
            f.ids, f.slot.pending, f.gated, r.glue = was;
            g.active = false;
        }
        f.ids = ids[0..s.rows];
        f.slot.pending = .{ .start = 0, .n = s.rows };
        f.gated = try f.gatesRows(s.rows);
        g.active = true;
        r.glue = .{ .ctx = f, .run = fwd_glue.glueFn };
        g.current = cs;
        const fp = f.fingerprint(g);
        return pre.warmParts(&g.cache, &w.floor, graphs.Parts.of(cs, g.split orelse .{}), key, &g.span, r.stream, fp, .{ .ctx = f, .run = fwd_graph.graphBody });
    }

    pub fn facts(w: *Warmer) pre.Facts {
        const c = &w.g.cache;
        return .{ .rank = @intCast(w.f.comm.rank()), .graphs = c.stats.precaptured, .held = @intCast(c.count()), .cap = c.cap, .price_mib = w.floor.priceGib() * 1024, .avail_gib = graphs.memAvailableGiB(), .keep_gib = w.floor.room.keep_gib };
    }
};
