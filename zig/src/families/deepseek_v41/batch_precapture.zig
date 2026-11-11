//! Row windows' graphs at boot (TF_DSV41_GRAPH_PRECAPTURE): the shapes `Batch.runRows` keys, captured, never run.

const graphs = @import("graphs.zig");
const batch = @import("batch.zig");
const batch_graph = @import("batch_graph.zig");
const forward_follow = @import("forward_follow.zig");
const warm_fwd = @import("forward_graph_warm.zig");
const plan_mod = @import("graph_plan.zig");
const pre = @import("graph_precapture.zig");
const Batch = batch.Batch;

/// Rank 0 tells the followers (op_graph_warm) and every rank captures the same plan in order; a no-op without graphs.
pub fn precapture(b: *Batch, last: u64) !void {
    if (b.g == null) return;
    if (b.ss.leads()) try b.f.link.?.send(&.{ forward_follow.op_graph_warm, @intCast(last) });
    return warmPlan(b, last);
}

/// The follower's op_graph_warm: its row graphs are on when the leader's are (the one environment).
pub fn followWarm(b: *Batch, msg: []const i64) !void {
    if (msg.len != 2 or msg[1] < 0 or b.g == null) return error.BadPlan;
    return warmPlan(b, @intCast(msg[1]));
}

fn warmPlan(b: *Batch, last: u64) !void {
    try warm_fwd.settleBuffers(b.f);
    var rows: [plan_mod.rows_cap]u32 = undefined;
    const most = plan_mod.roundRows(b.cap, b.ss.n);
    const plan: plan_mod.Plan = .{ .rows = plan_mod.rowBuckets(&rows, most, b.cap), .ctx = plan_mod.ctxBuckets(last, @intCast(b.f.opts.limit), b.settings) };
    var w: Warmer = .{ .b = b, .floor = pre.Floor.init(graphs.rule orelse .{}, graphs.meter) };
    _ = try pre.run(&w, plan, "row graphs");
}

const Warmer = struct {
    b: *Batch,
    floor: pre.Floor,

    pub fn warm(w: *Warmer, s: plan_mod.Shape) !pre.Step {
        const b = w.b;
        const f = b.f;
        const r = f.runner;
        const g = &b.g.?;
        const key: graphs.Key = .{ .rows = s.rows, .ctx = s.ctx };
        const cs = try batch_graph.program(b, key, b.settings);
        if (graphs.audit(cs) != null) return .unavailable;
        r.guard(cs) catch |e| return if (e == error.MissingTritonVariant) .unavailable else e;
        // the host state `runRows` leaves for the body: nothing is staged or run, so all of it is put back
        const was = .{ r.glue, r.rows, f.slot.pending, f.gated, b.R, b.current };
        defer {
            r.glue, r.rows, f.slot.pending, f.gated, b.R, b.current = was;
            if (r.taps_mark) |*m| m.armed = false;
        }
        b.R = s.rows;
        b.current = cs;
        f.slot.pending = .{ .start = 0, .n = s.rows };
        r.glue = .{ .ctx = b, .run = batch_graph.glueFn };
        r.rows = s.rows;
        if (r.taps_mark) |*m| m.armed = true;
        f.gated = b.stages.items.len > 0 and try f.gatesRows(s.rows);
        const fp = batch_graph.fingerprint(b);
        return pre.warmParts(&g.cache, &w.floor, graphs.Parts.of(cs, b.rg), key, &b.span, r.stream, fp, .{ .ctx = b, .run = batch_graph.body });
    }

    pub fn facts(w: *Warmer) pre.Facts {
        const c = &w.b.g.?.cache;
        return .{ .rank = @intCast(w.b.f.comm.rank()), .graphs = c.stats.precaptured, .held = @intCast(c.count()), .cap = c.cap, .price_mib = w.floor.priceGib() * 1024, .avail_gib = graphs.memAvailableGiB(), .keep_gib = w.floor.room.keep_gib };
    }
};
