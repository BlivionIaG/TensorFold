//! CUDA-graph replay of the decode step (rows == 1) for exl3_fwd.zig's Engine.
//!
//! One decoder layer is two device segments with the host in between (the router pick needs the logits on the host):
//!   A: attention-or-GDN block (HC pre, mixer, HC post) + MoE HC pre + router dots      -> gate_logits
//!   [host: sync, D2H of the 513 logits, top-10 pick, pool staging H2D for misses, weights + line ids uploaded]
//!   B: routed experts (line ids read from `ids_all`) + shared expert + combine + MoE HC post
//! Every address a segment uses is fixed for the life of the engine (stream scratch, `b.pos0`, `b.ids_all`, `b.wts`, pool bases,
//! dense weights, KV and GDN state of this layer), and every value that varies per step is read through a device pointer
//! (`pos0` for position and KV write row, `ids_all` for the pool lines, `wts` for the weights). So each segment is captured once
//! per layer (exl3_graph.Key rank 0, layer, window 1) and replayed; a pool-line swap or a changed id only rewrites device memory.
//! Eager stays for: prefill and any call with rows != 1, the PLE bounce at layer 1 (host download / add-back / upload between
//! segments), the MTP head, trace mode, profile mode and an on-demand (pool-less) layer.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fwd = @import("exl3_fwd.zig");
const route = @import("exl3_route.zig");
const egraph = @import("exl3_graph.zig");

pub const Mode = enum {
    off,
    /// Capture each segment once, then replay.
    replay,
    /// The same segmented path, but every segment is launched eagerly (the enqueue-time reference).
    eager,
};

pub const Stats = struct {
    captures: u32 = 0,
    replays: u32 = 0,
    eager_segments: u32 = 0,
    /// Host time spent enqueueing segments (cuGraphLaunch for replays, the kernel launches for eager), ns.
    enqueue_ns: u64 = 0,
    capture_ns: u64 = 0,
    nodes_a: usize = 0,
    nodes_b: usize = 0,
    segments_run: u32 = 0,
};

pub const State = struct {
    mode: Mode,
    exec: [2][fwd.layers_total]?cuda.graph.Exec = @splat(@splat(null)),
    keys: std.ArrayList(egraph.Key) = .empty,
    stats: Stats = .{},

    pub fn init(mode: Mode) State {
        return .{ .mode = mode };
    }

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        for (&s.exec) |*row| for (row) |*x| if (x.*) |*ex| ex.deinit();
        s.keys.deinit(gpa);
    }
};

const Seg = enum(u1) { a, b };

fn bodyA(e: *fwd.Engine, lay: *fwd.Layer) !void {
    const s = e.stream;
    const hw = lay.hc_dev.?;
    const bb = &e.b;
    try e.hcg.pre(s, hw[0].weights(), e.offset, fwd.eps, bb.h.ptr, e.hc_scratch, bb.x.ptr, bb.inj.ptr, 1);
    try e.mixerRun(lay, 1);
    try e.hcg.post(s, bb.h.ptr, bb.br.ptr, bb.inj.ptr, 1);
    try e.hcg.pre(s, hw[1].weights(), e.offset, fwd.eps, bb.h.ptr, e.hc_scratch, bb.x.ptr, bb.inj.ptr, 1);
    try e.hcg.gate(s, bb.x.ptr, lay.gate_dev.?.ptr, lay.sgate_dev.?.ptr, bb.gate_logits.ptr, 1);
}

fn bodyB(e: *fwd.Engine, lay: *fwd.Layer) !void {
    const s = e.stream;
    const bb = &e.b;
    const ep = &lay.ep.?;
    var pk: [3]cuda.exl3.Packed = undefined;
    for (&pk, 0..) |*p, i| p.* = try ep.pool.packed_(i);
    try e.ex.hit(s, bb.x.ptr, bb.ids_all.ptr, pk[0], pk[1], pk[2], bb.y_gate.ptr, bb.y_up.ptr, bb.y_glu.ptr, bb.y_down.ptr, 1, ep.pool.lines, fwd.topk);
    try lay.shared[0].run(&e.ex, s, bb.x.ptr, bb.s_gate.ptr, 1);
    try lay.shared[1].run(&e.ex, s, bb.x.ptr, bb.s_up.ptr, 1);
    try e.ex.swiglu(s, bb.s_gate.ptr, bb.s_up.ptr, bb.s_glu.ptr, 1, fwd.inter);
    try lay.shared[2].run(&e.ex, s, bb.s_glu.ptr, bb.y_down.ptr + fwd.topk * fwd.hidden * 2, 1);
    try e.ex.combine(s, bb.y_down.ptr, bb.wts.ptr, bb.br.ptr, 1, fwd.hidden, fwd.slots);
    try e.hcg.post(s, bb.h.ptr, bb.br.ptr, bb.inj.ptr, 1);
}

fn body(e: *fwd.Engine, lay: *fwd.Layer, seg: Seg) !void {
    switch (seg) {
        .a => try bodyA(e, lay),
        .b => try bodyB(e, lay),
    }
}

fn segment(e: *fwd.Engine, st: *State, lay: *fwd.Layer, seg: Seg) !void {
    const io = e.gpu.io;
    const s = e.stream;
    st.stats.segments_run += 1;
    if (st.mode == .eager) {
        const t0 = check.now(io);
        try body(e, lay, seg);
        st.stats.enqueue_ns += @intCast(check.now(io) - t0);
        st.stats.eager_segments += 1;
        return;
    }
    const slot = &st.exec[@intFromEnum(seg)][lay.index];
    if (slot.* == null) {
        const key: egraph.Key = .{ .rank = 0, .layer = @intCast(lay.index * 2 + @intFromEnum(seg)), .window = 1 };
        if (egraph.promote(st.keys.items, key) != .recapture) return error.GraphKeyRepeated;
        const t0 = check.now(io);
        try cuda.graph.beginCapture(s, .thread_local);
        body(e, lay, seg) catch |err| {
            if (cuda.graph.endCapture(s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        const nodes = try g.nodeCount();
        if (seg == .a) st.stats.nodes_a = nodes else st.stats.nodes_b = nodes;
        slot.* = try g.instantiate();
        try slot.*.?.upload(s);
        try st.keys.append(e.gpu.gpa, key);
        st.stats.captures += 1;
        st.stats.capture_ns += @intCast(check.now(io) - t0);
    }
    const t1 = check.now(io);
    try slot.*.?.launchOn(s);
    st.stats.enqueue_ns += @intCast(check.now(io) - t1);
    st.stats.replays += 1;
}

/// True when this call/layer may take the segmented path.
pub fn applies(e: *const fwd.Engine, lay: *const fwd.Layer, rows: usize) bool {
    return e.graph != null and rows == 1 and lay.ep != null and e.trace == null and !e.opt.profile;
}

/// One decoder layer for a single row. `h` is the engine's device stream buffer; the PLE bounce (if any) already happened.
pub fn layerStep(e: *fwd.Engine, lay: *fwd.Layer) !void {
    const st = &e.graph.?;
    const bb = &e.b;
    try segment(e, st, lay, .a);
    try e.stream.synchronize();
    var glog: [cuda.exl3_hc.gate_rows]f32 = undefined;
    try bb.gate_logits.download(0, std.mem.sliceAsBytes(&glog));
    var picks: [fwd.slots]u16 = undefined;
    var weights: [fwd.slots]f32 = undefined;
    var rscratch: [fwd.experts]f32 = undefined;
    try route.select(&glog, fwd.topk, &picks, &weights, &rscratch);
    e.setGap(&glog, &picks);
    try e.moeHost(lay, &picks, &weights);
    try segment(e, st, lay, .b);
    try e.done.record(e.stream);
    e.have_done = true;
}
