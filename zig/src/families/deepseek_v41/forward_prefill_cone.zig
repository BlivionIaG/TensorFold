//! CED's exact decoder replay (TF_DSV41_REPLAY_EXACT, docs/deepseek-v41-disaggregated.md): the decoder over the cone.

const std = @import("std");
const cuda = @import("cuda");
const buffers = @import("buffers.zig");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const image_opts = @import("image_opts.zig");
const fwd = @import("forward.zig");
const ced = @import("ced.zig");
const Forward = fwd.Forward;
const pf = @import("forward_prefill.zig");
const pf_glue = @import("forward_prefill_glue.zig");
const pf_k4 = @import("forward_prefill_k4.zig");
const K4 = pf_k4.K4;
const log = pf.log;

/// One ticket a prompt: taken when the slot's does not continue the cone ring (another prompt, a restore, a rewind).
pub fn begin(f: *Forward) void {
    const st = pf.state(f);
    if (f.slot.cone_id != 0 and f.slot.cone_id == st.cone_id and st.cone.hi == f.slot.pos) return;
    st.cone_id +%= 1;
    if (st.cone_id == 0) st.cone_id = 1;
    f.slot.cone_id = st.cone_id;
    st.cone = .{ .from = f.slot.pos, .hi = f.slot.pos };
}

/// An encoder segment's rows [start, start + ids.len) are in the cone ring.
pub fn stashed(f: *Forward, start: u64, ids: []const u32) void {
    const st = pf.state(f);
    if (f.slot.cone_id == st.cone_id) st.cone.extend(start, ids);
}

/// The cone's rows behind a prompt ending at `end`.
fn rowsOf(f: *const Forward, end: u64) !u32 {
    return ced.groupRows(f.cfg.window, f.cfg.layers - try ced.decoderStart(f.cfg), 0, end);
}

/// Whether the slot's ring holds the cone of a prompt ending at `end` (all of it, or from position 0) and it can run.
pub fn usable(f: *const Forward, end: u64) bool {
    const st = pf.state(f);
    if (!st.cone_on or end <= ced.keep) return false;
    const rows = rowsOf(f, end) catch return false;
    // a cone past the forward's rows runs in the workspace (this prompt call's agreement)
    if (rows > f.opts.prefill_rows and !(st.k4 and st.ws_ok)) return false;
    if (f.slot.cone_id == 0 or f.slot.cone_id != st.cone_id or st.cone.hi != end) return false;
    return st.cone.lo() == 0 or end - st.cone.lo() >= rows;
}

/// The exact decoder replay for the prompt ending at `f.slot.pos`; false (nothing run) when the cone cannot.
pub fn finish(f: *Forward) !bool {
    const st = pf.state(f);
    const end = f.slot.pos;
    const ok = usable(f, end);
    f.slot.cone_id = 0;
    if (!ok) return false;
    const t0 = pf_k4.nowMs();
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try pf.decoderLayers(f, try f.backbone(a));
    const nd: u32 = @intCast(layers.len);
    var gbuf: [32]ced.Group = undefined;
    const gs = ced.groups(f.cfg, layers, st.cone_group, &gbuf);
    const W = f.cfg.window;
    const n0 = ced.groupRows(W, nd, 0, end);
    const all = st.tail_ids[0..n0];
    st.cone.tail(end - n0, end, all);
    // the programs all in the workspace or all in the forward's buffers: one set of candidates and taps
    const big = n0 > f.opts.prefill_rows;
    var keep: block_prefill.Keep = .empty;
    defer keep.deinit(f.gpa);
    if (big) {
        var it = st.k4ws.keep.keyIterator();
        while (it.next()) |k| try keep.put(f.gpa, k.*, {});
        try keep.put(f.gpa, "w.taps", {});
        try pf_k4.k4Acquire(f);
    }
    f.runner.glue = .{ .ctx = f, .run = pf_glue.glueFn };
    defer f.slot.pending = null;
    var layer_rows: u64 = 0;
    var last_n: u32 = n0;
    {
        // the workspace's send / gathered buffers while the programs run (as a 4K segment's), back before its release
        if (big) {
            std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
            std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
        }
        defer if (big) {
            std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
            std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
        };
        for (gs, 0..) |g, gi| {
            const n = ced.groupRows(W, nd, g.first, end);
            const last = gi + 1 == gs.len;
            const ids = all[n0 - n ..];
            const o0 = image_opts.segmentOptions(f, ids);
            const o = if (big) pf.k4Options(o0) else o0;
            // a program not the last ends on the next one's first site, handing its rows on through the ring
            const lay = layers[g.first .. g.first + g.count + @intFromBool(!last)];
            const cand_at: ?i64 = if (gi == 0) null else @intCast(n0 - n);
            const plain = try block_prefill.emitCone(a, f.cfg, f.widths, o, lay, n, @intCast(end - n), if (last) .cone_last else .cone_mid, cand_at);
            const cs = if (big) try block_prefill.ownCallsKeep(a, plain, K4.tag, &keep) else plain;
            f.ids = ids;
            f.slot.pending = .{ .start = end - n, .n = n };
            st.layers = lay;
            st.moe = 0;
            try longpf.window(&st.long, f.gpa, f.runner, f.kv, end - n, cs);
            if (f.runner.prof) |pr| pr.prefillRows(@intCast(n), true);
            layer_rows += @as(u64, n) * g.count;
            last_n = n;
        }
    }
    f.last_rows = 0;
    try f.runner.stream.synchronize();
    try pf_k4.k4Release(f);
    try longpf.release(&st.long, f.runner);
    st.taps_at = end - last_n;
    st.taps_rows = last_n;
    st.replay_exact = true;
    if (f.comm.rank() == 0) log.info("replay cone: end {d}: {d} rows, {d} programs of {d} layers, {d} layer-rows, {d:.1} ms", .{ end, n0, gs.len, st.cone_group, layer_rows, pf_k4.nowMs() - t0 });
    return true;
}

/// The exact replay's programs for a plan, at the first and the last start each emits at (forward's or workspace's).
pub fn plan(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32, ws: bool) !void {
    const st = pf.state(f);
    const dec = try pf.decoderLayers(f, layers);
    const nd: u32 = @intCast(dec.len);
    const W = f.cfg.window;
    var gbuf: [32]ced.Group = undefined;
    const gs = ced.groups(f.cfg, dec, st.cone_group, &gbuf);
    const n0 = ced.groupRows(W, nd, 0, if (ws) K4.rows else rows);
    var keep: block_prefill.Keep = .empty;
    defer keep.deinit(f.gpa);
    try keep.put(f.gpa, "w.taps", {});
    const o = if (ws) pf.k4Options(image_opts.planOptions(f)) else image_opts.planOptions(f);
    for (gs, 0..) |g, gi| {
        const n = @min(n0, ced.groupRows(W, nd, g.first, std.math.maxInt(u64)));
        const last = gi + 1 == gs.len;
        const lay = dec[g.first .. g.first + g.count + @intFromBool(!last)];
        const part: block_prefill.Part = if (last) .cone_last else .cone_mid;
        const cand_at: ?i64 = if (gi == 0) null else @intCast(n0 - n);
        const top = try pf.lastStartWith(f, o, lay, n, part);
        for ([_]i64{ 0, top }) |start| {
            var ea = std.heap.ArenaAllocator.init(f.gpa);
            defer ea.deinit();
            const cs = try block_prefill.emitCone(ea.allocator(), f.cfg, f.widths, o, lay, n, start, part, cand_at);
            try p.add(if (ws) try block_prefill.ownCallsKeep(ea.allocator(), cs, K4.tag, &keep) else cs);
        }
    }
    // the last program's taps ("w.taps", the forward's either way): all rows of its cone, the drafter reads the window
    if (!ws) {
        const rows_last = ced.groupRows(W, nd, gs[gs.len - 1].first, std.math.maxInt(u64));
        try p.need("w.taps", @as(u64, rows_last) * f.cfg.dspark_targets.items().len * f.cfg.hidden * 2);
    }
}
