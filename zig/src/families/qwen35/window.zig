//! A lane round's forward (window.py): every stream's window in one pass, each row with its serial decode step's bits.
//! Projections, norms and the MLP take all rows; RoPE, the cache write, attention, the conv and the recurrence take a
//! stream's rows in one launch each at their own positions.

const std = @import("std");
const hip = @import("hip");
const view = @import("view.zig");
const state = @import("state.zig");
const fwd = @import("forward.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;

/// A linear layer's states after each row of a window of two rows or more: conv (rows, K - 1, ch), delta (rows, Hv,
/// dv, dk), both fp32; none for one row (its states advance in place and every commit keeps them).
pub const Snapshot = struct { convs: u64 = 0, deltas: u64 = 0 };

/// One stream's rows this round: its last token and drafts from slot `pos`, over its caches.
pub const Window = struct {
    caches: *state.Caches,
    pos: usize,
    rows: usize,
    /// The rows' positions on the device, int32 (pos .. pos + rows - 1).
    at32: u64,
    /// Filled by the forward: a snapshot a layer (zero for full-attention layers).
    snaps: []Snapshot,
};

/// The final-normed rows of every window, in order (total rows, hidden); `ids` int32 on the device, all rows.
pub fn forward(o: Ops, m: *const view.Model, windows: []Window, ids: u64, trace: ?fwd.Trace) fwd.Error!Tensor {
    const s = m.spec;
    var total: usize = 0;
    for (windows) |w| total += w.rows;
    const x = try fwd.take(o, m.act, total * s.hidden);
    try o.embedRows(m.embed, ids, total, x);
    // a window of several rows keeps every linear layer's per-row states until its commit
    for (windows) |w| for (w.snaps, 0..) |*snap, index| {
        snap.* = .{};
        if (w.rows < 2 or s.full(index)) continue;
        snap.convs = try o.arena.of(f32, w.rows * (s.conv - 1) * view.convChannels(s));
        snap.deltas = try o.arena.of(f32, w.rows * s.value_heads * s.value_dim * s.key_dim);
    };
    const normed = try fwd.take(o, m.act, total * s.hidden);
    for (m.layers, 0..) |layer, index| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const input_norm, const post_norm, const mlp = switch (layer) {
            .full => |f| .{ f.input_norm, f.post_norm, f.mlp },
            .linear => |l| .{ l.input_norm, l.post_norm, l.mlp },
        };
        try o.rms(x, input_norm, normed, total, s.hidden, @floatCast(s.eps));
        const y = switch (layer) {
            .full => |f| try attentionRows(o, m, f, windows, index, normed, total),
            .linear => |l| try linearRows(o, m, l, windows, index, normed, total),
        };
        try fwd.residual(o, m, x, y, total * s.hidden);
        try o.rms(x, post_norm, normed, total, s.hidden, @floatCast(s.eps));
        const z = try fwd.mlpRows(o, m, mlp, normed, total, true);
        try fwd.residual(o, m, x, z, total * s.hidden);
        if (trace) |t| t.layer(t.ctx, index, x, total) catch return error.KernelFailed;
    }
    const out = try fwd.take(o, m.act, total * s.hidden);
    try o.rms(x, m.final_norm, out, total, s.hidden, @floatCast(s.eps));
    return out;
}

/// Keep a window's first `rows` rows: attention lengths, and the linear states as after those serial steps.
pub fn commit(o: Ops, m: *const view.Model, w: Window, rows: usize) hip.Error!void {
    const s = m.spec;
    for (w.caches.layers, 0..) |*cache, index| switch (cache.*) {
        .full => |*f| f.len = w.pos + rows,
        .linear => |l| if (rows < w.rows) {
            const conv_bytes = (s.conv - 1) * view.convChannels(s) * 4;
            const delta_bytes = s.value_heads * s.value_dim * s.key_dim * 4;
            const snap = w.snaps[index];
            try l.conv.copyFrom(0, snap.convs + (rows - 1) * conv_bytes, conv_bytes, o.stream);
            try l.state.copyFrom(0, snap.deltas + (rows - 1) * delta_bytes, delta_bytes, o.stream);
        },
    };
}

fn attentionRows(o: Ops, m: *const view.Model, f: view.Full, windows: []Window, index: usize, x: Tensor, total: usize) fwd.Error!Tensor {
    const s = m.spec;
    const hd = s.head_dim;
    const qg = try o.affine(x, f.q, total, false);
    const keys = try o.affine(x, f.k, total, false);
    const values = try o.affine(x, f.v, total, false);
    const q_rows = total * s.heads;
    const eps: f32 = @floatCast(s.eps);
    const theta: f32 = @floatCast(s.rope_theta);
    const att = try o.arena.of(f32, q_rows * hd);
    var start: usize = 0;
    for (windows) |w| {
        const c = w.caches.attention(m, index);
        // q and k each normed, rotated at the rows' positions and rounded in one launch: q into fp32, k into the cache
        const q32 = try o.arena.of(f32, w.rows * s.heads * hd);
        try o.qkRope(fwd.at(qg, start * s.heads * 2 * hd), s.heads * 2 * hd, 2 * hd, f.q_norm, eps, w.rows, s.heads, hd, s.rotary_dim, theta, w.at32, q32, null, 0);
        try o.qkRope(fwd.at(keys, start * s.kv_heads * hd), s.kv_heads * hd, hd, f.k_norm, eps, w.rows, s.kv_heads, hd, s.rotary_dim, theta, w.at32, null, c.k, c.total);
        try o.kvWriteAt(fwd.at(values, start * s.kv_heads * hd), c.v, w.rows, s.kv_heads, hd, c.total, w.at32);
        try o.causalAt(q32, c, att + start * s.heads * hd * 4, w.rows, s.heads, fwd.scaleOf(hd), w.at32);
        start += w.rows;
    }
    const gated = try fwd.take(o, m.act, q_rows * hd);
    try o.attnGate(att, qg, gated, total, s.heads, hd, true);
    return o.affine(gated, f.o, total, f.o.partial);
}

fn linearRows(o: Ops, m: *const view.Model, l: view.Linear, windows: []Window, index: usize, x: Tensor, total: usize) fwd.Error!Tensor {
    const s = m.spec;
    const qkv = try o.affine(x, l.qkv, total, false);
    const z = try o.affine(x, l.z, total, false);
    const a = try o.affine(x, l.a, total, false);
    const b = try o.affine(x, l.b, total, false);
    const ch = view.convChannels(s);
    const y = try o.arena.of(f32, total * s.valueWidth());
    var start: usize = 0;
    for (windows) |w| {
        const cache = w.caches.layers[index].linear;
        const rows = w.rows;
        const xr = try o.arena.of(f32, rows * ch);
        try o.cast(fwd.at(qkv, start * ch), .{ .ptr = xr, .kind = .f32 }, rows * ch);
        const mixed = try o.arena.of(f32, rows * ch);
        const snap = w.snaps[index];
        if (rows == 1) {
            try o.convDecode(xr, l.conv, cache.conv.ptr, mixed, ch, s.conv);
        } else {
            try o.convRows(xr, l.conv, cache.conv.ptr, mixed, snap.convs, rows, ch, s.conv);
        }
        const q, const k, const v = try fwd.splitQkv(o, m, .{ .ptr = mixed, .kind = .f32 }, rows);
        const gate = try o.arena.of(f32, rows * s.value_heads);
        const beta = try o.arena.of(f32, rows * s.value_heads);
        try o.gdnGate(fwd.at(a, start * s.value_heads), fwd.at(b, start * s.value_heads), l.a_log, l.dt_bias, gate, beta, rows * s.value_heads, s.value_heads);
        try o.gatedDelta(q, k, v, gate, beta, cache.state.ptr, y + start * s.valueWidth() * 4, rows, s.key_heads, s.value_heads, s.key_dim, s.value_dim, if (rows > 1) snap.deltas else null);
        start += rows;
    }
    return fwd.gatedOut(o, m, l, y, z, total);
}
