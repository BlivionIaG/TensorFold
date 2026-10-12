//! Flash Next's two mixers on a HIP rank: the Gated DeltaNet and the sparse attention with its indexer.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;

/// normalize_qk's constants: q and k scaled by rsqrt(sum + 1e-6) (q also by dk^-0.5) as an RMS norm with a weight.
pub fn qkConstants(key_dim: usize) struct { q: f32, k: f32, eps: f32 } {
    const inv = std.math.pow(f64, @floatFromInt(key_dim), -0.5);
    return .{ .q = @floatCast(inv * inv), .k = @floatCast(inv), .eps = @floatCast(1e-6 * inv * inv) };
}

/// The Gated DeltaNet over every segment's rows: products and norms on all rows, conv and recurrence a stream at a time.
pub fn linear(o: Ops, m: *const model.Model, l: model.Linear, segs: []const forward.Seg, index: usize, x: Tensor, rows: usize, window: bool) !Tensor {
    const d = m.dims;
    const ch = d.convWidth();
    const kw = d.key_heads * d.key_dim;
    const vw = d.value_heads * d.value_dim;
    const qkv = try o.project(x, l.qkv, rows, false);
    const z = try o.project(x, l.z, rows, false);
    const a = try o.project(x, l.a, rows, false);
    const b = try o.project(x, l.b, rows, false);
    const mixed = try o.arena.of(f32, rows * ch);
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const st = g.c.linearAt(index);
        const out = mixed + off * ch * 4;
        if (window) {
            const wide = try o.arena.of(f32, n * ch);
            try o.cast(forward.at(qkv, off * ch), .{ .ptr = wide, .kind = .f32 }, n * ch);
            try o.convRows(wide, l.conv, st.conv, out, g.c.snapshot(index, 0).conv, n, ch, d.conv);
        } else {
            const conv_bytes = (d.conv - 1) * ch * 4;
            const next = try o.arena.take(conv_bytes);
            try o.convPrefill(forward.at(qkv, off * ch), l.conv, st.conv, out, next, n, ch, d.conv);
            if (o.stream != hip.counting) try hip.runtime.check(o.l.r.api.hipMemcpyDtoDAsync(@ptrFromInt(st.conv), @ptrFromInt(next), conv_bytes, o.stream));
        }
        off += n;
    }
    const mixed_t: Tensor = .{ .ptr = mixed, .kind = .f32 };
    const qc = try o.arena.of(f32, rows * kw);
    const kc = try o.arena.of(f32, rows * kw);
    const v = try o.arena.of(f32, rows * vw);
    try o.copyCols(mixed_t, ch, 0, qc, rows, kw);
    try o.copyCols(mixed_t, ch, kw, kc, rows, kw);
    try o.copyCols(mixed_t, ch, 2 * kw, v, rows, vw);
    const q = try o.arena.of(f32, rows * kw);
    const k = try o.arena.of(f32, rows * kw);
    try o.rms(.{ .ptr = qc, .kind = .f32 }, m.qk.q, .{ .ptr = q, .kind = .f32 }, rows * d.key_heads, d.key_dim, m.qk.eps);
    try o.rms(.{ .ptr = kc, .kind = .f32 }, m.qk.k, .{ .ptr = k, .kind = .f32 }, rows * d.key_heads, d.key_dim, m.qk.eps);
    const gate = try o.arena.of(f32, rows * d.value_heads);
    const beta = try o.arena.of(f32, rows * d.value_heads);
    try o.gdnGatePrefill(a, b, l.a_log, l.dt_bias, gate, beta, rows, d.value_heads);
    const y = try o.arena.of(f32, rows * vw);
    off = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const snaps: ?u64 = if (window) g.c.snapshot(index, 0).state else null;
        try o.gatedDelta(q + off * kw * 4, k + off * kw * 4, v + off * vw * 4, gate + off * d.value_heads * 4, beta + off * d.value_heads * 4, g.c.linearAt(index).state, y + off * vw * 4, n, d.key_heads, d.value_heads, d.key_dim, d.value_dim, snaps);
        off += n;
    }
    const out = try take(o, m.act, rows * vw);
    try o.gnormOut(y, l.norm, z, out, rows * d.value_heads, d.value_dim, d.eps, .sigmoid);
    return o.project(out, l.out, rows, l.out.partial);
}

/// Rows of a prompt's attention a slice: the indexer's scores and the attention's parts stay bounded.
const slice_rows = 64;
const parts = 16;

/// The sparse attention over every segment's rows: products on all rows, each stream's caches, blocks and keys its own.
pub fn sparse(o: Ops, m: *const model.Model, d: model.Dims, s: model.Sparse, segs: []const forward.Seg, layer: usize, x: Tensor, rows: usize, dev_pos: u64, host: *forward.Host) !Tensor {
    const hd = d.head_dim;
    const di = d.index_dim;
    const width = (d.index_heads + 1) * di;
    const kind = @backingInt(m.act);
    const qg = try o.project(x, s.q, rows, false);
    const keys = try o.project(x, s.k, rows, false);
    const values = try o.project(x, s.v, rows, false);
    const idx = try o.project(x, s.index, rows, false);
    const q32 = try o.arena.of(f32, rows * d.heads * hd);
    try o.qkRope(qg, d.heads * 2 * hd, 2 * hd, s.q_norm, d.eps, rows, d.heads, hd, d.rotary, d.theta, dev_pos, q32, null, 0);
    const iq = try o.arena.of(f32, rows * d.index_heads * di);
    try o.qkRope(idx, width, di, s.iq_norm, d.eps, rows, d.index_heads, di, d.rotary, d.theta, dev_pos, iq, null, 0);
    const att = try o.arena.of(f32, rows * d.heads * hd);
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const kv = g.c.sparse(layer);
        const cap = g.c.layout.cap;
        const pos = g.pos;
        try o.qkRope(forward.at(keys, off * d.kv_heads * hd), d.kv_heads * hd, hd, s.k_norm, d.eps, n, d.kv_heads, hd, d.rotary, d.theta, dev_pos + off * 4, null, kv.k, cap);
        try o.kvWrite(forward.at(values, off * d.kv_heads * hd), kv.v, n, d.kv_heads, hd, cap, pos);
        try o.copyCols(forward.at(idx, off * width), width, d.index_heads * di, kv.raw + pos * di * m.act.size(), n, di);
        const blocks = (pos + n) / d.index_ratio;
        const pooled = g.c.pooled;
        if (blocks > pooled) try o.l.fn_idx_pool(@ptrFromInt(kv.raw), @ptrFromInt(kv.pooled + pooled * di * m.act.size()), @ptrFromInt(s.ik_norm), kind, @intCast(pooled), @intCast(blocks - pooled), @intCast(di), @intCast(d.index_ratio), @intCast(d.rotary), d.theta, d.eps, o.stream);
        try attend(o, m, d, kv, cap, q32 + off * d.heads * hd * 4, iq + off * d.index_heads * di * 4, att + off * d.heads * hd * 4, n, pos, host);
        off += n;
    }
    const gated = try take(o, m.act, rows * d.heads * hd);
    try o.attnGate(att, qg, gated, rows, d.heads, hd, true);
    return o.project(gated, s.o, rows, s.o.partial);
}

/// One stream's `rows` rows from `pos` over its caches, in slices: each row's keys chosen, then attended.
fn attend(o: Ops, m: *const model.Model, d: model.Dims, kv: cache.SparseView, cap: usize, q32: u64, iq: u64, att: u64, rows: usize, pos: usize, host: *forward.Host) !void {
    const hd = d.head_dim;
    const di = d.index_dim;
    const kind = @backingInt(m.act);
    const key_stride = d.index_top * d.index_ratio + d.index_ratio;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += slice_rows) {
        const n = @min(slice_rows, rows - r0);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        var complete: [slice_rows]i32 = undefined;
        var ends: [slice_rows]i32 = undefined;
        var sparse_rows = false;
        for (0..n) |r| {
            ends[r] = @intCast(pos + r0 + r + 1);
            complete[r] = @divFloor(ends[r], @as(i32, @intCast(d.index_ratio)));
            sparse_rows = sparse_rows or complete[r] > d.index_top;
        }
        const dev_complete = try host.put(o, std.mem.sliceAsBytes(complete[0..n]));
        const dev_ends = try host.put(o, std.mem.sliceAsBytes(ends[0..n]));
        var ids: ?u64 = null;
        var sparse_flags: ?u64 = null;
        var count = dev_ends;
        if (sparse_rows) {
            const nb: usize = @intCast(complete[n - 1]);
            const scores = try o.arena.of(f32, n * nb);
            try o.l.fn_idx_scores(@ptrFromInt(iq + r0 * d.index_heads * di * 4), @ptrFromInt(kv.pooled), @ptrFromInt(dev_complete), @ptrFromInt(scores), kind, @intCast(n), @intCast(d.index_heads), @intCast(di), @intCast(nb), @intCast(nb), @intCast(d.index_top), o.stream);
            const list = try o.arena.of(i32, n * key_stride);
            const counts = try o.arena.of(i32, n);
            const flags = try o.arena.of(i32, n);
            try o.l.fn_idx_select(@ptrFromInt(scores), @intCast(nb), @ptrFromInt(dev_complete), @ptrFromInt(dev_ends), @ptrFromInt(list), @intCast(key_stride), @ptrFromInt(counts), @ptrFromInt(flags), @intCast(n), @intCast(d.index_top), @intCast(d.index_ratio), o.stream);
            ids = list;
            sparse_flags = flags;
            count = counts;
        }
        const po = try o.arena.of(f32, n * d.heads * parts * hd);
        const pm = try o.arena.of(f32, n * d.heads * parts * 2);
        const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(hd))));
        try o.l.fn_sparse_attention(@ptrFromInt(q32 + r0 * d.heads * hd * 4), @ptrFromInt(kv.k), @ptrFromInt(kv.v), @backingInt(m.act), if (ids) |i| @ptrFromInt(i) else null, @intCast(key_stride), @ptrFromInt(count), if (sparse_flags) |f| @ptrFromInt(f) else null, @ptrFromInt(po), @ptrFromInt(pm), @ptrFromInt(att + r0 * d.heads * hd * 4), @intCast(n), @intCast(d.heads), @intCast(d.kv_heads), @intCast(hd), @intCast(cap), scale, parts, o.stream);
    }
}
