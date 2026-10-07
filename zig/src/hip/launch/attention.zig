//! Attention launches: causal prefill and decode, the gate and the cache write.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

pub fn tf_causal(l: *const Launcher, q: CF, k: C, v: C, out: F, batch: c_int, qlen: c_int, span: c_int, heads: c_int, kv_heads: c_int, d: c_int, scale: f32, q_pos0: c_int, k_sb: c_longlong, k_sh: c_longlong, k_ss: c_longlong, v_sb: c_longlong, v_sh: c_longlong, v_ss: c_longlong, cache_kind: c_int, scores: F, stats: F, partials: F, s: S, pos: CI) Error!void {
    if (d < 1 or d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or batch < 1 or qlen < 1 or span < 1 or q_pos0 < 0) {
        return invalid("causal attention");
    }
    if (cache_kind < 0 or cache_kind > 2) return invalid("causal attention cache kind");
    const kind: usize = @intCast(cache_kind);
    const visible: c_int = @min(q_pos0 + 1, span);
    if (scores != null) {
        if (qlen != 1 or stats == null or partials == null) return invalid("the decode walk");
        // With a device position the keys in use are read on the device and the grid covers the whole cache.
        const tiles: c_int = @intCast(if (pos != null) cdiv(span, 128) else cdiv(visible, 128));
        const head_grid = dim(heads, batch, 1);
        var a: Args = .{};
        a.add(ad(q));
        a.add(ad(k));
        a.add(ad(scores));
        a.add(span);
        a.add(heads);
        a.add(kv_heads);
        a.add(d);
        a.add(scale);
        a.add(visible);
        a.add(k_sb);
        a.add(k_sh);
        a.add(k_ss);
        a.add(ad(pos));
        // a warp scores a key: four warps a block
        try l.go(l.score_keys[kind], dim(cdiv(span, 4), heads, batch), dim(128, 1, 1), 0, s, &a);
        var b: Args = .{};
        b.add(ad(scores));
        b.add(ad(stats));
        b.add(span);
        b.add(visible);
        b.add(ad(pos));
        try l.go(l.softmax_stats, head_grid, dim(256, 1, 1), 0, s, &b);
        var c: Args = .{};
        c.add(ad(scores));
        c.add(ad(stats));
        c.add(ad(v));
        c.add(ad(partials));
        c.add(span);
        c.add(heads);
        c.add(kv_heads);
        c.add(d);
        c.add(visible);
        c.add(v_sb);
        c.add(v_sh);
        c.add(v_ss);
        c.add(ad(pos));
        try l.go(l.apply_values[kind], dim(tiles, heads, batch), dim(256, 1, 1), 0, s, &c);
        var e: Args = .{};
        e.add(ad(partials));
        e.add(ad(out));
        e.add(tiles);
        e.add(d);
        try l.go(l.sum_partials, head_grid, dim(256, 1, 1), 0, s, &e);
        return;
    }
    var a: Args = .{};
    a.add(ad(q));
    a.add(ad(k));
    a.add(ad(v));
    a.add(ad(out));
    a.add(qlen);
    a.add(span);
    a.add(heads);
    a.add(kv_heads);
    a.add(d);
    a.add(scale);
    a.add(q_pos0);
    a.add(k_sb);
    a.add(k_sh);
    a.add(k_ss);
    a.add(v_sb);
    a.add(v_sh);
    a.add(v_ss);
    // a 16-bit cache of whole 64-wide heads takes the 64-row tile (attention=f32: the 16-row one); odd heads a wave a query
    if (kind < 2 and @rem(d, 64) == 0 and d <= 256 and @rem(k_ss, 8) == 0 and @rem(v_ss, 8) == 0 and l.wide) {
        try l.go(l.fa_wide[kind], dim(cdiv(qlen, 64), heads, batch), dim(256, 1, 1), 0, s, &a);
    } else if (d >= 2 and @rem(d, 2) == 0) {
        try l.go(l.fa_prefill[kind], dim(cdiv(qlen, 16), heads, batch), dim(256, 1, 1), 0, s, &a);
    } else {
        try l.go(l.causal[kind], dim(qlen, heads, batch), dim(32, 1, 1), 0, s, &a);
    }
}

pub fn tf_attn_gate(l: *const Launcher, att: CF, qg: C, out: P, kind: c_int, len: c_int, heads: c_int, d: c_int, rows_major: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(att));
    a.add(ad(qg));
    a.add(ad(out));
    a.add(kind);
    a.add(len);
    a.add(heads);
    a.add(d);
    a.add(rows_major);
    try l.flat(l.op.attn_gate, @as(i64, len) * heads * d, s, &a);
}

pub fn tf_kv_write(l: *const Launcher, src: C, cache: P, kind: c_int, len: c_int, kv_heads: c_int, d: c_int, total: c_int, pos0: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(src));
    a.add(ad(cache));
    a.add(kind);
    a.add(len);
    a.add(kv_heads);
    a.add(d);
    a.add(total);
    a.add(pos0);
    try l.flat(l.op.kv_write, @as(i64, len) * kv_heads * d, s, &a);
}
