//! A lane round's launches over its device plan: cache writes, the attention walk, the recurrence and the keep.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// plan.hpp's PlanArgs: device addresses of the round's per-row and per-slot arrays (all zero: no plan).
pub const PlanArgs = extern struct {
    pos: u64 = 0,
    slot: u64 = 0,
    first: u64 = 0,
    count: u64 = 0,
    desc: u64 = 0,
    snaps: u64 = 0,
};

/// A plan's shape: `rows` rows (a bucket, padding included) over `slots` slots.
pub const PlanRef = struct { args: PlanArgs, rows: usize, slots: usize };

/// Rows of keys (fp32, rotated) and values (fp16 or bf16) into each row's slot at its position.
pub fn planKvWrite(l: *const Launcher, keys: u64, values: u64, kind: c_int, p: PlanRef, layer: usize, kv_heads: usize, d: usize, s: S) Error!void {
    var a: Args = .{};
    a.add(keys);
    a.add(values);
    a.add(kind);
    a.add(@as(c_int, @intCast(p.rows)));
    a.add(@as(c_int, @intCast(kv_heads)));
    a.add(@as(c_int, @intCast(d)));
    a.add(p.args);
    a.add(@as(c_int, @intCast(layer)));
    try l.flat(l.plan.kv_write, @intCast(p.rows * kv_heads * d), s, &a);
}

/// One query a row (rows, heads, d) fp32 over its slot's caches up to its position; the walk covers `span` keys.
pub fn planCausal(l: *const Launcher, q: u64, out: u64, scores: u64, stats: u64, partials: u64, p: PlanRef, layer: usize, heads: usize, kv_heads: usize, d: usize, span: usize, scale: f32, kind: c_int, s: S) Error!void {
    if (d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or kind < 1 or kind > 2) return invalid("planned attention");
    const which: usize = @intCast(kind - 1);
    const rows = p.rows;
    const tiles: c_int = @intCast(cdiv(span, 128));
    const span_c: c_int = @intCast(span);
    const heads_c: c_int = @intCast(heads);
    const kv_c: c_int = @intCast(kv_heads);
    const d_c: c_int = @intCast(d);
    const layer_c: c_int = @intCast(layer);
    var a: Args = .{};
    a.add(q);
    a.add(scores);
    a.add(span_c);
    a.add(heads_c);
    a.add(kv_c);
    a.add(d_c);
    a.add(scale);
    a.add(p.args);
    a.add(layer_c);
    // a warp scores a key: four warps a block
    try l.go(l.plan.score[which], dim(cdiv(span, 4), heads, rows), dim(128, 1, 1), 0, s, &a);
    var b: Args = .{};
    b.add(scores);
    b.add(stats);
    b.add(span_c);
    b.add(@as(c_int, 0));
    b.add(p.args.pos);
    try l.go(l.softmax_stats, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &b);
    var c: Args = .{};
    c.add(scores);
    c.add(stats);
    c.add(partials);
    c.add(span_c);
    c.add(heads_c);
    c.add(kv_c);
    c.add(d_c);
    c.add(p.args);
    c.add(layer_c);
    try l.go(l.plan.apply[which], dim(tiles, heads, rows), dim(256, 1, 1), 0, s, &c);
    var e: Args = .{};
    e.add(partials);
    e.add(out);
    e.add(tiles);
    e.add(d_c);
    try l.go(l.sum_partials, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &e);
}

/// The DeltaNet recurrence of every slot's rows (q, k, v, gate, beta, y flat over the round's rows), state in the slot's caches.
pub fn planGatedDelta(l: *const Launcher, q: u64, k: u64, v: u64, gate: u64, beta: u64, y: u64, p: PlanRef, layer: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize, s: S) Error!void {
    if ((dk != 16 and dk != 128) or @rem(dv, 8) != 0 or key_heads < 1 or @rem(value_heads, key_heads) != 0) return invalid("planned gated delta");
    var a: Args = .{};
    a.add(q);
    a.add(k);
    a.add(v);
    a.add(gate);
    a.add(beta);
    a.add(y);
    a.add(@as(c_int, @intCast(key_heads)));
    a.add(@as(c_int, @intCast(value_heads)));
    a.add(@as(c_int, @intCast(dv)));
    a.add(p.args);
    a.add(@as(c_int, @intCast(layer)));
    const wide: usize = if (dk == 128) 0 else 1;
    try l.go(l.plan.gdn[wide], dim(@divExact(dv, 8), value_heads, p.slots), dim(32, 8, 1), 0, s, &a);
}

/// Row r of `dst` (`words` words each) from the address in `srcs[r]` (device u64s), for `rows` rows.
pub fn planGather(l: *const Launcher, srcs: u64, dst: u64, words: usize, rows: usize, s: S) Error!void {
    if (words % 4 != 0 or rows == 0) return invalid("planned gather");
    var a: Args = .{};
    a.add(srcs);
    a.add(dst);
    a.add(@as(c_int, @intCast(words)));
    try l.go(l.plan.gather, dim(cdiv(words, 256), rows, 1), dim(256, 1, 1), 0, s, &a);
}

/// What a keep reads: the plan, each slot's kept row count (negative: not listed), the round's final rows, the words of
/// one final row, of one conv snapshot and of one DeltaNet snapshot (multiples of four), and the layers.
pub const Keep = struct { keep: u64, hidden: u64, hidden_words: usize, conv_words: usize, delta_words: usize, layers: usize };

/// Keeps the listed slots' rows: linear states from the snapshot of the last kept row, and its final row.
pub fn planKeep(l: *const Launcher, p: PlanRef, k: Keep, s: S) Error!void {
    if (k.hidden_words % 4 != 0 or k.conv_words % 4 != 0 or k.delta_words % 4 != 0) return invalid("planned keep");
    var a: Args = .{};
    a.add(p.args);
    a.add(k.keep);
    a.add(k.hidden);
    a.add(@as(c_int, @intCast(k.hidden_words)));
    a.add(@as(c_int, @intCast(k.conv_words)));
    a.add(@as(c_int, @intCast(k.delta_words)));
    a.add(@as(c_int, @intCast(k.layers)));
    try l.go(l.plan.keep, dim(64, k.layers + 1, p.slots), dim(256, 1, 1), 0, s, &a);
}
