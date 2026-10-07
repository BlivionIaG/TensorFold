//! RoPE launches: decode, prefill and the window's fused q / k norm and rotation.

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

pub fn tf_qk_rope(l: *const Launcher, src: C, kind: c_int, s_row: c_longlong, s_head: c_int, weight: CF, eps: f32, rows: c_int, heads: c_int, width: c_int, rotary: c_int, theta: f32, pos: CI, wide: F, cache: P, total: c_int, s: S) Error!void {
    if (width > 512) return invalid("qk_rope");
    var a: Args = .{};
    a.add(ad(src));
    a.add(kind);
    a.add(s_row);
    a.add(s_head);
    a.add(ad(weight));
    a.add(eps);
    a.add(rows);
    a.add(heads);
    a.add(width);
    a.add(rotary);
    a.add(theta);
    a.add(ad(pos));
    a.add(ad(wide));
    a.add(ad(cache));
    a.add(total);
    // a wave a (row, head), 8 a block
    try l.go(l.op.qk_rope, dim(cdiv(@as(i64, rows) * heads, 8), 1, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_rope_decode(l: *const Launcher, x: CF, y: F, rows: c_int, width: c_int, rotary: c_int, pos: c_int, theta: f32, s: S, pos_dev: CI, per: c_int) Error!void {
    var a: Args = .{};
    a.add(ad(x));
    a.add(ad(y));
    a.add(width);
    a.add(rotary);
    a.add(pos);
    a.add(theta);
    a.add(ad(pos_dev));
    a.add(per);
    try l.go(l.rope_decode, dim(rows, 1, 1), dim(32, 1, 1), 0, s, &a);
}

pub fn tf_rope_prefill(l: *const Launcher, x: C, kind: c_int, out: P, out_kind: c_int, s_head: c_longlong, s_row: c_longlong, len: c_int, heads: c_int, d: c_int, rotary: c_int, pos0: c_int, theta: f32, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(x));
    a.add(kind);
    a.add(ad(out));
    a.add(out_kind);
    a.add(s_head);
    a.add(s_row);
    a.add(len);
    a.add(heads);
    a.add(d);
    a.add(rotary);
    a.add(pos0);
    a.add(theta);
    try l.flat(l.op.rope_prefill, @as(i64, len) * heads * d, s, &a);
}
