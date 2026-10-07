//! Element-wise launches: embedding rows, casts, sums, column copies and the draft projection.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;

pub fn tf_embed_rows(l: *const Launcher, words: C, scale: C, bias: C, scale_kind: c_int, ids: CI, n: c_int, bits: c_int, group: c_int, k: c_int, out: P, out_kind: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(words));
    a.add(ad(scale));
    a.add(ad(bias));
    a.add(scale_kind);
    a.add(ad(ids));
    a.add(n);
    a.add(bits);
    a.add(group);
    a.add(k);
    a.add(ad(out));
    a.add(out_kind);
    try l.flat(l.op.embed_rows, @as(i64, n) * k, s, &a);
}

pub fn tf_cast(l: *const Launcher, src: C, skind: c_int, dst: P, dkind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(src));
    a.add(skind);
    a.add(ad(dst));
    a.add(dkind);
    a.add(n);
    try l.flat(l.op.cast, n, s, &a);
}

pub fn tf_silu_mul(l: *const Launcher, gate: C, up: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(gate));
    a.add(ad(up));
    a.add(ad(out));
    a.add(kind);
    a.add(n);
    try l.flat(l.op.silu_mul, n, s, &a);
}

pub fn tf_add(l: *const Launcher, x: C, y: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(x));
    a.add(ad(y));
    a.add(ad(out));
    a.add(kind);
    a.add(n);
    try l.flat(l.op.add, n, s, &a);
}

pub fn tf_copy_cols(l: *const Launcher, src: C, stride: c_longlong, offset: c_int, dst: P, kind: c_int, rows: c_int, cols: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(src));
    a.add(stride);
    a.add(offset);
    a.add(ad(dst));
    a.add(kind);
    a.add(rows);
    a.add(cols);
    try l.flat(l.op.copy_cols, @as(i64, rows) * cols, s, &a);
}

pub fn tf_dense_rows(l: *const Launcher, x: C, kind: c_int, w: CF, out: P, rows: c_int, n: c_int, k: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(x));
    a.add(kind);
    a.add(ad(w));
    a.add(ad(out));
    a.add(rows);
    a.add(n);
    a.add(k);
    try l.go(l.op.dense_rows, dim(n, rows, 1), dim(32, 1, 1), 0, s, &a);
}
