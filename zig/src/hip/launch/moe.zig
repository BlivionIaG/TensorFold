//! MoE launches: router, select, plan, activation and combine.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const I = util.I;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const tri = util.tri;
const invalid = util.invalid;

/// The router's 64 x 64 tiles at any row count (prefill's, so a prompt's rows do not depend on its cuts).
pub fn routerTile(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(x));
    a.add(ad(rows));
    a.add(ad(logits));
    a.add(r);
    a.add(d);
    a.add(e);
    try l.go(l.op.router_tile[if (kind == 1) 0 else 1], dim(cdiv(e, 64), cdiv(r, 64), 1), dim(256, 1, 1), 0, s, &a);
}

/// A lane round's router at any row count; false (nothing launched) where the shape keeps the other kernels.
pub fn routerWindow(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!bool {
    if (!l.fuse or r < 1 or @rem(d, 4) != 0 or ad(rows) % 16 != 0 or ad(x) % 8 != 0) return false;
    var a: Args = .{};
    a.add(ad(x));
    a.add(ad(rows));
    a.add(ad(logits));
    a.add(r);
    a.add(d);
    a.add(e);
    try l.go(l.dec.router[if (kind == 1) 0 else 1], dim(cdiv(e, 4), cdiv(r, 16), 1), dim(128, 1, 1), 0, s, &a);
    return true;
}

pub fn tf_moe_router(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    // a prompt's rows take the 64 x 64 tiles, a round's few rows a wave an expert; each logit's sum is the same
    var a: Args = .{};
    a.add(ad(x));
    a.add(ad(rows));
    a.add(ad(logits));
    a.add(r);
    a.add(d);
    a.add(e);
    const k: usize = if (kind == 1) 0 else 1;
    if (l.fuse and r <= 16 and @rem(d, 4) == 0 and ad(rows) % 16 == 0 and ad(x) % 8 == 0) {
        return l.go(l.dec.router[k], dim(cdiv(e, 4), 1, 1), dim(128, 1, 1), 0, s, &a);
    }
    if (r >= 64 and @rem(d, 32) == 0) return l.go(l.op.router_tile[k], dim(cdiv(e, 64), cdiv(r, 64), 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.op.router_rows[k], dim(cdiv(e, 8), cdiv(r, 8), 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_select(l: *const Launcher, logits: CF, pick: I, wts: F, items: I, members: I, capacity: c_int, r: c_int, experts: c_int, top_k: c_int, s: S) Error!void {
    if (experts > 1024 or top_k < 1 or top_k > 31) return invalid("moe select");
    var a: Args = .{};
    a.add(ad(logits));
    a.add(ad(pick));
    a.add(ad(wts));
    a.add(ad(items));
    a.add(ad(members));
    a.add(capacity);
    a.add(experts);
    a.add(top_k);
    if (l.fuse) return l.go(l.dec.select, dim(r, 1, 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.moe_select, dim(r, 1, 1), dim(32, 1, 1), 0, s, &a);
}

pub fn tf_moe_act(l: *const Launcher, both: CF, out: P, kind: c_int, pairs: c_int, width: c_int, limit: f32, s: S) Error!void {
    const total = @as(i64, pairs) * width;
    var a: Args = .{};
    a.add(ad(both));
    a.add(ad(out));
    a.add(width);
    a.add(limit);
    a.add(total);
    try l.go(l.moe_act[if (kind == 1) 0 else 1], dim(cdiv(total, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_combine(l: *const Launcher, y: CF, wts: CF, out: P, kind: c_int, r: c_int, slots: c_int, d: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(y));
    a.add(ad(wts));
    a.add(ad(out));
    a.add(slots);
    a.add(d);
    try l.go(l.moe_combine[tri(kind)], dim(cdiv(d, 256), r, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_route(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, tile: c_int, members: I, items: I, capacity: c_int, s: S) Error!void {
    var a: Args = .{};
    a.add(ad(picks));
    a.add(pairs);
    a.add(experts);
    a.add(tile);
    a.add(ad(members));
    a.add(ad(items));
    a.add(capacity);
    // every expert's count, start and end, and each pair segment's 16-bit counts, in dynamic shared memory
    const ex: usize = @intCast(experts);
    const segs: usize = if (ex <= 256) 64 else 16;
    if (ex > 1024) return invalid("moe_route");
    try l.go(l.op.moe_route, dim(1, 1, 1), dim(256, 1, 1), @intCast(3 * ex * 4 + segs * ex * 2), s, &a);
}
