//! MoE launches: router, select, plan, activation and combine.

const std = @import("std");
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

/// The most rows the router's wave-an-expert kernel takes in prefill.
pub const router_small_rows = 32;

/// The router's logits at any row count so prompt rows ignore cuts: 64 x 64 tiles, or a wave an expert, same bits.
pub fn routerTile(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    return routerWith(l, x, kind, rows, logits, r, d, e, s, r <= router_small_rows);
}

pub fn routerWith(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S, small: bool) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    const k: usize = if (kind == 1) 0 else 1;
    if (small) return l.go(l.op.router_small[k], dim(cdiv(e, 8), cdiv(r, 32), 1), dim(256, 1, 1), 0, s, &a);
    try l.go(l.op.router_tile[k], dim(cdiv(e, 64), cdiv(r, 64), 1), dim(256, 1, 1), 0, s, &a);
}

/// A lane round's router at any row count; false (nothing launched) where the shape keeps the other kernels.
pub fn routerWindow(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!bool {
    if (!l.fuse or r < 1 or @rem(d, 4) != 0 or ad(rows) % 16 != 0 or ad(x) % 8 != 0) return false;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    try l.go(l.dec.router[if (kind == 1) 0 else 1], dim(cdiv(e, 4), cdiv(r, 16), 1), dim(128, 1, 1), 0, s, &a);
    return true;
}

/// A lane round's router over the stored bf16 rows, which are the fp32 rows' values held narrower.
pub fn routerBf16(l: *const Launcher, x: C, bf16: C, kind: c_int, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!bool {
    if (!l.fuse or r < 1 or r > 16 or @rem(d, 4) != 0 or ad(x) % 8 != 0 or ad(bf16) % 8 != 0) return false;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(bf16));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
    try l.go(l.dec.router_bf16[if (kind == 1) 0 else 1], dim(cdiv(e, 4), cdiv(r, 16), 1), dim(128, 1, 1), 0, s, &a);
    return true;
}

pub fn tf_moe_router(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
    // a prompt's rows take the 64 x 64 tiles, a round's few rows a wave an expert; each logit's sum is the same
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(rows));
    try a.add(ad(logits));
    try a.add(r);
    try a.add(d);
    try a.add(e);
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
    try a.add(ad(logits));
    try a.add(ad(pick));
    try a.add(ad(wts));
    try a.add(ad(items));
    try a.add(ad(members));
    try a.add(capacity);
    try a.add(experts);
    try a.add(top_k);
    if (l.fuse) return l.go(l.dec.select, dim(r, 1, 1), dim(512, 1, 1), 0, s, &a);
    try l.go(l.moe_select, dim(r, 1, 1), dim(32, 1, 1), 0, s, &a);
}

pub fn tf_moe_act(l: *const Launcher, both: CF, out: P, kind: c_int, pairs: c_int, width: c_int, limit: f32, s: S) Error!void {
    const total = @as(i64, pairs) * width;
    var a: Args = .{};
    try a.add(ad(both));
    try a.add(ad(out));
    try a.add(width);
    try a.add(limit);
    try a.add(total);
    try l.go(l.moe_act[if (kind == 1) 0 else 1], dim(cdiv(total, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// kernels/hip/ops/act.hip tf_moe_act_items: the activation of the pair rows of `count` plan items alone (`tile` the most pairs an item holds), same arithmetic as `tf_moe_act`.
pub fn tf_moe_act_items(l: *const Launcher, both: CF, out: P, kind: c_int, items: CI, members: CI, count: c_int, tile: c_int, width: c_int, limit: f32, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(both));
    try a.add(ad(out));
    try a.add(ad(items));
    try a.add(ad(members));
    try a.add(width);
    try a.add(limit);
    try l.go(l.moe_act_items[if (kind == 1) 0 else 1], dim(cdiv(@as(i64, tile) * width, 256), count, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_combine(l: *const Launcher, y: CF, wts: CF, out: P, kind: c_int, r: c_int, slots: c_int, d: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(wts));
    try a.add(ad(out));
    try a.add(slots);
    try a.add(d);
    try l.go(l.moe_combine[tri(kind)], dim(cdiv(d, 256), r, 1), dim(256, 1, 1), 0, s, &a);
}

/// The slots' sum plus a shared expert run beside the stack, weighted by wts' last slot; `shared` null adds nothing.
pub fn tf_moe_combine_gate(l: *const Launcher, y: CF, wts: CF, shared: CF, out: P, kind: c_int, r: c_int, slots: c_int, d: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(wts));
    try a.add(ad(shared));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(slots);
    try a.add(d);
    try l.go(l.moe_combine_gate, dim(cdiv(d, 256), r, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_moe_route(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, tile: c_int, members: I, items: I, capacity: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(experts);
    try a.add(tile);
    try a.add(ad(members));
    try a.add(ad(items));
    try a.add(capacity);
    // every expert's count, start and end, and each pair segment's 16-bit counts, in dynamic shared memory
    const ex: usize = @intCast(experts);
    const segs: usize = if (ex <= 256) 64 else 16;
    if (ex > 1024) return invalid("moe_route");
    // and the block's scan of two ints a thread, past the segments (4-byte aligned)
    const seg_bytes = std.mem.alignForward(usize, segs * ex * 2, 4);
    try l.go(l.op.moe_route, dim(1, 1, 1), dim(256, 1, 1), @intCast(3 * ex * 4 + seg_bytes + 2 * 256 * 4), s, &a);
}

/// A layer's `pairs` picks into the trace at `dst` (host memory mapped into the device); one block.
pub fn tf_expert_trace(l: *const Launcher, picks: CI, pairs: c_int, dst: I, s: S) Error!void {
    if (comptime !@import("../kernels.zig").expert_ptrs) return invalid("expert_trace");
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(ad(dst));
    try l.go(l.op.expert_trace, dim(1, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// The hit/miss check of an offloaded decode layer (kernels/hip/ops/moe.hpp tf_expert_check; one wave): `picks` (`pairs` ids) to the layer's trace slot `trace`, each looked up in
/// `resident` (a u64 a key of this layer's experts); a hit stores `seen` and `ack` = `n`, a miss `req` = `n`.
/// `split` (hit-first): the plan's `items` (`count` entries) are also written to `hit` and `miss` by class, each with the other's pair counts zeroed, and the layer's VRAM table `row`
/// gets the resident picks' entries (line, 0) at once (all 0: not split, and `tf_expert_rows` writes them after the wait).
pub const ItemSplit = struct { items: u64 = 0, count: c_int = 0, hit: u64 = 0, miss: u64 = 0, row: u64 = 0 };

pub fn tf_expert_check(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, resident: u64, trace: u64, ack: u64, seen: u64, req: u64, n: u32, split: ItemSplit, s: S) Error!void {
    if (comptime !@import("../kernels.zig").expert_ptrs) return invalid("expert_check");
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(experts);
    try a.add(resident);
    try a.add(split.row);
    try a.add(trace);
    try a.add(ack);
    try a.add(seen);
    try a.add(req);
    try a.add(n);
    try a.add(split.items);
    try a.add(split.count);
    try a.add(split.hit);
    try a.add(split.miss);
    // one wave: a decode layer has at most a few dozen picks, and every wave of the block would wait on its own system-scope fence
    try l.go(l.op.expert_check, dim(1, 1, 1), dim(64, 1, 1), 0, s, &a);
}

/// The plan's `items` (`count` entries) written `nchunks` times at `out` (`stride` ints apart), each keeping the pair counts of the experts `chunk` (host memory, a u32 an expert) puts in that chunk
/// (kernels/hip/ops/moe.hpp tf_expert_chunks; one block).
pub fn tf_expert_chunks(l: *const Launcher, items: CI, count: c_int, experts: c_int, chunk: u64, nchunks: c_int, stride: c_int, out: u64, s: S) Error!void {
    if (comptime !@import("../kernels.zig").expert_ptrs) return invalid("expert_chunks");
    var a: Args = .{};
    try a.add(ad(items));
    try a.add(count);
    try a.add(experts);
    try a.add(chunk);
    try a.add(nchunks);
    try a.add(stride);
    try a.add(out);
    try l.go(l.op.expert_chunks, dim(1, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// After the wait on the ack: the VRAM table `row` for the picked experts (all `experts` when `all`): (line, 0) from `resident` when `seen` says the card passed job `n`, the host's row otherwise.
pub fn tf_expert_rows(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, host_row: u64, row: u64, resident: u64, seen: u64, n: u32, all: bool, s: S) Error!void {
    if (comptime !@import("../kernels.zig").expert_ptrs) return invalid("expert_rows");
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(experts);
    try a.add(host_row);
    try a.add(row);
    try a.add(resident);
    try a.add(seen);
    try a.add(n);
    try a.add(@as(c_int, @intFromBool(all)));
    try l.go(l.op.expert_rows, dim(1, 1, 1), dim(64, 1, 1), 0, s, &a);
}

/// TF_EXPERT_VERIFY: the record of the `pairs` picks' table entries, the residency words and the blobs' ends into the host-mapped ring slot `out` as job `n`'s.
pub fn tf_expert_snap(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, resident: u64, row: u64, out: u64, n: u32, blob: c_int, s: S) Error!void {
    if (comptime !@import("../kernels.zig").expert_ptrs) return invalid("expert_snap");
    var a: Args = .{};
    try a.add(ad(picks));
    try a.add(pairs);
    try a.add(experts);
    try a.add(resident);
    try a.add(row);
    try a.add(out);
    try a.add(n);
    try a.add(blob);
    try l.go(l.op.expert_snap, dim(1, 1, 1), dim(64, 1, 1), 0, s, &a);
}
