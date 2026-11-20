//! GLP steering's calls (tf_dsv41_glp_v1, glp.cu): the two spliced values of a steered block N, as emitted around it.
//! Fold: the mean of the four streams after block N's FFN write folded in (before the next block's Engram or site; for
//! the last block, before the head). Writer: block N's attention output after the full TP reduction, before hc post.
//! A run without a plan (no vector, or strength 0) never reaches here: every caller asks `steers` first.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const glp_plan = @import("glp_plan.zig");
const moe = @import("block_moe.zig");
const E = block.Emitter;
const Arg = calls.Arg;

const ns = "tf_dsv41_glp_v1.";

/// Whether block `layer` is steered in this run: a plan is held and the vector has a direction for the layer.
pub fn steers(e: *const E, layer: u32) bool {
    const p = e.o.glp orelse return false;
    return p.has(layer);
}

/// A steered run's program shape: every block from 0, whole, without the trace copies or the TBO split (each would
/// show or skip an unsteered fold). Refused, never silently unsteered.
pub fn check(e: *const E, layers: []const u32, whole: bool) !void {
    if (e.o.glp == null) return;
    if (e.o.trace or e.marks != null or !whole) return error.InvalidBlock;
    for (layers, 0..) |L, i| if (L != i) return error.InvalidBlock;
}

fn mode(p: *const glp_plan.Plan) Arg {
    return .{ .i = @intFromEnum(p.mode) };
}

/// The layer's direction: its row of the plan's table (fp32 [width]).
fn dirOf(e: *E, layer: u32) !Arg {
    const p = e.o.glp.?;
    const w: i64 = p.width;
    return e.view(.{ .buf = glp_plan.dirs_role }, .f32, &.{w}, &.{1}, @as(i64, p.slot(layer).?) * w * 4);
}

/// DSpark's tap of block `layer`'s entry (the steered mean), when the run takes taps and the layer is a target.
pub fn tap(e: *E, layer: u32) !?Arg {
    if (!e.o.taps) return null;
    const t = e.cfg.dspark_targets.items();
    const j = std.mem.indexOfScalar(u32, t, layer) orelse return null;
    const D: i64 = e.cfg.hidden;
    return try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, D }, &.{ @as(i64, @intCast(t.len)) * D, 1 }, @as(i64, @intCast(j)) * D * 2);
}

/// Block `layer`'s fold edit on the streams in place (post already applied), the tap of the steered mean beside.
pub fn fold(e: *E, layer: u32, tap_to: ?Arg) !void {
    const p = e.o.glp.?;
    try e.ext(ns ++ "stream", &.{
        try e.streams(),
        tap_to orelse try e.empty(.bf16),
        try dirOf(e, layer),
        .{ .f = p.alpha },
        mode(p),
        .{ .i = e.n },
        .{ .i = p.width },
    });
}

/// Block `layer`'s writer edit on the gathered attention partials `g` [world, n, hidden], in place.
pub fn writer(e: *E, layer: u32, g: Arg) !void {
    const p = e.o.glp.?;
    try e.ext(ns ++ "writer", &.{ g, try dirOf(e, layer), .{ .f = p.alpha }, mode(p), .{ .i = e.n }, .{ .i = p.width } });
}

/// The head's collapse and final norm of the steered streams into "w.out" (mHC final's work, for a steered last block).
pub fn final(e: *E) !void {
    const D: i64 = e.cfg.hidden;
    try e.ext(ns ++ "final", &.{
        try e.streams(),
        try e.buf("w.c{d}.pre", .{e.coef}, .f32, &.{ e.n, 4 }),
        try e.weight("norm", .{}, .f32, &.{D}),
        try e.buf("w.out", .{}, .bf16, &.{ e.n, D }),
        .{ .f = E.dec(e.cfg.eps) },
        .{ .i = e.n },
        .{ .i = D },
    });
}

/// A decode window's end when the last block is steered: its post alone, its fold edit, the steered final, the head.
pub fn finish(e: *E, layer: u32) !void {
    try moe.postOnly(e, null);
    try fold(e, layer, null);
    try final(e);
    try moe.head(e);
}
