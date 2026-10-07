//! The products on MLX affine weights: the plain, grouped and routed matmuls, the embedding rows and the dense draft projection.

const t = @import("types.zig");
const launches = @import("../launches.zig");
const affine_launch = @import("../launch/affine.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const Affine = t.Affine;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// Whether the stream tile takes `m` rows of x against `w`: up to 16 rows, or any number in a lane round.
fn takesStream(o: Ops, z: anytype, m: usize, w: Affine, x: Tensor) bool {
    const kind = w.tables.table();
    if (o.window) return z.affine.windowTakes(int(w.n), int(w.k), int(w.bits), int(w.group), kind, kind, x.ptr);
    return z.affine.streamTakes(int(m), int(w.n), int(w.k), int(w.bits), int(w.group), kind, kind, x.ptr);
}

/// matmul(x, ...) as the Python wrapper runs it on the auto schedule: an fp32 product (`f32`), else the input
/// dtype; fp16 x of at most 8 rows on RDNA2 takes the decode tile's own fp16 rounding.
pub fn affine(o: Ops, x: Tensor, w: Affine, m: usize, f32_out: bool) Error!Tensor {
    try w.check();
    if (m == 0) return error.BadShape;
    const fp16 = x.kind == .f16;
    if (x.kind == .f32 or (fp16 and o.wmma()) or (!fp16 and !o.wmma())) return error.BadShape;
    // the decode tiles round to the activation type themselves: RDNA2's up to 8 rows, and the stream tile's either type
    const stream = if (o.prefill) false else if (o.lib.zig) |z| takesStream(o, z, m, w, x) else false;
    const half = !f32_out and !o.prefill and ((fp16 and m <= 8 and !o.wmma()) or stream);
    const n: usize = w.n;
    const out = try o.arena.take(m * n * @as(usize, if (half) 2 else 4));
    const groups: usize = w.k / w.group;
    var splits: c_int = 1;
    if (fp16 and !o.prefill) splits = launches.affineSplits(int(m), int(n), int(w.k), int(w.group), 0);
    const partial: u64 = if (splits > 1) try o.arena.of(f32, m * n * groups * 2) else 0;
    // prefill's tile at any row count is the Zig launches' (schedule 3); the library keeps its own rule
    const schedule: c_int = if (o.prefill and o.lib.zig != null) 3 else if (o.window and stream) 4 else 0;
    const args = .{ p(x.ptr), p(w.words), p(w.scale), p(w.bias), w.tables.table(), p(out), int(m), int(n), int(w.k), w.bits, w.group, schedule, @intFromBool(fp16), o.stream, f(partial), splits, @intFromBool(half) };
    try o.lib.call("tf_affine", args);
    if (half) return .{ .ptr = out, .kind = x.kind };
    if (f32_out) return .{ .ptr = out, .kind = .f32 };
    const narrow = try o.arena.take(m * n * 2);
    try o.cast(.{ .ptr = out, .kind = .f32 }, .{ .ptr = narrow, .kind = x.kind }, m * n);
    return .{ .ptr = narrow, .kind = x.kind };
}

/// Up to four products of the same `m` rows of x in one launch of the stream tile, each rounded to x's kind into
/// `outs`; false (nothing launched) when the products differ in K, width, group or tables, or the tile does not take
/// them, and the caller launches them one by one.
pub fn affineGroup(o: Ops, x: Tensor, ws: []const Affine, m: usize, outs: []Tensor) Error!bool {
    const z = o.lib.zig orelse return false;
    if (o.prefill or !o.fused() or ws.len < 2 or ws.len > 4 or m == 0 or (m > 16 and !o.window) or x.kind == .f32) return false;
    var widest: u32 = 0;
    for (ws) |w| {
        try w.check();
        if (w.k != ws[0].k or w.bits != ws[0].bits or w.group != ws[0].group or w.tables != ws[0].tables or w.partial) return false;
        widest = @max(widest, w.n);
    }
    const kind = ws[0].tables.table();
    const takes = if (o.window) z.affine.windowTakes(int(widest), int(ws[0].k), int(ws[0].bits), int(ws[0].group), kind, kind, x.ptr) else z.affine.streamTakes(int(m), int(widest), int(ws[0].k), int(ws[0].bits), int(ws[0].group), kind, kind, x.ptr);
    if (!takes) return false;
    var sides: [4]affine_launch.Side = undefined;
    for (ws, outs[0..ws.len], sides[0..ws.len]) |w, *out, *side| {
        out.* = .{ .ptr = try o.arena.take(m * w.n * 2), .kind = x.kind };
        side.* = .{ .words = w.words, .scale = w.scale, .bias = w.bias, .n = int(w.n), .out = out.ptr };
    }
    const arg: affine_launch.Arg = .{
        .x = x.ptr,
        .words = 0,
        .scale = .{ .p = 0, .kind = kind },
        .bias = .{ .p = 0, .kind = kind },
        .out = 0,
        .m = int(m),
        .n = int(widest),
        .k = int(ws[0].k),
        .bits = int(ws[0].bits),
        .group = int(ws[0].group),
        .fp16 = @intFromBool(x.kind == .f16),
    };
    try z.affine.groupRun(z.d, arg, sides[0..ws.len], true, o.stream);
    return true;
}

/// The routed gate and up in one launch with the activation as its epilogue: every item's stacked (gate | up) product
/// and silu(gate) * up in x's kind, out (pairs, width). Null (nothing launched) when the tile does not take the shape.
pub fn affineRoutedAct(o: Ops, x: Tensor, w: Affine, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize, limit: f32) Error!?Tensor {
    const z = o.lib.zig orelse return null;
    if (o.prefill or !o.fused() or x.kind == .f32 or w.n % 2 != 0) return null;
    try w.check();
    const kind = w.tables.table();
    if (!z.affine.streamTakes(int(rows), int(w.n), int(w.k), int(w.bits), int(w.group), kind, kind, x.ptr)) return null;
    const out = try o.arena.take(pairs * (w.n / 2) * 2);
    const arg: affine_launch.Arg = .{
        .x = x.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = kind },
        .bias = .{ .p = w.bias, .kind = kind },
        .out = 0,
        .m = int(rows),
        .n = int(w.n),
        .k = int(w.k),
        .bits = int(w.bits),
        .group = int(w.group),
        .fp16 = @intFromBool(x.kind == .f16),
        .route = .{ .items = items, .members = members, .x_div = int(x_div) },
        .out16 = out,
    };
    if (!try z.affine.pairRun(z.d, arg, limit, int(count), o.stream)) return null;
    return .{ .ptr = out, .kind = x.kind };
}

/// matmul_routed: every item (expert, first, count) in one launch over stacked weights; out (pairs, N) fp32.
pub fn affineRouted(o: Ops, x: Tensor, w: Affine, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize) Error!u64 {
    try w.check();
    const out = try o.arena.of(f32, pairs * w.n);
    if (o.prefill) if (o.lib.zig) |*z| {
        const kind = w.tables.table();
        try z.affine.prefillLaunch(z.d, .{ .x = x.ptr, .words = w.words, .scale = .{ .p = w.scale, .kind = kind }, .bias = .{ .p = w.bias, .kind = kind }, .out = out, .m = int(rows), .n = int(w.n), .k = int(w.k), .bits = int(w.bits), .group = int(w.group), .fp16 = @intFromBool(x.kind == .f16), .route = .{ .items = items, .members = members, .x_div = int(x_div) } }, o.stream, int(count));
        return out;
    };
    try o.lib.call("tf_affine_routed", .{ p(x.ptr), p(w.words), p(w.scale), p(w.bias), w.tables.table(), p(out), i(items), int(count), i(members), int(x_div), int(rows), int(w.n), int(w.k), w.bits, w.group, @intFromBool(x.kind == .f16), o.stream });
    return out;
}

/// gather_rows: `n` embedding rows of width `table.k` by device ids, dequantized into `out`.
pub fn embedRows(o: Ops, table: Affine, ids: u64, n: usize, out: Tensor) Error!void {
    try table.check();
    try o.lib.call("tf_embed_rows", .{ p(table.words), p(table.scale), p(table.bias), @backingInt(table.tables), @ptrFromInt(ids), int(n), table.bits, table.group, int(table.k), p(out.ptr), @backingInt(out.kind), o.stream });
}

/// x (rows, k) of x's kind times fp32 weights (n, k), out (rows, n) in x's kind: an unquantized draft projection.
pub fn denseRows(o: Ops, x: Tensor, w: u64, out: Tensor, rows: usize, n: usize, k: usize) Error!void {
    try o.lib.call("tf_dense_rows", .{ p(x.ptr), @backingInt(x.kind), f(w), p(out.ptr), int(rows), int(n), int(k), o.stream });
}
