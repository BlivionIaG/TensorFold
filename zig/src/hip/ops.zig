//! Typed launches of the ROCm kernel library on device addresses: shapes checked here (the C launchers trust them),
//! scratch from the forward's arena, the Python wrappers' choices (splits, fp16 decode output) made the same way.

const std = @import("std");
const abi = @import("runtime/abi.zig");
const rocm = @import("rocm.zig");
const launches = @import("launches.zig");
const affine_launch = @import("affine_launch.zig");
const Arena = @import("runtime/arena.zig").Arena;

/// act.hpp's numbering: the activation and fp32 buffers the torch-op kernels read and write.
pub const Kind = enum(c_int) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,

    pub fn size(k: Kind) usize {
        return if (k == .f32) 4 else 2;
    }

    /// The attention kernel's cache numbering: 0 fp16, 1 bf16, 2 fp32.
    fn cache(k: Kind) c_int {
        return switch (k) {
            .f16 => 0,
            .bf16 => 1,
            .f32 => 2,
        };
    }

    /// The affine kernels' table numbering: 0 fp32, 1 bf16, 2 fp16.
    fn table(k: Kind) c_int {
        return switch (k) {
            .f32 => 0,
            .bf16 => 1,
            .f16 => 2,
        };
    }
};

/// Rows from which the DeltaNet prefill runs chunked, and the rows of one chunked segment (a multiple of the chunk).
const gdn_min_rows = 64;
const gdn_segment = 4096;

fn chunkedOff() bool {
    const text = std.c.getenv("TF_GDN_CHUNKED") orelse return false;
    return text[0] == '0';
}

pub const Error = rocm.Error || error{ OutOfDeviceMemory, BadShape };

/// A device buffer of `kind` values.
pub const Tensor = struct { ptr: u64, kind: Kind };

/// One MLX affine matrix (N, K): packed words (N, K * bits / 32), scale and bias (N, K / group) of `tables`.
pub const Affine = struct {
    words: u64,
    scale: u64,
    bias: u64,
    tables: Kind,
    n: u32,
    k: u32,
    bits: u8,
    group: u16,
    /// A tensor-parallel slice along K: the product stays fp32, one rank's share of a sum.
    partial: bool = false,

    fn check(a: Affine) Error!void {
        const ok_bits = switch (a.bits) {
            2, 3, 4, 5, 6, 8 => true,
            else => false,
        };
        if (!ok_bits or (a.group != 32 and a.group != 64 and a.group != 128)) return error.BadShape;
        if (a.k % a.group != 0 or (@as(u64, a.k) * a.bits) % 32 != 0) return error.BadShape;
    }
};

fn p(addr: u64) ?*anyopaque {
    return @ptrFromInt(addr);
}

fn f(addr: u64) ?[*]f32 {
    return @ptrFromInt(addr);
}

fn i(addr: u64) ?[*]i32 {
    return @ptrFromInt(addr);
}

fn int(v: anytype) c_int {
    return @intCast(v);
}

pub const Ops = struct {
    lib: *const rocm.Library,
    stream: abi.Stream,
    arena: *Arena,
    /// A prompt's span: every product, the router and the recurrence take prefill's one kernel at any row count, so a
    /// prompt's rows have the same bits however it is cut (fresh, resumed or in steps).
    prefill: bool = false,
    /// A lane round's forward: the decode tiles at any row count (the stream tile over 4-row blocks, the router a wave an
    /// expert, 8-row routed items), so a row's bits do not depend on the rows it shares the round with.
    window: bool = false,

    fn wmma(o: Ops) bool {
        return o.lib.family == .rdna3;
    }

    pub fn rms(o: Ops, x: Tensor, weight: ?u64, y: Tensor, rows: usize, width: usize, eps: f32) Error!void {
        if (x.kind != y.kind or width < 1 or width > 8192) return error.BadShape;
        if (rows == 0) return;
        try o.lib.call("tf_rms", .{ p(x.ptr), if (weight) |w| f(w) else null, p(y.ptr), @backingInt(x.kind), int(rows), int(width), eps, o.stream });
    }

    /// matmul(x, ...) as the Python wrapper runs it on the auto schedule: an fp32 product (`f32`), else the input
    /// dtype; fp16 x of at most 8 rows on RDNA2 takes the decode tile's own fp16 rounding.
    /// Whether the stream tile takes `m` rows of x against `w`: up to 16 rows, or any number in a lane round.
    fn takesStream(o: Ops, z: anytype, m: usize, w: Affine, x: Tensor) bool {
        const kind = w.tables.table();
        if (o.window) return z.affine.windowTakes(int(w.n), int(w.k), int(w.bits), int(w.group), kind, kind, x.ptr);
        return z.affine.streamTakes(int(m), int(w.n), int(w.k), int(w.bits), int(w.group), kind, kind, x.ptr);
    }

    pub fn affine(o: Ops, x: Tensor, w: Affine, m: usize, f32_out: bool) Error!Tensor {
        try w.check();
        if (m == 0) return error.BadShape;
        const fp16 = x.kind == .f16;
        if (x.kind == .f32 or (fp16 and o.wmma()) or (!fp16 and !o.wmma())) return error.BadShape;
        // the decode tiles round to the activation type themselves: RDNA2's up to 8 rows, and the stream tile's either type
        const stream = if (o.prefill) false else if (o.lib.zig) |z| o.takesStream(z, m, w, x) else false;
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

    /// Whether decode.hip's merged launches are on: Zig launches, and TF_DECODE_FUSE is not `old`.
    pub fn fused(o: Ops) bool {
        return if (o.lib.zig) |z| z.fuse else false;
    }

    /// Two sets of fp32 rows (width at most 1024), each normed with its own weight, in one launch.
    pub fn rms2(o: Ops, x0: u64, w0: u64, y0: u64, x1: u64, w1: u64, y1: u64, rows: usize, width: usize, eps: f32) Error!void {
        const z = o.lib.zig orelse return error.BadShape;
        try z.tf_rms2(f(x0), f(w0), f(y0), f(x1), f(w1), f(y1), int(rows), int(width), eps, o.stream);
    }

    /// The linear attention's gated norm: out = rms(y) * weight * round(silu(z)) in z's kind, rows of `width` (at most 1024).
    pub fn gnormOut(o: Ops, y: u64, weight: u64, z: Tensor, out: Tensor, rows: usize, width: usize, eps: f32) Error!void {
        if (z.kind != out.kind or z.kind == .f32) return error.BadShape;
        const zig = o.lib.zig orelse return error.BadShape;
        try zig.tf_gnorm_out(f(y), f(weight), p(z.ptr), p(out.ptr), @backingInt(z.kind), int(rows), int(width), eps, o.stream);
    }

    /// x = round(x + y) and normed = rms(x) * weight (fp32) in one launch: a residual and the norm after it.
    pub fn addRms(o: Ops, x: Tensor, y: Tensor, weight: u64, normed: Tensor, rows: usize, width: usize, eps: f32) Error!void {
        if (x.kind != y.kind or x.kind != normed.kind or x.kind == .f32) return error.BadShape;
        const z = o.lib.zig orelse return error.BadShape;
        try z.tf_tail(p(x.ptr), p(y.ptr), null, f(weight), p(normed.ptr), @backingInt(x.kind), int(rows), 0, int(width), eps, o.stream);
    }

    /// The MoE combine, the residual and the next norm in one launch: x = round(x + round(sum_s wts[r, s] * y[r, s])),
    /// y (rows * slots, width) fp32, then normed = rms(x) * weight.
    pub fn moeTail(o: Ops, x: Tensor, y: u64, wts: u64, weight: u64, normed: Tensor, rows: usize, slots: usize, width: usize, eps: f32) Error!void {
        if (x.kind != normed.kind or x.kind == .f32 or slots == 0) return error.BadShape;
        const z = o.lib.zig orelse return error.BadShape;
        try z.tf_tail(p(x.ptr), f(y), f(wts), f(weight), p(normed.ptr), @backingInt(x.kind), int(rows), int(slots), int(width), eps, o.stream);
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

    pub fn cast(o: Ops, src: Tensor, dst: Tensor, n: usize) Error!void {
        try o.lib.call("tf_cast", .{ p(src.ptr), @backingInt(src.kind), p(dst.ptr), @backingInt(dst.kind), @intCast(n), o.stream });
    }

    /// gather_rows: `n` embedding rows of width `table.k` by device ids, dequantized into `out`.
    pub fn embedRows(o: Ops, table: Affine, ids: u64, n: usize, out: Tensor) Error!void {
        try table.check();
        try o.lib.call("tf_embed_rows", .{ p(table.words), p(table.scale), p(table.bias), @backingInt(table.tables), @ptrFromInt(ids), int(n), table.bits, table.group, int(table.k), p(out.ptr), @backingInt(out.kind), o.stream });
    }

    pub fn siluMul(o: Ops, gate: Tensor, up: Tensor, out: Tensor, n: usize) Error!void {
        if (gate.kind != up.kind or gate.kind != out.kind) return error.BadShape;
        try o.lib.call("tf_silu_mul", .{ p(gate.ptr), p(up.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream });
    }

    pub fn add(o: Ops, x: Tensor, y: Tensor, out: Tensor, n: usize) Error!void {
        if (x.kind != y.kind or x.kind != out.kind) return error.BadShape;
        try o.lib.call("tf_add", .{ p(x.ptr), p(y.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream });
    }

    /// The gated attention's o input from att fp32, (heads, len, d) or with `rows_major` (len, heads, d), and the gate half.
    pub fn attnGate(o: Ops, att: u64, qg: Tensor, out: Tensor, len: usize, heads: usize, d: usize, rows_major: bool) Error!void {
        try o.lib.call("tf_attn_gate", .{ f(att), p(qg.ptr), p(out.ptr), @backingInt(out.kind), int(len), int(heads), int(d), @intFromBool(rows_major), o.stream });
    }

    pub fn gnormSilu(o: Ops, y: u64, z: Tensor, out: Tensor, n: usize) Error!void {
        try o.lib.call("tf_gnorm_silu", .{ f(y), p(z.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream });
    }

    pub fn copyCols(o: Ops, src: Tensor, stride: usize, offset: usize, dst: u64, rows: usize, cols: usize) Error!void {
        try o.lib.call("tf_copy_cols", .{ p(src.ptr), @intCast(stride), int(offset), p(dst), @backingInt(src.kind), int(rows), int(cols), o.stream });
    }

    /// Prefill RoPE of x (len, heads, d), rounded to x's kind, stored as `out` kind at out[h * s_head + r * s_row + j].
    pub fn ropePrefill(o: Ops, x: Tensor, out: Tensor, s_head: usize, s_row: usize, len: usize, heads: usize, d: usize, rotary: usize, pos0: usize, theta: f32) Error!void {
        try o.lib.call("tf_rope_prefill", .{ p(x.ptr), @backingInt(x.kind), p(out.ptr), @backingInt(out.kind), @intCast(s_head), @intCast(s_row), int(len), int(heads), int(d), int(rotary), int(pos0), theta, o.stream });
    }

    /// The decode RoPE over fp32 rows; `pos_dev` (int32, one a group of `per` rows) or the host `pos`.
    /// A window's q or k heads (row stride `s_row`, head stride `s_head` elements of `src`): each head RMS-normed with
    /// `weight`, rounded, rotated at its row's position (`pos`, int32 a row), rounded; into fp32 `wide` and/or the
    /// cache slots of those positions (`cache`, `total` slots a head).
    pub fn qkRope(o: Ops, src: Tensor, s_row: usize, s_head: usize, weight: u64, eps: f32, rows: usize, heads: usize, width: usize, rotary: usize, theta: f32, pos: u64, wide: ?u64, cache: ?u64, total: usize) Error!void {
        if (width > 512 or src.kind == .f32) return error.BadShape;
        try o.lib.call("tf_qk_rope", .{ p(src.ptr), @backingInt(src.kind), @intCast(s_row), int(s_head), f(weight), eps, int(rows), int(heads), int(width), int(rotary), theta, @ptrFromInt(pos), if (wide) |w| f(w) else null, if (cache) |c| p(c) else null, int(total), o.stream });
    }

    pub fn ropeDecode(o: Ops, x: u64, y: u64, rows: usize, width: usize, rotary: usize, pos: usize, theta: f32, pos_dev: ?u64, per: usize) Error!void {
        try o.lib.call("tf_rope_decode", .{ f(x), f(y), int(rows), int(width), int(rotary), int(pos), theta, o.stream, if (pos_dev) |a| @ptrFromInt(a) else null, int(per) });
    }

    pub fn kvWrite(o: Ops, src: Tensor, cache: u64, len: usize, kv_heads: usize, d: usize, total: usize, pos0: usize) Error!void {
        try o.lib.call("tf_kv_write", .{ p(src.ptr), p(cache), @backingInt(src.kind), int(len), int(kv_heads), int(d), int(total), int(pos0), o.stream });
    }

    /// kv_write with the first slot read from the device (one int32), for graphs replayed at new positions.
    pub fn kvWriteAt(o: Ops, src: Tensor, cache: u64, len: usize, kv_heads: usize, d: usize, total: usize, pos: u64) Error!void {
        try o.lib.check(o.lib.api.tf_kv_write_at(p(src.ptr), p(cache), @backingInt(src.kind), int(len), int(kv_heads), int(d), int(total), @ptrFromInt(pos), o.stream), "kv_write_at");
    }

    pub fn convPrefill(o: Ops, x: Tensor, weight: u64, state: ?u64, out: u64, new_state: u64, len: usize, channels: usize, kernel: usize) Error!void {
        if (kernel < 1 or kernel > 8) return error.BadShape;
        try o.lib.call("tf_conv_prefill", .{ p(x.ptr), @backingInt(x.kind), f(weight), if (state) |s| f(s) else null, f(out), f(new_state), int(len), int(channels), int(kernel), o.stream });
    }

    pub fn convDecode(o: Ops, x: u64, weight: u64, state: u64, y: u64, channels: usize, kernel: usize) Error!void {
        try o.lib.call("tf_conv_decode", .{ f(x), f(weight), f(state), f(y), 1, int(channels), int(kernel), o.stream });
    }

    pub fn convRows(o: Ops, x: u64, weight: u64, state: u64, y: u64, states: ?u64, rows: usize, channels: usize, kernel: usize) Error!void {
        try o.lib.call("tf_conv_rows", .{ f(x), f(weight), f(state), f(y), if (states) |s| f(s) else null, int(rows), int(channels), int(kernel), o.stream });
    }

    pub fn gdnGatePrefill(o: Ops, a: Tensor, b: Tensor, a_log: u64, dt_bias: u64, gate: u64, beta: u64, count: usize, heads: usize) Error!void {
        try o.lib.call("tf_gdn_gate_prefill", .{ p(a.ptr), p(b.ptr), @backingInt(a.kind), f(a_log), f(dt_bias), f(gate), f(beta), int(count), int(heads), o.stream });
    }

    /// The fused gate over `elements` values of a and b (rows times heads).
    pub fn gdnGate(o: Ops, a: Tensor, b: Tensor, a_log: u64, dt_bias: u64, gate: u64, beta: u64, elements: usize, heads: usize) Error!void {
        try o.lib.call("tf_gdn_gate", .{ p(a.ptr), p(b.ptr), @backingInt(a.kind), f(a_log), f(dt_bias), f(gate), f(beta), int(elements), int(heads), o.stream });
    }

    /// The chunked recurrence in segments of `gdn_segment` rows, whose scratch the arena hands back after each.
    fn gatedDeltaChunked(o: Ops, z: *const launches.Launcher, q: u64, k: u64, v: u64, gate: u64, beta: u64, state: u64, y: u64, length: usize, key_heads: usize, value_heads: usize) Error!void {
        var start: usize = 0;
        while (start < length) : (start += gdn_segment) {
            const rows = @min(gdn_segment, length - start);
            const mark = o.arena.mark();
            defer o.arena.release(mark);
            const scratch = try o.arena.take(launches.gdnScratch(rows, key_heads, value_heads).total);
            const qk = start * key_heads * 128 * 4;
            const vy = start * value_heads * 128 * 4;
            const gb = start * value_heads * 4;
            try z.gdnChunked(f(q + qk), f(k + qk), f(v + vy), f(gate + gb), f(beta + gb), f(state), f(y + vy), rows, key_heads, value_heads, scratch, o.stream);
        }
    }

    /// The DeltaNet recurrence: q, k (L, Hk, dk), v, y (L, Hv, dv), gate and beta (L, Hv) fp32; state in place.
    /// A prefill (no snapshots) of at least `gdn_min_rows` rows runs chunked on the Zig launches; TF_GDN_CHUNKED=0 keeps
    /// the token-serial kernel.
    pub fn gatedDelta(o: Ops, q: u64, k: u64, v: u64, gate: u64, beta: u64, state: u64, y: u64, length: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize, states: ?u64) Error!void {
        if (states == null and (length >= gdn_min_rows or o.prefill) and dk == 128 and dv == 128 and value_heads % key_heads == 0) {
            if (o.lib.zig) |*z| if (!chunkedOff()) return o.gatedDeltaChunked(z, q, k, v, gate, beta, state, y, length, key_heads, value_heads);
        }
        try o.lib.call("tf_gated_delta", .{ f(q), f(k), f(v), f(gate), f(beta), f(state), f(y), 1, int(length), int(key_heads), int(value_heads), int(dk), int(dv), o.stream, if (states) |s| f(s) else null });
    }

    /// The cache's layout: `kv_heads` heads of `total` slots of `d` values (batch 1).
    pub const Cache = struct { k: u64, v: u64, kind: Kind, kv_heads: usize, total: usize, d: usize };

    /// Prefill attention on the prefill tile (any query count): q (heads, qlen, d) fp32 over the first `span` slots.
    pub fn causalPrefill(o: Ops, q: u64, c: Cache, out: u64, qlen: usize, span: usize, heads: usize, scale: f32, q_pos0: usize) Error!void {
        if (c.d > 256 or heads % c.kv_heads != 0) return error.BadShape;
        const sh: c_longlong = @intCast(c.total * c.d);
        const ss: c_longlong = @intCast(c.d);
        try o.lib.call("tf_causal", .{ f(q), p(c.k), p(c.v), f(out), 1, int(qlen), int(span), int(heads), int(c.kv_heads), int(c.d), scale, int(q_pos0), sh * @as(c_longlong, @intCast(c.kv_heads)), sh, ss, sh * @as(c_longlong, @intCast(c.kv_heads)), sh, ss, c.kind.cache(), null, null, null, o.stream, null });
    }

    /// causal_at: `rows` queries (rows, heads, 1, d) fp32 each at its device position over one shared cache.
    pub fn causalAt(o: Ops, q: u64, c: Cache, out: u64, rows: usize, heads: usize, scale: f32, pos: u64) Error!void {
        if (c.d > 256 or heads % c.kv_heads != 0) return error.BadShape;
        const span = c.total;
        const scores = try o.arena.of(f32, rows * heads * span);
        const stats = try o.arena.of(f32, rows * heads * 2);
        const partials = try o.arena.of(f32, rows * heads * ((span + 127) / 128) * c.d);
        const sh: c_longlong = @intCast(c.total * c.d);
        const ss: c_longlong = @intCast(c.d);
        try o.lib.call("tf_causal", .{ f(q), p(c.k), p(c.v), f(out), int(rows), 1, int(span), int(heads), int(c.kv_heads), int(c.d), scale, 0, 0, sh, ss, 0, sh, ss, c.kind.cache(), f(scores), f(stats), f(partials), o.stream, @ptrFromInt(pos) });
    }

    /// x (rows, k) of x's kind times fp32 weights (n, k), out (rows, n) in x's kind: an unquantized draft projection.
    pub fn denseRows(o: Ops, x: Tensor, w: u64, out: Tensor, rows: usize, n: usize, k: usize) Error!void {
        try o.lib.call("tf_dense_rows", .{ p(x.ptr), @backingInt(x.kind), f(w), p(out.ptr), int(rows), int(n), int(k), o.stream });
    }

    /// torch.argmax of each of `rows` logits rows (width `n`, fp16 or bf16) into device i32 `out`.
    pub fn argmaxRows(o: Ops, logits: Tensor, rows: usize, n: usize, out: u64) Error!void {
        if (logits.kind == .f32) return error.BadShape;
        try o.lib.check(o.lib.api.tf_argmax_rows(p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), i(out), o.stream), "argmax_rows");
    }

    /// softmax of each of `rows` logits rows at the token the device i32 `ids` names (0: the row's largest), into f32 `out`.
    pub fn tokenProb(o: Ops, logits: Tensor, rows: usize, n: usize, ids: u64, out: u64) Error!void {
        if (logits.kind == .f32) return error.BadShape;
        try o.lib.check(o.lib.api.tf_token_prob(p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), if (ids == 0) null else @ptrFromInt(ids), f(out), o.stream), "token_prob");
    }

    /// Each row's `ks[r]` largest by (value desc, id asc): ids (i32) and the values' 16-bit patterns, `stride` apart.
    pub fn topkRows(o: Ops, logits: Tensor, rows: usize, n: usize, ks: u64, stride: usize, ids: u64, values: u64) Error!void {
        if (logits.kind == .f32) return error.BadShape;
        try o.lib.check(o.lib.api.tf_topk_rows(p(logits.ptr), int(rows), int(n), i(ks), int(stride), i(ids), p(values), o.stream), "topk_rows");
    }

    pub fn moeRouter(o: Ops, x: Tensor, rows32: u64, logits: u64, r: usize, d: usize, e: usize) Error!void {
        if (o.window) if (o.lib.zig) |*z| if (z.routerWindow(p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream)) |ran| {
            if (ran) return;
        } else |err| return err;
        if (o.prefill) if (o.lib.zig) |*z| if (d % 32 == 0) return z.routerTile(p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream);
        try o.lib.call("tf_moe_router", .{ p(x.ptr), @backingInt(x.kind), f(rows32), f(logits), int(r), int(d), int(e), o.stream });
    }

    /// The pick rule per row; with `plan` (one row) it also writes the plan's items and members.
    pub fn moeSelect(o: Ops, logits: u64, pick: u64, wts: u64, plan: ?struct { items: u64, members: u64 }, capacity: usize, r: usize, experts: usize, top_k: usize) Error!void {
        try o.lib.call("tf_moe_select", .{ f(logits), i(pick), f(wts), if (plan) |pl| i(pl.items) else null, if (plan) |pl| i(pl.members) else null, int(capacity), int(r), int(experts), int(top_k), o.stream });
    }

    pub fn moeRoute(o: Ops, picks: u64, pairs: usize, experts: usize, tile: usize, members: u64, items: u64, capacity: usize) Error!void {
        try o.lib.call("tf_moe_route", .{ @ptrFromInt(picks), int(pairs), int(experts), int(tile), i(members), i(items), int(capacity), o.stream });
    }

    pub fn moeAct(o: Ops, both: u64, out: Tensor, pairs: usize, width: usize, limit: f32) Error!void {
        try o.lib.call("tf_moe_act", .{ f(both), p(out.ptr), @backingInt(out.kind), int(pairs), int(width), limit, o.stream });
    }

    pub fn moeCombine(o: Ops, y: u64, wts: u64, out: Tensor, r: usize, slots: usize, d: usize) Error!void {
        try o.lib.call("tf_moe_combine", .{ f(y), f(wts), p(out.ptr), @backingInt(out.kind), int(r), int(slots), int(d), o.stream });
    }

    /// out = round(x + y): an activation `x` plus an fp32 sum `y`, rounded once to x's kind (a tp residual).
    pub fn addWide(o: Ops, x: Tensor, y: u64, out: Tensor, n: usize) Error!void {
        if (x.kind == .f32 or x.kind != out.kind) return error.BadShape;
        try o.lib.check(o.lib.api.tf_add_wide(p(x.ptr), f(y), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream), "add_wide");
    }

    /// out[i] = remap[pick[i]], or `skip` where it is negative: a tp rank's local expert ids.
    pub fn moeLocalize(o: Ops, pick: u64, remap: u64, out: u64, n: usize, skip: usize) Error!void {
        try o.lib.check(o.lib.api.tf_moe_localize(i(pick), i(remap), i(out), int(n), int(skip), o.stream), "moe_localize");
    }

    /// Zeroes the pair count of every plan item of the `skip` expert.
    pub fn moeForeignItems(o: Ops, items: u64, count: usize, skip: usize) Error!void {
        try o.lib.check(o.lib.api.tf_moe_foreign_items(i(items), int(count), int(skip), o.stream), "moe_foreign_items");
    }

    /// Zeroes the rows of `y` (pairs, d) fp32 whose pick is `skip`.
    pub fn moeZeroForeign(o: Ops, y: u64, picks: u64, skip: usize, pairs: usize, d: usize) Error!void {
        try o.lib.check(o.lib.api.tf_moe_zero_foreign(f(y), i(picks), int(skip), @intCast(pairs), int(d), o.stream), "moe_zero_foreign");
    }
};

test "dtype numberings of the three kernel families" {
    try std.testing.expectEqual(@as(c_int, 1), @backingInt(Kind.f16));
    try std.testing.expectEqual(@as(c_int, 0), Kind.f16.cache());
    try std.testing.expectEqual(@as(c_int, 2), Kind.f16.table());
    try std.testing.expectEqual(@as(c_int, 1), Kind.bf16.table());
}

test "every launch compiles" {
    std.testing.refAllDecls(Ops);
}
