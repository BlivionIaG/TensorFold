//! Qwen3.8-Flash-Next full-attention layer glue on the device (kernels/cuda/exl3_attn.cu).
//! The five EXL3 projections stay in `exl3.zig`'s `linear`. These kernels sit between them:
//! `prep` (norm + RoPE + KV write at a device position), `pool`, `scores`, `select` (the indexer), `decode`.
//! Every position comes from the int32 at `pos0`, so a captured graph replays with a one-int update.
//! The KV table is `exl3_kv.Table`'s layout for one layer: [token][kv head][K 256 bf16 | V 256 bf16].

const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const head_dim: usize = 256;
pub const index_heads: usize = 4;
pub const index_dim: usize = 128;
pub const ratio: usize = 4;
pub const top_blocks: usize = 512;
/// ids stride per row in the select output.
pub const ids_stride: usize = 2056;
/// Bytes of one (token, kv head) slot in the table: K and V, bf16.
pub const slot_bytes: usize = 2 * head_dim * 2;

pub const Dims = struct {
    /// q heads and kv heads this rank holds (tp=1: 24 and 2; tp=2: 12 and 1).
    q_heads: usize = 24,
    kv_heads: usize = 2,
    /// KV table capacity in tokens.
    tokens: usize,
    eps: f32 = 1e-6,
};

pub const Attn = struct {
    mod: Module,
    prep_fn: Function,
    pool_fn: Function,
    scores_fn: Function,
    select_fn: Function,
    decode_fn: Function,

    pub fn load(d: *const Driver) !Attn {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.exl3_attn);
        errdefer m.unload();
        return .{
            .mod = m,
            .prep_fn = try m.function(kernels.exl3_attn_symbols.prep),
            .pool_fn = try m.function(kernels.exl3_attn_symbols.pool),
            .scores_fn = try m.function(kernels.exl3_attn_symbols.scores),
            .select_fn = try m.function(kernels.exl3_attn_symbols.select),
            .decode_fn = try m.function(kernels.exl3_attn_symbols.decode),
        };
    }

    pub fn unload(a: *Attn) void {
        a.mod.unload();
    }

    /// Device pointers one layer's prep needs. Norm scales are fp32 with the centred offset already added.
    pub const PrepArgs = struct {
        qp: u64, // fp16 [rows][q_heads * 512], q then gate per head
        kp: u64, // fp16 [rows][kv_heads * 256]
        vp: u64, // fp16 [rows][kv_heads * 256]
        ip: u64, // fp16 [rows][640]: 4 index q heads then the index key
        q_scale: u64, // f32 [256]
        k_scale: u64, // f32 [256]
        i_scale: u64, // f32 [128]: the q_layernorm (queries)
        inv_freq: u64, // f32 [32]
        pos0: u64, // int32 device pointer
        q_out: u64, // bf16 [rows][q_heads][256]
        kv: u64, // bf16 table for this layer
        iq_out: u64, // bf16 [rows][4][128]
        ik: u64, // bf16 [tokens][128] raw index keys
    };

    pub fn prep(g: *const Attn, s: Stream, dims: Dims, rows: usize, p: PrepArgs) !void {
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(p.qp);
        a.add(p.kp);
        a.add(p.vp);
        a.add(p.ip);
        a.add(p.q_scale);
        a.add(p.k_scale);
        a.add(p.i_scale);
        a.add(p.inv_freq);
        a.add(p.pos0);
        a.add(p.q_out);
        a.add(p.kv);
        a.add(p.iq_out);
        a.add(p.ik);
        a.add(try i32_(rows));
        a.add(try i32_(dims.tokens));
        a.add(try i32_(dims.q_heads));
        a.add(try i32_(dims.kv_heads));
        a.add(dims.eps);
        try launch_.launch(g.prep_fn, .{
            .grid = .{ .x = @intCast(rows), .y = @intCast(dims.q_heads + 2 * dims.kv_heads + index_heads + 1), .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// Pool every 4-key block that rows [pos0, pos0 + rows) complete. `k_scale` is the indexer k_layernorm scale.
    pub fn pool(g: *const Attn, s: Stream, dims: Dims, rows: usize, ik: u64, pooled: u64, k_scale: u64, inv_freq: u64, pos0: u64) !void {
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(ik);
        a.add(pooled);
        a.add(k_scale);
        a.add(inv_freq);
        a.add(pos0);
        a.add(try i32_(rows));
        a.add(try i32_(dims.tokens));
        a.add(dims.eps);
        try launch_.launch(g.pool_fn, .{
            .grid = .{ .x = @intCast(rows / ratio + 2), .y = 1, .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// Block scores for rows past `top_blocks` complete blocks. `nb` is the scores row stride (>= blocks_bound).
    /// `blocks_bound` bounds the grid: ceil(context / 4) for the longest context this launch can see (a graph uses its capacity).
    pub fn scores(g: *const Attn, s: Stream, rows: usize, iq: u64, pooled: u64, out: u64, pos0: u64, nb: usize, blocks_bound: usize) !void {
        if (rows == 0 or blocks_bound <= top_blocks) return;
        var a: launch_.Args = .{};
        a.add(iq);
        a.add(pooled);
        a.add(out);
        a.add(pos0);
        a.add(try i32_(nb));
        try launch_.launch(g.scores_fn, .{
            .grid = .{ .x = @intCast(rows), .y = @intCast((blocks_bound + 127) / 128), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
        }, s, &a);
    }

    pub fn select(g: *const Attn, s: Stream, rows: usize, scores_: u64, ids: u64, nk: u64, pos0: u64, nb: usize) !void {
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(scores_);
        a.add(ids);
        a.add(nk);
        a.add(pos0);
        a.add(try i32_(nb));
        try launch_.launch(g.select_fn, .{
            .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 },
            .block = .{ .x = 1024, .y = 1, .z = 1 },
        }, s, &a);
    }

    /// out fp16 [rows][q_heads * 256] = bf16(softmax(q K / 16) V) * sigmoid(gate), the o_proj input.
    pub fn decode(g: *const Attn, s: Stream, dims: Dims, rows: usize, q: u64, qp: u64, kv: u64, ids: u64, out: u64, pos0: u64) !void {
        if (rows == 0) return;
        if (dims.kv_heads == 0 or dims.q_heads % dims.kv_heads != 0) return error.Exl3Shape;
        var a: launch_.Args = .{};
        a.add(q);
        a.add(qp);
        a.add(kv);
        a.add(ids);
        a.add(out);
        a.add(pos0);
        a.add(try i32_(dims.q_heads));
        a.add(try i32_(dims.kv_heads));
        try launch_.launch(g.decode_fn, .{
            .grid = .{ .x = @intCast(rows), .y = @intCast(dims.q_heads), .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &a);
    }
};

fn i32_(v: usize) !i32 {
    return std.math.cast(i32, v) orelse error.Exl3Shape;
}
