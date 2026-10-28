//! Batched LFM3 rows use the same projections and attention reduction for prompt, verify and plain decode.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("metal_weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const pairs = @import("row_projection.zig");
const Buffer = mtl.Buffer;
pub const counters = @import("../../core/gpu_counters.zig");

pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };
pub var gpu_seconds: f64 = 0;
pub const Trace = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, name: []const u8, value: Buffer, rows: usize, width: usize) anyerror!void,
};

const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 };
const AttnArgs = @import("kernels.zig").AttentionArgs;

const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    record: bool,

    fn begin(self: *Encoder, kind: counters.Class) !void {
        if (self.m.profile) |profile| try profile.beginSpan(&self.e, kind);
    }

    fn end(self: *Encoder) !void {
        if (self.m.profile) |profile| try profile.endSpan(&self.e);
    }

    fn pipe(self: Encoder, comptime key: []const u8) void {
        self.e.setPipeline(self.m.kernels.get(key));
    }

    fn tensor(self: Encoder, t: wts.Tensor, slot: usize) void {
        self.e.setBuffer(t.buffer, t.offset, slot);
    }

    fn threads(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    fn groups(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    fn projection(self: *Encoder, w: row.Weights, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) !void {
        try self.begin(.dense);
        try pairs.encode(self.e, &self.m.kernels.rows, w, x, x_off, y, y_off, rows);
        try self.end();
    }

    fn norm(self: *Encoder, input: Buffer, weight: wts.Tensor, out: Buffer, rows: usize) !void {
        try self.begin(.norm);
        self.pipe("lfm3_norm");
        self.e.setBuffer(input, 0, 0);
        self.e.setBuffer(input, 0, 1);
        self.tensor(weight, 2);
        self.e.setValue(c.eps, 3);
        self.e.setValue(@as(u32, 0), 4);
        self.e.setBuffer(out, 0, 5);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
        try self.end();
    }

    fn post(self: *Encoder, input: Buffer, weight: wts.Tensor, residual: Buffer, rows: usize) !void {
        try self.begin(.norm);
        self.pipe("lfm3_post_add");
        self.e.setBuffer(input, 0, 0);
        self.tensor(weight, 1);
        self.e.setBuffer(residual, 0, 2);
        self.e.setValue(c.eps, 3);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
        try self.end();
    }
};

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    return runWithImages(m, s, segments, tokens, record, head, null);
}

/// Developer projection fixture through the production selector, with no tokenizer or model-state update.
pub fn project(m: *Model, s: *st.Scratch, w: row.Weights, input: Buffer, output: Buffer, rows: usize) !void {
    if (m.profile != null or rows == 0 or rows > s.rows) return error.InvalidLfmProjectionFixture;
    const cb = m.queue.commandBuffer();
    var enc = Encoder{ .e = cb.compute(.serial), .m = m, .record = false };
    var ended = false;
    errdefer if (!ended) enc.e.end();
    try enc.projection(w, input, 0, output, 0, rows);
    enc.e.end();
    ended = true;
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

pub fn runWithImages(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head, images: ?@import("image_rows.zig").Images) !void {
    return execute(m, s, segments, tokens, record, head, images, null);
}

pub fn runTrace(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, head: Head, trace: Trace) !void {
    return execute(m, s, segments, tokens, false, head, null, trace);
}

fn execute(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head, images: ?@import("image_rows.zig").Images, trace: ?Trace) !void {
    if (trace != null and m.profile != null) return error.ConflictingLfmDiagnostics;
    if (images != null and (record or segments.len != 1)) return error.InvalidLfmImageSegments;
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > st.batch_rows)) return error.LfmRowsExceeded;
    var total: usize = 0;
    for (segments, 0..) |seg, i| {
        if (seg.rows == 0 or seg.rows > seg.cache.capacity -| seg.cache.len) return error.LfmContextFull;
        for (segments[0..i]) |prior| if (prior.cache == seg.cache) return error.InvalidLfmSegments;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidLfmSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidLfmToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var cb = m.queue.commandBuffer();
    var enc = Encoder{ .e = cb.compute(.serial), .m = m, .record = record };
    var ended = false;
    if (m.profile) |profile| profile.beginCommand(cb) catch |err| {
        enc.e.end();
        return err;
    };
    errdefer if (!ended) {
        if (m.profile) |profile| profile.cancelEncoding(&enc.e) catch {} else enc.e.end();
    };
    try enc.begin(.embedding);
    const embedding = m.weights.embedding;
    enc.e.setPipeline(m.kernels.embed(embedding.bits));
    enc.e.setBuffer(s.ids, 0, 0);
    enc.e.setBuffer(embedding.w, embedding.w_off, 1);
    enc.e.setBuffer(embedding.scales, embedding.s_off, 2);
    enc.e.setBuffer(embedding.biases, embedding.b_off, 3);
    enc.e.setBuffer(s.h, 0, 4);
    enc.e.setValue(@as(u32, @intCast(embedding.group)), 5);
    enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    if (images) |input| try input.encode(enc.e, s.h, segments[0].cache.len, total);
    try enc.end();
    try stage(&enc, trace, "embedding", s.h, total, c.hidden, &cb, &ended);
    for (m.weights.blocks, 0..) |block, i| {
        var name: [64]u8 = undefined;
        try enc.norm(s.h, block.operator_norm, s.x, total);
        try layerStage(&enc, trace, &name, i, "operator_norm", s.x, total, c.hidden, &cb, &ended);
        switch (block.mixer) {
            .conv => |cv| {
                try conv(&enc, s, cv, i, segments, total);
                try layerStage(&enc, trace, &name, i, "conv_in", s.bcx, total, c.hidden * 3, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "conv_gated", s.y, total, c.hidden, &cb, &ended);
            },
            .attention => |a| {
                try attention(&enc, s, a, segments, i, total);
                try layerStage(&enc, trace, &name, i, "attn_q", s.q, total, c.hidden, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "attn_k", s.k, total, c.kv_heads * c.head_dim, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "attn_v", s.v, total, c.kv_heads * c.head_dim, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "attn_gate", s.head_gate, total, c.query_heads, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "attn_queries", s.queries, total, c.hidden, &cb, &ended);
                try layerStage(&enc, trace, &name, i, "attn_gated", s.y, total, c.hidden, &cb, &ended);
            },
        }
        try layerStage(&enc, trace, &name, i, "mixer", s.r, total, c.hidden, &cb, &ended);
        try enc.post(s.r, block.post_operator_norm, s.h, total);
        try layerStage(&enc, trace, &name, i, "post_operator", s.h, total, c.hidden, &cb, &ended);
        try enc.norm(s.h, block.ffn_norm, s.x, total);
        try layerStage(&enc, trace, &name, i, "ffn_norm", s.x, total, c.hidden, &cb, &ended);
        const ffn = block.ffn;
        try enc.projection(ffn.gate, s.x, 0, s.gate, 0, total);
        try layerStage(&enc, trace, &name, i, "gate", s.gate, total, c.intermediate, &cb, &ended);
        try enc.projection(ffn.up, s.x, 0, s.up, 0, total);
        try layerStage(&enc, trace, &name, i, "up", s.up, total, c.intermediate, &cb, &ended);
        try enc.begin(.activation);
        enc.pipe("lfm3_swiglu");
        enc.e.setBuffer(s.gate, 0, 0);
        enc.e.setBuffer(s.up, 0, 1);
        enc.e.setBuffer(s.act, 0, 2);
        enc.e.setValue(@as(u32, @intCast(total * c.intermediate)), 3);
        enc.threads(.{ total * c.intermediate / 8, 1, 1 }, .{ 256, 1, 1 });
        try enc.end();
        try layerStage(&enc, trace, &name, i, "act", s.act, total, c.intermediate, &cb, &ended);
        try enc.projection(ffn.down, s.act, 0, s.r, 0, total);
        try layerStage(&enc, trace, &name, i, "down", s.r, total, c.hidden, &cb, &ended);
        try enc.post(s.r, block.post_ffn_norm, s.h, total);
        try layerStage(&enc, trace, &name, i, "output", s.h, total, c.hidden, &cb, &ended);
    }
    try enc.norm(s.h, m.weights.norm, s.x, total);
    try stage(&enc, trace, "final_norm", s.x, total, c.hidden, &cb, &ended);
    if (head == .all) try enc.projection(embedding, s.x, 0, s.logits, 0, total) else {
        try enc.projection(embedding, s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
    if (m.profile) |profile| try profile.finishEncoding(&enc.e) else enc.e.end();
    ended = true;
    cb.commit();
    cb.wait();
    gpu_seconds += cb.gpuSeconds();
    if (cb.failure()) |msg| {
        std.log.err("LFM3 forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (m.profile) |profile| _ = try profile.readAfterCompletion();
    var base: usize = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        @memcpy(seg.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
        seg.cache.commit(s, base, seg.rows, record);
        base += seg.rows;
    }
}

fn stage(enc: *Encoder, trace: ?Trace, name: []const u8, value: Buffer, rows: usize, width: usize, cb: *mtl.CommandBuffer, ended: *bool) !void {
    const hook = trace orelse return;
    enc.e.end();
    ended.* = true;
    cb.commit();
    cb.wait();
    gpu_seconds += cb.gpuSeconds();
    if (cb.failure() != null) return error.GpuFailed;
    try hook.take(hook.ctx, name, value, rows, width);
    cb.* = enc.m.queue.commandBuffer();
    enc.e = cb.compute(.serial);
    ended.* = false;
}

fn layerStage(enc: *Encoder, trace: ?Trace, name: []u8, index: usize, suffix: []const u8, value: Buffer, rows: usize, width: usize, cb: *mtl.CommandBuffer, ended: *bool) !void {
    if (trace == null) return;
    return stage(enc, trace, try std.fmt.bufPrint(name, "layer_{d:0>2}_{s}", .{ index, suffix }), value, rows, width, cb, ended);
}

fn conv(enc: *Encoder, s: *st.Scratch, cv: wts.Conv, layer: usize, segments: []const Segment, total: usize) !void {
    try enc.projection(cv.in, s.x, 0, s.bcx, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        try enc.begin(.activation);
        enc.pipe("lfm3_conv");
        enc.e.setBuffer(s.bcx, base * 3 * c.hidden * 2, 0);
        enc.e.setBuffer(seg.cache.blocks[layer].conv, 0, 1);
        enc.tensor(cv.weight, 2);
        enc.e.setBuffer(s.y, base * c.hidden * 2, 3);
        enc.e.setBuffer(s.snapshots[layer].?, if (enc.record) base * st.conv_bytes else 0, 4);
        enc.e.setValue(@as(u32, @intCast(if (enc.record) 0 else seg.rows - 1)), 5);
        enc.threads(.{ c.hidden, seg.rows, 1 }, .{ 256, 1, 1 });
        try enc.end();
        base += seg.rows;
    }
    try enc.projection(cv.out, s.y, 0, s.r, 0, total);
}

fn attention(enc: *Encoder, s: *st.Scratch, a: wts.Attention, segments: []const Segment, layer: usize, total: usize) !void {
    try enc.projection(a.q, s.x, 0, s.q, 0, total);
    try enc.projection(a.k, s.x, 0, s.k, 0, total);
    try enc.projection(a.v, s.x, 0, s.v, 0, total);
    try enc.projection(a.gate, s.x, 0, s.head_gate, 0, total);
    const sliding = enc.m.config.layer_types[layer] == .sliding_attention;
    const rotary = if (sliding) enc.m.config.sliding_rotary else enc.m.config.full_rotary;
    var base: usize = 0;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].attention;
        const cap: u32 = @intCast(seg.cache.capacity);
        const qk = QkArgs{
            .len = @intCast(seg.cache.len),
            .cap = cap,
            .eps = c.eps,
            .log2_theta = std.math.log2(rotary.theta),
            .rotary_pairs = @intFromFloat(c.head_dim * rotary.partial / 2),
        };
        try enc.begin(.attention_prepare);
        enc.pipe("lfm3_qk");
        enc.e.setBuffer(s.q, base * c.hidden * 2, 0);
        enc.e.setBuffer(s.k, base * c.kv_heads * c.head_dim * 2, 1);
        enc.e.setBuffer(s.v, base * c.kv_heads * c.head_dim * 2, 2);
        enc.tensor(a.q_norm, 3);
        enc.tensor(a.k_norm, 4);
        enc.e.setBytes(std.mem.asBytes(&qk), 5);
        enc.e.setBuffer(s.queries, base * c.hidden * 2, 6);
        enc.e.setBuffer(cache.keys, 0, 7);
        enc.e.setBuffer(cache.values, 0, 8);
        enc.groups(.{ c.query_heads + c.kv_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        try enc.end();
        var at: usize = 0;
        while (at < seg.rows) : (at += st.attn_slice) {
            const rows = @min(seg.rows - at, st.attn_slice);
            const first = seg.cache.len + at;
            const chunk_base = if (sliding) (first + 1 -| c.sliding_window) / st.chunk_keys else 0;
            const last_chunk = (first + rows + st.chunk_keys - 1) / st.chunk_keys;
            const args = AttnArgs{ .len = @intCast(first), .cap = cap, .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .window = if (sliding) c.sliding_window else 0, .chunk_base = @intCast(chunk_base) };
            try enc.begin(.attention);
            enc.pipe("lfm3_attn_partial");
            enc.e.setBuffer(s.queries, (base + at) * c.hidden * 2, 0);
            enc.e.setBuffer(cache.keys, 0, 1);
            enc.e.setBuffer(cache.values, 0, 2);
            enc.e.setBytes(std.mem.asBytes(&args), 3);
            enc.e.setBuffer(s.pm, 0, 4);
            enc.e.setBuffer(s.pl, 0, 5);
            enc.e.setBuffer(s.po, 0, 6);
            enc.groups(.{ c.query_heads, rows, last_chunk - chunk_base }, .{ 128, 1, 1 });
            try enc.end();
            try enc.begin(.attention);
            enc.pipe("lfm3_attn_merge");
            enc.e.setBuffer(s.pm, 0, 0);
            enc.e.setBuffer(s.pl, 0, 1);
            enc.e.setBuffer(s.po, 0, 2);
            enc.e.setBytes(std.mem.asBytes(&args), 3);
            enc.e.setBuffer(s.y, (base + at) * c.hidden * 2, 4);
            enc.groups(.{ c.query_heads, rows, 1 }, .{ 32, 1, 1 });
            try enc.end();
        }
        base += seg.rows;
    }
    try enc.begin(.activation);
    enc.pipe("lfm3_gate");
    enc.e.setBuffer(s.y, 0, 0);
    enc.e.setBuffer(s.head_gate, 0, 1);
    enc.e.setValue(@as(u32, @intCast(total * c.hidden)), 2);
    enc.threads(.{ total * c.hidden, 1, 1 }, .{ 256, 1, 1 });
    try enc.end();
    try enc.projection(a.out, s.y, 0, s.r, 0, total);
}
