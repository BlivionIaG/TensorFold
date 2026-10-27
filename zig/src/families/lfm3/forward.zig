//! Batched LFM3 rows use the same projections and attention reduction for prompt, verify and plain decode.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("metal_weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const Buffer = mtl.Buffer;

pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };
pub var gpu_seconds: f64 = 0;

const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 };
const AttnArgs = extern struct { len: u32, cap: u32, scale: f32, window: u32 };

const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    record: bool,

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

    fn projection(self: Encoder, w: row.Weights, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) !void {
        const call = try row.call(&self.m.kernels.rows, w, rows, false);
        self.e.setPipeline(call.pipeline);
        self.e.setBuffer(x, x_off, 0);
        self.e.setBuffer(w.w, w.w_off, 1);
        self.e.setBuffer(w.scales, w.s_off, 2);
        self.e.setBuffer(w.biases, w.b_off, 3);
        self.e.setBytes(std.mem.asBytes(&call.dims), 4);
        self.e.setBuffer(y, y_off, 5);
        self.groups(.{ call.groups, 1, 1 }, .{ call.threads, 1, 1 });
    }

    fn norm(self: Encoder, input: Buffer, weight: wts.Tensor, out: Buffer, rows: usize) void {
        self.pipe("lfm3_norm");
        self.e.setBuffer(input, 0, 0);
        self.e.setBuffer(input, 0, 1);
        self.tensor(weight, 2);
        self.e.setValue(c.eps, 3);
        self.e.setValue(@as(u32, 0), 4);
        self.e.setBuffer(out, 0, 5);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }

    fn post(self: Encoder, input: Buffer, weight: wts.Tensor, residual: Buffer, rows: usize) void {
        self.pipe("lfm3_post_add");
        self.e.setBuffer(input, 0, 0);
        self.tensor(weight, 1);
        self.e.setBuffer(residual, 0, 2);
        self.e.setValue(c.eps, 3);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }
};

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    return runWithImages(m, s, segments, tokens, record, head, null);
}

pub fn runWithImages(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head, images: ?@import("image_rows.zig").Images) !void {
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
    const cb = m.queue.commandBuffer();
    const enc = Encoder{ .e = cb.compute(.serial), .m = m, .record = record };
    const e = enc.e;
    var ended = false;
    errdefer if (!ended) e.end();
    const embedding = m.weights.embedding;
    e.setPipeline(m.kernels.embed(embedding.bits));
    e.setBuffer(s.ids, 0, 0);
    e.setBuffer(embedding.w, embedding.w_off, 1);
    e.setBuffer(embedding.scales, embedding.s_off, 2);
    e.setBuffer(embedding.biases, embedding.b_off, 3);
    e.setBuffer(s.h, 0, 4);
    e.setValue(@as(u32, @intCast(embedding.group)), 5);
    enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    if (images) |input| try input.encode(e, s.h, segments[0].cache.len, total);
    for (m.weights.blocks, 0..) |block, i| {
        enc.norm(s.h, block.operator_norm, s.x, total);
        switch (block.mixer) {
            .conv => |cv| try conv(enc, s, cv, i, segments, total),
            .attention => |a| try attention(enc, s, a, segments, i, total),
        }
        enc.post(s.r, block.post_operator_norm, s.h, total);
        enc.norm(s.h, block.ffn_norm, s.x, total);
        const ffn = block.ffn;
        try enc.projection(ffn.gate, s.x, 0, s.gate, 0, total);
        try enc.projection(ffn.up, s.x, 0, s.up, 0, total);
        enc.pipe("lfm3_swiglu");
        e.setBuffer(s.gate, 0, 0);
        e.setBuffer(s.up, 0, 1);
        e.setBuffer(s.act, 0, 2);
        e.setValue(@as(u32, @intCast(total * c.intermediate)), 3);
        enc.threads(.{ total * c.intermediate / 8, 1, 1 }, .{ 256, 1, 1 });
        try enc.projection(ffn.down, s.act, 0, s.r, 0, total);
        enc.post(s.r, block.post_ffn_norm, s.h, total);
    }
    enc.norm(s.h, m.weights.norm, s.x, total);
    if (head == .all) try enc.projection(embedding, s.x, 0, s.logits, 0, total) else {
        try enc.projection(embedding, s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
    e.end();
    ended = true;
    cb.commit();
    cb.wait();
    gpu_seconds += cb.gpuSeconds();
    if (cb.failure()) |msg| {
        std.log.err("LFM3 forward: {s}", .{msg});
        return error.GpuFailed;
    }
    var base: usize = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        @memcpy(seg.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
        seg.cache.commit(s, base, seg.rows, record);
        base += seg.rows;
    }
}

fn conv(enc: Encoder, s: *st.Scratch, cv: wts.Conv, layer: usize, segments: []const Segment, total: usize) !void {
    const e = enc.e;
    try enc.projection(cv.in, s.x, 0, s.bcx, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        enc.pipe("lfm3_conv");
        e.setBuffer(s.bcx, base * 3 * c.hidden * 2, 0);
        e.setBuffer(seg.cache.blocks[layer].conv, 0, 1);
        enc.tensor(cv.weight, 2);
        e.setBuffer(s.y, base * c.hidden * 2, 3);
        e.setBuffer(s.snapshots[layer].?, if (enc.record) base * st.conv_bytes else 0, 4);
        e.setValue(@as(u32, @intCast(if (enc.record) 0 else seg.rows - 1)), 5);
        enc.threads(.{ c.hidden, seg.rows, 1 }, .{ 256, 1, 1 });
        base += seg.rows;
    }
    try enc.projection(cv.out, s.y, 0, s.r, 0, total);
}

fn attention(enc: Encoder, s: *st.Scratch, a: wts.Attention, segments: []const Segment, layer: usize, total: usize) !void {
    const e = enc.e;
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
        enc.pipe("lfm3_qk");
        e.setBuffer(s.q, base * c.hidden * 2, 0);
        e.setBuffer(s.k, base * c.kv_heads * c.head_dim * 2, 1);
        e.setBuffer(s.v, base * c.kv_heads * c.head_dim * 2, 2);
        enc.tensor(a.q_norm, 3);
        enc.tensor(a.k_norm, 4);
        e.setBytes(std.mem.asBytes(&qk), 5);
        e.setBuffer(s.queries, base * c.hidden * 2, 6);
        e.setBuffer(cache.keys, 0, 7);
        e.setBuffer(cache.values, 0, 8);
        enc.groups(.{ c.query_heads + c.kv_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        var at: usize = 0;
        while (at < seg.rows) : (at += st.attn_slice) {
            const rows = @min(seg.rows - at, st.attn_slice);
            const first = seg.cache.len + at;
            const args = AttnArgs{ .len = @intCast(first), .cap = cap, .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .window = if (sliding) c.sliding_window else 0 };
            enc.pipe("lfm3_attn_partial");
            e.setBuffer(s.queries, (base + at) * c.hidden * 2, 0);
            e.setBuffer(cache.keys, 0, 1);
            e.setBuffer(cache.values, 0, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(s.pm, 0, 4);
            e.setBuffer(s.pl, 0, 5);
            e.setBuffer(s.po, 0, 6);
            enc.groups(.{ c.query_heads, rows, (first + rows + st.chunk_keys - 1) / st.chunk_keys }, .{ 128, 1, 1 });
            enc.pipe("lfm3_attn_merge");
            e.setBuffer(s.pm, 0, 0);
            e.setBuffer(s.pl, 0, 1);
            e.setBuffer(s.po, 0, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(s.y, (base + at) * c.hidden * 2, 4);
            enc.groups(.{ c.query_heads, rows, 1 }, .{ 32, 1, 1 });
        }
        base += seg.rows;
    }
    enc.pipe("lfm3_gate");
    e.setBuffer(s.y, 0, 0);
    e.setBuffer(s.head_gate, 0, 1);
    e.setValue(@as(u32, @intCast(total * c.hidden)), 2);
    enc.threads(.{ total * c.hidden, 1, 1 }, .{ 256, 1, 1 });
    try enc.projection(a.out, s.y, 0, s.r, 0, total);
}
