//! One command buffer a call: segment-local conv state and causal attention, every row's arithmetic its own.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const Buffer = mtl.Buffer;
pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };
/// Rows from which a MoE layer sorts its picks by expert (each expert's weights read once for its rows).
const grouped_rows = 4;

/// 4-bit experts take the simdgroup tiles, or at few rows their checked scalar twin: a row's bits never depend on its window.
fn tiled(x: row.Experts) bool {
    return x.one.bits == 4 and x.one.group == 64 and x.one.sum == .f32;
}

/// Whether the twin equals the tiles for every expert shape of this layer (else the tiles at every width).
fn self_twin(enc: Encoder, x: wts.Moe) bool {
    const k = &enc.m.kernels;
    return (k.twin(x.gate.one) orelse false) and (k.twin(x.up.one) orelse false) and (k.twin(x.down.one) orelse false);
}

const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32 };
const AttnArgs = extern struct { len: u32, cap: u32, scale: f32 };

const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    s: *st.Scratch,

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
    fn weights(self: Encoder, w: row.Weights) void {
        self.e.setBuffer(w.w, w.w_off, 1);
        self.e.setBuffer(w.scales, w.s_off, 2);
        self.e.setBuffer(w.biases, w.b_off, 3);
    }
    /// y[rows] = x[rows] W^T through the core's row kernel.
    fn projection(self: Encoder, w: row.Weights, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        if (self.m.kernels.twin(w)) |scalar| {
            const m = row.simd.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k };
            row.simd.encode(self.e, row.simd.call(&self.m.kernels.simd, w.n, w.k, rows, scalar) catch unreachable, m, x, x_off, y, y_off);
            return;
        }
        const call = row.call(&self.m.kernels.rows, w, rows, false) catch unreachable;
        self.e.setPipeline(call.pipeline);
        self.e.setBuffer(x, x_off, 0);
        self.weights(w);
        self.e.setBytes(std.mem.asBytes(&call.dims), 4);
        self.e.setBuffer(y, y_off, 5);
        self.threads(.{ call.groups * call.threads, 1, 1 }, .{ call.threads, 1, 1 });
    }
    /// y[slot] = x[slot / repeat] W[ids[slot]]^T.
    fn experts(self: Encoder, x: row.Experts, in: Buffer, ids: Buffer, y: Buffer, slots: usize) void {
        if (tiled(x)) {
            const w = x.one;
            const m = row.simd.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k };
            row.simd.encodeIndexed(self.e, row.simd.callIndexed(&self.m.kernels.simd, w.n, w.k, slots, x.repeat) catch unreachable, m, in, y, ids);
            return;
        }
        const call = row.callIndexed(&self.m.kernels.rows, x, slots, false) catch unreachable;
        self.e.setPipeline(call.pipeline);
        self.e.setBuffer(in, 0, 0);
        self.weights(x.one);
        self.e.setBytes(std.mem.asBytes(&call.dims), 4);
        self.e.setBuffer(y, 0, 5);
        self.e.setBuffer(ids, 0, 6);
        self.threads(.{ call.groups * call.threads, slots, 1 }, .{ call.threads, 1, 1 });
    }
    /// y[sorted rows] = x[sorted rows] W[e]^T, expert e's rows from starts[e].
    fn grouped(self: Encoder, x: row.Experts, in: Buffer, starts: Buffer, y: Buffer, rows: usize) void {
        if (tiled(x)) {
            const w = x.one;
            const m = row.simd.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k };
            row.simd.encodeGrouped(self.e, row.simd.callGrouped(&self.m.kernels.simd, w.n, w.k, rows, x.experts) catch unreachable, m, in, y, starts, self.s.work);
            return;
        }
        const call = row.callGrouped(&self.m.kernels.rows, x, rows) catch unreachable;
        self.e.setPipeline(call.pipeline);
        self.e.setBuffer(in, 0, 0);
        self.weights(x.one);
        self.e.setBytes(std.mem.asBytes(&call.dims), 4);
        self.e.setBuffer(y, 0, 5);
        self.e.setBuffer(starts, 0, 6);
        self.groups(.{ call.groups, c.experts, 1 }, .{ call.threads, 1, 1 });
    }
    /// h += r unless first, then x = norm(h).
    fn norm(self: Encoder, s: *st.Scratch, weight: wts.Tensor, first: bool, rows: usize) void {
        self.pipe("lfm2_norm");
        self.e.setBuffer(s.h, 0, 0);
        self.e.setBuffer(s.r, 0, 1);
        self.tensor(weight, 2);
        self.e.setValue(c.eps, 3);
        self.e.setValue(@as(u32, @intFromBool(!first)), 4);
        self.e.setBuffer(s.x, 0, 5);
        self.groups(.{ rows, 1, 1 }, .{ c.hidden / 4, 1, 1 });
    }
    fn swiglu(self: Encoder, s: *st.Scratch, count: usize) void {
        self.pipe("lfm2_swiglu");
        self.e.setBuffer(s.gate, 0, 0);
        self.e.setBuffer(s.up, 0, 1);
        self.e.setBuffer(s.act, 0, 2);
        self.e.setValue(@as(u32, @intCast(count)), 3);
        self.threads(.{ count, 1, 1 }, .{ 256, 1, 1 });
    }
};

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > st.batch_rows)) return error.LfmRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.LfmContextFull;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidLfmSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidLfmToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = Encoder{ .e = cb.compute(.serial), .m = m, .s = s };
    const e = enc.e;
    const embedding = m.weights.embedding;
    e.setPipeline(m.kernels.embed(embedding.bits));
    e.setBuffer(s.ids, 0, 0);
    enc.weights(embedding);
    e.setBuffer(s.h, 0, 4);
    const units = c.hidden / @as(usize, if (embedding.bits == 6) 4 else 8 / embedding.bits);
    enc.threads(.{ units, total, 1 }, .{ 256, 1, 1 });
    for (m.weights.blocks, 0..) |block, i| {
        enc.norm(s, block.operator_norm, i == 0, total);
        switch (block.mixer) {
            .conv => |cv| conv(enc, s, cv, i, segments, total),
            .attention => |a| attention(enc, s, a, &m.config, segments, i, total),
        }
        enc.norm(s, block.ffn_norm, false, total);
        switch (block.ffn) {
            .dense => |d| {
                enc.projection(d.gate, s.x, 0, s.gate, 0, total);
                enc.projection(d.up, s.x, 0, s.up, 0, total);
                enc.swiglu(s, total * c.intermediate);
                enc.projection(d.down, s.act, 0, s.r, 0, total);
            },
            .moe => |x| moe(enc, s, x, m.config.scale, total),
        }
    }
    enc.norm(s, m.weights.norm, false, total);
    if (head == .all) enc.projection(embedding, s.x, 0, s.logits, 0, total) else {
        enc.projection(embedding, s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("LFM2 forward: {s}\n", .{msg});
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

fn conv(enc: Encoder, s: *st.Scratch, cv: wts.Conv, layer: usize, segments: []const Segment, total: usize) void {
    const e = enc.e;
    enc.projection(cv.in, s.x, 0, s.bcx, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        enc.pipe("lfm2_conv");
        e.setBuffer(s.bcx, base * 3 * c.hidden * 2, 0);
        e.setBuffer(seg.cache.blocks[layer].conv, 0, 1);
        enc.tensor(cv.weight, 2);
        e.setBuffer(s.y, base * c.hidden * 2, 3);
        e.setBuffer(s.snapshots[layer].?, base * st.conv_bytes, 4);
        enc.threads(.{ c.hidden, seg.rows, 1 }, .{ 256, 1, 1 });
        base += seg.rows;
    }
    enc.projection(cv.out, s.y, 0, s.r, 0, total);
}

fn attention(enc: Encoder, s: *st.Scratch, a: wts.Attention, config: *const c.Config, segments: []const Segment, layer: usize, total: usize) void {
    const e = enc.e;
    enc.projection(a.q, s.x, 0, s.q, 0, total);
    enc.projection(a.k, s.x, 0, s.k, 0, total);
    enc.projection(a.v, s.x, 0, s.v, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].attention;
        const len: u32 = @intCast(seg.cache.len);
        const cap: u32 = @intCast(seg.cache.capacity);
        enc.pipe("lfm2_qk");
        e.setBuffer(s.q, base * c.hidden * 2, 0);
        e.setBuffer(s.k, base * c.kv_heads * c.head_dim * 2, 1);
        e.setBuffer(s.v, base * c.kv_heads * c.head_dim * 2, 2);
        enc.tensor(a.q_norm, 3);
        enc.tensor(a.k_norm, 4);
        e.setBytes(std.mem.asBytes(&QkArgs{ .len = len, .cap = cap, .eps = c.eps, .log2_theta = std.math.log2(config.theta) }), 5);
        e.setBuffer(s.queries, base * c.hidden * 2, 6);
        e.setBuffer(cache.keys, 0, 7);
        e.setBuffer(cache.values, 0, 8);
        enc.groups(.{ c.query_heads + c.kv_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        const args = AttnArgs{ .len = len, .cap = cap, .scale = 1.0 / @sqrt(@as(f32, c.head_dim)) };
        enc.pipe("lfm2_attn_partial");
        e.setBuffer(s.queries, base * c.hidden * 2, 0);
        e.setBuffer(cache.keys, 0, 1);
        e.setBuffer(cache.values, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(s.pm, 0, 4);
        e.setBuffer(s.pl, 0, 5);
        e.setBuffer(s.po, 0, 6);
        enc.groups(.{ st.chunks, c.query_heads, seg.rows }, .{ 32, 1, 1 });
        enc.pipe("lfm2_attn_merge");
        e.setBuffer(s.pm, 0, 0);
        e.setBuffer(s.pl, 0, 1);
        e.setBuffer(s.po, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(s.y, base * c.hidden * 2, 4);
        enc.groups(.{ c.query_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        base += seg.rows;
    }
    enc.projection(a.out, s.y, 0, s.r, 0, total);
}

fn moe(enc: Encoder, s: *st.Scratch, x: wts.Moe, scale: f32, total: usize) void {
    const e = enc.e;
    const slots = total * c.topk;
    enc.pipe("lfm2_route_logits");
    e.setBuffer(s.x, 0, 0);
    enc.tensor(x.router, 1);
    e.setBuffer(s.route, 0, 2);
    enc.groups(.{ c.experts / 8, total, 1 }, .{ 256, 1, 1 });
    enc.pipe("lfm2_route_select");
    e.setBuffer(s.route, 0, 0);
    enc.tensor(x.bias, 1);
    e.setValue(scale, 2);
    e.setBuffer(s.picks, 0, 3);
    e.setBuffer(s.weights, 0, 4);
    enc.groups(.{ total, 1, 1 }, .{ 32, 1, 1 });
    if (total < grouped_rows and (!tiled(x.gate) or self_twin(enc, x))) {
        enc.experts(x.gate, s.x, s.picks, s.gate, slots);
        enc.experts(x.up, s.x, s.picks, s.up, slots);
        enc.swiglu(s, slots * c.moe_intermediate);
        enc.experts(x.down, s.act, s.picks, s.experts, slots);
        enc.pipe("lfm2_combine");
        e.setBuffer(s.experts, 0, 0);
        e.setBuffer(s.weights, 0, 1);
        e.setBuffer(s.r, 0, 2);
        enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
        return;
    }
    enc.pipe("lfm2_sort");
    e.setBuffer(s.picks, 0, 0);
    e.setValue(@as(u32, @intCast(slots)), 1);
    e.setBuffer(s.starts, 0, 2);
    e.setBuffer(s.order, 0, 3);
    e.setBuffer(s.pos, 0, 4);
    enc.groups(.{ 1, 1, 1 }, .{ c.experts, 1, 1 });
    if (tiled(x.gate)) row.simd.encodeWork(e, &enc.m.kernels.simd, s.starts, c.experts, slots, s.work);
    enc.pipe("lfm2_gather");
    e.setBuffer(s.x, 0, 0);
    e.setBuffer(s.order, 0, 1);
    e.setBuffer(s.sorted, 0, 2);
    enc.threads(.{ c.hidden, slots, 1 }, .{ 256, 1, 1 });
    enc.grouped(x.gate, s.sorted, s.starts, s.gate, slots);
    enc.grouped(x.up, s.sorted, s.starts, s.up, slots);
    enc.swiglu(s, slots * c.moe_intermediate);
    enc.grouped(x.down, s.act, s.starts, s.experts, slots);
    enc.pipe("lfm2_combine_sorted");
    e.setBuffer(s.experts, 0, 0);
    e.setBuffer(s.weights, 0, 1);
    e.setBuffer(s.r, 0, 2);
    e.setBuffer(s.pos, 0, 3);
    enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
}
