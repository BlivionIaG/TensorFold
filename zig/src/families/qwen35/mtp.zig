//! The checkpoint's MTP head drafting a chain (mtp.py): each step reads the last final row and the token after it,
//! runs the head's gated attention over the chain's own cache and its MLP, and draws the next draft (greedy chains
//! on the device without a host sync, sampled ones over downloaded candidates).
//! Drafts only choose which rows a window verifies; the verify draws every token, so a draft's bits are free.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const view = @import("view.zig");
const weights = @import("weights.zig");
const bridge = @import("bridge.zig");
const fwd = @import("forward.zig");
const draw = @import("draw.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;

/// Most drafts a chain (the Python engine's depth) and the confidence below which it stops.
pub const max_depth = 3;
pub const confidence = 0.3;

/// Byte offset of the drafts in the scalars (the token, then each slot, come first).
const drafts_at = 32;

pub const Head = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Driver,
    w: *const weights.MtpHead,
    heads: usize,
    kv_heads: usize,
    mlp: ?view.Mlp,
    logits_head: view.Affine,
    k: hip.DeviceBuffer,
    v: hip.DeviceBuffer,
    arena: hip.Arena,
    scalars: hip.HostBuffer, // the first token, the chain's slots, then the drafts (at `drafts_at`), pinned
    scalars_dev: hip.DeviceBuffer,
    last: u64 = 0, // the last step's final row, the next step's hidden input

    /// The head of `m`, or null when the checkpoint has none.
    pub fn init(gpa: std.mem.Allocator, d: *const hip.Driver, m: *const weights.Model, model: *const view.Model) !?*Head {
        const w = if (m.mtp) |*x| x else return null;
        const s = m.spec;
        const qa = try affineOf(w.q);
        const ka = try affineOf(w.k);
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.* = .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .heads = qa.n / s.head_dim / @as(usize, if (w.gated) 2 else 1),
            .kv_heads = ka.n / s.head_dim,
            .mlp = if (w.mlp) |x| try bridge.mlpView(x) else null,
            .logits_head = if (w.head) |hp| try affineOf(hp) else model.head,
            .k = undefined,
            .v = undefined,
            .arena = undefined,
            .scalars = undefined,
            .scalars_dev = undefined,
        };
        const cache = h.kv_heads * (max_depth + 1) * s.head_dim * model.act.size();
        h.k = try hip.DeviceBuffer.alloc(d, cache);
        errdefer h.k.free();
        h.v = try hip.DeviceBuffer.alloc(d, cache);
        errdefer h.v.free();
        h.arena = try hip.Arena.init(d, 256 << 20);
        errdefer h.arena.deinit();
        h.scalars = try hip.HostBuffer.alloc(d, 64);
        errdefer h.scalars.free();
        h.scalars_dev = try hip.DeviceBuffer.alloc(d, 64);
        errdefer h.scalars_dev.free();
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.scalars_dev.free();
        h.scalars.free();
        h.arena.deinit();
        h.v.free();
        h.k.free();
        h.gpa.destroy(h);
    }

    fn affineOf(p: weights.Projection) !view.Affine {
        return switch (p) {
            .affine => |a| bridge.affine(a),
            .dense => error.DenseProjection,
        };
    }

    /// A projection of the head: affine, or the unquantized fp32 weight an MTP fc may keep.
    fn project(o: Ops, p: weights.Projection, x: Tensor, rows: usize) !Tensor {
        switch (p) {
            .affine => |a| return o.affine(x, try bridge.affine(a), rows, false),
            .dense => |dn| {
                const n = dn.weight.dim(0);
                const out = try fwd.take(o, x.kind, rows * n);
                try o.denseRows(x, dn.weight.ptr, out, rows, n, dn.weight.dim(1));
                return out;
            },
        }
    }

    /// Up to `depth` drafts from `hidden` (the last kept final row) and `token` (the one after it), the first at slot
    /// `position + 1`; each drawn with `sampling` at its slot, the chain cut after a draft below `confidence`.
    pub fn chain(h: *Head, lib: *const hip.rocm.Library, stream: hip.Stream, drawer: *draw.Drawer, m: *const view.Model, hidden: Tensor, token: u32, position: usize, depth: usize, sampling: ?lanes.Sampling, out: []u32) !usize {
        const o: Ops = .{ .lib = lib, .stream = stream.handle, .arena = &h.arena };
        h.arena.reset();
        const count = @min(depth, max_depth, out.len);
        const sc = h.scalars.slice(i32);
        sc[0] = @intCast(token);
        for (0..max_depth) |i| sc[1 + i] = @intCast(i);
        try h.scalars_dev.uploadAsync(0, h.scalars.bytes[0 .. 4 * (1 + max_depth)], o.stream);
        const greedy = if (sampling) |s| s.temperature <= 0.0 else true;
        const drafts = h.scalars_dev.ptr + drafts_at;
        var cur_hidden = hidden;
        for (0..count) |n| {
            // a greedy draft feeds the next step from the device; a sampled one comes back through the host
            const ids = if (n == 0) h.scalars_dev.ptr else if (greedy) drafts + 4 * (n - 1) else h.scalars_dev.ptr;
            const logits = try h.step(o, m, cur_hidden, ids, h.scalars_dev.ptr + 4 * (1 + n), position + n, n);
            if (greedy) {
                try o.argmaxRows(logits, 1, h.logits_head.n, drafts + 4 * n);
            } else {
                try drawer.draw(o, stream, logits, &.{.{ .sampling = sampling, .position = position + n + 1 }}, out[n..][0..1]);
                if (n + 1 < count) {
                    sc[0] = @intCast(out[n]);
                    try h.scalars_dev.uploadAsync(0, h.scalars.bytes[0..4], o.stream);
                }
            }
            cur_hidden = .{ .ptr = h.last, .kind = m.act };
        }
        if (greedy) {
            const got = h.scalars.slice(u32)[drafts_at / 4 ..][0..count];
            try h.scalars_dev.downloadAsync(drafts_at, std.mem.sliceAsBytes(got), o.stream);
            try stream.synchronize();
            @memcpy(out[0..count], got);
        }
        return count;
    }

    /// One head step at rope position `pos`, writing chain slot `slot` (its last visible slot at device `slot_at`), its
    /// token at device `ids`: its logits (one row, activation dtype).
    fn step(h: *Head, o: Ops, m: *const view.Model, hidden: Tensor, ids: u64, slot_at: u64, pos: usize, slot: usize) !Tensor {
        const s = m.spec;
        const w = h.w;
        const eps: f32 = @floatCast(s.eps);
        const hd = s.head_dim;
        const emb = try fwd.take(o, m.act, s.hidden);
        try o.embedRows(m.embed, ids, 1, emb);
        const emb_e = try fwd.take(o, m.act, s.hidden);
        try o.rms(emb, w.fc_e_norm.ptr, emb_e, 1, s.hidden, eps);
        const emb_h = try fwd.take(o, m.act, s.hidden);
        try o.rms(hidden, w.fc_h_norm.ptr, emb_h, 1, s.hidden, eps);
        const e_proj = try project(o, w.fc_e, emb_e, 1);
        const h_proj = try project(o, w.fc_h, emb_h, 1);
        const x = try fwd.take(o, m.act, s.hidden);
        try o.add(e_proj, h_proj, x, s.hidden);
        var normed = x;
        if (w.input_norm) |n| {
            normed = try fwd.take(o, m.act, s.hidden);
            try o.rms(x, n.ptr, normed, 1, s.hidden, eps);
        }
        // the head's gated attention over the chain's cache (slots 0 .. slot), rope at the chain's position
        const qg = try project(o, w.q, normed, 1);
        const keys = try project(o, w.k, normed, 1);
        const values = try project(o, w.v, normed, 1);
        const qc = try fwd.take(o, m.act, h.heads * hd);
        try o.copyCols(qg, if (w.gated) 2 * hd else hd, 0, qc.ptr, h.heads, hd);
        const qn = try fwd.take(o, m.act, h.heads * hd);
        try o.rms(qc, w.q_norm.ptr, qn, h.heads, hd, eps);
        const kn = try fwd.take(o, m.act, h.kv_heads * hd);
        try o.rms(keys, w.k_norm.ptr, kn, h.kv_heads, hd, eps);
        const q32 = try ropeOne(o, m, qn, h.heads, pos);
        const k32 = try ropeOne(o, m, kn, h.kv_heads, pos);
        const kr = try fwd.take(o, m.act, h.kv_heads * hd);
        try o.cast(.{ .ptr = k32, .kind = .f32 }, kr, h.kv_heads * hd);
        const c: Ops.Cache = .{ .k = h.k.ptr, .v = h.v.ptr, .kind = m.act, .kv_heads = h.kv_heads, .total = max_depth + 1, .d = hd };
        try o.kvWrite(kr, c.k, 1, h.kv_heads, hd, c.total, slot);
        try o.kvWrite(values, c.v, 1, h.kv_heads, hd, c.total, slot);
        const att = try o.arena.of(f32, h.heads * hd);
        try o.causalAt(q32, c, att, 1, h.heads, fwd.scaleOf(hd), slot_at);
        const gated = try fwd.take(o, m.act, h.heads * hd);
        if (w.gated) {
            try o.attnGate(att, qg, gated, 1, h.heads, hd, true);
        } else try o.cast(.{ .ptr = att, .kind = .f32 }, gated, h.heads * hd);
        const attn_out = try project(o, w.o, gated, 1);
        try o.add(x, attn_out, x, s.hidden);
        if (w.post_norm) |pn| if (h.mlp) |mlp| {
            const xn = try fwd.take(o, m.act, s.hidden);
            try o.rms(x, pn.ptr, xn, 1, s.hidden, eps);
            const y = try fwd.mlpRows(o, m, mlp, xn, 1, false);
            try o.add(x, y, x, s.hidden);
        };
        const residual = try fwd.take(o, m.act, s.hidden);
        try o.rms(x, w.final_norm.ptr, residual, 1, s.hidden, eps);
        h.last = residual.ptr;
        return o.affine(residual, h.logits_head, 1, false);
    }
};

/// The decode RoPE of one row's heads at `pos`: widened, rotated, rounded to the activation dtype, widened again.
fn ropeOne(o: Ops, m: *const view.Model, x: Tensor, heads: usize, pos: usize) !u64 {
    const s = m.spec;
    const n = heads * s.head_dim;
    const wide = try o.arena.of(f32, n);
    try o.cast(x, .{ .ptr = wide, .kind = .f32 }, n);
    const turned = try o.arena.of(f32, n);
    try o.ropeDecode(wide, turned, heads, s.head_dim, s.rotary_dim, pos, @floatCast(s.rope_theta), null, 0);
    const narrow = try fwd.take(o, m.act, n);
    try o.cast(.{ .ptr = turned, .kind = .f32 }, narrow, n);
    const back = try o.arena.of(f32, n);
    try o.cast(narrow, .{ .ptr = back, .kind = .f32 }, n);
    return back;
}
