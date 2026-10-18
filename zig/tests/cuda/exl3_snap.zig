//! Snapshot / restore of ALL per-sequence state of the assembled EXL3 forward (`exl3_fwd.Engine`), kept in this file so the
//! engine itself stays untouched (every field read here is a public field of `Engine`).
//!
//! What a sequence of `pos` tokens owns:
//!   GDN, 36 layers:  conv tail [3][10240] fp16 and recurrent state [48][128][128] fp32, per layer, whole (cannot be rewound)
//!   attention, 12:   KV [pos][2 kv heads][K256|V256] bf16, raw index keys [pos][128] bf16, pooled blocks [pos/4 + 2][128] bf16
//!                    (prefix only; the pooled kernel recomputes a partial block from the raw keys)
//!   PLE:             the 2-token id history and the 9-row conv tail (host f64 in this engine)
//!   engine:          `pos`
//! Scratch (scores, ids, activations) is rewritten by every call and is not part of a sequence.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("exl3_fwd.zig");
const fple = @import("exl3_fwd_ple.zig");

pub const Snapshot = struct {
    pos: usize,
    layers: usize,
    conv: []u8,
    rec: []u8,
    kv: []u8,
    ik: []u8,
    pooled: []u8,
    hist: ?[fple.context]i64,
    tail: ?[]f64,

    pub fn deinit(s: *Snapshot, gpa: std.mem.Allocator) void {
        gpa.free(s.conv);
        gpa.free(s.rec);
        gpa.free(s.kv);
        gpa.free(s.ik);
        gpa.free(s.pooled);
        if (s.tail) |t| gpa.free(t);
    }

    pub fn bytes(s: *const Snapshot) u64 {
        const tail: usize = if (s.tail) |t| t.len * 8 else 0;
        return s.conv.len + s.rec.len + s.kv.len + s.ik.len + s.pooled.len + tail + @sizeOf(usize);
    }
};

const slot = cuda.exl3_attn.slot_bytes;
const kv_token = 2 * slot; // both kv heads
const ik_token = 128 * 2;
const block = 128 * 2;

fn counts(e: *const fwd.Engine) struct { gdn: usize, attn: usize } {
    var g: usize = 0;
    var a: usize = 0;
    for (e.layers) |l| {
        if (l.gdn != null) g += 1;
        if (l.attn != null) a += 1;
    }
    return .{ .gdn = g, .attn = a };
}

fn pooledBlocks(e: *const fwd.Engine, pos: usize) usize {
    return @min(pos / 4 + 2, e.opt.cap / 4 + 2);
}

/// Copies the sequence state out of the device. The engine must be idle (`forward` synchronizes before returning).
pub fn capture(e: *fwd.Engine) !Snapshot {
    const gpa = e.gpu.gpa;
    try e.stream.synchronize();
    const n = counts(e);
    const pos = e.pos;
    const cap = e.opt.cap;
    const nb = cap / 4;
    var s: Snapshot = .{
        .pos = pos,
        .layers = e.layers.len,
        .conv = try gpa.alloc(u8, n.gdn * cuda.exl3_gdn.conv_bytes),
        .rec = &.{},
        .kv = &.{},
        .ik = &.{},
        .pooled = &.{},
        .hist = null,
        .tail = null,
    };
    errdefer gpa.free(s.conv);
    s.rec = try gpa.alloc(u8, n.gdn * cuda.exl3_gdn.rec_bytes);
    errdefer gpa.free(s.rec);
    s.kv = try gpa.alloc(u8, n.attn * pos * kv_token);
    errdefer gpa.free(s.kv);
    s.ik = try gpa.alloc(u8, n.attn * pos * ik_token);
    errdefer gpa.free(s.ik);
    s.pooled = try gpa.alloc(u8, n.attn * pooledBlocks(e, pos) * block);
    errdefer gpa.free(s.pooled);

    var gi: usize = 0;
    var ai: usize = 0;
    for (e.layers) |*lay| {
        if (lay.gdn != null) {
            const idx = fwd.Layer.gdnIndex(lay.index);
            const cb = cuda.exl3_gdn.conv_bytes;
            const rb = cuda.exl3_gdn.rec_bytes;
            try e.table.conv.download((idx) * cb, s.conv[gi * cb ..][0..cb]);
            try e.table.rec.download(idx * rb, s.rec[gi * rb ..][0..rb]);
            gi += 1;
        } else {
            const kvi = fwd.Layer.kvIndex(lay.index);
            const pb = pooledBlocks(e, pos);
            try e.kv_all.download(kvi * cap * kv_token, s.kv[ai * pos * kv_token ..][0 .. pos * kv_token]);
            try e.ik_all.download(kvi * cap * ik_token, s.ik[ai * pos * ik_token ..][0 .. pos * ik_token]);
            try e.pooled_all.download(kvi * (nb + 2) * block, s.pooled[ai * pb * block ..][0 .. pb * block]);
            ai += 1;
        }
    }
    if (e.ple) |p| {
        s.hist = p.history;
        s.tail = try gpa.dupe(f64, p.tail);
    }
    return s;
}

/// Puts a snapshot back into an engine with the same layers: resets it first (new sequence), then writes the state.
pub fn restore(e: *fwd.Engine, s: *const Snapshot) !void {
    const n = counts(e);
    if (s.layers != e.layers.len or s.pos > e.opt.cap) return error.SnapshotMismatch;
    if ((s.tail != null) != (e.ple != null)) return error.SnapshotMismatch;
    try e.reset();
    const cap = e.opt.cap;
    const nb = cap / 4;
    const pos = s.pos;
    var gi: usize = 0;
    var ai: usize = 0;
    for (e.layers) |*lay| {
        if (lay.gdn != null) {
            const idx = fwd.Layer.gdnIndex(lay.index);
            const cb = cuda.exl3_gdn.conv_bytes;
            const rb = cuda.exl3_gdn.rec_bytes;
            try e.table.conv.upload(idx * cb, s.conv[gi * cb ..][0..cb]);
            try e.table.rec.upload(idx * rb, s.rec[gi * rb ..][0..rb]);
            gi += 1;
        } else {
            const kvi = fwd.Layer.kvIndex(lay.index);
            const pb = pooledBlocks(e, pos);
            try e.kv_all.upload(kvi * cap * kv_token, s.kv[ai * pos * kv_token ..][0 .. pos * kv_token]);
            try e.ik_all.upload(kvi * cap * ik_token, s.ik[ai * pos * ik_token ..][0 .. pos * ik_token]);
            try e.pooled_all.upload(kvi * (nb + 2) * block, s.pooled[ai * pb * block ..][0 .. pb * block]);
            ai += 1;
        }
    }
    _ = n;
    e.pos = pos;
    if (e.ple) |*p| {
        p.history = s.hist.?;
        @memcpy(p.tail, s.tail.?);
    }
}
