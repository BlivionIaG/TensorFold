//! KV memory of an EXL3 pack's streams (K10): bytes by piece, the scratch format, where kept states live.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const gdn = @import("../../cuda_gdn.zig");

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const index_row: usize = wt.index_dim * 2;

/// One token's K and V in one attention layer for `heads` KV heads, bf16.
pub fn kvRowBytes(heads: usize) usize {
    return 2 * heads * head_dim * 2;
}

/// The levers that change a sequence's device bytes.
pub const Policy = struct {
    /// KV heads a rank keeps (2: replicated as upstream; 1: the attention head split across two ranks).
    heads: usize = kv_heads,
};

/// Bytes a sequence of `cap` tokens holds, by piece.
pub const SeqBytes = struct {
    gdn: u64,
    kv: u64,
    ikc: u64,
    pooled: u64,
    tail: u64,

    pub fn device(s: SeqBytes) u64 {
        return s.gdn + s.kv + s.ikc + s.pooled + s.tail;
    }
};

pub fn pooledBytes(cap: usize) u64 {
    return (cap + wt.index_ratio - 1) / wt.index_ratio * index_row;
}

/// cuda_prompt.zig seqBytes split by piece (the default policy gives upstream's number: a host test checks it).
pub fn seqBytes(cap: usize, p: Policy) SeqBytes {
    const attn: u64 = wt.attn_n;
    return .{
        .gdn = wt.linear_n * ((gdn.taps - 1) * gdn.conv_dim * 2 + wt.state_n * 4),
        .kv = attn * cap * kvRowBytes(p.heads),
        .ikc = attn * cap * index_row,
        .pooled = attn * pooledBytes(cap),
        .tail = wt.ple_tail_n * wt.wide * 2,
    };
}

pub const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// What an engine's new sequences hold (set once at load by `setup`, before any sequence; kept here by engine address).
pub const EngineKv = struct {
    /// KV heads a sequence holds: 1 when the attention heads are split across two ranks (cuda_exl3_heads.zig).
    heads: usize = kv_heads,

    pub fn policy(k: EngineKv) Policy {
        return .{ .heads = k.heads };
    }

    /// Whether new sequences differ from upstream's newSeq.
    pub fn placed(k: EngineKv) bool {
        return k.heads != kv_heads;
    }

    /// A snapshot's K (and V) bytes of one layer a token.
    pub fn rowBytes(k: EngineKv) usize {
        return k.heads * head_dim * 2;
    }
};

/// Widest logits pass the server runs (a decode round of 32 streams, an MTP window): `head_local`'s rows.
pub const head_local_serve_rows: usize = 64;
/// K10 lever 3: the pass's chunk for an EXL3 checkpoint unless TF_FLASHNEXT_CHUNK says.
pub const exl3_chunk: usize = 1024;

/// The server's load hook (cuda_lanes_load.zig, right after the engine loads, before any scratch or sequence).
pub fn setup(e: anytype, exl3: bool) !void {
    const heads = @import("../model/cuda_exl3_heads.zig").kvHeads(e.layers[0..e.n_layers]);
    if (heads != kv_heads) (try entryOf(e)).heads = heads;
    if (heads != kv_heads) std.log.info("flashnext: attention heads split across the ranks: {d} query heads and {d} KV head a rank", .{ wt.q_heads / 2, heads });
    if (!exl3) return;
    if (e.batch_s != null) return error.ScratchAlreadyReserved;
    const decoder_exl3 = e.n_layers != 0 and for (e.layers[0..e.n_layers]) |layer| {
        if (layer.exl3 == null) break false;
    } else true;
    const draft_exl3 = @import("../pack/cuda_exl3_load.zig").hasExperts(&e.store, "language_model.mtp.layers.0.mlp");
    e.scratch_format = if (decoder_exl3 and draft_exl3) .exl3 else .classic;
    if (std.c.getenv("TF_FLASHNEXT_CHUNK") == null) e.chunk = exl3_chunk;
    @import("../../cuda_batch_state.zig").head_local_rows = if (e.head_q.n == wt.vocab) 0 else head_local_serve_rows;
}

/// One log line: a stream's device bytes at the window by piece.
pub fn logStream(e: anytype, window: usize, stream_bytes: usize) !void {
    const b = seqBytes(window, of(e).policy());
    std.log.info("flashnext: a stream at a {d}-token window holds {d:.3} GiB on the device (K/V {d:.3} bf16, indexer keys {d:.3}, pooled {d:.3}, DeltaNet {d:.3}, draft and the rest {d:.3}); chunk {d} rows, scratch {d:.3} GiB", .{
        window,
        @as(f64, @floatFromInt(stream_bytes)) / gib,
        @as(f64, @floatFromInt(b.kv)) / gib,
        @as(f64, @floatFromInt(b.ikc)) / gib,
        @as(f64, @floatFromInt(b.pooled)) / gib,
        @as(f64, @floatFromInt(b.gdn)) / gib,
        @as(f64, @floatFromInt(stream_bytes -| b.device())) / gib,
        e.chunk,
        @as(f64, @floatFromInt(@import("../../cuda_batch_state.zig").Scratch.bytesForFormat(e.chunk, e.scratch_format))) / gib,
    });
}

/// Whether kept prompt states go to pinned host memory (cuda_snap.on_host, set here):
pub fn snapHost(context: usize, exl3: bool) bool {
    var on = exl3 and context > 32768;
    if (std.c.getenv("TF_FLASHNEXT_SNAP_HOST")) |v| on = std.mem.eql(u8, std.mem.span(v), "1");
    @import("../../cuda_snap.zig").on_host = on;
    return on;
}

/// cuda_lanes.zig rankRoom: the ranks agree how many STREAMS fit (each its own stream_bytes: rank 1 holds no MTP head
pub fn rankRoom(r: anytype, local: u64, stream_bytes: usize) !u64 {
    if (stream_bytes == 0) return r.room(local);
    const sb: u64 = stream_bytes;
    const q = (local / sb) << 16 | ((local % sb) << 16) / sb;
    const m = try r.room(q);
    return (m >> 16) * sb + ((m & 0xffff) * sb >> 16);
}

/// Device bytes of the sequence caches under the same placement used by newSeq.
pub fn engineSeqBytes(k: EngineKv, cap: usize) usize {
    return @intCast(seqBytes(cap, k.policy()).device());
}

const Entry = struct { e: ?*const anyopaque = null, kv: EngineKv = .{} };
var table: [8]Entry = @splat(.{});

/// The placement `setup` gave engine `e` (upstream's when none: two KV heads).
pub fn of(e: *const anyopaque) EngineKv {
    for (&table) |*t| if (t.e == e) return t.kv;
    return .{};
}

pub fn entryOf(e: *const anyopaque) !*EngineKv {
    for (&table) |*t| if (t.e == e) return &t.kv;
    for (&table) |*t| if (t.e == null) {
        t.* = .{ .e = e };
        return &t.kv;
    };
    return error.OutOfMemory;
}

/// The engine's deinit: its placement.
pub fn forget(e: *const anyopaque) void {
    for (&table) |*t| if (t.e == e) {
        t.* = .{};
    };
}

/// prompt.Engine.newSeq under a non-default placement: the same buffers, K/V of the heads a rank keeps.
pub fn newSeq(e: anytype, comptime Seq: type, cap: usize) !Seq {
    const gpa = e.gpa;
    const d = e.driver;
    const st = e.stream.*;
    const attn = wt.attn_n;
    const row = of(e).rowBytes();
    const logits = try gpa.alloc(u8, wt.vocab * 2);
    errdefer gpa.free(logits);
    var held: Held = .{ .gpa = gpa };
    errdefer held.free();
    const conv = try held.bufs(d, st, wt.linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
    const state = try held.bufs(d, st, wt.linear_n, wt.state_n * 4);
    const kc = try held.bufs(d, st, attn, cap * row);
    const vc = try held.bufs(d, st, attn, cap * row);
    const ikc = try held.bufs(d, st, attn, cap * index_row);
    const pooled = try held.bufs(d, st, attn, @intCast(pooledBytes(cap)));
    var tail = try cuda.DeviceBuffer.alloc(d, wt.ple_tail_n * wt.wide * 2);
    errdefer tail.free();
    try tail.fill8(0, st.handle);
    try st.synchronize();
    return .{
        .conv = conv,
        .state = state,
        .kc = kc,
        .vc = vc,
        .ikc = ikc,
        .pooled = pooled,
        .tail = tail,
        .history = .{ e.eos, e.eos },
        .fed = 0,
        .logits = logits,
        .cap = cap,
    };
}

/// Zeroed device buffers made so far, freed together on a failure.
const Held = struct {
    gpa: std.mem.Allocator,
    lists: [8][]cuda.DeviceBuffer = undefined,
    n: usize = 0,

    fn bufs(h: *Held, d: *const cuda.Driver, st: cuda.Stream, count: usize, len: usize) ![]cuda.DeviceBuffer {
        const out = try h.gpa.alloc(cuda.DeviceBuffer, count);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |*b| b.free();
            h.gpa.free(out);
        }
        while (made < count) : (made += 1) {
            out[made] = try cuda.DeviceBuffer.alloc(d, len);
            out[made].fill8(0, st.handle) catch |err| {
                out[made].free();
                return err;
            };
        }
        h.lists[h.n] = out;
        h.n += 1;
        return out;
    }

    fn free(h: *Held) void {
        for (h.lists[0..h.n]) |l| {
            for (l) |*b| b.free();
            h.gpa.free(l);
        }
        h.n = 0;
    }
};
