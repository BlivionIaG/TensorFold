//! Host-side lane for one EXL3 rank pair. No driver, no peer access, no serve loop.
//! `pinned` is `pinRank` and `kv` is `contextKvRank`; this file does not walk headers or split heads.
//! Rank 0's footprint is those two plus the learned lines. A failed walk stays the file total so
//! admission can refuse before open. Rank 1's context id is recorded on its own thread.
//! `replay` is one host layer of the programs that already exist. Empty slices skip that stage.
//! `bounceBytes` counts the hidden reductions those programs exchange. No CUDA graph is captured.

const std = @import("std");
const attn = @import("exl3_attn.zig");
const hc = @import("exl3_hc.zig");
const gdn = @import("exl3_gdn.zig");
const reduce = @import("exl3_reduce.zig");
const router = @import("exl3_router.zig");
const norm = @import("exl3_norm.zig");
const embed = @import("exl3_embed.zig");
const ple_launch = @import("exl3_ple_launch.zig");
const admit = @import("exl3_admit.zig");
const head = @import("exl3_head.zig");
const sample = @import("exl3_sample.zig");

/// Current context id on this thread. Rank 0 leaves 0 on the caller. The CUDA lane is not linked here.
threadlocal var current_context: u32 = 0;

pub const Loaded = struct {
    backend: struct {} = .{},
    facts: struct {} = .{},
    rows: u32 = 0,
    stream_bytes: usize = 0,
    free_streams: u32 = 0,
    learn_bytes: usize = 0,
    /// Empty and zero: this host stand-in does not serve.
    model_type: []const u8 = "",
    formats: []const []const u8 = &.{},
    default_context: i64 = 0,
    max_segments: u32 = 0,
    prompt_rows: u32 = 0,
    /// Rank 1's context id. Its thread writes this; rank 0 stays current on the caller.
    context_id: u32 = 0,
    /// Peer access is unsupported on the pair. This stays false; nothing enables it.
    peer_access: bool = false,
    dead: bool = false,

    pub fn deinit(self: *Loaded) void {
        self.dead = true;
    }
};

/// Rank 0's device footprint when the walk succeeded: pin + KV + `lines` * `stride`.
/// Not a fraction of the checkpoint. `lines` and `stride` stay arguments so a measured line is not frozen.
/// A failed walk returns `file_total` unchanged. Adding the pin or the KV there would hide the size refused before open.
pub fn weightShare(pinned: u64, kv: u64, lines: u64, stride: u64, walk_ok: bool, file_total: u64) u64 {
    if (!walk_ok) return file_total;
    return pinned + kv + lines * stride;
}

/// True when the share does not fit the card. A file total from a failed walk refuses the same way.
pub fn openRefuses(share: u64, card_bytes: u64) bool {
    return share > card_bytes;
}

/// Refuse before any `Loaded` exists. A share that fits returns without opening ranks.
pub fn openWith(share: u64, card_bytes: u64) error{Refused}!void {
    if (openRefuses(share, card_bytes)) return error.Refused;
}

/// `error.Refused` before any `Loaded` exists. A share that fits returns the rank pair.
pub fn open(share: u64, card_bytes: u64, gpa: std.mem.Allocator) !Loaded {
    try openWith(share, card_bytes);
    return openRanks(gpa);
}

/// Rank 0 is context 0 on this thread. Rank 1's id is stored from a joined thread. `peer_access` is left false.
/// The host stand-in only records ids, so `gpa` waits for the CUDA lane that will actually allocate.
pub fn openRanks(gpa: std.mem.Allocator) !Loaded {
    _ = gpa;
    current_context = 0;
    var loaded: Loaded = .{};
    const thread = try std.Thread.spawn(.{}, rank1, .{&loaded});
    thread.join();
    return loaded;
}

fn rank1(loaded: *Loaded) void {
    // The CUDA lane will call `cuCtxSetCurrent` here.
    current_context = 1;
    loaded.context_id = current_context;
}

/// MoE has no counter of its own, so one hop uses `attn.bounceBytes` and the width stays there.
/// A replicated GDN out-proj exchanges nothing (`gdn.bounceBytes(false)` is 0) and is not added.
/// The lm_head concat is the vocab fp32 join, not one of these four hidden reductions.
pub fn bounceBytes(moe_layers: usize, attn_layers: usize, hc_layers: usize, gdn_split_layers: usize, tokens: usize) u64 {
    return @as(u64, @intCast(moe_layers)) * attn.bounceBytes(tokens) +
        @as(u64, @intCast(attn_layers)) * attn.bounceBytes(tokens) +
        @as(u64, @intCast(hc_layers)) * hc.bounceBytes(tokens) +
        @as(u64, @intCast(gdn_split_layers)) * gdn.bounceBytes(true, tokens);
}

/// Host forward for one layer, in program order. An empty partial is a stage that did not run.
/// A length error returns before a later stage stores. The caller passes the next input;
/// this does not walk the 48 layers. Rank 1 does not launch the router, and rank 1's shared
/// vector is not an `addHidden` argument. No capture and no `fitWorkspace`: eager is the
/// startup window, when a graph is not ready.
pub fn replay(
    norm_x: []const f32,
    norm_w: []const f32,
    norm_means: []const f32,
    eps: f32,
    norm_out: []f32,
    embed_table: []const u8,
    embed_token: i64,
    embed_out: []u8,
    rank: usize,
    gate: []const u8,
    hidden: []const f32,
    shared_gate: []const u8,
    shared_x: []const f32,
    topk: usize,
    router_logits: []f32,
    picks: []u16,
    weights: []f32,
    scratch: []f32,
    moe0: []const f32,
    moe1: []const f32,
    moe_out: []f32,
    attn0: []const f32,
    attn1: []const f32,
    attn_out: []f32,
    hc0: []const f32,
    hc1: []const f32,
    hc_out: []f32,
    gdn0: []const f32,
    gdn1: []const f32,
    gdn_split: bool,
    gdn_out: []f32,
    admitted: ?admit.Admitted,
    ple_layers: []const ple_launch.Layer,
    words_per_row: u32,
    ple_dest: []i16,
    logit0: []const f32,
    logit1: []const f32,
    logit_out: []f32,
    reader: anytype,
) !?u32 {
    if (norm_x.len != 0 and norm_w.len != 0 and norm_means.len != 0) {
        try norm.apply(norm_x, norm_w, norm_means, eps, norm_out);
    }
    if (embed_table.len != 0) {
        const row = try embed.gather(embed_table, embed_token);
        if (embed_out.len < row.len) return error.Length;
        // One host row. There is no device pointer on this path.
        @memcpy(embed_out[0..row.len], &row);
    }
    // Rank 1 must not launch, or the caller sees `NotRank0` instead of an unchanged sentinel.
    if (rank == 0 and gate.len != 0) {
        try router.launch(rank, gate, hidden, shared_gate, shared_x, topk, router_logits, picks, weights, scratch);
    }
    if (moe0.len != 0 and moe1.len != 0) {
        try reduce.addHidden(moe0, moe1, moe_out);
    }
    if (attn0.len != 0 and attn1.len != 0) {
        try attn.exchangeO(attn0, attn1, attn_out);
    }
    if (hc0.len != 0 and hc1.len != 0) {
        try hc.add(hc0, hc1, hc_out);
    }
    if (gdn0.len != 0 and gdn1.len != 0) {
        try gdn.reduceOut(gdn0, gdn1, gdn_split, gdn_out);
    }
    if (admitted) |listed| {
        if (ple_layers.len != 0) {
            try ple_launch.launch(listed, ple_layers, words_per_row, reader, ple_dest);
        }
    }
    if (logit0.len != 0 and logit1.len != 0) {
        try head.concat(logit0, logit1, logit_out);
        return try sample.sample(logit_out);
    }
    return null;
}

test "weightShare is the pin, the KV, and the lines, or the file total when the walk failed" {
    try std.testing.expectEqual(@as(u64, 10 + 20 + 12), weightShare(10, 20, 3, 4, true, 999));
    try std.testing.expectEqual(@as(u64, 999), weightShare(10, 20, 3, 4, false, 999));
}

test "openWith refuses a file total bigger than the card and accepts a share that fits" {
    const missed = weightShare(10, 20, 3, 4, false, 999);
    try std.testing.expectError(error.Refused, openWith(missed, 100));
    try openWith(42, 100);
}

test "openRanks joins with rank 1's context id, no peer access, and deinit marks the lane dead" {
    var loaded = try openRanks(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), current_context);
    try std.testing.expectEqual(@as(u32, 1), loaded.context_id);
    try std.testing.expect(!loaded.peer_access);
    try std.testing.expect(!loaded.dead);
    loaded.deinit();
    try std.testing.expect(loaded.dead);
}

const Quiet = struct {
    pub fn read(_: @This(), _: u32, _: u64, _: []u8) anyerror!void {
        return error.NotUsed;
    }
};

/// Test bundle. Null outputs are empty slices, so that stage is absent.
const Forward = struct {
    norm_x: []const f32 = &.{},
    norm_w: []const f32 = &.{},
    norm_means: []const f32 = &.{},
    eps: f32 = 0,
    norm_out: ?[]f32 = null,
    embed_table: []const u8 = &.{},
    embed_token: i64 = 0,
    embed_out: ?[]u8 = null,
    rank: usize = 0,
    gate: []const u8 = &.{},
    hidden: []const f32 = &.{},
    shared_gate: []const u8 = &.{},
    shared_x: []const f32 = &.{},
    topk: usize = 0,
    router_logits: ?[]f32 = null,
    picks: ?[]u16 = null,
    weights: ?[]f32 = null,
    scratch: ?[]f32 = null,
    moe0: []const f32 = &.{},
    moe1: []const f32 = &.{},
    moe_out: ?[]f32 = null,
    attn0: []const f32 = &.{},
    attn1: []const f32 = &.{},
    attn_out: ?[]f32 = null,
    hc0: []const f32 = &.{},
    hc1: []const f32 = &.{},
    hc_out: ?[]f32 = null,
    gdn0: []const f32 = &.{},
    gdn1: []const f32 = &.{},
    gdn_split: bool = false,
    gdn_out: ?[]f32 = null,
    admitted: ?admit.Admitted = null,
    ple_layers: []const ple_launch.Layer = &.{},
    words_per_row: u32 = 0,
    ple_dest: ?[]i16 = null,
    logit0: []const f32 = &.{},
    logit1: []const f32 = &.{},
    logit_out: ?[]f32 = null,
};

fn runReader(f: Forward, reader: anytype) !?u32 {
    var blank_f32: [0]f32 = .{};
    var blank_u8: [0]u8 = .{};
    var blank_u16: [0]u16 = .{};
    var blank_i16: [0]i16 = .{};
    const no_f32: []f32 = &blank_f32;
    const no_u8: []u8 = &blank_u8;
    const no_u16: []u16 = &blank_u16;
    const no_i16: []i16 = &blank_i16;
    return replay(
        f.norm_x,
        f.norm_w,
        f.norm_means,
        f.eps,
        f.norm_out orelse no_f32,
        f.embed_table,
        f.embed_token,
        f.embed_out orelse no_u8,
        f.rank,
        f.gate,
        f.hidden,
        f.shared_gate,
        f.shared_x,
        f.topk,
        f.router_logits orelse no_f32,
        f.picks orelse no_u16,
        f.weights orelse no_f32,
        f.scratch orelse no_f32,
        f.moe0,
        f.moe1,
        f.moe_out orelse no_f32,
        f.attn0,
        f.attn1,
        f.attn_out orelse no_f32,
        f.hc0,
        f.hc1,
        f.hc_out orelse no_f32,
        f.gdn0,
        f.gdn1,
        f.gdn_split,
        f.gdn_out orelse no_f32,
        f.admitted,
        f.ple_layers,
        f.words_per_row,
        f.ple_dest orelse no_i16,
        f.logit0,
        f.logit1,
        f.logit_out orelse no_f32,
        reader,
    );
}

fn run(f: Forward) !?u32 {
    return runReader(f, Quiet{});
}

fn listedPle(ids: []const u16, on: bool) admit.Admitted {
    return .{
        .model_type = "qwen4_exp",
        .quant_method = "exl3",
        .hidden = 0,
        .layers = 0,
        .experts = 0,
        .topk = 0,
        .expert_intermediate = 0,
        .vocab = 0,
        .ngram_shards = 0,
        .ngram_bits = 5,
        .ple = .page_cache,
        .kv = .bf16,
        .ranks = 1,
        .cache_bytes = 0,
        .ssd_experts = false,
        .ngram_host = true,
        .ple_layer_ids = ids,
        .ple_listed = on,
    };
}

test "open refuses a share above the card and a share that fits returns rank 1" {
    try std.testing.expectError(error.Refused, open(101, 100, std.testing.allocator));
    var loaded = try open(100, 100, std.testing.allocator);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(u32, 0), current_context);
    try std.testing.expectEqual(@as(u32, 1), loaded.context_id);
    try std.testing.expect(!loaded.peer_access);
}

test "bounceBytes counts moe, o_proj, hc, and a split gdn, not the lm_head" {
    const hop = attn.bounceBytes(1);
    const four = bounceBytes(1, 1, 1, 1, 1);
    const three = bounceBytes(1, 1, 1, 0, 1);
    try std.testing.expectEqual(hop * 4, four);
    try std.testing.expectEqual(hop * 3, three);
    try std.testing.expectEqual(attn.bounceBytes(1) + attn.bounceBytes(1) + hc.bounceBytes(1) + gdn.bounceBytes(true, 1), four);
    try std.testing.expectEqual(attn.bounceBytes(1) + attn.bounceBytes(1) + hc.bounceBytes(1), three);
    try std.testing.expectEqual(gdn.bounceBytes(true, 1), four - three);
    try std.testing.expectEqual(@as(u64, 0), gdn.bounceBytes(false, 1));
    const head_bytes = head.bounceBytes(1);
    try std.testing.expect(head_bytes != hop);
    try std.testing.expect(four != hop * 4 + head_bytes);
    try std.testing.expect(three != hop * 3 + head_bytes);
    try std.testing.expectEqual(attn.bounceBytes(4) * 2, bounceBytes(2, 0, 0, 0, 4));
    try std.testing.expectEqual(attn.bounceBytes(4), bounceBytes(0, 1, 0, 0, 4));
    try std.testing.expectEqual(hc.bounceBytes(4), bounceBytes(0, 0, 1, 0, 4));
    try std.testing.expectEqual(gdn.bounceBytes(true, 4), bounceBytes(0, 0, 0, 1, 4));
}

test "moe replay matches addHidden and a length mismatch stores nothing" {
    const rank0 = [_]f32{1};
    const routed1 = [_]f32{@as(f32, 4097) * 0x1p-20};
    const shared1 = [_]f32{1};
    var out = [_]f32{0};
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .moe0 = &rank0, .moe1 = &routed1, .moe_out = &out }));
    var via = [_]f32{0};
    try reduce.addHidden(&rank0, &routed1, &via);
    try std.testing.expectEqualSlices(f32, &via, &out);
    try std.testing.expectEqual(@as(f32, 1.0078125), out[0]);
    // Rank 1's shared vector is not an argument. Folding it in leaves this rounded sum.
    try std.testing.expect(out[0] != rank0[0] + routed1[0]);
    try std.testing.expect(out[0] != rank0[0] + routed1[0] + shared1[0]);

    const wide = [_]f32{ 1, 2, 3 };
    const short = [_]f32{ 4, 5 };
    var held = [_]f32{ 9, 9, 9 };
    const attn0 = [_]f32{ 1, 1 };
    const attn1 = [_]f32{ 2, 2 };
    var attn_held = [_]f32{ 7, 7 };
    try std.testing.expectError(error.Length, run(.{
        .moe0 = &wide,
        .moe1 = &short,
        .moe_out = &held,
        .attn0 = &attn0,
        .attn1 = &attn1,
        .attn_out = &attn_held,
    }));
    try std.testing.expectEqual(@as(f32, 9), held[0]);
    try std.testing.expectEqual(@as(f32, 9), held[2]);
    try std.testing.expectEqual(@as(f32, 7), attn_held[0]);
    try std.testing.expectEqual(@as(f32, 7), attn_held[1]);

    var untouched = [_]f32{ 4, 5 };
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .moe_out = &untouched }));
    try std.testing.expectEqual(@as(f32, 4), untouched[0]);
    try std.testing.expectEqual(@as(f32, 5), untouched[1]);
}

test "a second replay reduces the first call's hidden" {
    const first0 = [_]f32{10};
    const first1 = [_]f32{20};
    var reduced: [1]f32 = .{0};
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .moe0 = &first0, .moe1 = &first1, .moe_out = &reduced }));
    try std.testing.expectEqual(@as(f32, 30), reduced[0]);

    const next1 = [_]f32{1};
    var second: [1]f32 = .{0};
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .moe0 = &reduced, .moe1 = &next1, .moe_out = &second }));
    var expect: [1]f32 = .{0};
    try reduce.addHidden(&reduced, &next1, &expect);
    try std.testing.expectEqualSlices(f32, &expect, &second);
    try std.testing.expectEqual(@as(f32, 31), second[0]);

    var from_original: [1]f32 = .{0};
    try reduce.addHidden(&first0, &next1, &from_original);
    try std.testing.expect(from_original[0] != second[0]);
}

test "rank 1 does not launch and leaves the sentinel picks" {
    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, router.gate_bytes);
    defer gpa.free(gate);
    @memset(gate, 0);
    const winner: usize = 42;
    const one_bf16: u16 = 0x3f80;
    const at = winner * router.gate_cols * @sizeOf(u16);
    std.mem.writeInt(u16, gate[at..][0..2], one_bf16, .little);

    var shared_gate: [6]u8 = @splat(0);
    for (0..3) |i| std.mem.writeInt(u16, shared_gate[i * 2 ..][0..2], one_bf16, .little);
    const shared_x = [_]f32{ 0, 5, 4 };
    var hidden: [router.gate_cols]f32 = @splat(0);
    hidden[0] = 1;
    var logits: [router.gate_rows + 1]f32 = @splat(8);
    var weights = [_]f32{ 3, 3 };
    var scratch: [router.gate_rows]f32 = @splat(4);

    var sentinel = [_]u16{ 0xBEEF, 0xBEEF };
    try std.testing.expectEqual(@as(?u32, null), try run(.{
        .rank = 1,
        .gate = gate,
        .hidden = &hidden,
        .shared_gate = &shared_gate,
        .shared_x = &shared_x,
        .topk = 1,
        .router_logits = &logits,
        .picks = &sentinel,
        .weights = &weights,
        .scratch = &scratch,
    }));
    try std.testing.expectEqualSlices(u16, &.{ 0xBEEF, 0xBEEF }, &sentinel);
    try std.testing.expectEqual(@as(f32, 8), logits[0]);
    try std.testing.expectEqual(@as(f32, 8), logits[router.gate_rows]);
    try std.testing.expectEqual(@as(f32, 3), weights[0]);

    var picks = [_]u16{ 0xBEEF, 0xBEEF };
    _ = try run(.{
        .rank = 0,
        .gate = gate,
        .hidden = &hidden,
        .shared_gate = &shared_gate,
        .shared_x = &shared_x,
        .topk = 1,
        .router_logits = &logits,
        .picks = &picks,
        .weights = &weights,
        .scratch = &scratch,
    });
    try std.testing.expectEqual(@as(u16, winner), picks[0]);
    try std.testing.expectEqual(@as(u16, 512), picks[1]);

    var held = [_]u16{ 0xBEEF, 0xBEEF };
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .rank = 0, .picks = &held, .topk = 1 }));
    try std.testing.expectEqual(@as(u16, 0xBEEF), held[0]);
}

test "concat then sample reads rank 1's first logit as token 124160" {
    const tp = @import("exl3_tp.zig");
    const shape = @import("exl3_shape.zig");
    const gpa = std.testing.allocator;
    const low = try tp.range(.head, 0, 2);
    const high = try tp.range(.head, 1, 2);
    try std.testing.expectEqual(@as(usize, 124160), high.lo);

    const low_half = try gpa.alloc(f32, low.hi - low.lo);
    defer gpa.free(low_half);
    const high_half = try gpa.alloc(f32, high.hi - high.lo);
    defer gpa.free(high_half);
    const out = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(out);
    @memset(low_half, 0);
    @memset(high_half, 0);
    @memset(out, 0);
    high_half[0] = 1;

    try std.testing.expectEqual(@as(?u32, 124160), try run(.{
        .logit0 = low_half,
        .logit1 = high_half,
        .logit_out = out,
    }));
    try std.testing.expectEqual(@as(f32, 1), out[124160]);

    const short = [_]f32{1};
    @memset(out, 9);
    try std.testing.expectError(error.Length, run(.{
        .logit0 = low_half,
        .logit1 = &short,
        .logit_out = out,
    }));
    try std.testing.expect(std.mem.allEqual(f32, out, 9));
}

test "captureReady is false at startup and host replay still returns the sampled token" {
    const graph = @import("exl3_graph.zig");
    const tp = @import("exl3_tp.zig");
    const shape = @import("exl3_shape.zig");
    try std.testing.expect(!graph.captureReady(true, &.{}, &.{}));

    const gpa = std.testing.allocator;
    const low = try tp.range(.head, 0, 2);
    const high = try tp.range(.head, 1, 2);
    const low_half = try gpa.alloc(f32, low.hi - low.lo);
    defer gpa.free(low_half);
    const high_half = try gpa.alloc(f32, high.hi - high.lo);
    defer gpa.free(high_half);
    const out = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(out);
    @memset(low_half, 0);
    @memset(high_half, 0);
    @memset(out, 0);
    high_half[0] = 1;
    try std.testing.expectEqual(@as(?u32, 124160), try run(.{
        .logit0 = low_half,
        .logit1 = high_half,
        .logit_out = out,
    }));
}

test "attention, hyper-connection, and gdn replay call the host programs" {
    const shape = @import("exl3_shape.zig");
    const attn0 = [_]f32{ 1, 2, 3 };
    const attn1 = [_]f32{ 4, 5, 6 };
    var attn_out: [3]f32 = .{ 0, 0, 0 };
    try std.testing.expectEqual(@as(?u32, null), try run(.{
        .attn0 = &attn0,
        .attn1 = &attn1,
        .attn_out = &attn_out,
    }));
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, &attn_out);

    const gpa = std.testing.allocator;
    const n = shape.hidden;
    const hc0 = try gpa.alloc(f32, n);
    defer gpa.free(hc0);
    const hc1 = try gpa.alloc(f32, n);
    defer gpa.free(hc1);
    const hc_out = try gpa.alloc(f32, n);
    defer gpa.free(hc_out);
    const expect = try gpa.alloc(f32, n);
    defer gpa.free(expect);
    for (hc0, hc1, 0..) |*a, *b, i| {
        a.* = @floatFromInt(i);
        b.* = 1;
    }
    try hc.add(hc0, hc1, expect);
    @memset(hc_out, 0);
    _ = try run(.{ .hc0 = hc0, .hc1 = hc1, .hc_out = hc_out });
    try std.testing.expectEqualSlices(f32, expect, hc_out);

    var short_out = [_]f32{ 9, 9 };
    const s0 = [_]f32{ 1, 2 };
    const s1 = [_]f32{ 3, 4 };
    try std.testing.expectError(error.Length, run(.{ .hc0 = &s0, .hc1 = &s1, .hc_out = &short_out }));
    try std.testing.expectEqual(@as(f32, 9), short_out[0]);
    try std.testing.expectEqual(@as(f32, 9), short_out[1]);

    const g0 = [_]f32{ 1, -2 };
    const g1 = [_]f32{ 3, 4 };
    var summed: [2]f32 = .{ 0, 0 };
    var copied: [2]f32 = .{ 0, 0 };
    _ = try run(.{ .gdn0 = &g0, .gdn1 = &g1, .gdn_split = true, .gdn_out = &summed });
    _ = try run(.{ .gdn0 = &g0, .gdn1 = &g1, .gdn_split = false, .gdn_out = &copied });
    try std.testing.expectEqualSlices(f32, &.{ 4, 2 }, &summed);
    try std.testing.expectEqualSlices(f32, &g0, &copied);
}

test "centred norm and the embed row run only when their inputs are present" {
    const near_zero = [_]f32{ -0.02, 0.01, 0.0, 0.04 };
    const x = [_]f32{1};
    const w = [_]f32{2};
    var norm_out = [_]f32{0};
    try std.testing.expectEqual(@as(?u32, null), try run(.{
        .norm_x = &x,
        .norm_w = &w,
        .norm_means = &near_zero,
        .eps = 0,
        .norm_out = &norm_out,
    }));
    try std.testing.expectEqual(@as(f32, 3), norm_out[0]);

    var held = [_]f32{9};
    const x_only = [_]f32{1};
    const w_only = [_]f32{2};
    try std.testing.expectEqual(@as(?u32, null), try run(.{
        .norm_x = &x_only,
        .norm_w = &w_only,
        .norm_out = &held,
    }));
    try std.testing.expectEqual(@as(f32, 9), held[0]);

    var norm_held = [_]f32{ 9, 8 };
    var moe_held = [_]f32{5};
    const bad_x = [_]f32{ 1, 1 };
    const bad_w = [_]f32{1};
    const m0 = [_]f32{2};
    const m1 = [_]f32{3};
    try std.testing.expectError(error.Length, run(.{
        .norm_x = &bad_x,
        .norm_w = &bad_w,
        .norm_means = &near_zero,
        .norm_out = &norm_held,
        .moe0 = &m0,
        .moe1 = &m1,
        .moe_out = &moe_held,
    }));
    try std.testing.expectEqual(@as(f32, 9), norm_held[0]);
    try std.testing.expectEqual(@as(f32, 8), norm_held[1]);
    try std.testing.expectEqual(@as(f32, 5), moe_held[0]);

    const gpa = std.testing.allocator;
    const table = try gpa.alloc(u8, embed.row_bytes);
    defer gpa.free(table);
    @memset(table, 0x3C);
    const got = try gpa.alloc(u8, embed.row_bytes);
    defer gpa.free(got);
    @memset(got, 0);
    try std.testing.expectEqual(@as(usize, 5120), embed.row_bytes);
    try std.testing.expectEqual(@as(?u32, null), try run(.{
        .embed_table = table,
        .embed_token = 0,
        .embed_out = got,
    }));
    try std.testing.expectEqualSlices(u8, table, got);

    try std.testing.expectError(error.ShortTable, run(.{
        .embed_table = table,
        .embed_token = 1,
        .embed_out = got,
    }));
    try std.testing.expectError(error.TokenOutOfRange, run(.{
        .embed_table = table,
        .embed_token = -1,
        .embed_out = got,
    }));

    var tiny = [_]u8{0};
    var moe_row = [_]f32{6};
    try std.testing.expectError(error.Length, run(.{
        .embed_table = table,
        .embed_token = 0,
        .embed_out = &tiny,
        .moe0 = &m0,
        .moe1 = &m1,
        .moe_out = &moe_row,
    }));
    try std.testing.expectEqual(@as(u8, 0), tiny[0]);
    try std.testing.expectEqual(@as(f32, 6), moe_row[0]);

    @memset(got, 0x11);
    try std.testing.expectEqual(@as(?u32, null), try run(.{ .embed_out = got }));
    try std.testing.expect(std.mem.allEqual(u8, got, 0x11));
}

const Mem = struct {
    image: []const u8,
    pub fn read(self: @This(), file: u32, at: u64, dest: []u8) anyerror!void {
        if (file != 0) return error.BadFile;
        const start: usize = @intCast(at);
        if (start + dest.len > self.image.len) return error.ShortRead;
        @memcpy(dest, self.image[start..][0..dest.len]);
    }
};

test "ple launch gathers an admitted layer and a missing list is MissingField" {
    const ple = @import("exl3_ple.zig");
    const words: u32 = 51;
    const row_b: usize = 102;
    var image: [row_b]u8 = @splat(0);
    var w: usize = 0;
    while (w < words) : (w += 1) {
        const v: i16 = @intCast(7 * 64 + w);
        std.mem.writeInt(i16, image[w * 2 ..][0..2], v, .little);
    }
    const shard = [_]ple.Shard{.{ .file = 0, .offset = 0, .rows = 1 }};
    const rows = [_]u64{0};
    const layers = [_]ple_launch.Layer{.{ .id = 2, .shards = &shard, .rows = &rows }};
    const ids = [_]u16{2};
    var dest: [words]i16 = @splat(-1);
    try std.testing.expectEqual(@as(?u32, null), try runReader(.{
        .admitted = listedPle(&ids, true),
        .ple_layers = &layers,
        .words_per_row = words,
        .ple_dest = &dest,
    }, Mem{ .image = &image }));
    try std.testing.expectEqual(@as(i16, 7 * 64), dest[0]);
    try std.testing.expectEqual(@as(i16, 7 * 64 + 50), dest[50]);

    var held: [words]i16 = @splat(-1);
    try std.testing.expectError(error.MissingField, runReader(.{
        .admitted = listedPle(&.{}, false),
        .ple_layers = &layers,
        .words_per_row = words,
        .ple_dest = &held,
    }, Mem{ .image = &image }));
    try std.testing.expectEqual(@as(i16, -1), held[0]);
}
