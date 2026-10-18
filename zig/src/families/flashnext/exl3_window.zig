//! One decode window for this rank: when to capture, the prefix KV cache, and MTP accept.
//! The CUDA graph itself is captured by `tf-cuda-test exl3-window`. This file has no driver calls.
//! One key is rank, layer, and window. A stored key rewrites replay buffers and does not recapture.
//! Startup, a `.copy`, or an inflight line stays eager. That is the only eager window.

const std = @import("std");
const cache = @import("exl3_cache.zig");
const graph = @import("exl3_graph.zig");
const offload = @import("exl3_offload.zig");
const sample = @import("exl3_sample.zig");
const shape = @import("exl3_shape.zig");

/// Logit row `draftAndVerify` reads. Both heads are this wide.
pub const logit_len: usize = shape.vocab;

pub const Action = enum { eager, capture, rewrite_ids };

/// Eager through the first reap wave. After that, the first stable key captures
/// and every later visit of that key rewrites `expert_ids`, `ple_rows`, and `kv_index`.
pub fn action(
    startup: bool,
    decisions: []const offload.Decision,
    lines: []const offload.Line,
    keys: []const graph.Key,
    key: graph.Key,
) Action {
    if (!graph.captureReady(startup, decisions, lines)) return .eager;
    return switch (graph.promote(keys, key)) {
        .rewrite_ids => .rewrite_ids,
        .recapture => .capture,
    };
}

/// First token still computed. A hit of `n` cached tokens resumes one before
/// the end, which is `cache.resumeAt`. A miss starts at 0.
pub fn startAt(hit: ?u32) u32 {
    return cache.resumeAt(hit orelse 0);
}

const Entry = struct {
    tokens: []u32,
    kv_len: u32,
};

/// Exact stored sequences. A lookup hits the longest stored sequence that is a
/// prefix of the query and returns that entry's KV length.
pub const PrefixCache = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(Entry) = .empty,

    pub fn init(gpa: std.mem.Allocator) PrefixCache {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *PrefixCache) void {
        for (self.items.items) |e| self.gpa.free(e.tokens);
        self.items.deinit(self.gpa);
    }

    pub fn store(self: *PrefixCache, tokens: []const u32, kv_len: u32) error{ Length, OutOfMemory }!void {
        if (tokens.len == 0) return error.Length;
        for (self.items.items) |*e| {
            if (!std.mem.eql(u32, e.tokens, tokens)) continue;
            e.kv_len = kv_len;
            return;
        }
        const copy = try self.gpa.alloc(u32, tokens.len);
        @memcpy(copy, tokens);
        try self.items.append(self.gpa, .{ .tokens = copy, .kv_len = kv_len });
    }

    pub fn lookup(self: *const PrefixCache, tokens: []const u32) ?u32 {
        var best: ?u32 = null;
        var best_len: usize = 0;
        for (self.items.items) |e| {
            if (e.tokens.len == 0 or e.tokens.len > tokens.len) continue;
            if (!std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len])) continue;
            if (e.tokens.len < best_len) continue;
            best = e.kv_len;
            best_len = e.tokens.len;
        }
        return best;
    }
};

pub const Verdict = struct {
    token: u32,
    accepted: bool,
    /// 2 when the draft matches the verified token, otherwise 1.
    advance: u8,
};

/// The main head already sampled `verified`. The MTP head sampled `draft`.
/// A match keeps that token and advances over both the verified position and the draft.
/// A mismatch keeps the verified token and drops the draft.
pub fn accept(draft: u32, verified: u32) Verdict {
    if (draft == verified) return .{ .token = verified, .accepted = true, .advance = 2 };
    return .{ .token = verified, .accepted = false, .advance = 1 };
}

/// Greedy sample of both heads, then `accept`. A short logit row is `error.Length`
/// before a token is chosen.
pub fn draftAndVerify(main_logits: []const f32, mtp_logits: []const f32) error{Length}!Verdict {
    const verified = try sample.sample(main_logits);
    const draft = try sample.sample(mtp_logits);
    return accept(draft, verified);
}

test "startup stays eager and a stable key captures once" {
    const key: graph.Key = .{ .rank = 0, .layer = 3, .window = 1 };
    const copy = [_]offload.Decision{.copy};
    const hit = [_]offload.Decision{.hit};
    const inflight = [_]offload.Line{.inflight};
    const resident = [_]offload.Line{.resident};
    try std.testing.expectEqual(Action.eager, action(true, &hit, &resident, &.{}, key));
    try std.testing.expectEqual(Action.eager, action(false, &copy, &resident, &.{}, key));
    try std.testing.expectEqual(Action.eager, action(false, &hit, &inflight, &.{}, key));
    try std.testing.expectEqual(Action.capture, action(false, &hit, &resident, &.{}, key));
    const stored = [_]graph.Key{key};
    try std.testing.expectEqual(Action.rewrite_ids, action(false, &hit, &resident, &stored, key));
    const other: graph.Key = .{ .rank = 1, .layer = 3, .window = 1 };
    try std.testing.expectEqual(Action.capture, action(false, &hit, &resident, &stored, other));
}

test "a repeated prefix resumes at its KV length" {
    const gpa = std.testing.allocator;
    var prefixes = PrefixCache.init(gpa);
    defer prefixes.deinit();
    try std.testing.expectEqual(@as(?u32, null), prefixes.lookup(&.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(u32, 0), startAt(null));
    try prefixes.store(&.{ 1, 2 }, 2);
    try prefixes.store(&.{ 1, 2, 3 }, 3);
    try std.testing.expectEqual(@as(?u32, 3), prefixes.lookup(&.{ 1, 2, 3, 9 }));
    try std.testing.expectEqual(@as(u32, 2), startAt(prefixes.lookup(&.{ 1, 2, 3, 9 })));
    try std.testing.expectEqual(@as(?u32, 2), prefixes.lookup(&.{ 1, 2, 8 }));
    try std.testing.expectEqual(@as(?u32, null), prefixes.lookup(&.{ 4, 5 }));
    try prefixes.store(&.{ 1, 2, 3 }, 7);
    try std.testing.expectEqual(@as(?u32, 7), prefixes.lookup(&.{ 1, 2, 3 }));
    try std.testing.expectError(error.Length, prefixes.store(&.{}, 1));
}

test "MTP keeps a matching draft and drops a mismatch" {
    try std.testing.expectEqual(@as(u8, 2), accept(40, 40).advance);
    try std.testing.expect(accept(40, 40).accepted);
    try std.testing.expectEqual(@as(u32, 40), accept(40, 40).token);
    const miss = accept(9, 40);
    try std.testing.expect(!miss.accepted);
    try std.testing.expectEqual(@as(u32, 40), miss.token);
    try std.testing.expectEqual(@as(u8, 1), miss.advance);

    const gpa = std.testing.allocator;
    const main_logits = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(main_logits);
    const mtp_logits = try gpa.alloc(f32, shape.vocab);
    defer gpa.free(mtp_logits);
    @memset(main_logits, 0);
    @memset(mtp_logits, 0);
    main_logits[40] = 1;
    mtp_logits[40] = 1;
    const same = try draftAndVerify(main_logits, mtp_logits);
    try std.testing.expect(same.accepted);
    try std.testing.expectEqual(@as(u32, 40), same.token);
    mtp_logits[40] = 0;
    mtp_logits[9] = 1;
    const dropped = try draftAndVerify(main_logits, mtp_logits);
    try std.testing.expect(!dropped.accepted);
    try std.testing.expectEqual(@as(u32, 40), dropped.token);
    try std.testing.expectError(error.Length, draftAndVerify(main_logits[0..4], mtp_logits));
}
