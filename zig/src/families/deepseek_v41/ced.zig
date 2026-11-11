//! CED bounded replay prefill (`TF_DSV41_PREFILL=replay`): the replay rules and the stash bookkeeping (host only).

const std = @import("std");
const Config = @import("config.zig").Config;
const ft = @import("forward_types.zig");

pub const replay_rows = ft.replay_rows;
pub const keep = ft.keep;
pub const ring = ft.ring;
/// snapshot and piece alignment
pub const grid: u64 = 16;
/// a prompt snapshot needs this many tokens
pub const session_min: u64 = 64;
/// session tag prefill field: the fast prefill path, prompt snapshots only; exact-tag (0) turn entries never resume
pub const tag_fast: u32 = 1 << 8;
/// with the CED field (replay mode index << 4): replay entries never resume full ones
pub const tag: u32 = tag_fast | 1 << 4;

pub const Error = error{ NoDecoder, EngramInDecoder, StashGap };

/// The first ratio-1 layer: must be a kv source, and no decoder layer may be an Engram layer.
pub fn decoderStart(cfg: *const Config) Error!u32 {
    var d: u32 = cfg.layers;
    for (0..cfg.layers) |i| if (cfg.compressRatio(@intCast(i)) == 1) {
        d = @intCast(i);
        break;
    };
    if (d >= cfg.layers or !cfg.isKvSource(d)) return error.NoDecoder;
    for (cfg.engram_layers.items()) |L| if (L >= d) return error.EngramInDecoder;
    return d;
}

/// The decoder replay of a prompt whose prefilled part ends at `end` (= n - 1): rows [r0, end), m = end - r0.
pub const Tail = struct { r0: u64, m: u32 };

pub fn tailOf(end: u64) Tail {
    const m: u32 = @intCast(@min(end, keep));
    return .{ .r0 = end - m, .m = m };
}

/// The last grid point strictly before a prompt's end.
pub fn snapshotPoint(n: u64) u64 {
    return if (n > 0) (n - 1) / grid * grid else 0;
}

/// Where a prompt of `n` tokens resumed at `cached` is snapshotted, or null.
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s = snapshotPoint(n);
    return if (s > cached and s >= session_min) s else null;
}

pub const Stash = ft.Stash;

/// A suffix rule: how many rows the decoder's first layer consumes, and whether those rows are the full forward's.
pub const Rule = struct { rows: u32, exact: bool };

/// The exact rule for `kept` rows: the cone's rows, which the full forward's bits match.
pub fn rule(cfg: *const Config, kept: u32) Error!Rule {
    return .{ .rows = try suffixRows(cfg, kept), .exact = true };
}

/// The decoder suffix cone: the rows the encoder's last layer must produce for the decoder to stay exact.
pub fn suffixRows(cfg: *const Config, kept: u32) Error!u32 {
    const d = try decoderStart(cfg);
    return kept + (cfg.layers - d) * (cfg.window - 1);
}

/// The exact decoder replay's rows: layer k of `nd` needs `window + (nd - 1 - k) * (window - 1)` rows behind the end.
pub fn coneRows(window: u32, nd: u32) u32 {
    return window + (nd -| 1) * (window - 1);
}

/// The release's cone (window 128, 20 decoder layers): the stash's and the device ring's rows.
pub const cone_keep = ft.cone_keep;
pub const ConeStash = ft.ConeStash;

/// The cone ring's roles (persistent, one set a process): the entering stream state of the layer a program starts at.
pub const cone_roles = [_][]const u8{ "s.cone.x", "s.cone.out", "s.cone.pre", "s.cone.post", "s.cone.comb" };

/// One program of the exact replay: decoder layers [first, first + count) of the pass.
pub const Group = struct { first: u32, count: u32 };

/// The programs: a cut before an index source below every drafter tap once `group` layers have run since the last.
pub fn groups(cfg: *const Config, dec: []const u32, group: u32, out: []Group) []const Group {
    var n: usize = 0;
    var first: u32 = 0;
    const below: u32 = if (cfg.dspark_targets.items().len == 0) std.math.maxInt(u32) else std.mem.min(u32, cfg.dspark_targets.items());
    for (1..dec.len) |i| {
        const k: u32 = @intCast(i);
        if (k - first < group or !cfg.isIndexSource(dec[i]) or dec[i] >= below or n + 1 >= out.len) continue;
        out[n] = .{ .first = first, .count = k - first };
        n += 1;
        first = k;
    }
    out[n] = .{ .first = first, .count = @as(u32, @intCast(dec.len)) - first };
    return out[0 .. n + 1];
}

/// The rows a program starting at decoder layer `first` runs over, given `avail` rows of stash behind the prompt's end.
pub fn groupRows(window: u32, nd: u32, first: u32, avail: u64) u32 {
    return @intCast(@min(avail, window + (nd - 1 - first) * (window - 1)));
}

/// Every row: an encoder layer is not cone-limited.
pub const all_rows: u32 = std.math.maxInt(u32);

/// The rows of layer `layer`'s output the cone needs: the decoder's cone, or every row of an encoder layer.
pub fn layerRows(cfg: *const Config, kept: u32, layer: u32) Error!u32 {
    const d = try decoderStart(cfg);
    if (layer < d) return all_rows;
    return kept + (cfg.layers - 1 - layer) * (cfg.window - 1);
}

/// The rows layer `layer` computes over a prompt of `n` rows: its cone's, or all of them for an encoder layer.
pub fn layerRowsOf(cfg: *const Config, kept: u32, layer: u32, n: u32) Error!u32 {
    const rows = try layerRows(cfg, kept, layer);
    return if (rows == all_rows) n else @min(rows, n);
}

/// The replay before a save at S: the cone's rows [S - R, S), or null when the segment since `cached` is shorter.
pub const Replay = struct { r0: u64, s: u64 };

pub fn snapshotReplayWith(cfg: *const Config, n: u64, cached: u64, kept: u32) Error!?Replay {
    const s = savePoint(n, cached) orelse return null;
    if (s < cached) return null;
    const r: u64 = try suffixRows(cfg, kept);
    if (s - cached < r) return null;
    return .{ .r0 = s - r, .s = s };
}

/// The replay before a save at S under the release config: the cone's rows [S - R, S), or null when too short.
pub fn snapshotReplay(n: u64, cached: u64, kept: u32) Error!?Replay {
    const cfg: Config = .{};
    return snapshotReplayWith(&cfg, n, cached, kept);
}

/// The drafter hand-off before the first verify row: one ingest of the last `window` taps rows from `row0` at `start`.
pub const Handoff = struct { start: u64, row0: u32, n: u32 };

pub fn handoff(at: u64, rows: u32, window: u32) ?Handoff {
    if (rows == 0 or window == 0) return null;
    const skip = rows -| window;
    return .{ .start = at + skip, .row0 = skip, .n = rows - skip };
}

/// One contiguous run of positions in the ring: rows [row, row + n) hold positions [at, at + n) of a range.
pub const Span = struct { row: u32, n: u32, at: u32 };

/// Positions [a, b) (at most `ring`) as at most two contiguous ring runs; `at` counts from a.
pub fn spans(a: u64, b: u64, out: *[2]Span) []const Span {
    return spansIn(ring, a, b, out);
}

/// `spans` over a ring of `rows` rows (the cone ring's).
pub fn spansIn(rows: u32, a: u64, b: u64, out: *[2]Span) []const Span {
    std.debug.assert(b >= a and b - a <= rows);
    const total: u32 = @intCast(b - a);
    if (total == 0) return out[0..0];
    const r: u32 = @intCast(a % rows);
    const first = @min(total, rows - r);
    out[0] = .{ .row = r, .n = first, .at = 0 };
    if (first == total) return out[0..1];
    out[1] = .{ .row = 0, .n = total - first, .at = first };
    return out[0..2];
}

/// The stash roles (streams [ring, 4 D], normed input, fp32 pre / post [ring, 4], comb [ring, 16]) in glue-step order.
pub const roles = [_][]const u8{ "s.ced.x", "s.ced.out", "s.ced.pre", "s.ced.post", "s.ced.comb" };

test "replay rules as replay.py / sessions.py / batch.py" {
    const cfg: Config = .{};
    try std.testing.expectEqual(@as(u32, 20), try decoderStart(&cfg));
    // n <= 128: R0 = 0, the decoder sees the whole prefilled part
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 0 }, tailOf(0));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 44 }, tailOf(44));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 127 }, tailOf(127));
    // n = 129 (end 128): R0 = 1 = n - 128
    try std.testing.expectEqual(Tail{ .r0 = 1, .m = 127 }, tailOf(128));
    try std.testing.expectEqual(Tail{ .r0 = 32768 - 128, .m = 127 }, tailOf(32767));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(33));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(48));
    try std.testing.expectEqual(@as(u64, 48), snapshotPoint(49));
    try std.testing.expectEqual(@as(?u64, null), savePoint(60, 0)); // S = 48 < session_min
    try std.testing.expectEqual(@as(?u64, 64), savePoint(65, 0));
    try std.testing.expectEqual(@as(?u64, null), savePoint(65, 64)); // nothing past the resumed position
    try std.testing.expectEqual(@as(?u64, 4096), savePoint(4100, 1024));
    // a decoder layer with Engram, or no ratio-1 layer: refused
    var bad = cfg;
    bad.engram_layers = @TypeOf(cfg.engram_layers).of(&.{ 1, 24 });
    try std.testing.expectError(error.EngramInDecoder, decoderStart(&bad));
}

test "the DSpark hand-off: the decoder pass's rows whole, a longer run's last window" {
    // replay, n = 1,536: the tail [1408, 1535), 127 rows, one ingest at R0
    try std.testing.expectEqual(Handoff{ .start = 1408, .row0 = 0, .n = 127 }, handoff(tailOf(1535).r0, tailOf(1535).m, 128).?);
    // a prompt of 45 tokens: its 44 prefilled rows from 0
    try std.testing.expectEqual(Handoff{ .start = 0, .row0 = 0, .n = 44 }, handoff(tailOf(44).r0, tailOf(44).m, 128).?);
    // full mode, a last segment of 2,048 rows at 4,096: its last 128
    try std.testing.expectEqual(Handoff{ .start = 4096 + 1920, .row0 = 1920, .n = 128 }, handoff(4096, 2048, 128).?);
    try std.testing.expectEqual(@as(?Handoff, null), handoff(0, 0, 128));
}

test "the stash: the last rows of every segment, a gap restarts it, two ring runs at most" {
    var s: Stash = .{};
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    s.extend(0, ids[0..100]);
    try std.testing.expect(s.covers(0, 100));
    s.extend(100, ids[100..300]);
    try std.testing.expectEqual(@as(u64, 300 - keep), s.lo());
    try std.testing.expect(s.covers(173, 300));
    try std.testing.expect(!s.covers(172, 300));
    try std.testing.expect(!s.covers(173, 299));
    var got: [keep]u32 = undefined;
    s.tail(173, 300, &got);
    try std.testing.expectEqualSlices(u32, ids[173..300], &got);
    // a segment that does not start at hi: the stash starts over there (a restored slot whose stash ended elsewhere)
    s.extend(500, ids[0..10]);
    try std.testing.expectEqual(@as(u64, 500), s.lo());
    try std.testing.expect(!s.covers(495, 510));
    var sp: [2]Span = undefined;
    try std.testing.expectEqualSlices(Span, &.{ .{ .row = 44, .n = 84, .at = 0 }, .{ .row = 0, .n = 43, .at = 84 } }, spans(300, 427, &sp));
    try std.testing.expectEqualSlices(Span, &.{.{ .row = 0, .n = 127, .at = 0 }}, spans(256, 383, &sp));
    try std.testing.expectEqual(@as(usize, 0), spans(7, 7, &sp).len);
}

test "CED compact tail and ring copies match a linear sliding-window oracle across chunk boundaries" {
    const t = std.testing;
    const cfg: Config = .{};
    var ids: [521]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(10000 + 17 * i);
    // The oracle scans absolute positions; it neither wraps a ring nor calls tailOf to choose visible keys.
    for ([_]u64{ 0, 65520, (1 << 20) - ids.len }) |origin| {
        for ([_]usize{ 1, 15, 16, 17, 127, 128, 129, 257, ids.len }) |chunk| {
            var stash: Stash = .{};
            var device: [ring]u32 = @splat(0);
            var used: usize = 0;
            while (used < ids.len) {
                const next = @min(used + chunk, ids.len);
                stash.extend(origin + used, ids[used..next]);
                // Model the independent device stash, whose compact reads are driven by spans().
                for (used..next) |i| device[(origin + i) % ring] = ids[i];
                used = next;
                const end = origin + used;
                var expected: [keep]u32 = undefined;
                var count: usize = 0;
                for (ids[0..used], 0..) |id, i| {
                    if (end - (origin + i) < cfg.window) {
                        expected[count] = id;
                        count += 1;
                    }
                }
                const from = end - count;
                try t.expect(stash.covers(from, end));
                var got: [keep]u32 = undefined;
                stash.tail(from, end, &got);
                try t.expectEqualSlices(u32, expected[0..count], got[0..count]);
                var copied: [keep]u32 = @splat(0);
                var pieces: [2]Span = undefined;
                var total: usize = 0;
                for (spans(from, end, &pieces)) |piece| {
                    @memcpy(copied[piece.at..][0..piece.n], device[piece.row..][0..piece.n]);
                    total += piece.n;
                }
                try t.expectEqual(count, total);
                try t.expectEqualSlices(u32, expected[0..count], copied[0..count]);
                if (origin == 0) {
                    const tail = tailOf(end);
                    try t.expectEqual(from, tail.r0);
                    try t.expectEqual(count, tail.m);
                }
                if (from > 0) try t.expect(!stash.covers(from - 1, end));
                try t.expect(!stash.covers(from, end + 1));
            }
        }
    }
}

test "CED restore or rewind cannot reuse stale rows before the replacement segment" {
    const t = std.testing;
    var old: [300]u32 = undefined;
    for (&old, 0..) |*id, i| id.* = @intCast(i + 1);
    const replacement = [_]u32{ 901, 902, 903, 904 };
    // Both a backward restore and a forward gap invalidate the old contiguous history.
    for ([_]u64{ 16, 127, 128, 255, 512 }) |restart| {
        var stash: Stash = .{};
        stash.extend(0, &old);
        stash.extend(restart, &replacement);
        try t.expect(!stash.covers(restart - 1, restart + replacement.len));
        try t.expect(stash.covers(restart, restart + replacement.len));
        var got: [replacement.len]u32 = undefined;
        stash.tail(restart, restart + replacement.len, &got);
        try t.expectEqualSlices(u32, &replacement, &got);
    }
}

test "the exact replay's cone: rows per program, the cuts at index sources below the drafter's taps, the stash ring" {
    const t = std.testing;
    const cfg: Config = .{};
    try t.expectEqual(cone_keep, coneRows(cfg.window, 20));
    try t.expectEqual(try layerRows(&cfg, cfg.window, 20), coneRows(cfg.window, 20));
    var dec: [20]u32 = undefined;
    for (&dec, 0..) |*l, i| l.* = @intCast(20 + i);
    var buf: [20]Group = undefined;
    // four layers a program: cuts before 24, 28, 32 and 36 (the taps 37-39 stay in the last program)
    const g4 = groups(&cfg, &dec, 4, &buf);
    try t.expectEqualSlices(Group, &.{ .{ .first = 0, .count = 4 }, .{ .first = 4, .count = 4 }, .{ .first = 8, .count = 4 }, .{ .first = 12, .count = 4 }, .{ .first = 16, .count = 4 } }, g4);
    // a size between index sources rounds up to the next; 20 and more: one program
    try t.expectEqualSlices(Group, &.{ .{ .first = 0, .count = 8 }, .{ .first = 8, .count = 8 }, .{ .first = 16, .count = 4 } }, groups(&cfg, &dec, 5, &buf));
    try t.expectEqualSlices(Group, &.{.{ .first = 0, .count = 20 }}, groups(&cfg, &dec, 20, &buf));
    // the rows: the cone's, less where the prompt is shorter
    try t.expectEqual(@as(u32, 2541), groupRows(cfg.window, 20, 0, 1 << 20));
    try t.expectEqual(@as(u32, 2033), groupRows(cfg.window, 20, 4, 1 << 20));
    try t.expectEqual(@as(u32, 509), groupRows(cfg.window, 20, 16, 1 << 20));
    try t.expectEqual(@as(u32, 700), groupRows(cfg.window, 20, 0, 700));
    // the stash: the last cone_keep rows of consecutive segments, two ring runs at most
    var s: ConeStash = .{};
    var ids: [3000]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    s.extend(0, ids[0..1000]);
    s.extend(1000, ids[1000..3000]);
    try t.expectEqual(@as(u64, 3000 - cone_keep), s.lo());
    try t.expect(s.covers(3000 - cone_keep, 3000));
    try t.expect(!s.covers(3000 - cone_keep - 1, 3000));
    var sp: [2]Span = undefined;
    const runs = spansIn(cone_keep, 3000 - 700, 3000, &sp);
    var total: u32 = 0;
    for (runs) |r| total += r.n;
    try t.expectEqual(@as(u32, 700), total);
    try t.expectEqual(@as(u32, @intCast(2300 % cone_keep)), runs[0].row);
}
