//! Plain types the forward shares with its later layers (CED stash, multi-segment runs, gate items, round graph).
const std = @import("std");
const eh = @import("engram_host.zig");

/// the decoder's rows over a prompt's end, its last token (the verify row) included
pub const replay_rows: u32 = 128;
/// the stash rows a slot needs (the prefilled part of the tail)
pub const keep: u32 = replay_rows - 1;
/// the stash ring's rows (position p at row p % ring): at least `keep`, a power of two
pub const ring: u32 = 128;

/// A slot's stash bookkeeping over a ring of `ring_rows` device rows (row p % ring_rows), `keep_rows` of them live.
pub fn StashOf(comptime ring_rows: u32, comptime keep_rows: u32) type {
    comptime std.debug.assert(ring_rows >= keep_rows);
    return struct {
        from: u64 = 0,
        hi: u64 = 0,
        ids: [ring_rows]u32 = @splat(0),

        /// A segment at `start`; a gap starts the stash over there.
        pub fn extend(s: *@This(), start: u64, ids: []const u32) void {
            if (s.hi != start) s.from = start;
            for (ids, 0..) |t, i| s.ids[@intCast((start + i) % ring_rows)] = t;
            s.hi = start + ids.len;
        }

        /// The first position held.
        pub fn lo(s: *const @This()) u64 {
            return @max(s.from, s.hi -| keep_rows);
        }

        /// The replay check: the stash ends at `end` and holds [r0, end).
        pub fn covers(s: *const @This(), r0: u64, end: u64) bool {
            return s.hi == end and s.lo() <= r0 and end - r0 <= keep_rows;
        }

        /// The ids of positions [r0, end) (`covers` first).
        pub fn tail(s: *const @This(), r0: u64, end: u64, out: []u32) void {
            for (out[0..@intCast(end - r0)], 0..) |*o, i| o.* = s.ids[@intCast((r0 + i) % ring_rows)];
        }
    };
}

/// A slot's stash bookkeeping (rows on the device, "s.ced.*" row p % ring): positions [max(from, hi - keep), hi), ids.
pub const Stash = StashOf(ring, keep);

/// The exact replay's stash (the cone ring "s.cone.*"): the last `cone_keep` rows of the encoder pass's segments.
pub const cone_keep: u32 = 128 + 19 * 127;
pub const ConeStash = StashOf(cone_keep, cone_keep);

/// One segment of a multi-segment run: a slot's prompt ids from its position `start`.
pub const MultiSeg = struct { slot: u32, start: u64, ids: []const u32 };

/// An Engram tail (a slot's last ids before the run).
pub const Tail = struct {
    ids: [eh.max_ngram]u32 = undefined,
    len: usize = 0,

    pub fn items(t: *const Tail) []const u32 {
        return t.ids[0..t.len];
    }
};

/// The run in flight: its segments, each one's first row and Engram tail at the start, every segment's ids in order.
pub const Multi = struct { segs: []const MultiSeg, row0: []const usize, tails: []const Tail, ids: []const u32 };

/// One slot's decoder replay in a batched run: the tail's first position and rows, its first row in the run.
pub const Replayed = struct { slot: u32, at: u64, rows: u32, row0: u32 };

/// A window's part on one slot: its ids and the slot's Engram tail (lookback) before them.
pub const Item = struct { ids: []const u32, tail: []const u32 };

/// TF_DSV41_ROUND_GRAPH (round_graph.zig): the settings read once.
pub const RoundGraph = struct {
    split: bool = false,
    gate: bool = false,
    /// layers in the head graph
    head: u32 = 1,
};

pub const stream_min: i64 = 4096; // TF_DSV41_INDEX_STREAM_MIN (block.zig: a decode window past it would stream)
