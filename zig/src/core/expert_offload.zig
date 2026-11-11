//! Expert-offload policy: which routed experts hold the device's cache lines while every expert stays in a host tier. No device types here;
//! a backend (HIP: hip/expert_store.zig) owns the memory and the copies and asks this for decisions.
//!
//! This is the CUDA EXL3 host expert cache (families/flashnext/exl3_offload.zig, BlivionIaG's offload series) carried over unchanged in its
//! rules and names: `Decision`, `Policy`, `Line`, `Vram`, `stage`, `schedule`, `reap`, `touch`, `onDevice`, `observe`, `seedHottest`. A VRAM line
//! is a copy that is in flight until `reap`, then resident; eviction is LRU among resident lines; an in-flight line is never a victim; a resident
//! named in the round is touched before any victim is chosen; `copy_bytes` bounds the copies a round starts and the misses past it stay `cpu`.
//! What differs, because the pool is global instead of one table a layer:
//!   * ids are keys (`layer * experts + expert`), the caller leaves the always-resident shared experts out;
//!   * `Vram.where` finds a key's line in one step (a 40-layer pool holds thousands of lines);
//!   * `stage` reports the key it evicted, so the backend can point that expert's table entry back at the host tier before the copy is issued;
//!   * `schedule` writes the copies it started to a caller's buffer, and orders its misses with a tally instead of rescanning the round's ids.
//! `Decision.cpu` is "served from the host tier this round": the CUDA path computes the expert on the host, the HIP path reads it in place over
//! the link, and the split path hands it to the Mac's GPU.

const std = @import("std");

pub const Id = u32;
pub const empty_id: Id = std.math.maxInt(Id);

pub const Decision = enum { hit, copy, cpu };

pub const Policy = struct {
    /// Lines of the pool.
    slots: u32,
    expert_bytes: u64,
    /// Bytes the copy engine may move this round. copy_count = expert_bytes == 0 ? 0 : copy_bytes / expert_bytes.
    copy_bytes: u64,
};

pub const Line = enum { empty, inflight, resident };

/// The pool's lines. Host RAM still holds every routed expert. `stamp` is the last reap or touch time; a higher stamp is newer.
///
/// With `segments` (a bool a line and a cap, `enableSegments`) the pool is an SLRU: a line admitted is on probation; a resident hit on a probation
/// line promotes it to the protected segment, which holds at most `protected_cap` lines (past that the protected line with the oldest stamp goes back
/// to probation, stamp unchanged); eviction takes the oldest probation line, a protected one only when no probation line can go.
pub const Vram = struct {
    slots: []Id,
    line: []Line,
    stamp: []u32,
    /// Per key: its line, or -1.
    where: []i32,
    /// Lines used so far: eviction never empties one, so the empty lines are the tail.
    used: usize = 0,
    /// SLRU: per line, whether it is protected (null: plain LRU), the segment's cap and its size.
    protected: ?[]bool = null,
    protected_cap: usize = 0,
    protected_count: usize = 0,

    pub fn init(slots: []Id, line: []Line, stamp: []u32, where: []i32) Vram {
        std.debug.assert(slots.len == line.len and slots.len == stamp.len);
        @memset(slots, empty_id);
        @memset(line, .empty);
        @memset(stamp, 0);
        @memset(where, -1);
        return .{ .slots = slots, .line = line, .stamp = stamp, .where = where };
    }

    /// SLRU with a protected segment of at most `cap` lines (`flags` has a cell a line).
    pub fn enableSegments(v: *Vram, flags: []bool, cap: usize) void {
        std.debug.assert(flags.len == v.slots.len);
        @memset(flags, false);
        v.protected = flags;
        v.protected_cap = cap;
        v.protected_count = 0;
    }

    pub fn full(v: Vram) bool {
        return v.used == v.slots.len;
    }
};

/// A copy `schedule` or `seedHottest` started: `id` goes to line `slot`, which held `evicted` (null: it was empty).
pub const Copy = struct { id: Id, slot: u32, evicted: ?Id };

/// Where `schedule` keeps its working sets: `unique` holds the round's distinct misses (at least as long as `ids`); `tally` has a cell
/// per key and is zero between calls.
pub const Scratch = struct { unique: []Id, tally: []u16 };

const queued: u16 = 0x8000;

fn lineOf(v: Vram, id: Id) ?usize {
    if (id >= v.where.len) return null;
    const at = v.where[id];
    return if (at < 0) null else @intCast(at);
}

/// True once `id` has been reaped into VRAM.
pub fn onDevice(v: Vram, id: Id) bool {
    const i = lineOf(v, id) orelse return false;
    return v.line[i] == .resident;
}

/// A resident hit. Moves that line to `now` so LRU keeps it; an SLRU promotes a probation line.
pub fn touch(v: *Vram, id: Id, now: u32) bool {
    const i = lineOf(v.*, id) orelse return false;
    if (v.line[i] != .resident) return false;
    if (v.protected) |p| if (!p[i] and v.protected_cap > 0) {
        if (v.protected_count >= v.protected_cap) {
            var oldest: ?usize = null;
            for (p, 0..) |flag, j| {
                if (!flag) continue;
                if (oldest == null or v.stamp[j] < v.stamp[oldest.?]) oldest = j;
            }
            if (oldest) |j| {
                p[j] = false;
                v.protected_count -= 1;
            }
        }
        p[i] = true;
        v.protected_count += 1;
    };
    v.stamp[i] = now;
    return true;
}

/// The key of a line, or null when it is empty.
pub fn keyAt(v: Vram, slot: usize) ?Id {
    return if (v.line[slot] == .empty) null else v.slots[slot];
}

/// Start a host-to-VRAM copy of `id`. An empty line is used first. Otherwise the resident line with the oldest stamp is replaced; a line whose
/// stamp is `now` or newer (touched this round) is not. An in-flight line is left alone. `error.Busy` means no line can be taken.
pub fn stageAt(v: *Vram, id: Id, now: u32) error{Busy}!Copy {
    if (id >= v.where.len) return error.Busy;
    if (lineOf(v.*, id)) |i| {
        if (v.line[i] != .inflight) v.line[i] = .inflight;
        return .{ .id = id, .slot = @intCast(i), .evicted = null };
    }
    if (!v.full()) {
        const slot = v.used;
        v.used += 1;
        v.slots[slot] = id;
        v.line[slot] = .inflight;
        v.stamp[slot] = 0;
        v.where[id] = @intCast(slot);
        return .{ .id = id, .slot = @intCast(slot), .evicted = null };
    }
    const slot = lru(v.*, now) orelse return error.Busy;
    if (v.protected) |p| if (p[slot]) {
        p[slot] = false;
        v.protected_count -= 1;
    };
    const out = v.slots[slot];
    v.where[out] = -1;
    v.slots[slot] = id;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    v.where[id] = @intCast(slot);
    return .{ .id = id, .slot = @intCast(slot), .evicted = out };
}

/// `stageAt` with no round: any resident may be the victim.
pub fn stage(v: *Vram, id: Id) error{Busy}!Copy {
    return stageAt(v, id, std.math.maxInt(u32));
}

/// Oldest resident stamp below `now`, on probation first when the pool is an SLRU. Equal stamps take the higher line.
fn lru(v: Vram, now: u32) ?usize {
    var best: ?usize = null;
    var best_protected: ?usize = null;
    for (v.line, 0..) |state, i| {
        if (state != .resident or v.stamp[i] >= now) continue;
        const slot = if (v.protected) |p| (if (p[i]) &best_protected else &best) else &best;
        if (slot.*) |b| {
            if (v.stamp[i] < v.stamp[b] or (v.stamp[i] == v.stamp[b] and i > b)) slot.* = i;
        } else slot.* = i;
    }
    return best orelse best_protected;
}

/// The copy of `id` has landed. The line becomes resident at `now`.
pub fn reap(v: *Vram, id: Id, now: u32) error{NotInflight}!void {
    const i = lineOf(v.*, id) orelse return error.NotInflight;
    if (v.line[i] != .inflight) return error.NotInflight;
    v.line[i] = .resident;
    v.stamp[i] = now;
}

/// Copies a round may start: `copy_bytes / expert_bytes`, at most the distinct misses.
pub fn copyCount(policy: Policy, unique_misses: usize) usize {
    if (policy.expert_bytes == 0 or unique_misses == 0) return 0;
    const raw = policy.copy_bytes / policy.expert_bytes;
    if (raw >= unique_misses) return unique_misses;
    return @intCast(raw);
}

/// Appearances descending, then the lower id.
fn hotterFirst(tally: []const u16, a: Id, b: Id) bool {
    const ca = tally[a] & ~queued;
    const cb = tally[b] & ~queued;
    if (ca != cb) return ca > cb;
    return a < b;
}

/// One step of the pool. Does not reap. A resident id is touched and decided `hit`. An in-flight id is `cpu` until reap. Unique misses, hottest
/// first (most rows of the round, then lower id), are staged until the copy budget runs out or no line can be taken; a staged miss is `copy`
/// (served from the host tier this round, resident after its `reap`) and is appended to `copies`, which must hold `copyCount` of them.
/// Returns how many copies started. Residents named in `ids` are touched to `now` first, so no one of them is a victim. A zero `expert_bytes`
/// stages nothing.
pub fn schedule(
    v: *Vram,
    ids: []const Id,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: Scratch,
    copies: []Copy,
) error{ Length, Scratch }!usize {
    if (decisions.len != ids.len) return error.Length;
    for (ids) |id| if (id >= scratch.tally.len or id >= v.where.len) return error.Scratch;
    for (ids) |id| {
        scratch.tally[id] +|= 1;
        if (scratch.tally[id] >= queued) scratch.tally[id] = queued - 1;
    }
    defer for (ids) |id| {
        scratch.tally[id] = 0;
    };
    var n: usize = 0;
    for (ids, decisions) |id, *decision| {
        if (onDevice(v.*, id)) {
            _ = touch(v, id, now);
            decision.* = .hit;
            continue;
        }
        decision.* = .cpu;
        if (lineOf(v.*, id) != null) continue;
        if (scratch.tally[id] & queued != 0) continue;
        if (n >= scratch.unique.len) return error.Scratch;
        scratch.tally[id] |= queued;
        scratch.unique[n] = id;
        n += 1;
    }
    std.mem.sort(Id, scratch.unique[0..n], @as([]const u16, scratch.tally), hotterFirst);
    const budget = copyCount(policy, n);
    var started: usize = 0;
    for (scratch.unique[0..n]) |id| {
        if (started >= budget or started >= copies.len) break;
        const c = stageAt(v, id, now) catch break;
        copies[started] = c;
        started += 1;
        for (ids, decisions) |asked, *decision| {
            if (asked == id) decision.* = .copy;
        }
    }
    return started;
}

/// Add one to each routed key in `ids`; `counts` has a cell per key.
pub fn observe(counts: []u32, ids: []const Id) error{Length}!void {
    for (ids) |id| {
        if (id >= counts.len) return error.Length;
        counts[id] +|= 1;
    }
}

fn hotterCount(counts: []const u32, a: Id, b: Id) bool {
    if (counts[a] != counts[b]) return counts[a] > counts[b];
    return a < b;
}

/// Fill empty lines with the hottest counted keys. No eviction; count 0 and keys already in the pool are skipped; ties take the lower key.
/// `order` is scratch with a cell per key. Staged lines stay in flight until `reap`; `copies` receives them (at most its length).
pub fn seedHottest(v: *Vram, counts: []const u32, order: []Id, copies: []Copy) error{Length}!usize {
    if (order.len != counts.len or counts.len != v.where.len) return error.Length;
    var n: usize = 0;
    for (counts, 0..) |c, id| {
        if (c == 0 or lineOf(v.*, @intCast(id)) != null) continue;
        order[n] = @intCast(id);
        n += 1;
    }
    std.mem.sort(Id, order[0..n], counts, hotterCount);
    var started: usize = 0;
    for (order[0..n]) |id| {
        if (v.full() or started >= copies.len) break;
        copies[started] = stage(v, id) catch break;
        started += 1;
    }
    return started;
}

fn testVram(comptime slots: usize, comptime keys: usize) struct {
    slots: [slots]Id = undefined,
    line: [slots]Line = undefined,
    stamp: [slots]u32 = undefined,
    where: [keys]i32 = undefined,
    unique: [keys * 4]Id = undefined,
    tally: [keys]u16 = @splat(0),
    decisions: [keys * 4]Decision = undefined,
    copies: [slots]Copy = undefined,
} {
    return .{};
}

test "a staged copy is not on device until reap, and LRU drops the older resident" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    try std.testing.expectEqual(@as(u32, 0), (try stage(&vram, 4)).slot);
    try std.testing.expectEqual(@as(u32, 1), (try stage(&vram, 7)).slot);
    try std.testing.expect(!onDevice(vram, 4));
    try std.testing.expectError(error.NotInflight, reap(&vram, 9, 1));
    try reap(&vram, 4, 1);
    try reap(&vram, 7, 2);
    try std.testing.expect(onDevice(vram, 4) and onDevice(vram, 7));
    const c = try stage(&vram, 9);
    try std.testing.expectEqual(@as(u32, 0), c.slot);
    try std.testing.expectEqual(@as(?Id, 4), c.evicted);
    try std.testing.expect(!onDevice(vram, 4) and !onDevice(vram, 9) and onDevice(vram, 7));
    try std.testing.expectEqual(@as(i32, -1), vram.where[4]);
}

test "an SLRU keeps a line that was hit again over a newer line that was not, and demotes the oldest protected past its cap" {
    var t = testVram(3, 16);
    var flags: [3]bool = undefined;
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    vram.enableSegments(&flags, 1);
    for ([_]Id{ 1, 2, 3 }, 1..) |id, now| {
        _ = try stage(&vram, id);
        try reap(&vram, id, @intCast(now));
    }
    // 1 is hit again: protected (stamp 4). 2 is now the oldest probation line though 1 was the oldest stamp
    try std.testing.expect(touch(&vram, 1, 4));
    try std.testing.expectEqual(@as(usize, 1), vram.protected_count);
    const c = try stage(&vram, 9);
    try std.testing.expectEqual(@as(?Id, 2), c.evicted);
    try reap(&vram, 9, 5);
    // a second promotion past the cap of one sends the oldest protected line (1) back to probation
    try std.testing.expect(touch(&vram, 3, 6));
    try std.testing.expectEqual(@as(usize, 1), vram.protected_count);
    const d = try stage(&vram, 10);
    try std.testing.expectEqual(@as(?Id, 1), d.evicted);
}

test "an in-flight line is not reused, and a full in-flight set is busy" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try std.testing.expectError(error.Busy, stage(&vram, 3));
    try reap(&vram, 2, 5);
    const c = try stage(&vram, 3);
    try std.testing.expectEqual(@as(u32, 1), c.slot);
    try std.testing.expectEqual(@as(?Id, 2), c.evicted);
    try std.testing.expectError(error.NotInflight, reap(&vram, 2, 6));
}

test "schedule touches residents, stages the hottest miss, and will not evict a hit" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 10, .copy_bytes = 10 };
    const ids = [_]Id{ 1, 5, 5, 6 };
    const n = try schedule(&vram, &ids, policy, 3, t.decisions[0..4], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqualSlices(Decision, &.{ .hit, .copy, .copy, .cpu }, t.decisions[0..4]);
    // 5 (two rows) is the hottest miss; it took the line of 2, which was older than the touched 1
    try std.testing.expectEqual(@as(?Id, 2), t.copies[0].evicted);
    try std.testing.expect(onDevice(vram, 1) and !onDevice(vram, 5));
    try std.testing.expectEqual(@as(u16, 0), t.tally[5]);
}

test "a zero copy budget stages nothing, and an older resident is the one that moves" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    var policy: Policy = .{ .slots = 2, .expert_bytes = 10, .copy_bytes = 0 };
    const ids = [_]Id{ 7, 1 };
    var n = try schedule(&vram, &ids, policy, 3, t.decisions[0..2], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 0), n);
    try std.testing.expectEqualSlices(Decision, &.{ .cpu, .hit }, t.decisions[0..2]);
    policy.copy_bytes = 10;
    n = try schedule(&vram, &ids, policy, 4, t.decisions[0..2], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(?Id, 2), t.copies[0].evicted);
    policy.expert_bytes = 0;
    const again = [_]Id{8};
    try std.testing.expectEqual(@as(usize, 0), try schedule(&vram, &again, policy, 5, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
}

test "misses past the copy budget are cpu, equal counts take the lower id, duplicates share a decision" {
    var t = testVram(4, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    const policy: Policy = .{ .slots = 4, .expert_bytes = 4, .copy_bytes = 8 };
    const ids = [_]Id{ 9, 3, 9, 5, 3, 7 };
    const n = try schedule(&vram, &ids, policy, 1, t.decisions[0..6], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 2), n);
    // 3 and 9 appear twice (lower id first), 5 and 7 once: only two copies fit
    try std.testing.expectEqual(@as(Id, 3), t.copies[0].id);
    try std.testing.expectEqual(@as(Id, 9), t.copies[1].id);
    try std.testing.expectEqualSlices(Decision, &.{ .copy, .copy, .copy, .cpu, .copy, .cpu }, t.decisions[0..6]);
    // an in-flight id is cpu until reap and is not staged twice
    const next = [_]Id{3};
    try std.testing.expectEqual(@as(usize, 0), try schedule(&vram, &next, policy, 2, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectEqual(Decision.cpu, t.decisions[0]);
}

test "a resident named this round is never the victim even when it is the oldest" {
    var t = testVram(2, 16);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try reap(&vram, 1, 1);
    try reap(&vram, 2, 2);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 1, .copy_bytes = 100 };
    // both residents and two misses in one round: nothing can be evicted, so no copy starts
    const ids = [_]Id{ 1, 2, 8, 9 };
    const n = try schedule(&vram, &ids, policy, 3, t.decisions[0..4], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies);
    try std.testing.expectEqual(@as(usize, 0), n);
    try std.testing.expectEqualSlices(Decision, &.{ .hit, .hit, .cpu, .cpu }, t.decisions[0..4]);
}

test "observed counts seed empty lines hottest first, without eviction" {
    var t = testVram(3, 8);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    var counts: [8]u32 = @splat(0);
    try observe(&counts, &.{ 5, 5, 5, 2, 2, 6, 1 });
    var order: [8]Id = undefined;
    const n = try seedHottest(&vram, &counts, &order, &t.copies);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(Id, 5), t.copies[0].id);
    try std.testing.expectEqual(@as(Id, 2), t.copies[1].id);
    try std.testing.expectEqual(@as(Id, 1), t.copies[2].id); // 1 and 6 tie on one: the lower
    try std.testing.expectEqual(@as(?Id, null), t.copies[0].evicted);
    try std.testing.expectEqual(@as(usize, 0), try seedHottest(&vram, &counts, &order, &t.copies));
    try std.testing.expectError(error.Length, observe(&counts, &.{9}));
}

test "a short decision buffer or an unknown key is refused without writes" {
    var t = testVram(2, 4);
    var vram = Vram.init(&t.slots, &t.line, &t.stamp, &t.where);
    const policy: Policy = .{ .slots = 2, .expert_bytes = 1, .copy_bytes = 1 };
    try std.testing.expectError(error.Length, schedule(&vram, &.{ 1, 2 }, policy, 1, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectError(error.Scratch, schedule(&vram, &.{9}, policy, 1, t.decisions[0..1], .{ .unique = &t.unique, .tally = &t.tally }, &t.copies));
    try std.testing.expectEqual(@as(usize, 0), vram.used);
}
