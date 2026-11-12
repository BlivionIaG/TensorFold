//! Expert-offload policy: which routed experts hold a device's cache lines while every expert stays in a host tier.
const std = @import("std");

pub const Id = u32;
pub const empty_id: Id = std.math.maxInt(Id);

/// `cpu` is "served from the host tier this round": read in place, computed on the host, or handed to another device.
pub const Decision = enum { hit, copy, cpu };

pub const Policy = struct {
    /// Lines of the pool.
    slots: u32,
    expert_bytes: u64,
    /// Bytes the copy engine may move this round: expert_bytes == 0 gives no copies, else copy_bytes / expert_bytes.
    copy_bytes: u64,
};

/// A line is empty, a copy in flight until `reap`, or resident; only a resident line is ever a victim.
pub const Line = enum { empty, inflight, resident };

/// The pool's lines, keyed `layer * experts + expert`; `stamp` is the last reap or touch time, a higher one is newer.
pub const Vram = struct {
    slots: []Id,
    line: []Line,
    stamp: []u32,
    /// Per key: its line, or -1.
    where: []i32,
    /// Lines used so far: eviction never empties one, so the empty lines are the tail.
    used: usize = 0,
    /// SLRU (`enableSegments`): per line whether it is protected; null is plain LRU.
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

    /// An SLRU: a line hit again after admission is protected, at most `cap` lines (`flags` has a cell a line).
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

/// What `schedule` works in: `unique` is as long as `ids`, `tally` has a cell a key and is zero between calls.
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

/// A resident hit moves its line to `now`; an SLRU promotes a probation line, demoting its oldest protected one.
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

/// Starts a host-to-VRAM copy of `id` on an empty line, else on the oldest resident one stamped before `now`.
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

/// Oldest resident stamp below `now`, on probation first when the pool is an SLRU; equal stamps take the higher line.
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

/// The copy of `id` has landed: the line becomes resident at `now`.
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

/// One step of the pool: residents of `ids` are touched (`hit`), in-flight ids are `cpu`, hot misses staged (`copy`).
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

/// Fills empty lines with the hottest counted keys, never evicting; count 0 and pooled keys are skipped.
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
