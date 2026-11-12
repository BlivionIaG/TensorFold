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
    if (v.protected) |p| {
        return scanOldest(v, now, p, false) orelse scanOldest(v, now, p, true);
    }
    return scanOldest(v, now, null, false);
}

/// `lruScalar`'s answer with the lines in vectors: the pool holds thousands of lines and an admission scans them all (about 3 us a miss scalar).
/// Pass one finds the least stamp among the eligible lines (resident, stamp below `now`, and in the wanted segment), pass two the highest line
/// that has it.
fn scanOldest(v: Vram, now: u32, segments: ?[]const bool, protected: bool) ?usize {
    const lanes = 16;
    const n = v.line.len;
    const state: []const u8 = std.mem.sliceAsBytes(v.line);
    const flag: []const u8 = if (segments) |s| std.mem.sliceAsBytes(s) else state;
    const resident: u8 = @intFromEnum(Line.resident);
    const want: u8 = @intFromBool(protected);
    const none = std.math.maxInt(u32);
    var best: u32 = none;
    var i: usize = 0;
    var low: @Vector(lanes, u32) = @splat(none);
    while (i + lanes <= n) : (i += lanes) {
        const st: @Vector(lanes, u32) = v.stamp[i..][0..lanes].*;
        const ln: @Vector(lanes, u8) = state[i..][0..lanes].*;
        var ok = (ln == @as(@Vector(lanes, u8), @splat(resident))) & (st < @as(@Vector(lanes, u32), @splat(now)));
        if (segments != null) {
            const fg: @Vector(lanes, u8) = flag[i..][0..lanes].*;
            ok = ok & (fg == @as(@Vector(lanes, u8), @splat(want)));
        }
        low = @min(low, @select(u32, ok, st, @as(@Vector(lanes, u32), @splat(none))));
    }
    best = @reduce(.Min, low);
    while (i < n) : (i += 1) {
        if (state[i] != resident or v.stamp[i] >= now) continue;
        if (segments != null and flag[i] != want) continue;
        best = @min(best, v.stamp[i]);
    }
    if (best == none) return null;
    // the highest eligible line with that stamp: from the tail
    var at = n;
    while (at > 0) {
        at -= 1;
        if (state[at] != resident or v.stamp[at] != best) continue;
        if (segments != null and flag[at] != want) continue;
        return at;
    }
    unreachable;
}

/// The rule `lru` implements, line by line (the reference of its tests).
fn lruScalar(v: Vram, now: u32) ?usize {
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

// The hit/miss split of the card's handshake (docs/LATENCY.md M1). A layer's stage is a job, numbered in the order the stream posts them. The card
// checks the layer's picks against `Cache.resident`: all resident, it stores `seen = n` itself and never asks (the job is a *hit*: the host only
// has to touch the lines in the policy, later, from the trace the card left); one is not, it stores `req = n` and waits for the host's `ack`
// (the job is a *miss*: the host serves it as before). The host serves jobs in order, so between two misses it replays the hits.

/// Whether the 32-bit counter `a` has reached `b` across the wrap.
pub fn reached(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) >= 0;
}

pub const Seen = enum { pending, hit, miss };

/// What the host knows of job `n` (the next one it has not served) from the card's two words. `seen` is the last job the card satisfied by
/// itself, `req` the last it asked for. The host alone answers misses, so no later job can have started while `n` is a miss it has not served:
/// a `seen` or `req` past `n` means `n` was a hit; `req == n` is the miss; nothing yet is pending.
pub fn classify(n: u32, seen: u32, req: u32) Seen {
    if (reached(seen, n)) return .hit;
    if (reached(req, n)) return if (req == n) .miss else .hit;
    return .pending;
}

/// The decode cache a host worker drives: the policy's pool (`Vram`) and its scratch, the admission budget of the round and the table the card
/// checks its picks against. The backend owns the memory behind the lines: `line_addr` is each line's device address, `resident` a word a key
/// in memory the card can read (the card's check loads it). `resident[key] != 0` exactly when the key's line is resident, and then it is the line's address.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    vram: Vram,
    slots: []Id,
    line: []Line,
    stamp: []u32,
    flags: []bool,
    where: []i32,
    unique: []Id,
    tally: []u16,
    decisions: []Decision,
    /// The keys of the stage being decided (the caller fills `ids[0..n]`).
    ids: []Id,
    copies: []Copy,
    resident: []u64,
    line_addr: []const u64,
    now: u32 = 0,
    /// Admissions the round's remaining stages may still make.
    budget: usize,
    copy_experts: usize,
    expert_bytes: u64,

    pub const Config = struct {
        keys: usize,
        lines: usize,
        /// The most keys a decode stage names (rows x routed picks).
        picks_max: usize,
        copy_experts: usize,
        expert_bytes: u64,
        /// Percent of the lines that are an SLRU's protected segment (0: plain LRU).
        protect_pct: u8 = 0,
    };

    pub const Decided = struct { started: usize, hits: usize };

    /// `resident` has a word a key and `line_addr` one a line; both are the caller's. `resident` is zeroed.
    pub fn init(gpa: std.mem.Allocator, cfg: Config, resident: []u64, line_addr: []const u64) std.mem.Allocator.Error!Cache {
        std.debug.assert(resident.len == cfg.keys and line_addr.len == cfg.lines);
        const slots = try gpa.alloc(Id, cfg.lines);
        errdefer gpa.free(slots);
        const line = try gpa.alloc(Line, cfg.lines);
        errdefer gpa.free(line);
        const stamp = try gpa.alloc(u32, cfg.lines);
        errdefer gpa.free(stamp);
        const flags = try gpa.alloc(bool, cfg.lines);
        errdefer gpa.free(flags);
        const where = try gpa.alloc(i32, cfg.keys);
        errdefer gpa.free(where);
        const unique = try gpa.alloc(Id, cfg.picks_max);
        errdefer gpa.free(unique);
        const tally = try gpa.alloc(u16, cfg.keys);
        errdefer gpa.free(tally);
        @memset(tally, 0);
        const decisions = try gpa.alloc(Decision, cfg.picks_max);
        errdefer gpa.free(decisions);
        const ids = try gpa.alloc(Id, cfg.picks_max);
        errdefer gpa.free(ids);
        const copies = try gpa.alloc(Copy, @min(@max(cfg.copy_experts, 1), cfg.picks_max));
        errdefer gpa.free(copies);
        @memset(resident, 0);
        var vram = Vram.init(slots, line, stamp, where);
        if (cfg.protect_pct > 0) vram.enableSegments(flags, cfg.lines * @as(usize, cfg.protect_pct) / 100);
        return .{
            .gpa = gpa,
            .vram = vram,
            .slots = slots,
            .line = line,
            .stamp = stamp,
            .flags = flags,
            .where = where,
            .unique = unique,
            .tally = tally,
            .decisions = decisions,
            .ids = ids,
            .copies = copies,
            .resident = resident,
            .line_addr = line_addr,
            .budget = cfg.copy_experts,
            .copy_experts = cfg.copy_experts,
            .expert_bytes = cfg.expert_bytes,
        };
    }

    pub fn deinit(c: *Cache) void {
        c.gpa.free(c.copies);
        c.gpa.free(c.ids);
        c.gpa.free(c.decisions);
        c.gpa.free(c.tally);
        c.gpa.free(c.unique);
        c.gpa.free(c.where);
        c.gpa.free(c.flags);
        c.gpa.free(c.stamp);
        c.gpa.free(c.line);
        c.gpa.free(c.slots);
        c.* = undefined;
    }

    /// A round begins: its stages may admit `copy_experts` experts again.
    pub fn beginRound(c: *Cache) void {
        c.budget = c.copy_experts;
    }

    /// The card's check, as the kernel makes it: every key of `ids` is resident.
    pub fn allResident(c: *const Cache, ids: []const Id) bool {
        for (ids) |id| if (c.resident[id] == 0) return false;
        return true;
    }

    /// The device address of `key`'s line when it is resident.
    pub fn address(c: *const Cache, key: Id) ?u64 {
        return if (onDevice(c.vram, key)) c.line_addr[@intCast(c.vram.where[key])] else null;
    }

    /// A miss stage over `ids[0..n]`: touches the residents, starts the admissions the budget allows (the keys they evict are no longer resident from here).
    /// The admitted keys are not resident until `commit`, after the caller has given them their table entries.
    pub fn decide(c: *Cache, n: usize) error{StageFailed}!Decided {
        c.now +%= 1;
        const started = schedule(&c.vram, c.ids[0..n], .{ .slots = @intCast(c.slots.len), .expert_bytes = c.expert_bytes, .copy_bytes = @as(u64, c.budget) * c.expert_bytes }, c.now, c.decisions[0..n], .{ .unique = c.unique, .tally = c.tally }, c.copies) catch return error.StageFailed;
        c.budget -= started;
        var hits: usize = 0;
        for (c.decisions[0..n]) |d| hits += @intFromBool(d == .hit);
        for (c.copies[0..started]) |cp| if (cp.evicted) |out| {
            c.resident[out] = 0;
        };
        return .{ .started = started, .hits = hits };
    }

    /// The admissions `decide` started are in place: their lines are resident (and named in the table) for the next stage.
    pub fn commit(c: *Cache, started: usize) error{StageFailed}!void {
        for (c.copies[0..started]) |cp| {
            reap(&c.vram, cp.id, c.now) catch return error.StageFailed;
            c.resident[cp.id] = c.line_addr[cp.slot];
        }
    }

    /// The stage of a hit job, late: what `decide` does to a stage whose every key is resident (they are touched, nothing starts), from the keys the
    /// card traced. A key that is not resident means the card and the host disagree about the table.
    pub fn replay(c: *Cache, ids: []const Id) error{NotResident}!void {
        c.now +%= 1;
        for (ids) |id| if (!touch(&c.vram, id, c.now)) return error.NotResident;
    }
};

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

/// Routing like a real trace's: a layer's picks lean on a few experts that drift, the rest are scattered; `per` distinct keys a layer.
const Synth = struct {
    prng: std.Random.DefaultPrng,
    layers: usize,
    experts: usize,
    per: usize,

    fn token(s: *Synth, out: []Id) void {
        const rnd = s.prng.random();
        for (0..s.layers) |layer| {
            const row = out[layer * s.per ..][0..s.per];
            var n: usize = 0;
            while (n < s.per) {
                const x: usize = if (rnd.uintLessThan(u8, 100) < 70) (layer * 7 + rnd.uintLessThan(usize, 12)) % s.experts else rnd.uintLessThan(usize, s.experts);
                const key: Id = @intCast(layer * s.experts + x);
                if (std.mem.indexOfScalar(Id, row[0..n], key) != null) continue;
                row[n] = key;
                n += 1;
            }
        }
    }
};

/// A cache of `lines` lines over `layers * experts` keys and the memory behind it.
fn testCache(gpa: std.mem.Allocator, layers: usize, experts: usize, lines: usize, copies: usize, protect: u8) !struct { c: Cache, resident: []u64, addr: []u64 } {
    const resident = try gpa.alloc(u64, layers * experts);
    const addr = try gpa.alloc(u64, lines);
    for (addr, 0..) |*a, i| a.* = 0x1000_0000 + i * 4096;
    return .{ .c = try Cache.init(gpa, .{ .keys = layers * experts, .lines = lines, .picks_max = 16, .copy_experts = copies, .expert_bytes = 4096, .protect_pct = protect }, resident, addr), .resident = resident, .addr = addr };
}

fn sameCache(a: *const Cache, b: *const Cache) !void {
    try std.testing.expectEqual(a.now, b.now);
    try std.testing.expectEqual(a.budget, b.budget);
    try std.testing.expectEqualSlices(Id, a.slots, b.slots);
    try std.testing.expectEqualSlices(Line, a.line, b.line);
    try std.testing.expectEqualSlices(u32, a.stamp, b.stamp);
    try std.testing.expectEqualSlices(i32, a.where, b.where);
    try std.testing.expectEqualSlices(bool, a.flags, b.flags);
    try std.testing.expectEqualSlices(u64, a.resident, b.resident);
}

/// resident[key] != 0 exactly for the keys on a line, and then it is that line's address.
fn residentMatches(c: *const Cache) !void {
    for (c.resident, 0..) |r, key| {
        if (onDevice(c.vram, @intCast(key))) {
            try std.testing.expectEqual(c.line_addr[@intCast(c.where[key])], r);
        } else try std.testing.expectEqual(@as(u64, 0), r);
    }
}

test "the vector scan of the oldest line is the scalar rule: LRU and SLRU, ties, in-flight lines and stamps at or past now" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    for ([_]usize{ 1, 7, 16, 17, 100, 333 }) |lines| {
        for ([_]bool{ false, true }) |segments| {
            const slots = try std.testing.allocator.alloc(Id, lines);
            defer std.testing.allocator.free(slots);
            const line = try std.testing.allocator.alloc(Line, lines);
            defer std.testing.allocator.free(line);
            const stamp = try std.testing.allocator.alloc(u32, lines);
            defer std.testing.allocator.free(stamp);
            const where = try std.testing.allocator.alloc(i32, lines);
            defer std.testing.allocator.free(where);
            const flags = try std.testing.allocator.alloc(bool, lines);
            defer std.testing.allocator.free(flags);
            var v = Vram.init(slots, line, stamp, where);
            if (segments) v.enableSegments(flags, lines / 3);
            for (0..200) |_| {
                for (line, stamp, flags) |*l, *s, *f| {
                    l.* = switch (rnd.uintLessThan(u8, 5)) {
                        0 => .empty,
                        1 => .inflight,
                        else => .resident,
                    };
                    s.* = rnd.uintLessThan(u32, 12);
                    f.* = rnd.boolean();
                }
                const now = rnd.uintLessThan(u32, 14);
                try std.testing.expectEqual(lruScalar(v, now), lru(v, now));
            }
        }
    }
}

test "a job is a hit when the card passed it by itself, a miss when it asked and waits, and pending before either" {
    try std.testing.expectEqual(Seen.pending, classify(5, 4, 4));
    try std.testing.expectEqual(Seen.miss, classify(5, 4, 5));
    try std.testing.expectEqual(Seen.hit, classify(5, 5, 4));
    try std.testing.expectEqual(Seen.hit, classify(5, 4, 6)); // a later job asked: this one was passed
    try std.testing.expectEqual(Seen.hit, classify(5, 9, 3));
    // across the counter's wrap
    const top: u32 = std.math.maxInt(u32);
    try std.testing.expectEqual(Seen.miss, classify(top, top - 1, top));
    try std.testing.expectEqual(Seen.hit, classify(top, 1, 0));
    try std.testing.expectEqual(Seen.miss, classify(2, top, 2));
    try std.testing.expectEqual(Seen.pending, classify(2, 1, top));
}

test "replaying a hit stage is what a stage whose keys are all resident does: touches, no copies, the same state" {
    const gpa = std.testing.allocator;
    var a = try testCache(gpa, 4, 16, 20, 2, 25);
    defer gpa.free(a.resident);
    defer gpa.free(a.addr);
    defer a.c.deinit();
    var b = try testCache(gpa, 4, 16, 20, 2, 25);
    defer gpa.free(b.resident);
    defer gpa.free(b.addr);
    defer b.c.deinit();
    var s: Synth = .{ .prng = .init(7), .layers = 4, .experts = 16, .per = 8 };
    var keys: [32]Id = undefined;
    for (0..300) |_| {
        s.token(&keys);
        a.c.beginRound();
        b.c.beginRound();
        for (0..4) |layer| {
            const ids = keys[layer * 8 ..][0..8];
            @memcpy(a.c.ids[0..8], ids);
            if (a.c.allResident(ids)) {
                @memcpy(b.c.ids[0..8], ids);
                const d = try b.c.decide(8);
                try std.testing.expectEqual(@as(usize, 0), d.started);
                try std.testing.expectEqual(@as(usize, 8), d.hits);
                try a.c.replay(ids);
                try sameCache(&a.c, &b.c);
            } else {
                @memcpy(b.c.ids[0..8], ids);
                const x = try a.c.decide(8);
                try a.c.commit(x.started);
                const y = try b.c.decide(8);
                try b.c.commit(y.started);
                try std.testing.expectEqual(x.started, y.started);
            }
            try residentMatches(&a.c);
        }
    }
    const cold: Id = @intCast(std.mem.indexOfScalar(u64, a.resident, 0).?);
    try std.testing.expectError(error.NotResident, a.c.replay(&.{cold}));
}

/// The always-handshake run (every layer decided at once) and the split run (the hits late, in order, before the next miss and at the round's end)
/// reach the same pool state after every round, whatever the policy and the budget.
fn splitMatchesHandshake(layers: usize, experts: usize, lines: usize, copies: usize, protect: u8, seed: u64, tokens: usize) !void {
    const gpa = std.testing.allocator;
    const per = 8;
    var base = try testCache(gpa, layers, experts, lines, copies, protect);
    defer gpa.free(base.resident);
    defer gpa.free(base.addr);
    defer base.c.deinit();
    var split = try testCache(gpa, layers, experts, lines, copies, protect);
    defer gpa.free(split.resident);
    defer gpa.free(split.addr);
    defer split.c.deinit();
    var s: Synth = .{ .prng = .init(seed), .layers = layers, .experts = experts, .per = per };
    const keys = try gpa.alloc(Id, layers * per);
    defer gpa.free(keys);
    var late: [64]usize = undefined;
    var hit_layers: usize = 0;
    var all: usize = 0;
    for (0..tokens) |_| {
        s.token(keys);
        base.c.beginRound();
        split.c.beginRound();
        var nlate: usize = 0;
        for (0..layers) |layer| {
            const ids = keys[layer * per ..][0..per];
            @memcpy(base.c.ids[0..per], ids);
            const x = try base.c.decide(per);
            try base.c.commit(x.started);
            all += 1;
            if (split.c.allResident(ids)) {
                // the card passes the layer by itself: the host replays it later
                late[nlate] = layer;
                nlate += 1;
                hit_layers += 1;
                // what the baseline decided agrees: nothing started
                try std.testing.expectEqual(@as(usize, 0), x.started);
            } else {
                for (late[0..nlate]) |l| try split.c.replay(keys[l * per ..][0..per]);
                nlate = 0;
                @memcpy(split.c.ids[0..per], ids);
                const y = try split.c.decide(per);
                try split.c.commit(y.started);
                try std.testing.expectEqual(x.started, y.started);
            }
        }
        for (late[0..nlate]) |l| try split.c.replay(keys[l * per ..][0..per]);
        try sameCache(&base.c, &split.c);
        try residentMatches(&split.c);
    }
    // the trace does produce both kinds of stage
    try std.testing.expect(hit_layers > 0 and hit_layers < all);
}

test "the split and the always-handshake run reach the same pool state after every round (LRU and SLRU, copy budgets, small and large pools)" {
    try splitMatchesHandshake(8, 32, 64, 2, 0, 1, 400);
    try splitMatchesHandshake(8, 32, 64, 2, 25, 2, 400);
    try splitMatchesHandshake(8, 32, 96, 8, 0, 3, 400);
    try splitMatchesHandshake(8, 32, 40, 1, 40, 4, 400);
    try splitMatchesHandshake(16, 64, 200, 3, 20, 5, 300);
}

test "the card's words and the host's replay and serve, on two threads, give the same pool state as the handshake every layer" {
    const gpa = std.testing.allocator;
    const layers = 8;
    const experts = 32;
    const per = 8;
    const tokens = 300;
    var base = try testCache(gpa, layers, experts, 64, 2, 25);
    defer gpa.free(base.resident);
    defer gpa.free(base.addr);
    defer base.c.deinit();
    var live = try testCache(gpa, layers, experts, 64, 2, 25);
    defer gpa.free(live.resident);
    defer gpa.free(live.addr);
    defer live.c.deinit();

    const Link = struct {
        seen: std.atomic.Value(u32) = .init(0),
        req: std.atomic.Value(u32) = .init(0),
        ack: std.atomic.Value(u32) = .init(0),
        published: std.atomic.Value(u32) = .init(0),
        done: std.atomic.Value(u32) = .init(0),
        stop: std.atomic.Value(bool) = .init(false),
        trace: [layers][per]Id = undefined,
        job_layer: [tokens * layers + 1]u8 = undefined,
        misses: u32 = 0,
        hits: u32 = 0,

        fn worker(l: *@This(), c: *Cache) void {
            while (!l.stop.load(.acquire)) {
                const n = l.done.load(.monotonic) +% 1;
                if (!reached(l.published.load(.acquire), n)) {
                    std.atomic.spinLoopHint();
                    continue;
                }
                const layer = l.job_layer[n];
                switch (classify(n, l.seen.load(.acquire), l.req.load(.acquire))) {
                    .pending => {
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    .hit => {
                        c.replay(&l.trace[layer]) catch @panic("hit job with a key that is not resident");
                        l.hits += 1;
                    },
                    .miss => {
                        @memcpy(c.ids[0..per], &l.trace[layer]);
                        const d = c.decide(per) catch @panic("decide");
                        c.commit(d.started) catch @panic("commit");
                        l.ack.store(n, .release);
                        l.misses += 1;
                    },
                }
                l.done.store(n, .release);
            }
        }
    };
    var link: Link = .{};
    const t = try std.Thread.spawn(.{}, Link.worker, .{ &link, &live.c });

    var s: Synth = .{ .prng = .init(11), .layers = layers, .experts = experts, .per = per };
    const keys = try gpa.alloc(Id, layers * per);
    defer gpa.free(keys);
    var n: u32 = 0;
    for (0..tokens) |_| {
        s.token(keys);
        // the host's endRound/beginRound: the worker has answered everything posted before the card starts a round
        while (link.done.load(.acquire) != n) std.atomic.spinLoopHint();
        live.c.beginRound();
        base.c.beginRound();
        for (0..layers) |layer| {
            const ids = keys[layer * per ..][0..per];
            @memcpy(base.c.ids[0..per], ids);
            const x = try base.c.decide(per);
            try base.c.commit(x.started);
            // the card: trace the picks, then check them against the table the host maintains
            n += 1;
            link.job_layer[n] = @intCast(layer);
            link.published.store(n, .release);
            @memcpy(&link.trace[layer], ids);
            if (live.c.allResident(ids)) {
                link.seen.store(n, .release);
            } else {
                link.req.store(n, .release);
                while (!reached(link.ack.load(.acquire), n)) std.atomic.spinLoopHint();
            }
        }
    }
    while (link.done.load(.acquire) != n) std.atomic.spinLoopHint();
    link.stop.store(true, .release);
    t.join();
    try sameCache(&base.c, &live.c);
    try std.testing.expect(link.hits > 0 and link.misses > 0);
    try std.testing.expectEqual(@as(u32, tokens * layers), link.hits + link.misses);
}
