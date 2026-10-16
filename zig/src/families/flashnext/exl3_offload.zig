//! Host expert-cache policy for Flash Next EXL3 offload.
//! Routed experts stay in host RAM. A VRAM line is a copy that is in flight until
//! `reap`, then resident. Eviction is LRU among resident lines; an in-flight line
//! is not reusable until it is reaped. `planRound` does not write either table.
//! Shared expert 512 is always a hit and never occupies a routed slot.

const std = @import("std");
const shape = @import("exl3_shape.zig");

const shared_id: u16 = @intCast(shape.shared_expert);

pub const Decision = enum { hit, copy, cpu };

pub const Policy = struct {
    slots: u16,
    expert_bytes: u64,
    /// Bytes the copy engine may move this round. copy_count = expert_bytes==0 ? 0 : copy_bytes / expert_bytes.
    copy_bytes: u64,
    swap_every: u16 = 4,
    swap_cap: u16 = 96,
};

pub const Cache = struct {
    slots: []u16, // expert id, or empty_id
    freq: []u32, // frequency of the resident id, parallel to slots
    pub const empty_id: u16 = 0xffff;
};

/// Clear every slot to `empty_id` and every frequency to 0. Lengths must match.
pub fn init(slots: []u16, freq: []u32) Cache {
    std.debug.assert(slots.len == freq.len);
    @memset(slots, Cache.empty_id);
    @memset(freq, 0);
    return .{ .slots = slots, .freq = freq };
}

/// Slot index of a resident expert, or null. `empty_id` is a vacant line, not an expert.
pub fn find(c: Cache, id: u16) ?usize {
    if (id == Cache.empty_id) return null;
    return indexOf(c.slots, id);
}

/// One decision per id. Does not write the slot table.
/// Unique misses sort by count descending, then id ascending. The first
/// `copy_bytes / expert_bytes` of those (0 when `expert_bytes` is 0) are `copy`;
/// the rest are `cpu`. Duplicate ids in `ids` share one decision.
pub fn planRound(
    c: Cache,
    ids: []const u16,
    policy: Policy,
    decisions: []Decision,
    scratch: []u16,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;
    std.debug.assert(c.slots.len == c.freq.len);

    var n: usize = 0;
    for (ids) |id| {
        if (id == shared_id) continue;
        if (find(c, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], ids, hotterFirst);

    const copies = copyCount(policy, n);
    for (ids, decisions) |id, *decision| {
        if (id == shared_id or find(c, id) != null) {
            decision.* = .hit;
            continue;
        }
        const rank = indexOf(scratch[0..n], id).?;
        decision.* = if (rank < copies) .copy else .cpu;
    }
}

/// Install on swap rounds only (`round > 0` and `round % swap_every == 0`).
/// Residents named in `requested` are heated before victims are chosen.
/// Candidates are unique non-resident, non-shared ids, hottest first, at most
/// `swap_cap`. Empty lines are filled before any eviction. A line placed earlier
/// in this call is not a victim of a later candidate. Returns how many experts
/// were installed.
pub fn swap(
    c: *Cache,
    requested: []const u16,
    round: u32,
    policy: Policy,
    scratch: []u16,
) error{Scratch}!u16 {
    if (policy.swap_every == 0 or round == 0 or round % @as(u32, policy.swap_every) != 0) return 0;
    std.debug.assert(c.slots.len == c.freq.len);

    var n: usize = 0;
    for (requested) |id| {
        if (id == shared_id) continue;
        if (find(c.*, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], requested, hotterFirst);
    heatResidents(c, requested);

    const cap = @min(n, @as(usize, policy.swap_cap));
    var installed: usize = 0;
    for (scratch[0..cap]) |id| {
        const slot = indexOf(c.slots, Cache.empty_id) orelse victim(c.*, scratch[0..installed]) orelse break;
        c.slots[slot] = id;
        c.freq[slot] = appearances(requested, id);
        installed += 1;
    }
    return @intCast(installed);
}

/// Fill slots from a routed histogram of length 512. Hottest count, then lower id.
/// `freq[slot] = histogram[id]`. Id 512 is not a histogram entry and is not stored.
pub fn installProfile(c: *Cache, histogram: []const u32) void {
    std.debug.assert(histogram.len == shape.experts);
    std.debug.assert(c.slots.len == c.freq.len);
    @memset(c.slots, Cache.empty_id);
    @memset(c.freq, 0);

    const n = @min(c.slots.len, histogram.len);
    var filled: usize = 0;
    while (filled < n) : (filled += 1) {
        var best: u16 = 0;
        var best_n: u32 = 0;
        var any = false;
        for (histogram, 0..) |count, i| {
            const id: u16 = @intCast(i);
            if (indexOf(c.slots[0..filled], id) != null) continue;
            // Scan is id-ascending, so the first id that reaches the max count wins ties.
            if (!any or count > best_n) {
                best = id;
                best_n = count;
                any = true;
            }
        }
        if (!any) break;
        c.slots[filled] = best;
        c.freq[filled] = best_n;
    }
}

fn copyCount(policy: Policy, unique_misses: usize) usize {
    if (policy.expert_bytes == 0 or unique_misses == 0) return 0;
    const raw = policy.copy_bytes / policy.expert_bytes;
    if (raw >= unique_misses) return unique_misses;
    return @intCast(raw);
}

fn appearances(ids: []const u16, id: u16) u32 {
    var n: u32 = 0;
    for (ids) |x| {
        if (x == id) n +|= 1;
    }
    return n;
}

/// Count descending, id ascending. `ctx` is the multiset the counts are taken from.
fn hotterFirst(ctx: []const u16, a: u16, b: u16) bool {
    const ca = appearances(ctx, a);
    const cb = appearances(ctx, b);
    if (ca != cb) return ca > cb;
    return a < b;
}

fn indexOf(ids: []const u16, id: u16) ?usize {
    for (ids, 0..) |x, i| {
        if (x == id) return i;
    }
    return null;
}

fn pushUnique(dst: []u16, n: *usize, id: u16) error{Scratch}!void {
    if (indexOf(dst[0..n.*], id) != null) return;
    if (n.* >= dst.len) return error.Scratch;
    dst[n.*] = id;
    n.* += 1;
}

fn heatResidents(c: *Cache, requested: []const u16) void {
    for (c.slots, 0..) |id, i| {
        if (id == Cache.empty_id) continue;
        const n = appearances(requested, id);
        if (n == 0) continue;
        c.freq[i] = c.freq[i] +| n;
    }
}

/// Lowest frequency, skipping lines already placed in this call.
/// Equal frequency evicts the higher slot index.
fn victim(c: Cache, skip: []const u16) ?usize {
    var best: ?usize = null;
    for (c.slots, 0..) |id, i| {
        if (id == Cache.empty_id) continue;
        if (indexOf(skip, id) != null) continue;
        if (best) |b| {
            if (c.freq[i] < c.freq[b] or (c.freq[i] == c.freq[b] and i > b)) best = i;
        } else best = i;
    }
    return best;
}

pub const Line = enum { empty, inflight, resident };

/// VRAM working set. Host RAM still holds every routed expert.
/// `stamp` is the last reap or touch time. A higher stamp is newer.
pub const Vram = struct {
    slots: []u16,
    line: []Line,
    stamp: []u32,

    pub fn init(slots: []u16, line: []Line, stamp: []u32) Vram {
        std.debug.assert(slots.len == line.len and slots.len == stamp.len);
        @memset(slots, Cache.empty_id);
        @memset(line, .empty);
        @memset(stamp, 0);
        return .{ .slots = slots, .line = line, .stamp = stamp };
    }
};

/// True once `id` has been reaped into VRAM.
pub fn onDevice(v: Vram, id: u16) bool {
    const i = vramIndex(v, id) orelse return false;
    return v.line[i] == .resident;
}

/// A resident hit. Moves that line to `now` so LRU keeps it.
pub fn touch(v: *Vram, id: u16, now: u32) bool {
    const i = vramIndex(v.*, id) orelse return false;
    if (v.line[i] != .resident) return false;
    v.stamp[i] = now;
    return true;
}

/// Start a host-to-VRAM copy of `id`. An empty line is used first.
/// Otherwise the resident line with the oldest stamp is replaced.
/// An in-flight line is left alone. `error.Busy` means every line is still in flight.
pub fn stage(v: *Vram, id: u16) error{ Busy, Shared }!usize {
    return stageKeeping(v, id, &.{});
}

/// Same as `stage`, except a resident whose id is in `keep` is not a victim.
/// `error.Busy` means every line is in flight or held by `keep`.
pub fn stageKeeping(v: *Vram, id: u16, keep: []const u16) error{ Busy, Shared }!usize {
    if (id == shared_id or id == Cache.empty_id) return error.Shared;
    if (vramIndex(v.*, id)) |i| {
        if (v.line[i] == .inflight) return i;
        v.line[i] = .inflight;
        return i;
    }
    const slot = vramEmpty(v.*) orelse vramLruKeep(v.*, keep) orelse return error.Busy;
    v.slots[slot] = id;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    return slot;
}

/// One step of the VRAM cache. Does not reap.
/// A resident id is touched and decided `hit`. An in-flight id is `cpu` until reap.
/// Unique misses, hottest first, are staged until the copy budget runs out or every
/// free line is gone. A staged miss is `copy` and stays off-device until `reap`.
/// Residents named in `ids` are not eviction victims. Shared id 512 is `hit`.
/// A zero `expert_bytes` stages nothing: every miss stays `cpu`.
pub fn schedule(
    v: *Vram,
    ids: []const u16,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: []u16,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;

    var n: usize = 0;
    for (ids, decisions) |id, *decision| {
        if (id == shared_id) {
            decision.* = .hit;
            continue;
        }
        if (onDevice(v.*, id)) {
            _ = touch(v, id, now);
            decision.* = .hit;
            continue;
        }
        decision.* = .cpu;
        if (vramIndex(v.*, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], ids, hotterFirst);

    const copies = copyCount(policy, n);
    var staged: usize = 0;
    for (scratch[0..n]) |id| {
        if (staged >= copies) break;
        _ = stageKeeping(v, id, ids) catch break;
        staged += 1;
        for (ids, decisions) |asked, *decision| {
            if (asked == id) decision.* = .copy;
        }
    }
}

/// The copy of `id` has landed. The line becomes resident at `now`.
pub fn reap(v: *Vram, id: u16, now: u32) error{NotInflight}!void {
    const i = vramIndex(v.*, id) orelse return error.NotInflight;
    if (v.line[i] != .inflight) return error.NotInflight;
    v.line[i] = .resident;
    v.stamp[i] = now;
}

fn vramIndex(v: Vram, id: u16) ?usize {
    if (id == Cache.empty_id) return null;
    for (v.slots, 0..) |slot, i| {
        if (slot == id and v.line[i] != .empty) return i;
    }
    return null;
}

fn vramEmpty(v: Vram) ?usize {
    for (v.line, 0..) |state, i| if (state == .empty) return i;
    return null;
}

/// Oldest resident stamp. Equal stamps take the higher slot index.
fn vramLru(v: Vram) ?usize {
    return vramLruKeep(v, &.{});
}

/// Oldest resident whose id is not in `keep`. Equal stamps take the higher slot index.
fn vramLruKeep(v: Vram, keep: []const u16) ?usize {
    var best: ?usize = null;
    for (v.line, 0..) |state, i| {
        if (state != .resident) continue;
        if (indexOf(keep, v.slots[i]) != null) continue;
        if (best) |b| {
            if (v.stamp[i] < v.stamp[b] or (v.stamp[i] == v.stamp[b] and i > b)) best = i;
        } else best = i;
    }
    return best;
}

fn countOf(counts: []const u32, id: u16) u32 {
    if (id >= counts.len) return 0;
    return counts[id];
}

/// Lowest lifetime count, then oldest stamp, then the higher slot index.
/// Equal counts are the stamp LRU, so a zeroed table matches `vramLruKeep`.
fn vramCold(v: Vram, counts: []const u32, keep: []const u16) ?usize {
    var best: ?usize = null;
    for (v.line, 0..) |state, i| {
        if (state != .resident) continue;
        if (indexOf(keep, v.slots[i]) != null) continue;
        if (best) |b| {
            const ci = countOf(counts, v.slots[i]);
            const cb = countOf(counts, v.slots[b]);
            const colder = ci < cb or
                (ci == cb and v.stamp[i] < v.stamp[b]) or
                (ci == cb and v.stamp[i] == v.stamp[b] and i > b);
            if (colder) best = i;
        } else best = i;
    }
    return best;
}

const UsedOrder = struct {
    counts: []const u32,
    ids: []const u16,
};

/// Lifetime count descending, then this round's appearances, then lower id.
fn usedFirst(ctx: UsedOrder, a: u16, b: u16) bool {
    const ca = countOf(ctx.counts, a);
    const cb = countOf(ctx.counts, b);
    if (ca != cb) return ca > cb;
    const aa = appearances(ctx.ids, a);
    const ab = appearances(ctx.ids, b);
    if (aa != ab) return aa > ab;
    return a < b;
}

/// Add one to each routed id in `ids`. Shared id 512 is not a histogram entry.
/// `counts` is one cell per routed expert (`shape.experts`).
pub fn observe(counts: []u32, ids: []const u16) error{Length}!void {
    if (counts.len != shape.experts) return error.Length;
    for (ids) |id| {
        if (id >= shape.experts) continue;
        counts[id] +|= 1;
    }
}

/// Expert ids this rank stores. Rank 0 is `0..256`, rank 1 is `256..512`.
pub const Owned = struct { lo: u16, hi: u16 };

pub fn allOwned() Owned {
    return .{ .lo = 0, .hi = @intCast(shape.experts) };
}

fn inOwned(owned: Owned, id: u16) bool {
    return id >= owned.lo and id < owned.hi;
}

/// Fill empty VRAM lines with the hottest counted ids. No eviction.
/// Ties take the lower id. Count 0 and ids already in the table are skipped.
/// Staged lines stay in flight until `reap`.
pub fn seedHottest(v: *Vram, counts: []const u32, limit: u16) error{Length}!u16 {
    return seedSpan(v, counts, limit, allOwned());
}

fn seedSpan(v: *Vram, counts: []const u32, limit: u16, owned: Owned) error{Length}!u16 {
    if (counts.len != shape.experts) return error.Length;
    var seeded: u16 = 0;
    while (seeded < limit) {
        if (vramEmpty(v.*) == null) break;
        var best: ?u16 = null;
        var best_n: u32 = 0;
        var id: u16 = owned.lo;
        while (id < owned.hi) : (id += 1) {
            const count = counts[id];
            if (count == 0) continue;
            if (vramIndex(v.*, id) != null) continue;
            if (best == null or count > best_n) {
                best = id;
                best_n = count;
            }
        }
        if (best == null) break;
        _ = stage(v, best.?) catch break;
        seeded += 1;
    }
    return seeded;
}

/// Use every empty line. Hottest owned ids go in first. Lines that are still
/// empty take the remaining owned ids, low id first, so the card is not left
/// idle while the histogram warms up. No eviction. The other rank's ids are skipped.
pub fn fillOwned(v: *Vram, counts: []const u32, owned: Owned) error{Length}!u16 {
    if (owned.hi > shape.experts or owned.lo > owned.hi) return error.Length;
    var n: u16 = try seedSpan(v, counts, std.math.maxInt(u16), owned);
    var id: u16 = owned.lo;
    while (id < owned.hi) : (id += 1) {
        if (vramEmpty(v.*) == null) break;
        if (vramIndex(v.*, id) != null) continue;
        _ = stage(v, id) catch break;
        n += 1;
    }
    return n;
}

/// `stageKeeping`, but the victim is the coldest lifetime count.
pub fn stageUsed(v: *Vram, id: u16, keep: []const u16, counts: []const u32) error{ Busy, Shared, Length }!usize {
    if (counts.len != shape.experts) return error.Length;
    if (id == shared_id or id == Cache.empty_id) return error.Shared;
    if (vramIndex(v.*, id)) |i| {
        if (v.line[i] == .inflight) return i;
        v.line[i] = .inflight;
        return i;
    }
    const slot = vramEmpty(v.*) orelse vramCold(v.*, counts, keep) orelse return error.Busy;
    v.slots[slot] = id;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    return slot;
}

/// `schedule`, ordered by lifetime counts from `observe`.
/// A high count keeps its line even when its stamp is older.
/// Residents named in `ids` are not victims. `schedule` itself is unchanged.
pub fn scheduleUsed(
    v: *Vram,
    ids: []const u16,
    counts: []const u32,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: []u16,
    owned: Owned,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;
    if (counts.len != shape.experts) return error.Length;

    var n: usize = 0;
    for (ids, decisions) |id, *decision| {
        if (id == shared_id) {
            decision.* = .hit;
            continue;
        }
        if (onDevice(v.*, id)) {
            _ = touch(v, id, now);
            decision.* = .hit;
            continue;
        }
        decision.* = .cpu;
        if (!inOwned(owned, id)) continue;
        if (vramIndex(v.*, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], UsedOrder{ .counts = counts, .ids = ids }, usedFirst);

    const copies = copyCount(policy, n);
    var staged: usize = 0;
    for (scratch[0..n]) |id| {
        if (staged >= copies) break;
        _ = stageUsed(v, id, ids, counts) catch |err| switch (err) {
            error.Length => return error.Length,
            error.Busy, error.Shared => break,
        };
        staged += 1;
        for (ids, decisions) |asked, *decision| {
            if (asked == id) decision.* = .copy;
        }
    }
}

fn benchPolicy(slot_n: u16, expert_bytes: u64, copy_bytes: u64) Policy {
    return .{
        .slots = slot_n,
        .expert_bytes = expert_bytes,
        .copy_bytes = copy_bytes,
    };
}

test "a staged copy is not on device until reap, and LRU drops the older resident" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);

    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 4));
    try std.testing.expectEqual(@as(usize, 1), try stage(&vram, 7));
    try std.testing.expect(!onDevice(vram, 4));
    try std.testing.expectError(error.NotInflight, reap(&vram, 9, 1));
    try reap(&vram, 4, 1);
    try reap(&vram, 7, 2);
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expect(touch(&vram, 7, 5));
    try std.testing.expect(onDevice(vram, 4));

    // 4 was reaped at 1 and not touched again. 7 was touched at 5. 4 is the victim.
    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 8));
    try std.testing.expect(!onDevice(vram, 8));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[1]);
    try reap(&vram, 8, 6);
    try std.testing.expect(onDevice(vram, 8));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 4));
}

test "an in-flight line is not reused, and a full in-flight set is busy" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try std.testing.expectError(error.Busy, stage(&vram, 3));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 2), slots[1]);
    try std.testing.expectError(error.Shared, stage(&vram, shared_id));
}

test "schedule touches residents, stages the hottest miss, and will not evict a hit" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    var decisions: [4]Decision = undefined;
    var scratch: [4]u16 = undefined;
    const policy = benchPolicy(2, 100, 100);

    const first = [_]u16{ 7, shared_id, 4, 7 };
    try schedule(&vram, &first, policy, 3, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[3]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(!onDevice(vram, 7));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(Line.empty, line[1]);

    try reap(&vram, 7, 4);
    const held = [_]u16{7};
    try schedule(&vram, &held, policy, 9, decisions[0..1], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(@as(u32, 9), stamp[0]);

    // Both lines are residents this step asked for, so the miss cannot take either.
    _ = try stage(&vram, 3);
    try reap(&vram, 3, 10);
    const protected = [_]u16{ 7, 3, 8 };
    try schedule(&vram, &protected, benchPolicy(2, 100, 10_000), 11, decisions[0..3], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(onDevice(vram, 3));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
}

test "a zero copy budget stages nothing, and an older resident is the one that moves" {
    var slots = [_]u16{ 4, 7 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 5 };
    var vram = Vram.init(&slots, &line, &stamp);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try schedule(&vram, &miss, benchPolicy(2, 0, 10_000), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);

    try schedule(&vram, &miss, benchPolicy(2, 100, 100), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(!onDevice(vram, 9));
}

test "resident ids are hits and planRound does not change slots" {
    var slots = [_]u16{ 4, 1, 9 };
    var freq = [_]u32{ 3, 8, 1 };
    const cache = init(&slots, &freq);
    slots = .{ 4, 1, 9 };
    freq = .{ 3, 8, 1 };
    const before_slots = slots;
    const before_freq = freq;

    const ids = [_]u16{ 4, shared_id, 1, 9, 4 };
    var decisions: [5]Decision = undefined;
    try planRound(cache, &ids, benchPolicy(3, 100, 10_000), &decisions, &[_]u16{});

    for (decisions) |d| try std.testing.expectEqual(Decision.hit, d);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
    try std.testing.expectEqual(@as(?usize, 1), find(cache, 1));
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
}

test "misses past the copy budget are cpu and slots stay put" {
    var slots = [_]u16{42};
    var freq = [_]u32{6};
    const cache = init(&slots, &freq);
    slots[0] = 42;
    freq[0] = 6;
    const before_slots = slots;
    const before_freq = freq;

    // Counts: 5 x4, 1 x3, 8 x1, 3 x1. Budget 250/100 = 2, so 5 and 1 are copied.
    const ids = [_]u16{ 8, 5, 1, 5, 3, 5, 1, 42, 5, 1 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [8]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(1, 100, 250), &decisions, &scratch);

    const want = [_]Decision{ .cpu, .copy, .copy, .copy, .cpu, .copy, .copy, .hit, .copy, .copy };
    try std.testing.expectEqualSlices(Decision, &want, &decisions);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
}

test "duplicate miss ids share one decision" {
    var slots: [1]u16 = undefined;
    var freq: [1]u32 = undefined;
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 6, 2, 6 };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    // One copy. 6 appears twice, 2 once, so both 6s are copy and 2 is cpu.
    try planRound(cache, &ids, benchPolicy(1, 100, 100), &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
}

test "equal miss counts break ties toward the lower id" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    // 10 x5, 6 x4, 2 x4. One copy would be 10; two copies are 10 and 2, not 6.
    const ids = [_]u16{ 6, 10, 2, 6, 10, 2, 10, 6, 2, 10, 6, 10, 2 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 50, 100), &decisions, &scratch);
    for (ids, decisions) |id, d| {
        const want: Decision = if (id == 10 or id == 2) .copy else .cpu;
        try std.testing.expectEqual(want, d);
    }
}

test "shared id is a hit and is never stored" {
    var slots: [4]u16 = @splat(Cache.empty_id);
    var freq: [4]u32 = @splat(0);
    var cache = init(&slots, &freq);
    slots[0] = 7;
    freq[0] = 2;

    const ids = [_]u16{ shared_id, 4, shared_id };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(4, 100, 0), &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.hit, decisions[2]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);

    const requested = [_]u16{ shared_id, 4, shared_id, 9 };
    const installed = try swap(&cache, &requested, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), installed);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);

    var hist: [shape.experts]u32 = @splat(0);
    hist[4] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);
}

test "round 2 installs nothing and round 4 installs min of cap and missing" {
    var slots: [8]u16 = @splat(Cache.empty_id);
    var freq: [8]u32 = @splat(0);
    var cache = init(&slots, &freq);
    const requested = [_]u16{ 3, 1, 3, 8 };
    var scratch: [8]u16 = undefined;
    var policy = benchPolicy(8, 1, 0);
    policy.swap_every = 4;

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 0, policy, &scratch));
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 2, policy, &scratch));
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[0]);

    // Three unique misses, under the cap.
    const n = try swap(&cache, &requested, 4, policy, &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual(@as(u16, 3), slots[0]);
    try std.testing.expectEqual(@as(u32, 2), freq[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[1]);
    try std.testing.expectEqual(@as(u16, 8), slots[2]);
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[3]);
}

test "a cold line is replaced by the missing id" {
    var slots = [_]u16{41};
    var freq = [_]u32{0};
    var cache = init(&slots, &freq);
    slots[0] = 41;
    freq[0] = 0;
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{9};
    const n = try swap(&cache, &requested, 4, benchPolicy(1, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(?usize, null), find(cache, 41));
}

test "eviction ties take the later slot" {
    var slots = [_]u16{ 30, 10, 20 };
    var freq = [_]u32{ 4, 4, 4 };
    var cache = init(&slots, &freq);
    slots = .{ 30, 10, 20 };
    freq = .{ 4, 4, 4 };
    var scratch: [2]u16 = undefined;
    const n = try swap(&cache, &[_]u16{7}, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    // Higher id would have dropped 30. Higher slot index drops 20.
    try std.testing.expectEqual([_]u16{ 30, 10, 7 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[2]);
}

test "a resident named this round is heated before victims are chosen" {
    var slots = [_]u16{ 2, 1 };
    var freq = [_]u32{ 0, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 2, 1 };
    freq = .{ 0, 0 };
    var scratch: [2]u16 = undefined;
    // Without the heat, both freqs stay 0 and the later slot (id 1) would be the victim.
    const n = try swap(&cache, &[_]u16{ 1, 99 }, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual([_]u16{ 99, 1 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "resident frequency adds this round's request count" {
    var slots = [_]u16{ 5, Cache.empty_id };
    var freq = [_]u32{ 4, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 5, Cache.empty_id };
    freq = .{ 4, 0 };
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{ 5, 5, 5, 4 };
    const n = try swap(&cache, &requested, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 5), slots[0]);
    try std.testing.expectEqual(@as(u32, 7), freq[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[1]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "empty lines are filled before a cold resident is evicted" {
    var slots = [_]u16{ 30, Cache.empty_id, 40 };
    var freq = [_]u32{ 5, 0, 1 };
    var cache = init(&slots, &freq);
    slots = .{ 30, Cache.empty_id, 40 };
    freq = .{ 5, 0, 1 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 7, 8 }, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual([_]u16{ 30, 7, 8 }, slots);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}

test "a candidate placed this call is not evicted by a later one" {
    var slots = [_]u16{ 1, 2, 3, 4 };
    var freq = [_]u32{ 1, 50, 2, 40 };
    var cache = init(&slots, &freq);
    slots = .{ 1, 2, 3, 4 };
    freq = .{ 1, 50, 2, 40 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 10, 11, 12 }, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual([_]u16{ 10, 2, 11, 12 }, slots);
    try std.testing.expectEqual(@as(u32, 50), freq[1]);
}

test "installProfile keeps the hottest routed ids" {
    var slots: [3]u16 = @splat(shared_id);
    var freq = [_]u32{ 9, 9, 9 };
    var cache = init(&slots, &freq);
    var hist: [shape.experts]u32 = @splat(0);
    hist[10] = 1;
    hist[4] = 7;
    hist[8] = 7;
    hist[2] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual([_]u16{ 4, 8, 2 }, slots);
    try std.testing.expectEqual([_]u32{ 7, 7, 3 }, freq);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));

    var wide_slots: [513]u16 = undefined;
    var wide_freq: [513]u32 = undefined;
    var wide = init(&wide_slots, &wide_freq);
    var flat: [shape.experts]u32 = @splat(0);
    flat[0] = 1;
    installProfile(&wide, &flat);
    for (wide_slots[0..shape.experts], 0..) |id, i| {
        try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    }
    try std.testing.expectEqual(Cache.empty_id, wide_slots[512]);
    try std.testing.expectEqual(@as(u32, 1), wide_freq[0]);
    try std.testing.expectEqual(@as(u32, 0), wide_freq[1]);
    try std.testing.expectEqual(@as(?usize, null), find(wide, shared_id));
}

test "swap installs at most the cap even when slots and misses are larger" {
    var slots: [200]u16 = @splat(Cache.empty_id);
    var freq: [200]u32 = @splat(0);
    var cache = init(&slots, &freq);
    var requested: [200]u16 = undefined;
    for (&requested, 0..) |*id, i| id.* = @intCast(i);
    var scratch: [200]u16 = undefined;
    const n = try swap(&cache, &requested, 4, benchPolicy(200, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 96), n);
    for (slots[0..96], 0..) |id, i| try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    try std.testing.expectEqual(Cache.empty_id, slots[96]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[96]);
}

test "a high count keeps an old stamp and a one-off is the victim" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[1] = 50;
    counts[4] = 1;
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(!onDevice(vram, 9));
}

test "zero lifetime counts fall back to stamp LRU" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var counts: [shape.experts]u32 = @splat(0);
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
}

test "lifetime count beats this round's appearances" {
    var slots = [_]u16{Cache.empty_id};
    var line = [_]Line{.empty};
    var stamp = [_]u32{0};
    var vram = Vram.init(&slots, &line, &stamp);
    var counts: [shape.experts]u32 = @splat(0);
    counts[8] = 5;
    const ids = [_]u16{ 3, 3, 3, 3, 8 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(1, 100, 100), 1, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[4]);
    try std.testing.expectEqual(@as(u16, 8), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
}

test "a resident named this step is kept even when its count is low" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[4] = 1;
    const ids = [_]u16{ 1, 4, 9 };
    var decisions: [3]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 9));
}

test "observed counts seed empty lines and a later round keeps the hot id" {
    var counts: [shape.experts]u32 = @splat(0);
    try observe(&counts, &.{ 1, shared_id, 1, 3 });
    try std.testing.expectEqual(@as(u32, 2), counts[1]);
    try std.testing.expectEqual(@as(u32, 1), counts[3]);
    try observe(&counts, &.{1});
    try std.testing.expectEqual(@as(u32, 3), counts[1]);
    var short: [1]u32 = .{9};
    try std.testing.expectError(error.Length, observe(&short, &.{1}));
    try std.testing.expectEqual(@as(u32, 9), short[0]);

    var seeded_counts: [shape.experts]u32 = @splat(0);
    seeded_counts[3] = 10;
    seeded_counts[1] = 10;
    seeded_counts[8] = 2;
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    const n = try seedHottest(&vram, &seeded_counts, 2);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 3), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(!onDevice(vram, 1));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
    try reap(&vram, 1, 1);
    try std.testing.expect(onDevice(vram, 1));

    var wide: [3]u16 = .{ Cache.empty_id, Cache.empty_id, Cache.empty_id };
    var wide_line: [3]Line = .{ .empty, .empty, .empty };
    var wide_stamp: [3]u32 = .{ 0, 0, 0 };
    var wide_vram = Vram.init(&wide, &wide_line, &wide_stamp);
    try std.testing.expectError(error.Length, seedHottest(&wide_vram, counts[0..3], 1));
    try std.testing.expectEqual(Line.empty, wide_line[0]);
}

test "a rank fills its own lines and does not copy the other rank's expert" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp);
    var counts: [shape.experts]u32 = @splat(0);
    counts[400] = 100;
    counts[7] = 3;
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    const n = try fillOwned(&vram, &counts, rank0);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(@as(u16, 0), slots[1]);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try reap(&vram, 7, 1);
    try reap(&vram, 0, 1);

    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const asked = [_]u16{ 400, 9 };
    try scheduleUsed(&vram, &asked, &counts, benchPolicy(2, 100, 10_000), 4, &decisions, &scratch, rank0);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expect(vramIndex(vram, 9) != null);
}

test "an empty slot table misses every routed id and installs nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    var cache = init(&slots, &freq);
    const ids = [_]u16{ shared_id, 4, 4, 8 };
    var decisions: [4]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const policy = benchPolicy(0, 10, 10);
    try planRound(cache, &ids, policy, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(Decision.cpu, decisions[3]);

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &ids, 4, policy, &scratch));
    try std.testing.expectEqual(@as(usize, 0), slots.len);
}

test "zero expert bytes copies nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 1, 2 };
    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 0, 1000), &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
}

test "a short scratch or a short decision buffer is refused without writes" {
    var slots = [_]u16{1};
    var freq = [_]u32{5};
    var cache = init(&slots, &freq);
    slots[0] = 1;
    freq[0] = 5;

    var decisions: [1]Decision = undefined;
    try std.testing.expectError(error.Length, planRound(cache, &[_]u16{ 1, 2 }, benchPolicy(1, 1, 1), &decisions, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var one: [2]Decision = undefined;
    try std.testing.expectError(error.Scratch, planRound(cache, &[_]u16{ 2, 3 }, benchPolicy(1, 1, 1), &one, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);

    var tiny: [1]u16 = undefined;
    try std.testing.expectError(error.Scratch, swap(&cache, &[_]u16{ 2, 3 }, 4, benchPolicy(1, 1, 0), &tiny));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var off = benchPolicy(1, 1, 0);
    off.swap_every = 0;
    var scratch: [2]u16 = undefined;
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &[_]u16{2}, 4, off, &scratch));
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}
