//! The pool's elastic capacity: ids past `mapped` stay out, the tail drains first, and every count stays exact.
const std = @import("std");
const kv = @import("kv_pool.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const gpa = std.testing.allocator;

const most = 24; // pages a walk's pool holds at most

/// Every count as a scan of the pages below `mapped`; those past it hold nothing, and each free id is listed once.
fn scanned(p: *const kv.Pool) !void {
    var free: usize = 0;
    var kept: usize = 0;
    var leased: usize = 0;
    var reclaimable: usize = 0;
    var listed: [most]u8 = @splat(0);
    for (p.free[0..p.nfree]) |id| {
        try expect(id < p.mapped);
        listed[id] += 1;
    }
    for (p.refs, p.leased, 0..) |r, l, id| {
        if (id >= p.mapped) {
            try equal(@as(u32, 0), r);
            try equal(@as(u32, 0), l);
            continue;
        }
        try expect(r >= l);
        free += @intFromBool(r == 0);
        kept += @intFromBool(r > l);
        leased += @intFromBool(l > 0);
        reclaimable += @intFromBool(r == 1 and l == 0);
        try equal(@as(u8, @intFromBool(r == 0)), listed[id]);
    }
    try equal(free, p.nfree);
    try expect(p.promised <= free);
    if (p.lowest and p.nfree > 1) for (1..p.nfree) |i| try expect(p.free[(i - 1) / 2] < p.free[i]);
    try equal(kv.Census{ .total = p.mapped, .free = free, .promised = p.promised, .kept = kept, .leased = leased }, p.census());
    try equal(reclaimable, p.reclaimable());
    for (0..p.capacity() + 2) |need| try equal(free - p.promised + reclaimable >= need, p.fits(need));
    try equal(p.promised == 0 and leased == 0 and free + kept == p.mapped, p.idle());
}

/// The first id of `k` free pages in a row at a multiple of `step` below `mapped`, lowest or highest, by brute force.
fn modelRun(p: *const kv.Pool, k: usize, step: usize, lowest: bool) ?u32 {
    var found: ?u32 = null;
    var start: usize = 0;
    while (start + k <= p.mapped) : (start += step) {
        if (std.mem.allEqual(u32, p.refs[start..][0..k], 0)) {
            found = @intCast(start);
            if (lowest) break;
        }
    }
    return if (k == 0) null else found;
}

fn lowestFree(p: *const kv.Pool) ?u32 {
    for (p.refs[0..p.mapped], 0..) |r, id| if (r == 0) return @intCast(id);
    return null;
}

test "elastic: a pool with no options is the stack it was: page 0 first, the last freed first, runs from the top" {
    var p = try kv.Pool.init(gpa, 8, 64);
    defer p.deinit();
    try equal(@as(usize, 8), p.total());
    try equal(@as(usize, 8), p.capacity());
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 4);
    try equal(@as(u32, 0), try p.take(&l));
    try equal(@as(u32, 1), try p.take(&l));
    try equal(@as(u32, 2), try p.take(&l));
}

test "elastic: the default order frees last-in first-out and finds runs from the top" {
    var p = try kv.Pool.init(gpa, 8, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 4);
    for (0..4) |_| _ = try p.take(&l);
    p.finish(&l);
    try p.promise(&l, 2);
    try equal(@as(u32, 3), try p.take(&l)); // the pages went back in order 0..3: the last in is the first out
    try equal(@as(u32, 2), try p.take(&l));
    try equal(@as(?u32, 5), p.findRun(3)); // pages 2 and 3 are held: the highest window of three is 5..7
    try equal(@as(?u32, 6), p.findAligned(2, 2));
}

test "elastic: lowest-first hands out the lowest free id, singles and runs alike" {
    var p = try kv.Pool.initWith(gpa, 12, 64, .{ .lowest_first = true });
    defer p.deinit();
    var a: kv.Lease = .{};
    var b: kv.Lease = .{};
    defer p.finish(&a);
    defer p.finish(&b);
    try p.promise(&a, 8);
    for (0..6) |i| try equal(@as(u32, @intCast(i)), try p.take(&a));
    try p.promise(&b, 3);
    p.finish(&a); // the pages come back in any order
    try p.promise(&a, 3);
    try equal(@as(u32, 0), try p.take(&a));
    try equal(@as(u32, 1), try p.take(&a));
    try equal(@as(u32, 2), try p.take(&b));
    try equal(@as(?u32, 3), p.findRun(3)); // 0..2 held: the lowest window that fits
    try equal(@as(?u32, 3), p.findRun(2));
    try equal(@as(?u32, 4), p.findAligned(2, 2)); // 0 and 2 are held
    try equal(@as(?u32, 3), p.findAligned(3, 3));
    const run = p.claimRun(&b, 2).?;
    try equal(@as(u32, 3), run);
    try equal(@as(u32, 5), try p.take(&a)); // past the run
    try scanned(&p);
    p.releaseRun(run, 2);
    try scanned(&p);
}

test "elastic: pages past mapped are never promised, taken, found or counted by fits" {
    var p = try kv.Pool.initWith(gpa, 10, 64, .{ .mapped = 4 });
    defer p.deinit();
    try equal(@as(usize, 4), p.total());
    try equal(@as(usize, 10), p.capacity());
    try scanned(&p);
    try expect(p.fits(4) and !p.fits(5));
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try std.testing.expectError(error.LaneMemory, p.promise(&l, 5));
    try p.promise(&l, 4);
    var seen: [10]bool = @splat(false);
    for (0..4) |_| seen[try p.take(&l)] = true;
    try expect(std.mem.allEqual(bool, seen[0..4], true) and std.mem.allEqual(bool, seen[4..], false));
    try equal(@as(?u32, null), p.findRun(1));
    p.finish(&l);
    try equal(@as(?u32, 0), p.findRun(4));
    try equal(@as(?u32, null), p.findRun(5));
    try equal(@as(?u32, null), p.findAligned(5, 1));
    try scanned(&p);
}

test "elastic: shrink is Busy exactly while the tail is held or the promises would be stranded, and then changes nothing" {
    var p = try kv.Pool.initWith(gpa, 8, 64, .{ .lowest_first = true });
    defer p.deinit();
    var a: kv.Lease = .{};
    var b: kv.Lease = .{};
    defer p.finish(&a);
    defer p.finish(&b);
    try p.promise(&a, 3);
    for (0..3) |_| _ = try p.take(&a); // pages 0, 1, 2
    try expect(!p.tailIdle(2) and p.tailIdle(3) and p.tailIdle(8));
    const before = p.census();
    try std.testing.expectError(error.Busy, p.shrink(2));
    try equal(before, p.census());
    try equal(@as(usize, 8), p.total());
    try p.shrink(3); // the free tail goes: nothing left to take
    try equal(@as(usize, 3), p.total());
    try equal(@as(usize, 0), p.nfree);
    try scanned(&p);
    p.grow(8);
    try equal(@as(usize, 8), p.total());
    try scanned(&p);
    try p.promise(&b, 4); // 5 free, 4 promised: one free page may leave
    try expect(!p.tailIdle(4) and !p.tailIdle(6) and p.tailIdle(7));
    try std.testing.expectError(error.Busy, p.shrink(4));
    try p.shrink(7);
    try scanned(&p);
    for (0..4) |i| try equal(@as(u32, @intCast(3 + i)), try p.take(&b)); // the promise holds on what is left
    try std.testing.expectError(error.Busy, p.shrink(6)); // page 6 is b's now
    try expect(p.fits(0) and !p.fits(1));
}

test "elastic: a page the store holds keeps its place in the pool until the store lets go" {
    var p = try kv.Pool.initWith(gpa, 6, 64, .{ .lowest_first = true });
    defer p.deinit();
    var l: kv.Lease = .{};
    try p.promise(&l, 4);
    for (0..4) |_| _ = try p.take(&l);
    p.retain(3); // a kept state's copy of the stream's last page
    p.finish(&l);
    try equal(@as(usize, 1), p.reclaimable());
    try std.testing.expectError(error.Busy, p.shrink(3));
    try std.testing.expectError(error.Busy, p.shrink(0));
    p.release(3);
    try p.shrink(0);
    try equal(@as(usize, 0), p.total());
    try expect(p.idle() and !p.fits(1));
    p.grow(2);
    try expect(p.idle() and p.fits(2) and !p.fits(3));
}

test "elastic: a shrink never drops a live page; a lease's, a run's or the store's page in the tail refuses it and changes nothing" {
    var p = try kv.Pool.initWith(gpa, 10, 64, .{ .lowest_first = true });
    defer p.deinit();
    var a: kv.Lease = .{};
    var r: kv.Lease = .{};
    defer p.finish(&a);
    defer p.finish(&r);
    try p.promise(&a, 5);
    for (0..5) |_| _ = try p.take(&a); // pages 0..4
    try p.promise(&r, 2);
    const run = p.claimRun(&r, 2).?; // pages 5 and 6
    try equal(@as(u32, 5), run);
    p.retain(4); // the store's copy of a stream's last page
    var highest: usize = 0; // the live page with the highest id
    for (p.refs, 0..) |count, id| if (count != 0) {
        highest = id;
    };
    try equal(@as(usize, 6), highest);
    for (0..p.capacity() + 1) |to| {
        const refs = try gpa.dupe(u32, p.refs);
        defer gpa.free(refs);
        const free = try gpa.dupe(u32, p.free[0..p.nfree]);
        defer gpa.free(free);
        const census = p.census();
        if (to <= highest) { // a live page lies at or past `to`
            try expect(!p.tailIdle(to));
            try std.testing.expectError(error.Busy, p.shrink(to));
            try equal(census, p.census());
            try std.testing.expectEqualSlices(u32, refs, p.refs);
            try std.testing.expectEqualSlices(u32, free, p.free[0..p.nfree]);
        } else {
            try expect(p.tailIdle(to));
            try p.shrink(to);
            try std.testing.expectEqualSlices(u32, refs, p.refs); // every live page is where it was, held as it was
            for (p.refs, 0..) |count, id| if (count != 0) try expect(id < p.mapped);
            p.grow(p.capacity());
        }
        try scanned(&p);
    }
    p.release(4);
    p.releaseRun(run, 2);
    try p.shrink(5); // the lease's own five pages are the least that stay
    try equal(@as(usize, 5), p.total());
}

test "elastic: grow in the default order puts the new lowest id on top, after the pages already free" {
    var p = try kv.Pool.initWith(gpa, 8, 64, .{ .mapped = 2 });
    defer p.deinit();
    p.grow(5);
    try scanned(&p);
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 5);
    for ([_]u32{ 2, 3, 4, 0, 1 }) |want| try equal(want, try p.take(&l));
}

/// Counts the notifications a change makes.
const Count = struct {
    n: usize = 0,
    fn call(ctx: *anyopaque, _: *const kv.Pool) void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.n += 1;
    }
};

test "elastic: shrink and grow notify once each, the counts the notice sees already moved" {
    var p = try kv.Pool.initWith(gpa, 8, 64, .{ .lowest_first = true });
    defer p.deinit();
    var c: Count = .{};
    p.notify = .{ .ctx = &c, .call = Count.call };
    try p.shrink(5);
    try equal(@as(usize, 1), c.n);
    p.grow(8);
    try equal(@as(usize, 2), c.n);
}

/// A pool, its leases and runs, and the store's holds beside them, with shrinks and grows between.
const Walk = struct {
    pool: kv.Pool,
    rnd: std.Random,
    leases: [4]kv.Lease = @splat(.{}),
    runs: [4]struct { start: u32, k: usize } = undefined,
    nruns: usize = 0,
    store: [most]u32 = @splat(0),
    in_run: [most]bool = @splat(false),
    shrinks: usize = 0,
    busy: usize = 0,
    grows: usize = 0,
    lowest_takes: usize = 0,
    runs_made: usize = 0,

    fn step(w: *Walk) !void {
        const p = &w.pool;
        const l = &w.leases[w.rnd.uintLessThan(usize, w.leases.len)];
        switch (w.rnd.uintLessThan(u8, 11)) {
            0 => p.promise(l, w.rnd.uintLessThan(usize, 5)) catch {},
            1 => {
                const want = lowestFree(p);
                if (p.take(l)) |id| {
                    try expect(id < p.mapped);
                    if (p.lowest) {
                        try equal(want.?, id);
                        w.lowest_takes += 1;
                    }
                } else |_| {}
            },
            2 => p.cover(l, l.ids.items.len + w.rnd.uintLessThan(usize, 4), null) catch {},
            3 => w.claim(l),
            4 => if (w.nruns > 0) w.unclaim(w.rnd.uintLessThan(usize, w.nruns)),
            5 => {
                const id = w.rnd.uintLessThan(u32, @intCast(p.capacity()));
                if (id < p.mapped and p.refs[id] > 0 and !w.in_run[id]) {
                    p.retain(id);
                    w.store[id] += 1;
                }
            },
            6 => {
                const id = w.rnd.uintLessThan(usize, p.capacity());
                if (w.store[id] > 0) {
                    p.release(@intCast(id));
                    w.store[id] -= 1;
                }
            },
            7 => p.finish(l),
            8 => p.unpromise(l),
            9 => {
                const to = w.rnd.uintAtMost(usize, p.mapped);
                var held = false; // a page of the tail held, or the head too short for the promises, by this model's own count
                var head_free: usize = 0;
                for (p.refs[0..p.mapped], 0..) |r, id| {
                    if (id >= to) {
                        held = held or r != 0;
                    } else head_free += @intFromBool(r == 0);
                }
                const refuse = held or head_free < p.promised;
                try equal(!refuse, p.tailIdle(to));
                if (p.shrink(to)) {
                    try expect(!refuse);
                    w.shrinks += 1;
                } else |_| {
                    try expect(refuse);
                    w.busy += 1;
                }
            },
            10 => {
                p.grow(p.mapped + w.rnd.uintAtMost(usize, p.capacity() - p.mapped));
                w.grows += 1;
            },
            else => unreachable,
        }
    }

    fn claim(w: *Walk, l: *kv.Lease) void {
        if (w.nruns == w.runs.len) return;
        const k = 1 + w.rnd.uintLessThan(usize, 3);
        const want = if (l.promised >= k) modelRun(&w.pool, k, 1, w.pool.lowest) else null;
        const start = w.pool.claimRun(l, k) orelse return std.debug.assert(want == null);
        std.debug.assert(want.? == start);
        for (start..start + k) |id| w.in_run[id] = true;
        w.runs[w.nruns] = .{ .start = start, .k = k };
        w.nruns += 1;
        w.runs_made += 1;
    }

    fn unclaim(w: *Walk, i: usize) void {
        const r = w.runs[i];
        w.pool.releaseRun(r.start, r.k);
        for (r.start..r.start + r.k) |id| w.in_run[id] = false;
        w.nruns -= 1;
        w.runs[i] = w.runs[w.nruns];
    }

    /// Holders and leases of a page are the store's holds, the leases mapping it and its run's.
    fn model(w: *const Walk) !void {
        var mapped: [most]u32 = @splat(0);
        var promised: usize = 0;
        for (w.leases) |l| {
            promised += l.promised;
            for (l.ids.items) |id| mapped[id] += 1;
        }
        try equal(promised, w.pool.promised);
        for (0..w.pool.capacity()) |id| {
            const run: u32 = @intFromBool(w.in_run[id]);
            try equal(mapped[id] + run, w.pool.leased[id]);
            try equal(w.store[id] + mapped[id] + run, w.pool.refs[id]);
        }
        for (1..5) |k| {
            try equal(modelRun(&w.pool, k, 1, w.pool.lowest), w.pool.findRun(k));
            for ([_]usize{ 1, 2, 3 }) |step_| try equal(modelRun(&w.pool, k, step_, w.pool.lowest), w.pool.findAligned(k, step_));
        }
    }

    /// Every lease, run and hold given back: the pool is idle, and all of it comes back when grown whole.
    fn drain(w: *Walk) !void {
        for (&w.leases) |*l| w.pool.finish(l);
        while (w.nruns > 0) w.unclaim(0);
        for (&w.store, 0..) |*n, id| while (n.* > 0) : (n.* -= 1) w.pool.release(@intCast(id));
        try scanned(&w.pool);
        try expect(w.pool.idle());
        try w.pool.shrink(0);
        try equal(@as(usize, 0), w.pool.nfree);
        w.pool.grow(w.pool.capacity());
        try scanned(&w.pool);
        try equal(w.pool.capacity(), w.pool.nfree);
    }
};

test "elastic: seeded walks over every transition and every shrink and grow keep the counts equal to a scan" {
    var shrinks: usize = 0;
    var busy: usize = 0;
    var grows: usize = 0;
    var lowest_takes: usize = 0;
    var runs: usize = 0;
    for (0..96) |seed| {
        var prng = std.Random.DefaultPrng.init(0x3c1e_0000 + seed);
        const count = 6 + seed % (most - 5);
        const rnd = prng.random();
        var w: Walk = .{ .pool = try kv.Pool.initWith(gpa, count, 4096, .{ .mapped = rnd.uintAtMost(usize, count), .lowest_first = seed % 2 == 0 }), .rnd = rnd };
        defer w.pool.deinit();
        for (0..600) |_| {
            try w.step();
            try scanned(&w.pool);
            try w.model();
        }
        try w.drain();
        shrinks += w.shrinks;
        busy += w.busy;
        grows += w.grows;
        lowest_takes += w.lowest_takes;
        runs += w.runs_made;
    }
    try expect(shrinks > 0 and busy > 0 and grows > 0 and lowest_takes > 0 and runs > 0);
}
