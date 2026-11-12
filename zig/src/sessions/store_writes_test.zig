//! Queued write-through saves: pinned in RAM at once, the file lands behind the engine and registers at a settle.

const std = @import("std");
const testing = std.testing;
const store_mod = @import("store.zig");
const PageStore = @import("pagestore.zig").PageStore;
const base = @import("store_test.zig");
const Rank = base.Rank;
const page = base.page;
const through = store_mod.through;

const queued: store_mod.Policy = .{ .write_through = true, .save_queue_bytes = 64 << 20 };

test "queued write-through: in RAM and pinned at once, the file registers at the settle, a restart finds it" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "queued");
    defer gpa.free(root);
    const a = try base.chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    var key: @import("prefix.zig").Digest = undefined;
    {
        var r: Rank = undefined;
        try r.init(gpa, root, 1, 0, queued);
        defer r.deinit();
        const s = try r.pool.newSlot(64 * page);
        try testing.expectEqualStrings("fresh", try base.turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]));
        const e = r.store.entry(0);
        try testing.expectEqual(@as(u64, 1), r.store.stats.write_throughs);
        try testing.expect(e.ram and !e.extra.file and e.extra.job != null);
        try testing.expect(r.disk.get(e.key) == null);
        try testing.expect((try r.store.evict(0)) == null); // a pinned entry is not parked under its own write
        r.store.writes.?.flush();
        try testing.expectEqual(@as(u32, 1), through.finished(&r.store));
        try testing.expect(!e.extra.file); // finished is not settled: the index changes only at the agreed point
        through.settleOldest(&r.store, 1);
        try testing.expect(e.ram and e.extra.file and e.extra.job == null);
        try testing.expect(r.disk.get(e.key) != null);
        key = e.key;
        try testing.expectEqualStrings("ram", try base.turn(gpa, &r, s, 1, a[0 .. 8 * page + 3]));
        const c = through.counters(&r.store).?;
        try testing.expectEqual(@as(u64, 2), c.queued);
        try testing.expect(c.written >= 1 and c.dropped == 0);
    }
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{});
    defer r.deinit();
    try testing.expectEqual(@as(u32, 2), try r.store.reconcile()); // the second turn's write ran at the stop
    try testing.expect(r.disk.get(key) != null);
}

test "a full queue drops the write, counts it, and the entry stays RAM-only" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "full");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{ .write_through = true, .save_queue_bytes = 1 });
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try base.chatIds(gpa, 1, 8 * page);
    defer gpa.free(a);
    const b = try base.chatIds(gpa, 2, 8 * page);
    defer gpa.free(b);
    _ = try base.turn(gpa, &r, s, 1, a[0 .. 5 * page]); // alone, a save is accepted whatever its size
    _ = try base.turn(gpa, &r, s, 2, b[0 .. 5 * page]); // behind it the byte bound refuses
    try testing.expectEqual(@as(u64, 1), r.store.stats.write_throughs);
    try testing.expectEqual(@as(u64, 1), through.counters(&r.store).?.dropped);
    const dropped = (try r.store.find(1, b)).?;
    try testing.expect(dropped.ram and !r.store.entry(dropped.id).extra.file and r.store.entry(dropped.id).extra.job == null);
    try testing.expectEqualStrings("ram", try base.turn(gpa, &r, s, 2, b[0 .. 6 * page])); // the dropped save still resumes
}

test "an entry that gives its pages back first settles its own write, then parks on its file" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "giveback");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, queued);
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try base.chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    _ = try base.turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]);
    try testing.expect(r.store.entry(0).extra.job != null);
    try r.store.leaveRam(0);
    const e = r.store.entry(0);
    try testing.expect(!e.ram and e.extra.file and e.extra.job == null);
    try testing.expectEqual(@as(u64, 1), r.store.stats.parks);
    try testing.expectEqualStrings("disk", try base.turn(gpa, &r, s, 1, a[0 .. 8 * page + 3]));
}

/// Three chats saved and settled in two groups; the settle counts are the leader's, the same on every rank.
fn twoGroups(gpa: std.mem.Allocator, root: []const u8, rank: u32, lead: bool) !u64 {
    var r: Rank = undefined;
    try r.init(gpa, root, 2, rank, queued);
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    for ([_]u32{ 1, 2 }) |chat| {
        const ids = try base.chatIds(gpa, chat, 8 * page);
        defer gpa.free(ids);
        _ = try base.turn(gpa, &r, s, chat, ids[0 .. 5 * page + chat]);
    }
    if (lead) r.store.writes.?.flush();
    through.settleOldest(&r.store, 2);
    const ids = try base.chatIds(gpa, 3, 8 * page);
    defer gpa.free(ids);
    _ = try base.turn(gpa, &r, s, 3, ids[0 .. 5 * page + 3]);
    through.settleOldest(&r.store, 1);
    try testing.expectEqual(@as(u64, 3), r.store.stats.parks);
    try testing.expectEqual(@as(u64, 0), r.store.stats.write_fails);
    return r.disk.used;
}

test "two split ranks settle the leader's counts and end alike whatever each disk's timing" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "pair");
    defer gpa.free(root);
    try testing.expectEqual(try twoGroups(gpa, root, 0, true), try twoGroups(gpa, root, 1, false));
}

/// Pages that cannot be read: the write fails after the entry was saved and pinned.
const Unreadable = struct {
    inner: PageStore,
    fn store(u: *Unreadable) PageStore {
        return .{ .ptr = u, .vtable = &vt };
    }
    const vt: PageStore.VTable = .{ .read = read, .write = write, .copyPage = copyPage, .fence = fence, .waitFence = waitFence, .publish = publish };
    fn read(_: *anyopaque, _: u32, _: []const u32, _: []u8) anyerror!void {
        return error.InputOutput;
    }
    fn write(p: *anyopaque, fam: u32, pages: []const u32, src: []const u8) anyerror!void {
        const u: *Unreadable = @ptrCast(@alignCast(p));
        return u.inner.write(fam, pages, src);
    }
    fn copyPage(p: *anyopaque, fam: u32, src: u32, dst: u32) anyerror!void {
        const u: *Unreadable = @ptrCast(@alignCast(p));
        return u.inner.copyPage(fam, src, dst);
    }
    fn fence(p: *anyopaque) anyerror!u64 {
        const u: *Unreadable = @ptrCast(@alignCast(p));
        return u.inner.fence();
    }
    fn waitFence(p: *anyopaque, token: u64) anyerror!void {
        const u: *Unreadable = @ptrCast(@alignCast(p));
        return u.inner.waitFence(token);
    }
    fn publish(p: *anyopaque) anyerror!void {
        const u: *Unreadable = @ptrCast(@alignCast(p));
        return u.inner.publish();
    }
};

test "a failed write is counted and leaves the entry RAM-only; the queue keeps its error and the engine goes on" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "failed");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, queued);
    defer r.deinit();
    var bad: Unreadable = .{ .inner = r.store.pages };
    r.store.pages = bad.store();
    const s = try r.pool.newSlot(64 * page);
    const a = try base.chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    _ = try base.turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]);
    through.settleEvery(&r.store);
    const e = r.store.entry(0);
    try testing.expect(e.ram and !e.extra.file and e.extra.job == null);
    try testing.expectEqual(@as(u64, 1), r.store.stats.write_fails);
    const c = through.counters(&r.store).?;
    try testing.expectEqual(@as(u64, 1), c.failed);
    try testing.expect(r.store.writes.?.first_error != null);
    try testing.expectEqualStrings("ram", try base.turn(gpa, &r, s, 1, a[0 .. 8 * page + 3]));
    r.store.pages = bad.inner;
}

test "stopping with unsettled saves runs them, frees them, and their files stand for the next boot" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try base.tmpRoot(gpa, &tmp, "stop");
    defer gpa.free(root);
    {
        var r: Rank = undefined;
        try r.init(gpa, root, 1, 0, queued);
        defer r.deinit();
        const s = try r.pool.newSlot(64 * page);
        for ([_]u32{ 1, 2, 3 }) |chat| {
            const ids = try base.chatIds(gpa, chat, 8 * page);
            defer gpa.free(ids);
            _ = try base.turn(gpa, &r, s, chat, ids[0 .. 5 * page + chat]);
        }
        try testing.expectEqual(@as(u64, 3), r.store.stats.write_throughs);
    }
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{});
    defer r.deinit();
    try testing.expectEqual(@as(u32, 3), try r.store.reconcile());
}
