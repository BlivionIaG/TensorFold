//! The lane core's HIP backend for Qwen3.5 / 3.6: every stream its own caches, every round's windows verified in one
//! forward, each row drawn on the host at its keyed position, the kept rows committed from the window's states.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const worker = @import("worker.zig");
const sample = @import("sample.zig");

const be = lanes.backend;

/// Drawn tokens a handle names, newest last.
const ring = 1024;

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    w: worker.Worker,
    /// Each stream's id on every rank.
    ids: std.AutoHashMapUnmanaged(*const lanes.Stream, u32) = .empty,
    next_id: u32 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it.
    link: ?*const hip.link.Link = null,
    msg: std.ArrayList(u32) = .empty,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .e = e, .w = try worker.Worker.init(gpa, e) };
        return h;
    }

    /// Rank 0 of a tensor-parallel group: every step also goes to the ranks behind `link` (`worker.follow`).
    pub fn withLink(h: *Hip, link: *const hip.link.Link) void {
        h.link = link;
    }

    pub fn deinit(h: *Hip) void {
        if (h.link != null) h.send(&.{@backingInt(worker.Op.stop)}) catch {};
        h.w.deinit();
        h.ids.deinit(h.gpa);
        h.msg.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    /// `words` to the other ranks, when there are any.
    fn send(h: *Hip, words: []const u32) !void {
        if (h.link) |l| try l.send(words);
    }

    pub fn backend(h: *Hip) be.Backend {
        return .{ .ptr = h, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
        } };
    }

    /// Rows a window holds: every width keeps a row's bits (the window forward), drafts come from the core.
    pub const max_window = 16;

    /// The facts the round loop reads at setup: shared rounds of exact windows, no draft head yet.
    pub fn facts(h: *const Hip) lanes.Model {
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = false,
            .speculate = false,
            .hidden_rows = true,
            .batch_rows = @intCast(h.e.o.batch_rows),
            .max_streams = @intCast(h.e.o.batch_rows),
        };
    }

    fn of(ptr: *anyopaque) *Hip {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(h: *Hip, token: u32) u64 {
        const at = h.next;
        h.drawn[at % ring] = token;
        h.next += 1;
        return at;
    }

    fn sampling(s: *const lanes.Stream) ?lanes.Sampling {
        return s.sampling;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const h = of(ptr);
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.ids.getOrPut(h.gpa, s);
        if (!gop.found_existing) {
            gop.value_ptr.* = h.next_id;
            h.next_id += 1;
        }
        const id = gop.value_ptr.*;
        h.msg.clearRetainingCapacity();
        try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.prefill), id, @intCast(prompt.len) });
        try h.msg.appendSlice(h.gpa, prompt);
        try h.send(h.msg.items);
        const row = try h.w.prefill(id, prompt);
        _ = h.take(try sample.draw(h.gpa, row, h.e.dtype, sampling(s), prompt.len));
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const h = of(ptr);
        if (position != s.prompt_len) {
            std.log.err("first draw at {d}, the prompt has {d} tokens", .{ position, s.prompt_len });
            return error.PositionMismatch;
        }
        return h.next - 1;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const h = of(ptr);
        if (handle >= h.next or h.next - handle > ring) return error.NoSuchToken;
        return h.drawn[handle % ring];
    }

    /// Each stream's window from its kept length: the pending token, then the host's drafts; one forward for all.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = of(ptr);
        var items: [64]worker.Item = undefined;
        var tokens: [64][16]u32 = undefined;
        if (windows.len > items.len or windows.len > h.e.o.batch_rows) return error.WindowTooWide;
        h.msg.clearRetainingCapacity();
        try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.verify), @intCast(windows.len) });
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.held > 0) return error.NoHeldDrafts;
            const id = h.ids.get(w.stream) orelse return error.UnknownStream;
            const n = w.rows();
            if (n > tokens[i].len) return error.WindowTooWide;
            const held = try h.w.settled(id);
            for (w.positions, 0..) |p, r| if (p != held + 1 + r) {
                std.log.err("row {d} keyed at {d}, the stream holds {d} slots", .{ r, p, held });
                return error.PositionMismatch;
            };
            tokens[i][0] = w.pending;
            @memcpy(tokens[i][1..n], w.tokens);
            items[i] = .{ .id = id, .tokens = tokens[i][0..n] };
            try h.msg.appendSlice(h.gpa, &.{ id, @intCast(n) });
            try h.msg.appendSlice(h.gpa, items[i].tokens);
        }
        try h.send(h.msg.items);
        const logits = try h.w.verify(items[0..windows.len]);
        const vocab = h.e.model().spec.vocab;
        var at: usize = 0;
        for (windows, out) |w, *o| {
            for (o.sampled, 0..) |*t, row| t.* = try sample.draw(h.gpa, logits[(at + row) * vocab ..][0..vocab], h.e.dtype, sampling(w.stream), w.positions[row]);
            @memcpy(o.drafts, w.tokens);
            at += w.rows();
        }
    }

    /// Keep each stream's accepted prefix: lengths, and the linear states of its last kept row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = of(ptr);
        h.msg.clearRetainingCapacity();
        try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.keep), @intCast(windows.len) });
        for (windows, paths) |w, path| {
            for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
            try h.msg.appendSlice(h.gpa, &.{ h.ids.get(w.stream) orelse return error.UnknownStream, @intCast(path.len) });
        }
        try h.send(h.msg.items);
        for (windows, paths) |w, path| try h.w.keep(h.ids.get(w.stream).?, path.len);
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        _ = ptr;
        if (requests.len > 0) return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const h = of(ptr);
        const kv = h.ids.fetchRemove(s) orelse return;
        h.send(&.{ @backingInt(worker.Op.release), kv.value }) catch {};
        h.w.release(kv.value);
    }
};
