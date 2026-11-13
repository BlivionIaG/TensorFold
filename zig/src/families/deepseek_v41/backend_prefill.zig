//! The V4.1 prompt pass a chunk at a time: the lane host runs the other streams' rounds between two chunks.
const std = @import("std");
const ds = @import("model.zig");
const lanes = @import("lanes");
const Stream = lanes.Stream;
const Backend = @import("backend.zig").Backend;
const split_feed = @import("split_feed.zig");
const split_run = @import("split_run.zig");
const entry_async = @import("entry_async.zig");

/// The stream's slot, taken, reset and (for a kept prompt state) restored on the first chunk; the pass's cursor.
fn begin(b: *Backend, s: *Stream, prompt: []const u32) !u32 {
    // a split feed the stream cannot start on is over, as a failed one (the prompt is a local pass if it runs again)
    errdefer dropFeed(s);
    const i = b.free() orelse return error.NoFreeSlot;
    const state = &b.model.caches[i];
    if (prompt.len + s.max_new + 1 > state.plan.capacity) return error.ContextFull;
    try b.by.put(b.gpa, s, i);
    try b.model.resetSlot(i);
    b.history[i].clearRetainingCapacity();
    s.cached = 0;
    b.runs[i] = null;
    // A LAYER-SPLIT FEED IS THE PASS'S WAY IN: the slot stays blank until its rows enter and its state restores.
    if (s.reuse.split) |p| {
        const feed: split_feed.Feed = @as(*const split_feed.Feed, @ptrCast(@alignCast(p))).*;
        s.reuse.split = null;
        b.runs[i] = split_run.Run.init(feed);
        if (!split_run.admits(feed, prompt.len)) b.runs[i].?.refuse();
        s.split_report = reportOf(&b.runs[i].?);
        s.fill_at.store(0, .monotonic);
        return i;
    }
    // A KEPT PROMPT STATE RESUMES THE PASS WHERE IT STOOD.
    if (s.reuse.saved) |saved| {
        const snap: *ds.Model.Snapshot = @ptrCast(@alignCast(saved));
        const want = s.reuse.at;
        if (want > 0 and want < prompt.len) blk: {
            b.model.snapshotRestore(i, snap) catch |err| {
                s.reuse_failed = true;
                s.reuse_error = err;
                break :blk;
            };
            s.cached = want;
            b.history[i].appendSlice(b.gpa, prompt[0..want]) catch {};
        }
    }
    s.fill_at.store(s.cached, .monotonic);
    return i;
}

/// One chunk of at most `max_rows` rows (0: the workspace's); true once the prompt is in and its first token drawn.
pub fn step(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
    const b = Backend.self(ptr);
    const prompt = s.prompt();
    if (prompt.len == 0) return error.EmptyPrompt;
    const i = b.by.get(s) orelse blk: {
        const slot = try begin(b, s, prompt);
        // a landed remote prefix the store marked is kept here, before the first chunk moves the slot
        if (s.cached > 0) if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, s.cached) != null) hook.at(hook.ptr, s, s.cached);
        break :blk slot;
    };
    if (s.isCancelled()) {
        endSplit(b, s, i);
        return error.Cancelled;
    }
    if (b.runs[i]) |*run| if (!run.finished and !try splitStep(b, s, i, run, prompt, max_rows)) return false;
    const at: u32 = b.model.caches[i].position;
    const plan_rows = b.model.workspace.plan.rows;
    const rows = if (max_rows != 0) @min(plan_rows, max_rows) else plan_rows;
    // THE TAIL MARK IS SNAPSHOTTED INSIDE THE CHUNK THAT RUNS PAST IT, not cut into a second forward.
    const merge_cold = std.c.getenv("TF_DSV41_SNAP_COLD") == null;
    const snap_at: u32 = if (std.c.getenv("TF_DSV41_NO_TAIL_SNAP") == null and b.model.snapTailAllowed() and (merge_cold or s.reuse.at > 0)) s.reuse.snap_at else 0;
    var end = @min(at + rows, @as(u32, @intCast(prompt.len)));
    // The store's marks always end a chunk: the pass stands exactly there when it calls the hook.
    for (s.reuse.marks) |m| if (m > at and m < end and m != snap_at) {
        end = m;
        break;
    };
    const last = end == prompt.len;
    // The next chunk's Engram rows are read while this chunk's GPU work runs.
    if (!last) b.model.lookahead(i, prompt[end..@min(prompt.len, end + plan_rows)]);
    var merge: u32 = if (snap_at > at and snap_at < end and end - at >= 64) snap_at else 0;
    if (merge != 0 and !b.model.beginSnapTail(i, prompt[@max(0, @as(usize, merge) - 4)..merge], merge, at)) merge = 0;
    if (last) {
        _ = try b.model.forwardOn(i, prompt[at..end], .last);
        b.first[i] = try b.draw(i, s, @intCast(prompt.len));
    } else {
        _ = try b.model.forwardOn(i, prompt[at..end], .none);
    }
    if (merge != 0) {
        // the state at the mark, copied from the stash the chunk's own publishes wrote
        if (s.reuse.hook) |hook| hook.at(hook.ptr, s, merge);
        b.model.endSnapTail();
    }
    // The prompt cache's keep hook, at the chunk end it asked to see.
    if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, end) != null) hook.at(hook.ptr, s, end);
    s.fill_at.store(end, .monotonic);
    return last;
}

/// A split feed no pass adopted (the stream could not start, or ended first) is over, as a failed one, exactly once.
pub fn dropFeed(s: *Stream) void {
    const p = s.reuse.split orelse return;
    const feed = @as(*const split_feed.Feed, @ptrCast(@alignCast(p))).*;
    s.reuse.split = null;
    feed.finish(false, .{});
    s.split_report = .{ .cut = feed.cut, .fell_back = true };
}

/// What the lane host reports of a layer-split pass (the feed's timings are read when it is finished).
pub fn reportOf(r: *const split_run.Run) @FieldType(lanes.stream.Stream, "split_report") {
    return .{
        .cut = r.feed.cut,
        .rows = r.stats.rows,
        .passes = r.stats.passes,
        .first_row_ns = r.pull.first_row_ns,
        .rows_ns = r.pull.rows_ns,
        .state_ns = r.pull.state_ns,
        .wait_ns = r.stats.wait_ns,
        .compute_ns = r.stats.compute_ns,
        .state_wait_ns = r.stats.state_wait_ns,
        .restore_ns = r.stats.restore_ns,
        .bytes = r.pull.bytes,
        .fell_back = r.fell_back,
    };
}

/// The stream's split feed, if it still holds one, is over (a cancel): the puller may stop.
fn endSplit(b: *Backend, s: *Stream, slot: u32) void {
    const run = &(b.runs[slot] orelse return);
    run.abort();
    s.split_report = reportOf(run);
}

/// The model's calls for the split machine, for one step of the pass.
const Calls = struct {
    b: *Backend,
    slot: u32,
    prompt: []const u32,
    cut: u32,

    fn ops(c: *Calls) split_run.Ops {
        return .{ .ptr = c, .reset = reset, .position = position, .enter = enter, .sync = sync, .restore = restore };
    }
    fn self(ptr: *anyopaque) *Calls {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) anyerror!void {
        const c = self(ptr);
        try c.b.model.resetSlot(c.slot);
        c.b.history[c.slot].clearRetainingCapacity();
    }
    fn position(ptr: *anyopaque) u32 {
        const c = self(ptr);
        return c.b.model.caches[c.slot].position;
    }
    fn enter(ptr: *anyopaque, at: u32, n: u32, rows: []const u8) anyerror!void {
        const c = self(ptr);
        try c.b.model.forwardEntered(c.slot, c.prompt[at..][0..n], rows, c.cut);
    }
    fn sync(ptr: *anyopaque) anyerror!void {
        try entry_async.settle(self(ptr).b.model);
    }
    fn restore(ptr: *anyopaque, snap: *anyopaque) anyerror!void {
        const c = self(ptr);
        try c.b.model.snapshotRestoreLayers(c.slot, @ptrCast(@alignCast(snap)), c.cut);
    }
};

/// One slice of a layer-split pass: false while its rows or state are still coming; true once it restored
fn splitStep(b: *Backend, s: *Stream, slot: u32, run: *split_run.Run, prompt: []const u32, max_rows: u32) !bool {
    var calls: Calls = .{ .b = b, .slot = slot, .prompt = prompt, .cut = run.feed.cut };
    const plan_rows = b.model.workspace.plan.rows;
    const cap = @min(plan_rows, @max(if (max_rows != 0) max_rows else plan_rows, 64));
    const next = run.advance(calls.ops(), cap) catch |err| {
        run.abort();
        s.split_report = reportOf(run);
        return err;
    };
    s.split_report = reportOf(run);
    s.fill_at.store(switch (next) {
        .more => run.at,
        .restored => run.upto,
        .fell_back => 0,
    }, .monotonic);
    return next != .more;
}

/// The whole prompt pass in one call, as the lane host runs it when it has nothing to interleave.
pub fn whole(ptr: *anyopaque, s: *Stream) anyerror!void {
    while (!try step(ptr, s, 0)) {}
}

/// TF_DSV41_PREFILL_INTERLEAVE: 0 runs each prompt whole (other streams stop for it); else by chunks.
pub fn interleaveOn() bool {
    const v = std.c.getenv("TF_DSV41_PREFILL_INTERLEAVE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}
