//! A layer-split export under way on the engine thread: the boundary rows in, then the state landed behind them.
const std = @import("std");
const fabric = @import("fabric");
const cx = @import("cuda_export.zig");
const serve = @import("cuda_export_serve.zig");
const Produce = serve.Produce;
const log = std.log.scoped(.dsv41);

/// A layer-split export under way on the engine thread.
pub const Early = struct {
    p: *Produce,
    held: *serve.Held,
    early: *cx.Early,
    id: [16]u8,
    leased: bool,

    /// The next region of boundary rows from `row0` for the producer to fill, then `publish`; waits for the window.
    pub fn rows(e: Early, row0: u64, n: u64) ![]u8 {
        return e.early.stage.reserve(row0, n, Produce.early_wait_ns, null);
    }

    /// Rows [0, end) are in: readers may take them. The entry's expiry clock restarts (it is not expired mid-run).
    pub fn publish(e: Early, end: u64) !void {
        try e.early.stage.publish(end);
        e.p.table.touch(e.id);
    }

    /// The decoder released the export, or it expired, while it ran: nobody is left to read it.
    pub fn gone(e: Early) bool {
        while (!e.p.lock.tryLock()) std.atomic.spinLoopHint();
        defer e.p.lock.unlock();
        return e.held.gone;
    }

    /// No rows or state will come: readers are told by the stage and latch (the lane host reports `failed` once).
    pub fn fail(e: Early) void {
        e.early.fail();
    }

    /// The run ends without handing the state over (`fail` first when it failed): the lease goes back.
    pub fn close(e: *Early) void {
        if (!e.leased) return;
        e.leased = false;
        e.p.unlease(e.held);
    }

    /// `cap` (a `saveCut` state, now this export's) lands in the early buffers on its own thread when one is free
    pub fn land(e: *Early, cap: cx.Captured) void {
        e.leased = false;
        const job = e.*;
        if (job.p.building.load(.acquire) >= Produce.max_building) return landRun(job, cap);
        _ = job.p.building.fetchAdd(1, .acq_rel);
        const t = std.Thread.spawn(.{}, landThread, .{ job, cap }) catch {
            _ = job.p.building.fetchSub(1, .acq_rel);
            return landRun(job, cap);
        };
        t.detach();
    }
};

fn landThread(e: Early, cap: cx.Captured) void {
    landRun(e, cap);
    _ = e.p.building.fetchSub(1, .acq_rel);
}

fn landRun(e: Early, cap: cx.Captured) void {
    const p = e.p;
    defer p.unlease(e.held);
    defer p.source.release(cap.ctx);
    const t0 = fabric.words.nowNs();
    cx.land(p.gpa, e.held.built, cap.state) catch |err| { // the run already returned: nobody else tells the decoder
        e.fail();
        return p.failed(e.id, @errorName(err));
    };
    p.table.touch(e.id); // the clock starts now: the decoder has the TTL to pull what is held
    log.info("handoff {x}: layer-split state landed in {d} ms", .{ &e.id, (fabric.words.nowNs() - t0) / std.time.ns_per_ms });
}
