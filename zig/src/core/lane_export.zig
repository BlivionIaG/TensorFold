//! #1594 step 2b, the producer's side: a handoff request's prompt state saved by its family for another machine.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const lifetime = @import("prompt_cache_lifetime.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = @import("lane_job.zig").Job;
const Request = api.Request;

/// kv_transfer_params.mcdma_handoff on a completion request: its prompt state exported under `id` for another machine.
pub const Handoff = struct { id: [16]u8, export_from: u64 = 0, split_cut: u32 = 0 };

/// Whether the request that runs a layer split has gone away (the client left, the server closes): the split stops.
pub const Cancel = struct {
    ptr: *anyopaque,
    is: *const fn (ptr: *anyopaque) bool,

    pub fn cancelled(c: Cancel) bool {
        return c.is(c.ptr);
    }
};

/// A layer-split producer: the exporter runs the prefill itself (it owns the slot and the early export).
pub const Split = struct {
    /// Whether this exporter takes a split cut at `cut` (the server opted in, the cut is one of its model's).
    ok: *const fn (ptr: *anyopaque, cut: u32) bool,
    /// Runs the split of `prompt` (all its tokens, `h.split_cut` the cut) on the engine thread with the slot empty
    run: *const fn (ptr: *anyopaque, h: Handoff, prompt: []const u32, cancel: ?Cancel) anyerror!u32,
};

/// A producer's exports: the family saves a handoff stream's state at its mark, and the exporter records it.
pub const Exporter = struct {
    ptr: *anyopaque,
    snapshots: pc.Snapshots,
    /// `saved` (now the exporter's) is the state after `tokens`, the prompt but its last token; recorded under `h`.
    exported: *const fn (ptr: *anyopaque, h: Handoff, tokens: []const u32, saved: pc.Saved) void,
    /// No state will come for `h`: the decoder is told why.
    failed: *const fn (ptr: *anyopaque, h: Handoff, why: []const u8) void,
    /// Held exports as `GET /v1/handoff/<id>` on the server's own port, for decoders with no fabric link (null: off).
    serve: ?Serve = null,
    /// Layer-split prefills (null: a request with `split_cut` is refused at submit).
    split: ?Split = null,
};

/// The host's held exports, when it serves them over HTTP (the engine's `handoff` hook).
pub fn serveHook(ctx: *anyopaque) ?Serve {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    const x = h.exporter orelse return null;
    return x.serve;
}

/// An export's bytes read over HTTP: the producer's replies to a decoder's OPEN and every PULL, back to back.
pub const Stream = struct {
    ptr: *anyopaque,
    /// the whole body, known before the first byte (the response's Content-Length)
    size: u64,
    /// The next message (valid until the next call), or null after the last.
    next: *const fn (ptr: *anyopaque) anyerror!?[]const u8,
    /// The export may be released again.
    close: *const fn (ptr: *anyopaque) void,
};

/// What the server reads of an exporter that serves its exports over HTTP (server/handoff_routes.zig).
pub const Serve = struct {
    ptr: *anyopaque,
    /// A held export's stream, held until closed. `NotHeld`: never held, released or expired. `Refused`: failed.
    open: *const fn (ptr: *anyopaque, id: [16]u8, why: *[]const u8) OpenError!Stream,
    /// The export released now (a decoder is done with it); false when none was held.
    release: *const fn (ptr: *anyopaque, id: [16]u8) bool,

    pub const OpenError = error{ NotHeld, Refused, OutOfMemory, Unreadable };
};

/// The exports `e` holds and serves over HTTP, when it serves them.
pub fn serving(e: api.Engine) ?Serve {
    const f = e.vtable.handoff orelse return null;
    return f(e.ctx);
}

/// A handoff request needs an exporter, and a prompt with a token past its mark.
pub fn exportable(h: *const LaneHost, r: *const Request) bool {
    const k = r.handoff orelse return true;
    const x = h.exporter orelse return false;
    if (k.split_cut != 0) {
        const s = x.split orelse return false;
        if (!s.ok(x.ptr, k.split_cut)) return false;
    }
    return r.prompt.len >= 2 and r.images.len == 0 and !r.promptOnly();
}

/// A layer-split handoff is run here by the exporter on the engine thread (slot empty); true: the job was finished.
pub fn runSplit(h: *LaneHost, job: *Job) bool {
    const k = job.request.handoff orelse return false;
    const x = h.exporter orelse return false;
    const s = x.split orelse return false;
    if (k.split_cut == 0) return false;
    job.exported = true; // the split reports its own state or failure to the decoder
    const cancel: Cancel = .{ .ptr = job, .is = LaneHost.cancelled };
    const token = s.run(x.ptr, k, job.request.prompt, cancel) catch |err| {
        x.failed(x.ptr, k, @errorName(err));
        _ = if (err == error.Cancelled) h.cancel(job) else @import("lane_host_end.zig").dropFailed(h, job, err);
        return true;
    };
    h.remove(job);
    LaneHost.emit(job, .{ .prefilled = 0 });
    const reply = [1]u32{token};
    LaneHost.emit(job, .{ .tokens = &reply });
    h.finish(job, .length, "");
    return true;
}

/// A handoff stream cuts before the prompt's last token (its row stays pending) and exports there; its marks keep.
pub fn exporting(h: *const LaneHost, job: *Job, reuse: *lanes.stream.Reuse) bool {
    if (job.request.handoff == null or h.exporter == null) return false;
    job.cut = .{@intCast(job.request.prompt.len - 1)};
    if (reuse.at >= job.cut[0]) below(h, job, reuse); // a restore at the cut leaves no chunk ending there, so no export
    reuse.marks = handoffMarks(h, job, reuse.marks, reuse.at);
    reuse.hook = .{ .ptr = job, .at = exportAt };
    return true;
}

/// The lookup's state at the cut gives way to the longest one kept before it, borrowed in its place (none: from 0).
fn below(h: *const LaneHost, job: *Job, reuse: *lanes.stream.Reuse) void {
    reuse.saved = null;
    reuse.at = 0;
    reuse.remote = false; // the provenance goes with the state: the one borrowed below brings its own
    const store = h.cache orelse return;
    const r = job.request;
    if (job.entry) |at_cut| store.release(at_cut); // first: a disk read may take its room, and the pass keeps it again
    job.entry = lifetime.borrow(store, resumable(r), r.bornLen(), r.chunks, h.spans(r));
    const e = job.entry orelse return;
    reuse.saved = e.saved;
    reuse.at = e.at;
    reuse.remote = e.remote;
}

/// The words a request's kept state is looked up in: a handoff's before its cut, where a restore would export nothing.
pub fn resumable(r: *const Request) []const u32 {
    const key = r.cacheKey();
    return if (r.handoff == null) key else key[0..@min(key.len, r.prompt.len -| 1)];
}

/// The store's marks past the restore and before the cut, and a planned store's last grid row before it; then the cut.
fn handoffMarks(h: *const LaneHost, job: *Job, planned: []const u32, from: u32) []const u32 {
    const cut = job.cut[0];
    const grid: u32 = if (h.cache) |store| (if (store.rules.planned) store.rules.grid else 0) else 0;
    // strictly before the cut: a repeat of this handoff resumes here (as does a next turn when the cut is off the grid)
    const row: u32 = if (grid > 0) (cut - 1) - (cut - 1) % grid else 0;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(h.gpa);
    for (planned) |m| if (m > from and m < cut) list.append(h.gpa, m) catch return &job.cut;
    if (row > from and row < cut and std.mem.indexOfScalar(u32, list.items, row) == null) list.append(h.gpa, row) catch return &job.cut;
    std.mem.sort(u32, list.items, {}, std.sort.asc(u32));
    const marks = h.gpa.alloc(u32, list.items.len + 1) catch return &job.cut;
    @memcpy(marks[0..list.items.len], list.items);
    marks[list.items.len] = cut;
    h.gpa.free(job.marks);
    job.marks = marks;
    return marks;
}

fn exportAt(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
    const job: *Job = @ptrCast(@alignCast(ptr));
    if (at != job.cut[0]) return Job.kept(ptr, s, at); // one of the store's marks: kept as for any request
    const x = job.host.exporter orelse return;
    const h = job.request.handoff orelse return;
    if (job.exported or s.isCancelled()) return; // a decoder that went away leaves nothing to hold; `unexported` tells the rest
    job.exported = true;
    const saved = x.snapshots.vtable.save(x.snapshots.ptr, s, at) catch |err| return x.failed(x.ptr, h, @errorName(err));
    x.exported(x.ptr, h, job.request.prompt[0..at], saved);
    if (keeps(job, at)) Job.kept(ptr, s, at);
}

/// Whether a stream's pass keeps its state at its mark `at`: a handoff's cut only where its store could resume it.
pub fn keeps(job: *const Job, at: u32) bool {
    if (job.request.handoff == null or at != job.cut[0]) return true;
    const store = job.host.cache orelse return false;
    return store.usable(at, job.request.chunks);
}

/// A handoff whose pass ended (or never ran) without its state: the decoder hears so rather than waiting it out.
pub fn unexported(job: *Job) void {
    const h = job.request.handoff orelse return;
    const x = job.host.exporter orelse return;
    if (job.exported or (job.started and !job.stream.finished and job.prefilled == null)) return; // still under way
    job.exported = true;
    x.failed(x.ptr, h, "the prompt pass ended before its state was saved");
}
