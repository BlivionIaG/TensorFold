//! Hand-off `Exporter` of the DeepSeek-V4.1 CUDA engine: `--mcdma-produce LINK` and TF_DSV41_HANDOFF_DUMP.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const fabric = @import("fabric");
const model = @import("cuda_model.zig");
const serve = @import("cuda_export_serve.zig");
const gpu = @import("cuda_export_gpu.zig");
const cx = @import("cuda_export.zig");
const split = @import("cuda_export_split.zig");
const ff = @import("forward_follow.zig");
const split_lib = @import("handoff/split.zig");
const geometry = @import("handoff/geometry.zig");
const pc = api.prompt_cache;
const Handoff = api.LaneHost.Export.Handoff;
const log = std.log.scoped(.dsv41);

/// Where the exports are written when set: a directory (TF_DSV41_HANDOFF_DUMP).
pub const dump_env: [:0]const u8 = "TF_DSV41_HANDOFF_DUMP";

/// The dump directory, or null when unset or blank; a malformed value sets `problem` and fails.
pub fn dumpDir(problem: *[]const u8) !?[]const u8 {
    const env = @import("sys").environment;
    return env.path(dump_env, (env.Process{}).get(dump_env)) catch {
        problem.* = "TF_DSV41_HANDOFF_DUMP is malformed: it must be a directory path without control characters";
        return error.InvalidEnvironment;
    };
}

pub const Export = struct {
    gpa: std.mem.Allocator,
    source: gpu.Source,
    produce: *serve.Produce,
    f: *@import("forward.zig").Forward,
    /// a KV_SPLIT pool: this export is shard 0 and the followers export theirs (cuda_export_split.zig)
    split: bool,

    /// The exporter for `m`, or null with `problem` set; `link` is a `--mcdma-produce` name or `http`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, link: ?[]const u8, checkpoint: []const u8, problem: *[]const u8) !?*Export {
        const dump: []const u8 = (try dumpDir(problem)) orelse "";
        if (link == null and dump.len == 0) return null;
        if (gpu.refusal(m)) |why| {
            problem.* = why;
            return null;
        }
        const cap = @import("sys").environment.read(u64, "TF_PRODUCE_HELD_GIB", 32) catch {
            problem.* = "--mcdma-produce: TF_PRODUCE_HELD_GIB is malformed; the line above names its forms";
            return error.InvalidEnvironment;
        };
        const sp = try split.spillOf(problem);
        const http = if (link) |l| std.mem.eql(u8, l, api.LaneHost.Remote.http_link) else false;
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        const shards: u32 = if (split.splitOf(m)) m.sessions.?.kx.world else 1;
        if (shards > 1 and http) {
            problem.* = "--mcdma-produce http exports a replicated KV pool: unset TF_DSV41_KV_SPLIT, or export on mcdma-rpcd links (one per rank)";
            return null;
        }
        x.* = .{ .gpa = gpa, .source = .{ .ss = m.sessions.? }, .produce = undefined, .f = &m.f, .split = shards > 1 };
        x.produce = try serve.Produce.open(gpa, io, x.source.source(), .{ .link = if (http) null else link, .http = http, .dump = if (dump.len > 0) dump else null, .model = checkpoint, .held_bytes = cap << 30, .shard = 0, .shards = shards, .spill_dir = sp.dir, .spill_min_bytes = sp.min_bytes, .stream = streamOn(link, shards, problem) catch return error.InvalidEnvironment }, problem);
        log.info("dsv41 handoff export: {s}{s}{s}", .{ if (link) |l| l else "no link", if (dump.len > 0) ", files under " else "", dump });
        return x;
    }

    /// TF_DSV41_EXPORT_STREAM (default on), one replicated pool: exports are built while the prompt prefills.
    fn streamOn(link: ?[]const u8, shards: u32, problem: *[]const u8) !bool {
        const set = @import("sys").environment.read(?bool, "TF_DSV41_EXPORT_STREAM", null) catch {
            problem.* = "TF_DSV41_EXPORT_STREAM is malformed: it must be 0 or 1";
            return error.InvalidEnvironment;
        };
        if (!(set orelse true)) return false;
        if (link == null or std.mem.eql(u8, link.?, api.LaneHost.Remote.http_link) or shards > 1) {
            if (set != null) log.info("dsv41 handoff export: TF_DSV41_EXPORT_STREAM needs an mcdma-rpcd link on one replicated pool; exports are built at the cut", .{});
            return false;
        }
        return true;
    }

    /// Stops serving and frees what is held (after the lane host stopped).
    pub fn stop(x: *Export) void {
        x.produce.stop();
        x.gpa.destroy(x);
    }

    pub fn exporter(x: *Export) api.LaneHost.Export.Exporter {
        const over_http = x.produce.opts.http;
        return .{ .ptr = x, .snapshots = .{ .ptr = x, .vtable = &snapshots }, .exported = exported, .failed = failed, .serve = if (over_http) .{ .ptr = x, .open = readOpen, .release = readRelease } else null, .begin = begin, .progress = progress, .save_for = saveFor, .split = .{ .ok = splitOk, .run = splitRun } };
    }

    /// The token a layer-split trigger's reply carries: the split drew none, and the decoder ignores it.
    pub const split_reply_token: u32 = 0;

    /// TF_LS_PRODUCE=1, one replicated KV pool on one slot, a link or HTTP to serve the early export
    fn splitOk(ptr: *anyopaque, cut: u32) bool {
        const x = self(ptr);
        const on = @import("sys").environment.read(bool, "TF_LS_PRODUCE", false) catch false;
        if (!on or x.split or x.produce.opts.shards != 1) return false;
        if (x.produce.opts.link == null and !x.produce.opts.http) return false;
        return x.source.ss.slot_set == null and split_lib.cutOk(geometry.release, cut);
    }

    fn splitRun(ptr: *anyopaque, h: Handoff, prompt: []const u32, cancel: ?api.LaneHost.Export.Cancel) anyerror!u32 {
        try self(ptr).splitPrefill(h, prompt, cancel);
        return split_reply_token;
    }

    /// The most rows of a split chunk: the forward's own segment rows (the 4K workspace has no split program), or less.
    fn splitRows(f: *const @import("forward.zig").Forward) !u32 {
        const seg: u32 = @intCast(f.opts.prefill_rows);
        return @min(seg, try @import("sys").environment.read(u32, "TF_LS_CHUNK", seg));
    }

    /// A layer-split prefill of `ids` (the whole prompt; its last token stays unexported) under `h` (`split_cut`)
    pub fn splitPrefill(x: *Export, h: Handoff, ids: []const u32, cancel: ?api.LaneHost.Export.Cancel) !void {
        const cut = h.split_cut;
        if (!splitOk(x, cut)) return error.SplitRefused;
        if (ids.len < 2 or ids.len - 1 < split_lib.min_tail_rows) return error.SplitTooShort;
        const upto: u64 = ids.len - 1;
        const max = try splitRows(x.f);
        const env = @import("sys").environment;
        const window = split_lib.windowRows(max, try env.read(u32, "TF_LS_WINDOW", split_lib.default_window));
        var run = try x.produce.openEarly(h.id, ids[0..upto], cut, window);
        defer run.close();
        x.splitBody(&run, h, ids, upto, max, cancel) catch |err| {
            run.fail();
            return err;
        };
    }

    fn splitBody(x: *Export, run: *serve.Produce.Early, h: Handoff, ids: []const u32, upto: u64, max: u32, cancel: ?api.LaneHost.Export.Cancel) !void {
        // the slot, empty before and after: pages and reservation back whatever happens (a split keeps no state)
        defer x.f.release() catch |err| log.err("handoff {x}: the split's slot was not released ({t})", .{ &h.id, err });
        try x.source.ss.beginFresh(ids.len, 1);
        // the boundary rows copy out asynchronously into pinned slots and are published, in order, as copies finish
        try x.f.splitBegin(.{ .ptr = run, .rows = sinkRows, .publish = sinkPublish }, max);
        var ended = false;
        defer if (!ended) x.f.splitEnd(false) catch {};
        var chunks: split_lib.Chunks = .{ .n = upto, .max = max };
        const trace = @import("sys").environment.read(bool, "TF_LS_TRACE", false) catch false;
        const t0 = fabric.words.nowNs();
        var issue_ns: u64 = 0;
        while (chunks.next()) |c| {
            if (cancel) |cc| if (cc.cancelled()) return error.Cancelled;
            if (run.gone()) return error.Released;
            const t1 = fabric.words.nowNs();
            try x.f.prefillSplit(ids[c.row0..][0..c.rows], h.split_cut);
            try x.f.splitPoll();
            const t2 = fabric.words.nowNs();
            issue_ns += t2 - t1;
            if (trace) log.info("split chunk {d}: rows {d}+{d}, issued in {d:.1} ms at {d:.0} ms, window waits {d:.1} ms", .{ chunks.i, c.row0, c.rows, nsMs(t2 - t1), nsMs(t2 - t0), nsMs(run.early.stage.blocked_ns.load(.monotonic)) });
        }
        ended = true;
        const t3 = fabric.words.nowNs();
        try x.f.splitEnd(true);
        const t4 = fabric.words.nowNs();
        log.info("handoff {x}: split producer {d} rows in {d} chunks: {d:.0} ms (issuing {d:.0} ms, last copies {d:.0} ms, waiting on the window {d:.0} ms)", .{ &h.id, upto, chunks.i, nsMs(t4 - t0), nsMs(issue_ns), nsMs(t4 - t3), nsMs(run.early.stage.blocked_ns.load(.monotonic)) });
        run.land(try x.produce.saveCut(upto, h.split_cut));
    }

    fn nsMs(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    fn sinkRows(ptr: *anyopaque, row0: u64, n: u64) anyerror![]u8 {
        const run: *serve.Produce.Early = @ptrCast(@alignCast(ptr));
        return run.rows(row0, n);
    }

    fn sinkPublish(ptr: *anyopaque, end: u64) anyerror!void {
        const run: *serve.Produce.Early = @ptrCast(@alignCast(ptr));
        return run.publish(end);
    }

    /// `GET /v1/handoff/<id>`: the held export's replies, a message at a time.
    fn readOpen(ptr: *anyopaque, id: [16]u8, why: *[]const u8) api.LaneHost.Export.Serve.OpenError!api.LaneHost.Export.Stream {
        const r = try self(ptr).produce.read(id, why);
        return .{ .ptr = r, .size = r.size(), .next = readNext, .close = readClose };
    }

    fn readNext(ptr: *anyopaque) anyerror!?[]const u8 {
        const r: *serve.Produce.Reading = @ptrCast(@alignCast(ptr));
        return r.next();
    }

    fn readClose(ptr: *anyopaque) void {
        const r: *serve.Produce.Reading = @ptrCast(@alignCast(ptr));
        r.close();
    }

    /// `DELETE /v1/handoff/<id>`.
    fn readRelease(ptr: *anyopaque, id: [16]u8) bool {
        return self(ptr).produce.discard(id);
    }

    const snapshots: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };

    fn self(ptr: *anyopaque) *Export {
        return @ptrCast(@alignCast(ptr));
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return cx.estimate(at);
    }

    /// The active slot read back; the stream is armed on the engine thread at its cut, so only one slot stands there.
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return @ptrCast(try self(ptr).produce.save(at));
    }

    fn begin(ptr: *anyopaque, h: Handoff, tokens: []const u32) void {
        self(ptr).produce.begin(h.id, tokens);
    }

    fn progress(ptr: *anyopaque, h: Handoff, at: u64) void {
        self(ptr).produce.progress(h.id, at);
    }

    /// The state at the cut: a streamed export's goes without its pool pages (the stream has read them).
    fn saveFor(ptr: *anyopaque, h: Handoff, _: *lanes.Stream, at: u32) anyerror!pc.Saved {
        return @ptrCast(try self(ptr).produce.saveFor(h.id, at));
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.ExportsAreNotRestored;
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        self(ptr).produce.drop(@ptrCast(@alignCast(saved)));
    }

    fn exported(ptr: *anyopaque, h: Handoff, tokens: []const u32, saved: pc.Saved) void {
        const x = self(ptr);
        // the followers read their pages now, at the cut, while this rank converts its own
        if (x.split) ff.sendExport(x.f, h.id, tokens.len) catch |err| log.warn("handoff {x}: the followers were not asked for their shards ({t})", .{ &h.id, err });
        x.produce.exportedBg(h.id, tokens, @ptrCast(@alignCast(saved)));
    }

    fn failed(ptr: *anyopaque, h: Handoff, why: []const u8) void {
        const x = self(ptr);
        x.produce.failed(h.id, why);
        if (x.split) ff.sendExportFail(x.f, h.id) catch {};
    }
};
