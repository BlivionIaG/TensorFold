//! Hand-off `Exporter` of the DeepSeek-V4.1 CUDA engine: `--mcdma-produce LINK` and TF_DSV41_HANDOFF_DUMP.
const std = @import("std");
const api = @import("engine_api");
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
        const http = if (link) |l| std.mem.eql(u8, l, api.LaneHost.Remote.http_link) else false;
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        const shards: u32 = if (split.splitOf(m)) m.sessions.?.kx.world else 1;
        if (shards > 1 and http) {
            problem.* = "--mcdma-produce http exports a replicated KV pool: unset TF_DSV41_KV_SPLIT, or export on mcdma-rpcd links (one per rank)";
            return null;
        }
        x.* = .{ .gpa = gpa, .source = .{ .ss = m.sessions.? }, .produce = undefined, .f = &m.f, .split = shards > 1 };
        x.produce = try serve.Produce.open(gpa, io, x.source.source(), .{ .link = if (http) null else link, .http = http, .dump = if (dump.len > 0) dump else null, .model = checkpoint, .held_bytes = cap << 30, .shard = 0, .shards = shards }, problem);
        log.info("dsv41 handoff export: {s}{s}{s}", .{ if (link) |l| l else "no link", if (dump.len > 0) ", files under " else "", dump });
        return x;
    }

    /// Stops serving and frees what is held (after the lane host stopped).
    pub fn stop(x: *Export) void {
        x.produce.stop();
        x.gpa.destroy(x);
    }

    pub fn exporter(x: *Export) api.LaneHost.Export.Exporter {
        const over_http = x.produce.opts.http;
        return .{ .ptr = x, .snapshots = .{ .ptr = x, .vtable = &snapshots }, .exported = exported, .failed = failed, .serve = if (over_http) .{ .ptr = x, .open = readOpen, .release = readRelease } else null, .split = .{ .ok = splitOk, .run = splitRun } };
    }

    /// The token a layer-split trigger's reply carries: the split drew none, and the decoder ignores it.
    pub const split_reply_token: u32 = 0;

    /// TF_LS_PRODUCE=1, one replicated KV pool on one slot, a link or HTTP to serve the early export, a cut of the model's.
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

    /// A layer-split prefill of `ids` (the whole prompt; its last token stays unexported) under `h` (`split_cut`), on
    /// the engine thread with the slot empty: the early export is added before any compute, so the decoder's OPEN
    /// succeeds at once; the layers below the cut then prefill chunk by chunk (`split_lib.Chunks`), each chunk's
    /// boundary rows published when its program ends, and the state of those layers lands at the end. The D2H of a
    /// chunk's rows is synchronous (no overlap with the next chunk's compute; the transfer to the decoder is what
    /// overlaps). No sampling and no full-model window run: nothing of the slot is kept. Failure, a closed export or a
    /// gone client: the stage and latch fail (readers get the producer's failure), `failed` is called, the slot is released.
    pub fn splitPrefill(x: *Export, h: Handoff, ids: []const u32, cancel: ?api.LaneHost.Export.Cancel) !void {
        const cut = h.split_cut;
        if (!splitOk(x, cut)) return x.refused(h, error.SplitRefused);
        if (ids.len < 2 or ids.len - 1 < split_lib.min_tail_rows) return x.refused(h, error.SplitTooShort);
        const upto: u64 = ids.len - 1;
        var run = x.produce.openEarly(h.id, ids[0..upto], cut) catch |err| return x.refused(h, err);
        defer run.close();
        x.splitBody(&run, h, ids, upto, cancel) catch |err| {
            run.fail(@errorName(err));
            return err;
        };
    }

    fn refused(x: *Export, h: Handoff, err: anyerror) anyerror {
        x.produce.failed(h.id, @errorName(err));
        return err;
    }

    fn splitBody(x: *Export, run: *serve.Produce.Early, h: Handoff, ids: []const u32, upto: u64, cancel: ?api.LaneHost.Export.Cancel) !void {
        // the slot, empty before and after: pages and reservation back whatever happens (a split keeps no state)
        defer x.f.release() catch |err| log.err("handoff {x}: the split's slot was not released ({t})", .{ &h.id, err });
        try x.source.ss.beginFresh(ids.len, 1);
        var chunks: split_lib.Chunks = .{ .n = upto, .max = try splitRows(x.f) };
        while (chunks.next()) |c| {
            if (cancel) |cc| if (cc.cancelled()) return error.Cancelled;
            if (run.gone()) return error.Released;
            try x.f.prefillSplit(ids[c.row0..][0..c.rows], h.split_cut, run.rows(c.row0, c.rows));
            try run.publish(c.row0 + c.rows);
        }
        run.land(try x.produce.saveCut(upto, h.split_cut));
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
