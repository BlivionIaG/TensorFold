//! The lane host's hand-off `Exporter` for the DeepSeek-V4.1 CUDA engine: `--mcdma-produce LINK` and TF_DSV41_HANDOFF_DUMP.
//!
//! A handoff request's stream is armed at its prompt's cut (draft/lanes.zig `Cut`, target.zig `arm`); the target calls the
//! lane host's mark hook there, which reaches `save` and `exported` below on the engine thread, the slot standing at the cut.
const std = @import("std");
const api = @import("engine_api");
const model = @import("cuda_model.zig");
const serve = @import("cuda_export_serve.zig");
const gpu = @import("cuda_export_gpu.zig");
const cx = @import("cuda_export.zig");
const pc = api.prompt_cache;
const Handoff = api.LaneHost.Export.Handoff;
const log = std.log.scoped(.dsv41);

/// Where the exports are written when set: a directory (TF_DSV41_HANDOFF_DUMP).
pub const dump_env = "TF_DSV41_HANDOFF_DUMP";

pub const Export = struct {
    gpa: std.mem.Allocator,
    source: gpu.Source,
    produce: *serve.Produce,

    /// The exporter for `m`, or null with `problem` set (what to change). `link`: `--mcdma-produce`'s name; the dump
    /// directory comes from TF_DSV41_HANDOFF_DUMP. With neither, no exporter (null, no problem).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, pieces: bool, link: ?[]const u8, checkpoint: []const u8, problem: *[]const u8) !?*Export {
        const dump: []const u8 = if (std.c.getenv(dump_env)) |v| std.mem.span(v) else "";
        if (link == null and dump.len == 0) return null;
        if (gpu.refusal(m, pieces)) |why| {
            problem.* = why;
            return null;
        }
        const cap = @import("sys").environment.read(u64, "TF_PRODUCE_HELD_GIB", 32) catch {
            problem.* = "--mcdma-produce: TF_PRODUCE_HELD_GIB is malformed; the line above names its forms";
            return error.InvalidEnvironment;
        };
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .source = .{ .ss = m.sessions.? }, .produce = undefined };
        x.produce = try serve.Produce.open(gpa, io, x.source.source(), .{ .link = link, .dump = if (dump.len > 0) dump else null, .model = checkpoint, .held_bytes = cap << 30 }, problem);
        log.info("dsv41 handoff export: {s}{s}{s}", .{ if (link) |l| l else "no link", if (dump.len > 0) ", files under " else "", dump });
        return x;
    }

    /// Stops serving and frees what is held (after the lane host stopped).
    pub fn stop(x: *Export) void {
        x.produce.stop();
        x.gpa.destroy(x);
    }

    pub fn exporter(x: *Export) api.LaneHost.Export.Exporter {
        return .{ .ptr = x, .snapshots = .{ .ptr = x, .vtable = &snapshots }, .exported = exported, .failed = failed };
    }

    const snapshots: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };

    fn self(ptr: *anyopaque) *Export {
        return @ptrCast(@alignCast(ptr));
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return cx.estimate(at);
    }

    /// The active slot's state, read back: the stream is armed on the engine thread at its cut, the only slot standing there.
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
        self(ptr).produce.exported(h.id, tokens, @ptrCast(@alignCast(saved)));
    }

    fn failed(ptr: *anyopaque, h: Handoff, why: []const u8) void {
        self(ptr).produce.failed(h.id, why);
    }
};
