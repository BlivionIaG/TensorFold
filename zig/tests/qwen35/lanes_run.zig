//! `lanes <model dir> <prompts.json> <max tokens> [--seed S --temperature T] [--solo] [--report out.json]
//! [--tp N --rank R [--master HOST] [--port P]]`: prompts through the lane core on the HIP backend, every stream at once
//! or one at a time, no end token (replies run out). With --tp one process runs each rank on the visible device of its
//! number; rank 0 runs the core and the others follow it, so every rank gets the same arguments.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen35");

fn readPrompts(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.ArrayHashMap([]u32)) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.ArrayHashMap([]u32), gpa, text, .{ .allocate = .alloc_always });
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    const max_tokens = try std.fmt.parseInt(u32, args[2], 10);
    var sampling: ?lanes.Sampling = null;
    var solo = false;
    var drafts = true;
    var report: ?[]const u8 = null;
    var world: usize = 1;
    var rank: usize = 0;
    var master: []const u8 = "127.0.0.1";
    var port: u16 = 29551;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--tp")) {
            i += 1;
            world = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--rank")) {
            i += 1;
            rank = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--master")) {
            i += 1;
            master = args[i];
        } else if (std.mem.eql(u8, args[i], "--port")) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--solo")) solo = true else if (std.mem.eql(u8, args[i], "--no-drafts")) drafts = false else if (std.mem.eql(u8, args[i], "--report")) {
            i += 1;
            report = args[i];
        } else if (std.mem.eql(u8, args[i], "--seed")) {
            i += 1;
            sampling = .{ .seed = try std.fmt.parseInt(u64, args[i], 10), .temperature = if (sampling) |s| s.temperature else 1.0 };
        } else if (std.mem.eql(u8, args[i], "--temperature")) {
            i += 1;
            var s = sampling orelse lanes.Sampling{ .seed = 0 };
            s.temperature = try std.fmt.parseFloat(f64, args[i]);
            sampling = s;
        } else return error.UnknownOption;
    }
    var prompts = try readPrompts(gpa, io, args[1]);
    defer prompts.deinit();
    const names = prompts.value.map.keys();
    const ids = prompts.value.map.values();
    var longest: usize = 0;
    for (ids) |p| longest = @max(longest, p.len);
    var link: hip.link.Link = undefined;
    var id: ?hip.rccl.UniqueId = null;
    // the unique id starts RCCL's bootstrap thread: the library stays loaded until the engine is gone
    var rccl: ?hip.rccl.Rccl = null;
    defer if (rccl) |*r| r.close();
    if (world > 1) {
        rccl = try hip.rccl.Rccl.open();
        const pair = try hip.link.Link.open(io, rank, world, master, port, if (rank == 0) try rccl.?.uniqueId() else undefined);
        link, id = pair;
    }
    defer if (world > 1) link.close();
    const e = try qwen35.engine.Engine.open(gpa, io, args[0], .{ .capacity = longest + max_tokens + 32, .batch_rows = @max(32, names.len * qwen35.hip_lanes.Hip.max_window), .device = @intCast(rank), .rank = rank, .world = world, .id = id });
    defer e.deinit();
    if (rank > 0) {
        var w = try qwen35.worker.Worker.init(gpa, e);
        defer w.deinit();
        return qwen35.worker.follow(gpa, &w, &link);
    }
    const h = try qwen35.hip_lanes.Hip.init(gpa, e);
    defer h.deinit();
    if (world > 1) h.withLink(&link);
    const rows = qwen35.hip_lanes.Hip.max_window;
    var cfg = try lanes.Config.init(gpa, h.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    var clock = lanes.backend.WallClock{ .io = io };
    const streams = try gpa.alloc(lanes.Stream, names.len);
    defer gpa.free(streams);
    for (streams, names, ids) |*s, name, p| s.* = try lanes.Stream.init(gpa, .{ .id = name, .prompt = p, .max_new = max_tokens, .sampling = sampling, .drafts = drafts });
    defer for (streams) |*s| s.deinit(gpa);
    const t0 = std.Io.Clock.awake.now(io);
    if (solo) {
        for (streams) |*s| try finish(gpa, &cfg, h, clock.clock(), &.{s});
    } else {
        const all = try gpa.alloc(*lanes.Stream, streams.len);
        defer gpa.free(all);
        for (all, streams) |*a, *s| a.* = s;
        try finish(gpa, &cfg, h, clock.clock(), all);
    }
    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
    var out: std.ArrayList(struct { name: []const u8, tokens: []const u32 }) = .empty;
    defer out.deinit(gpa);
    var total: usize = 0;
    for (streams, names) |*s, name| {
        try out.append(gpa, .{ .name = name, .tokens = s.emitted() });
        total += s.emitted().len;
        std.debug.print("{s}: {d} tokens, rounds {d}, accepted {d}\n", .{ name, s.emitted().len, s.rounds, s.accepted });
    }
    std.debug.print("{s}: {d} streams, {d} tokens in {d:.3} s, {d:.1} tok/s\n", .{ if (solo) "solo" else "together", streams.len, total, seconds, @as(f64, @floatFromInt(total)) / seconds });
    if (report) |path| {
        const json = try std.json.Stringify.valueAlloc(gpa, out.items, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
}

fn finish(gpa: std.mem.Allocator, cfg: *const lanes.Config, h: *qwen35.hip_lanes.Hip, clock: lanes.backend.Clock, streams: []const *lanes.Stream) !void {
    var engine = lanes.Engine.init(gpa, cfg, h.backend(), clock);
    defer engine.deinit();
    for (streams) |s| try engine.addStream(s);
    while (engine.live.items.len > 0) try engine.step();
}
