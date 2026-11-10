//! A dense Qwen model cut at a layer, the two halves on this Mac's GPU, joined by the split pipeline over the mock
//! device's host-memory window; the gate is token identity with the unsplit run of the same engine. The prompt goes
//! through the pipeline a position at a time (decode rows) or in chunks (prompt rows), one sequence or several at once.
const std = @import("std");
const tf = @import("tensorfold");
const split = @import("split");
const q = tf.qwen27;

const Range = q.range.Range;
const Model = q.model.Model;
const peer = q.split_peer;
const pl = split.pipeline;

/// Positions each runner holds: the prompt plus the generated tokens of any run here.
pub const capacity: u32 = 256;
/// Sequences one half serves at once.
pub const max_sequences = 2;

/// Where a layer range's model comes from: a checkpoint directory, or generated weights.
pub const Source = struct {
    ctx: *const anyopaque,
    load: *const fn (ctx: *const anyopaque, a: std.mem.Allocator, range: Range) anyerror!*Model,
    layers: usize,
};

/// One layer range: its model and a stage a sequence (each with its own state, all over the model's one frame), warmed
/// so that no pipeline deadline sees a shader compile.
pub const Half = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    serial: *peer.Serial,
    stages: []*peer.Stage,

    pub fn init(a: std.mem.Allocator, src: Source, range: Range) !Half {
        return initSequences(a, src, range, 1);
    }

    pub fn initSequences(a: std.mem.Allocator, src: Source, range: Range, n: usize) !Half {
        const m = try src.load(src.ctx, a, range);
        errdefer m.deinit();
        const serial = try a.create(peer.Serial);
        errdefer a.destroy(serial);
        serial.* = .{};
        const stages = try a.alloc(*peer.Stage, n);
        errdefer a.free(stages);
        var made: usize = 0;
        errdefer for (stages[0..made]) |s| s.deinit();
        while (made < n) : (made += 1) {
            stages[made] = try peer.Stage.init(a, m, capacity, serial);
            errdefer stages[made].deinit();
            try stages[made].warm();
        }
        return .{ .allocator = a, .model = m, .serial = serial, .stages = stages };
    }

    pub fn deinit(h: Half) void {
        for (h.stages) |s| s.deinit();
        h.allocator.free(h.stages);
        h.allocator.destroy(h.serial);
        h.model.deinit();
    }

    /// Every stage back to position 0.
    pub fn reset(h: Half) !void {
        for (h.stages) |s| try s.reset();
    }
};

/// Nanoseconds per position over a run, prompt positions included.
pub const Timing = struct { ns: u64 = 0, steps: usize = 0 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// The unsplit run of `m`'s engine over `prompt`, then `out.len` greedy tokens, a position at a time.
pub fn unsplit(a: std.mem.Allocator, m: *Model, prompt: []const u32, out: []u32) !Timing {
    return unsplitChunked(a, m, prompt, out, 1);
}

/// The same with the prompt in chunks of up to `chunk` positions, each run as the engine runs a prompt chunk (the
/// prompt kernels), except a chunk of one position, which is a decode step as in the split.
pub fn unsplitChunked(a: std.mem.Allocator, m: *Model, prompt: []const u32, out: []u32, chunk: usize) !Timing {
    var r = try q.decode_round.Runner.init(a, m, 1, capacity);
    defer r.deinit();
    const s = q.session.Session{ .runner = &r };
    try s.step(prompt[0]); // compiles and pages in before the clock
    try r.reset(0);
    const t0 = nowNs();
    var at: usize = 0;
    while (at < prompt.len) {
        const n = @min(chunk, prompt.len - at);
        if (n == 1) try s.step(prompt[at]) else try s.promptChunk(prompt[at..][0..n], true);
        at += n;
    }
    for (out) |*o| {
        o.* = try s.greedy();
        try s.step(o.*);
    }
    return .{ .ns = nowNs() - t0, .steps = prompt.len + out.len };
}

/// Pipelines over two halves: the mock device (its window, its stream thread) and the Metal stages on either side, one
/// pipeline a sequence over its own slots of the window.
const Rig = struct {
    a: std.mem.Allocator,
    mock: *split.mock.Mock,
    dev: split.device.ExternalDevice,
    window: split.device.Window,
    host: Half,
    lps: [max_sequences]peer.LayerPeer = undefined,
    exts: [max_sequences]peer.External = undefined,
    pipes: [max_sequences]pl.Pipeline = undefined,
    ptrs: [max_sequences]*pl.Pipeline = undefined,
    scratch: []u8,
    n: usize,

    /// `cfg.order` names who runs first; the halves are `low` = layers [0, k) and `high` = [k, n).
    fn create(a: std.mem.Allocator, low: Half, high: Half, cfg: pl.Config, mock: split.mock.Options, n: usize) !*Rig {
        std.debug.assert(n >= 1 and n <= max_sequences and low.stages.len >= n and high.stages.len >= n);
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.a = a;
        r.n = n;
        r.mock = try split.mock.Mock.create(a, mock);
        r.dev = r.mock.external();
        errdefer r.dev.deinit();
        r.window = try r.dev.allocWindow(pl.Pipeline.windowBytes(cfg, n));
        errdefer r.dev.freeWindow(r.window);
        // metal_first: the host runs the layers that come first. external_first: the device does.
        const host_low = cfg.order == .metal_first;
        r.host = if (host_low) low else high;
        const device_half = if (host_low) high else low;
        for (r.host.stages[0..n]) |s| try s.bindWindow(r.window.ptr[0..r.window.len]);
        errdefer for (r.host.stages[0..n]) |s| s.unbindWindow();
        const row = cfg.hidden_bytes * cfg.max_rows;
        r.scratch = try a.alloc(u8, row * n);
        errdefer a.free(r.scratch);
        for (0..n) |i| {
            var c = cfg;
            c.window_offset = i * 2 * pl.Pipeline.slotBytes(cfg);
            r.lps[i] = .{ .stage = r.host.stages[i] };
            r.exts[i] = .{ .stage = device_half.stages[i] };
            r.pipes[i] = .init(r.dev, r.window, c, r.exts[i].external(), r.lps[i].peer(), r.scratch[i * row ..][0..row]);
            r.ptrs[i] = &r.pipes[i];
        }
        return r;
    }

    /// The device's stream thread ends first, so no stage is in use once this returns.
    fn destroy(r: *Rig) void {
        for (r.host.stages[0..r.n]) |s| s.unbindWindow();
        r.dev.freeWindow(r.window);
        r.dev.deinit();
        r.a.free(r.scratch);
        r.a.destroy(r);
    }
};

fn config(m: *Model, order: pl.Order, level: pl.Level, rows: usize) pl.Config {
    return .{ .order = order, .level = level, .hidden_bytes = m.config.hidden * 4, .max_rows = @intCast(rows) };
}

/// The split run: stages reset to position 0, then the prompt (in chunks of up to `chunk`) and `out.len` greedy tokens
/// through the pipeline.
pub fn run(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, level: pl.Level, prompt: []const u32, out: []u32, chunk: usize) !Timing {
    try low.reset();
    try high.reset();
    const r = try Rig.create(a, low, high, config(low.model, order, level, chunk), .{}, 1);
    defer r.destroy();
    const t0 = nowNs();
    try split.toy.generateChunked(&r.pipes[0], prompt, out, chunk);
    return .{ .ns = nowNs() - t0, .steps = prompt.len + out.len };
}

/// One sequence a pipeline, all through the device at once (`split.streams`): each `outs[i]` is `works[i]`'s run.
pub fn runStreams(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, level: pl.Level, works: []const split.streams.Work, chunk: usize) !void {
    try low.reset();
    try high.reset();
    const r = try Rig.create(a, low, high, config(low.model, order, level, chunk), .{}, works.len);
    defer r.destroy();
    try split.streams.generate(r.ptrs[0..works.len], works, chunk);
}

/// A device whose writes take far longer than the host waits, then a cut cable: each a named error inside the
/// timeout. The next run on a healthy device from reset stages matches `want`.
pub fn failure(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, prompt: []const u32, want: []const u32) !void {
    try low.reset();
    try high.reset();
    {
        var cfg = config(low.model, order, .a, 1);
        cfg.timeout_ns = 50_000_000;
        const r = try Rig.create(a, low, high, cfg, .{ .write_delay_ns = 400_000_000 }, 1);
        defer r.destroy();
        const t0 = nowNs();
        try std.testing.expectError(error.Timeout, r.pipes[0].step(prompt[0]));
        try std.testing.expect(nowNs() - t0 < 2_000_000_000);
        r.mock.kill();
        r.pipes[0].reset();
        try std.testing.expectError(error.DeviceLost, r.pipes[0].step(prompt[1]));
    }
    const got = try a.alloc(u32, want.len);
    defer a.free(got);
    _ = try run(a, low, high, order, .a, prompt, got, 1);
    try std.testing.expectEqualSlices(u32, want, got);
}

pub const Report = struct { unsplit_ns: u64 = 0, split_ns: u64 = 0, split_steps: usize = 0, runs: usize = 0, chunked_runs: usize = 0, stream_runs: usize = 0 };

/// A prompt with its unsplit runs: a position at a time, and in chunks of `chunk_size`.
pub const Case = struct { prompt: []const u32, single: []const u32, chunked: []const u32 };
pub const chunk_size = 5;

fn expectSame(want: []const u32, got: []const u32, what: []const u8, k: usize, order: pl.Order, level: pl.Level) !void {
    std.testing.expectEqualSlices(u32, want, got) catch |err| {
        std.debug.print("  split at {d} {s} level {s}: {s} tokens differ\n", .{ k, @tagName(order), @tagName(level), what });
        return err;
    };
}

/// Every order and level at each cut. Decode rows: token identity with `cases[0].single`. Prompt chunks: identity with
/// `cases[0].chunked` and with `cases[0].single`. Two sequences through one pipeline each: each is identical to its own
/// chunked run (`cases[0]` and `cases[1]`). The timing of the decode runs is added to `report`.
pub fn cuts(a: std.mem.Allocator, src: Source, points: []const usize, cases: [2]Case, report: *Report) !void {
    const c0 = cases[0];
    const got = try a.alloc(u32, c0.single.len);
    defer a.free(got);
    const got1 = try a.alloc(u32, cases[1].chunked.len);
    defer a.free(got1);
    for (points) |k| {
        const low = try Half.initSequences(a, src, .{ .first = 0, .last = k }, max_sequences);
        defer low.deinit();
        const high = try Half.initSequences(a, src, .{ .first = k, .last = src.layers }, max_sequences);
        defer high.deinit();
        for ([_]pl.Order{ .external_first, .metal_first }) |order| for ([_]pl.Level{ .a, .b }) |level| {
            @memset(got, std.math.maxInt(u32));
            const t = try run(a, low, high, order, level, c0.prompt, got, 1);
            try expectSame(c0.single, got, "decode", k, order, level);
            report.split_ns += t.ns;
            report.split_steps += t.steps;
            report.runs += 1;
            @memset(got, std.math.maxInt(u32));
            _ = try run(a, low, high, order, level, c0.prompt, got, chunk_size);
            try expectSame(c0.chunked, got, "chunked", k, order, level);
            try expectSame(c0.single, got, "chunked vs position by position", k, order, level);
            report.chunked_runs += 1;
            @memset(got, std.math.maxInt(u32));
            @memset(got1, std.math.maxInt(u32));
            const works = [2]split.streams.Work{ .{ .prompt = c0.prompt, .out = got }, .{ .prompt = cases[1].prompt, .out = got1 } };
            try runStreams(a, low, high, order, level, &works, chunk_size);
            try expectSame(c0.chunked, got, "stream 0", k, order, level);
            try expectSame(cases[1].chunked, got1, "stream 1", k, order, level);
            report.stream_runs += 1;
        };
        try failure(a, low, high, .external_first, c0.prompt, c0.single);
        try failure(a, low, high, .metal_first, c0.prompt, c0.single);
        std.debug.print("  cut at {d}: decode, chunked and two-sequence runs identical over {d} generated tokens, both failure modes named\n", .{ k, c0.single.len });
    }
}

/// `prompt`'s unsplit runs on `m`, `n` generated tokens each (owned by `a`; the timing of the first is `report`'s).
pub fn reference(a: std.mem.Allocator, m: *Model, prompt: []const u32, n: usize, report: ?*Report) !Case {
    const single = try a.alloc(u32, n);
    errdefer a.free(single);
    const chunked = try a.alloc(u32, n);
    errdefer a.free(chunked);
    const t = try unsplit(a, m, prompt, single);
    if (report) |r| r.unsplit_ns = t.ns / t.steps;
    _ = try unsplitChunked(a, m, prompt, chunked, chunk_size);
    if (!std.mem.eql(u32, single, chunked)) std.debug.print("  the unsplit run in chunks of {d} differs from the one a position at a time\n", .{chunk_size});
    return .{ .prompt = prompt, .single = single, .chunked = chunked };
}

pub fn freeCase(a: std.mem.Allocator, c: Case) void {
    a.free(c.single);
    a.free(c.chunked);
}
