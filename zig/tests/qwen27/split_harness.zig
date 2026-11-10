//! A dense Qwen model cut at a layer, the two halves on this Mac's GPU, joined by the split pipeline over the mock
//! device's host-memory window; the gate is token identity with the unsplit run of the same engine. Prompts go through
//! the pipeline one token at a time (decode rows): batch prefill across the window is not part of this.
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

/// Where a layer range's model comes from: a checkpoint directory, or generated weights.
pub const Source = struct {
    ctx: *const anyopaque,
    load: *const fn (ctx: *const anyopaque, a: std.mem.Allocator, range: Range) anyerror!*Model,
    layers: usize,
};

/// One layer range: its model and its stage, warmed so that no pipeline deadline sees a shader compile.
pub const Half = struct {
    model: *Model,
    stage: *peer.Stage,

    pub fn init(a: std.mem.Allocator, src: Source, range: Range) !Half {
        const m = try src.load(src.ctx, a, range);
        errdefer m.deinit();
        const stage = try peer.Stage.init(a, m, capacity);
        errdefer stage.deinit();
        try stage.warm();
        return .{ .model = m, .stage = stage };
    }

    pub fn deinit(h: Half) void {
        h.stage.deinit();
        h.model.deinit();
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
    var r = try q.decode_round.Runner.init(a, m, 1, capacity);
    defer r.deinit();
    const s = q.session.Session{ .runner = &r };
    try s.step(prompt[0]); // compiles and pages in before the clock
    try r.reset(0);
    const t0 = nowNs();
    for (prompt) |t| try s.step(t);
    for (out) |*o| {
        o.* = try s.greedy();
        try s.step(o.*);
    }
    return .{ .ns = nowNs() - t0, .steps = prompt.len + out.len };
}

/// A pipeline over two halves: the mock device (its window, its stream thread) and the Metal stages on either side.
const Rig = struct {
    a: std.mem.Allocator,
    mock: *split.mock.Mock,
    dev: split.device.ExternalDevice,
    window: split.device.Window,
    host: *peer.Stage,
    lp: peer.LayerPeer,
    ext: peer.External,
    scratch: []u8,
    p: pl.Pipeline,

    /// `order` names who runs first; the halves are `low` = layers [0, k) and `high` = [k, n).
    fn create(a: std.mem.Allocator, low: Half, high: Half, cfg: pl.Config, mock: split.mock.Options) !*Rig {
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.a = a;
        r.mock = try split.mock.Mock.create(a, mock);
        r.dev = r.mock.external();
        errdefer r.dev.deinit();
        r.window = try r.dev.allocWindow(pl.Pipeline.slotBytes(cfg) * 2);
        errdefer r.dev.freeWindow(r.window);
        // metal_first: the host runs the layers that come first. external_first: the device does.
        const host_low = cfg.order == .metal_first;
        r.host = if (host_low) low.stage else high.stage;
        try r.host.bindWindow(r.window.ptr[0..r.window.len]);
        errdefer r.host.unbindWindow();
        r.lp = .{ .stage = r.host };
        r.ext = .{ .stage = if (host_low) high.stage else low.stage };
        r.scratch = try a.alloc(u8, cfg.hidden_bytes);
        r.p = .init(r.dev, r.window, cfg, r.ext.external(), r.lp.peer(), r.scratch);
        return r;
    }

    /// The device's stream thread ends first, so no stage is in use once this returns.
    fn destroy(r: *Rig) void {
        r.host.unbindWindow();
        r.dev.freeWindow(r.window);
        r.dev.deinit();
        r.a.free(r.scratch);
        r.a.destroy(r);
    }
};

fn config(m: *Model, order: pl.Order, level: pl.Level) pl.Config {
    return .{ .order = order, .level = level, .hidden_bytes = m.config.hidden * 4 };
}

/// The split run: stages reset to position 0, then the prompt and `out.len` greedy tokens through the pipeline.
pub fn run(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, level: pl.Level, prompt: []const u32, out: []u32) !Timing {
    try low.stage.reset();
    try high.stage.reset();
    const r = try Rig.create(a, low, high, config(low.model, order, level), .{});
    defer r.destroy();
    const t0 = nowNs();
    try split.toy.generate(&r.p, prompt, out);
    return .{ .ns = nowNs() - t0, .steps = prompt.len + out.len };
}

/// A device whose writes take far longer than the host waits, then a cut cable: each a named error inside the
/// timeout. The next run on a healthy device from reset stages matches `want`.
pub fn failure(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, prompt: []const u32, want: []const u32) !void {
    try low.stage.reset();
    try high.stage.reset();
    {
        var cfg = config(low.model, order, .a);
        cfg.timeout_ns = 50_000_000;
        const r = try Rig.create(a, low, high, cfg, .{ .write_delay_ns = 400_000_000 });
        defer r.destroy();
        const t0 = nowNs();
        try std.testing.expectError(error.Timeout, r.p.step(prompt[0]));
        try std.testing.expect(nowNs() - t0 < 2_000_000_000);
        r.mock.kill();
        r.p.reset();
        try std.testing.expectError(error.DeviceLost, r.p.step(prompt[1]));
    }
    const got = try a.alloc(u32, want.len);
    defer a.free(got);
    _ = try run(a, low, high, order, .a, prompt, got);
    try std.testing.expectEqualSlices(u32, want, got);
}

pub const Report = struct { unsplit_ns: u64 = 0, split_ns: u64 = 0, split_steps: usize = 0, runs: usize = 0 };

/// Every order and level at each cut: token identity with `want`, and the timing of the split runs.
pub fn cuts(a: std.mem.Allocator, src: Source, points: []const usize, prompt: []const u32, want: []const u32, report: *Report) !void {
    const got = try a.alloc(u32, want.len);
    defer a.free(got);
    for (points) |k| {
        const low = try Half.init(a, src, .{ .first = 0, .last = k });
        defer low.deinit();
        const high = try Half.init(a, src, .{ .first = k, .last = src.layers });
        defer high.deinit();
        for ([_]pl.Order{ .external_first, .metal_first }) |order| for ([_]pl.Level{ .a, .b }) |level| {
            @memset(got, std.math.maxInt(u32));
            const t = try run(a, low, high, order, level, prompt, got);
            std.testing.expectEqualSlices(u32, want, got) catch |err| {
                std.debug.print("  split at {d} {s} level {s}: tokens differ\n", .{ k, @tagName(order), @tagName(level) });
                return err;
            };
            report.split_ns += t.ns;
            report.split_steps += t.steps;
            report.runs += 1;
        };
        try failure(a, low, high, .external_first, prompt, want);
        try failure(a, low, high, .metal_first, prompt, want);
        std.debug.print("  cut at {d}: 4 runs token-identical over {d} generated tokens, both failure modes named\n", .{ k, want.len });
    }
}
