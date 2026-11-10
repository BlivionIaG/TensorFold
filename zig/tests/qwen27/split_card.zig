//! The card's side of the Qwen3.5-4B split between this Mac's GPU and a HIP card, in two processes: this one owns the
//! card and runs the pipeline, its layer range on the card's engine; `tf-offload-metal serve` holds the other range on the
//! Mac's GPU and is reached over a socket, the window being shared memory both map. Every configuration is run teacher
//! forced on the tokens in the reference file (`tf-offload-metal ref`) and compared position by position, then free
//! running. Opens the card: run it only with the card to yourself.
//!   tf-offload-card MODEL_DIR REF_FILE SOCKET metal_first|external_first CUT [streams=N] [lanes=L] [chunk=C] [repeat=R]
//!   tf-offload-card MODEL_DIR REF_FILE unsplit [streams=N] [lanes=L] [chunk=C] [repeat=R]
//! With `streams`, `lanes`, `chunk` or `repeat` (any of them, `streams=1` too) the run is the pipelined one instead: N sequences share the card and the Mac's GPU
//! (`tf-offload-metal serve SOCKET FIRST LAST L ceil(N/L)`), in L lanes (default N, one sequence each); the sequences of a lane decode in one
//! batched step, a row each. `unsplit` is the whole model resident on the card with only tokens crossing the link. Each prompt (the reference's, repeated R times) goes through in steps of
//! C positions, then the reference's tokens are generated; the prefill and the decode rates are printed, and every sequence's
//! tokens are checked (R = 1: against the reference file; else against the same run with one position a step).
const std = @import("std");
const split = @import("split");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const pl = split.pipeline;
const Engine = qwen35.engine.Engine;
const Stage = qwen35.split_stage.HipLayerStage;

/// Positions the engine holds: the prompt plus the generated tokens of any run here.
const capacity: u32 = 256;

fn nowNs() u64 {
    return split.device.nowNs();
}

/// One layer range on the card: its engine (opened for single-row decode steps and prompt chunks of up to `max_rows`),
/// one stage a sequence over it, and the adapter that queues the window's waits and writes on the engine's own stream.
const Half = struct {
    a: std.mem.Allocator,
    e: *Engine,
    stages: []Stage,
    dev: *hip.offload.HipExternal,

    /// `lanes` stages, each serving `slots` sequences (a decode batch is a row of each).
    fn init(a: std.mem.Allocator, io: std.Io, dir: []const u8, range: qwen35.view.Range, lanes: usize, slots: usize, max_rows: usize) !Half {
        const e = try Engine.open(a, io, dir, .{ .capacity = capacity, .streams = 1, .batch_rows = slots, .prompt_rows = max_rows, .range = range });
        errdefer e.deinit();
        const stages = try a.alloc(Stage, lanes);
        errdefer a.free(stages);
        var made: usize = 0;
        errdefer for (stages[0..made]) |*st| st.deinit();
        while (made < lanes) : (made += 1) {
            stages[made] = try Stage.init(e, capacity, .{ .max_rows = max_rows, .slots = slots });
            try stages[made].warm();
        }
        const dev = try hip.offload.HipExternal.create(a, &e.driver, 0, .{ .peak_bandwidth = 410_000_000_000, .stream = e.stream });
        return .{ .a = a, .e = e, .stages = stages, .dev = dev };
    }

    /// The first sequence's stage, the one the single-sequence runs use.
    fn stage(s: Half) *Stage {
        return &s.stages[0];
    }

    fn deinit(s: Half) void {
        for (s.stages) |*st| st.deinit();
        s.a.free(s.stages);
        s.e.drain();
        s.dev.external().deinit();
        s.e.deinit();
    }
};

fn lessU64(_: void, x: u64, y: u64) bool {
    return x < y;
}

fn agree(a: []const u32, b: []const u32) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(x == y);
    return n;
}

/// A pipeline over the window for one pass: both stages back at position 0, the window cleared.
const Pass = struct {
    a: std.mem.Allocator,
    half: Half,
    window: split.device.Window,
    cfg: pl.Config,
    client: ?*split.remote.Client,
    scratch: []u8,
    p: pl.Pipeline,

    fn begin(a: std.mem.Allocator, half: Half, window: split.device.Window, cfg: pl.Config, client: ?*split.remote.Client) !*Pass {
        const s = try a.create(Pass);
        errdefer a.destroy(s);
        try half.stage().reset();
        if (client) |cl| try cl.reset();
        @memset(window.bytes(), 0);
        s.* = .{ .a = a, .half = half, .window = window, .cfg = cfg, .client = client, .scratch = try a.alloc(u8, cfg.hidden_bytes), .p = undefined };
        s.p = .init(half.dev.external(), window, cfg, half.stage().external(), if (client) |cl| cl.peer() else null, s.scratch);
        return s;
    }

    fn end(s: *Pass) void {
        s.a.free(s.scratch);
        s.a.destroy(s);
    }
};

const Timed = struct { mean_us: f64, p50_us: f64, p99_us: f64, phases: pl.Phases, preds: []u32 };

/// The sequence teacher forced through a pass; the time of every position kept.
fn forced(a: std.mem.Allocator, half: Half, window: split.device.Window, cfg: pl.Config, client: ?*split.remote.Client, seq: []const u32) !Timed {
    const s = try Pass.begin(a, half, window, cfg, client);
    defer s.end();
    const preds = try a.alloc(u32, seq.len);
    errdefer a.free(preds);
    const times = try a.alloc(u64, seq.len);
    defer a.free(times);
    var sum: u64 = 0;
    for (seq, preds, times) |t, *o, *ns| {
        const t0 = nowNs();
        o.* = try s.p.step(t);
        ns.* = nowNs() - t0;
        sum += ns.*;
    }
    // the first position pays the engine's first launches: the rest are the steady state
    std.mem.sort(u64, times[1..], {}, lessU64);
    const rest = times[1..];
    const us = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e3;
        }
    }.of;
    return .{ .mean_us = us(sum) / @as(f64, @floatFromInt(seq.len)), .p50_us = us(rest[rest.len / 2]), .p99_us = us(rest[rest.len * 99 / 100]), .phases = s.p.phases, .preds = preds };
}

/// The prompt, then `out.len` greedy tokens free running.
fn free(a: std.mem.Allocator, half: Half, window: split.device.Window, cfg: pl.Config, client: ?*split.remote.Client, prompt: []const u32, out: []u32) !void {
    const s = try Pass.begin(a, half, window, cfg, client);
    defer s.end();
    try split.toy.generate(&s.p, prompt, out);
}

/// What the logits of a run at every position say against the reference's: the cosine of the two rows (min and mean), and
/// where the top-1 token differs, how far ahead of its runner-up the reference's was.
fn fidelity(a: std.mem.Allocator, half: Half, window: split.device.Window, cfg: pl.Config, client: ?*split.remote.Client, seq: []const u32, ref_logits: []const f32, ref_preds: []const u32) !void {
    const vocab = ref_logits.len / seq.len;
    const s = try Pass.begin(a, half, window, cfg, client);
    defer s.end();
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    var min_cos: f64 = 2;
    var sum_cos: f64 = 0;
    var worst: usize = 0;
    var differ: usize = 0;
    std.debug.print("  logits against the Metal run, {d} positions of {d}:", .{ seq.len, vocab });
    for (seq, 0..) |t, i| {
        const pred = try s.p.step(t);
        try half.stage().readLogits(row);
        const want = ref_logits[i * vocab ..][0..vocab];
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (row, want) |x, y| {
            dot += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
        }
        const cos = dot / @sqrt(na * nb);
        sum_cos += cos;
        if (cos < min_cos) {
            min_cos = cos;
            worst = i;
        }
        if (pred != ref_preds[i]) {
            var second: f32 = -std.math.inf(f32);
            for (want, 0..) |v, k| {
                if (k != ref_preds[i] and v > second) second = v;
            }
            if (differ == 0) std.debug.print("\n", .{});
            std.debug.print("    position {d}: card top-1 {d}, Metal top-1 {d} ahead of its runner-up by {d:.3} (logit {d:.3})\n", .{ i, pred, ref_preds[i], want[ref_preds[i]] - second, want[ref_preds[i]] });
            differ += 1;
        }
    }
    std.debug.print("  cosine min {d:.6} (position {d}), mean {d:.6}; top-1 differs at {d}\n", .{ min_cos, worst, sum_cos / @as(f64, @floatFromInt(seq.len)), differ });
}

/// `outs.len` sequences of the same prompt in `lanes` pipelines over the shared window (a lane's sequences decode in one batched
/// step), the prompt in steps of up to `chunk` positions, then `outs[i].len` greedy tokens; returns the wall time. Both sides
/// start every sequence at position 0. `client` null: the whole model is on the card.
fn pipelined(a: std.mem.Allocator, half: Half, window: split.device.Window, base: pl.Config, client: ?*split.remote.Client, chunk: usize, prompt: []const u32, outs: []const []u32) !u64 {
    const lanes = half.stages.len;
    const per = (outs.len + lanes - 1) / lanes;
    const used = (outs.len + per - 1) / per;
    for (half.stages[0..used]) |*st| try st.reset();
    if (client) |cl| try cl.reset();
    @memset(window.bytes(), 0);
    const slot = pl.Pipeline.slotBytes(base);
    const row = base.max_rows * base.hidden_bytes;
    const scratch = try a.alloc(u8, used * row);
    defer a.free(scratch);
    var pipes: [split.streams.max_streams]pl.Pipeline = undefined;
    var lane_list: [split.streams.max_streams]split.streams.Lane = undefined;
    var works: [split.streams.max_streams * pl.max_slots]split.streams.Work = undefined;
    for (outs, works[0..outs.len]) |o, *w| w.* = .{ .prompt = prompt, .out = o };
    for (0..used) |i| {
        var cfg = base;
        cfg.window_offset = i * 2 * slot;
        const peer: ?pl.Peer = if (client) |cl| cl.peerOf(@intCast(i)) else null;
        pipes[i] = .init(half.dev.external(), window, cfg, half.stages[i].external(), peer, scratch[i * row ..][0..row]);
        lane_list[i] = .{ .pipe = &pipes[i], .works = works[i * per .. @min((i + 1) * per, outs.len)] };
    }
    const t0 = nowNs();
    try split.streams.generateLanes(lane_list[0..used], chunk);
    return nowNs() - t0;
}

/// The pipelined run (see the header): one untimed pass, then the prompt alone and the prompt with the tokens, so the
/// decode time is the difference; every sequence's tokens against `want`.
fn pipelinedReport(a: std.mem.Allocator, half: Half, window: split.device.Window, cfg: pl.Config, client: ?*split.remote.Client, chunk: usize, prompt: []const u32, want: []const u32, n: usize) !void {
    const full = try a.alloc(u32, n * want.len);
    defer a.free(full);
    const outs = try a.alloc([]u32, n);
    defer a.free(outs);
    const none = try a.alloc([]u32, n);
    defer a.free(none);
    for (outs, none, 0..) |*o, *z, i| {
        o.* = full[i * want.len ..][0..want.len];
        z.* = full[0..0];
    }
    _ = try pipelined(a, half, window, cfg, client, chunk, prompt, outs);
    const pre = try pipelined(a, half, window, cfg, client, chunk, prompt, none);
    @memset(full, std.math.maxInt(u32));
    const all = try pipelined(a, half, window, cfg, client, chunk, prompt, outs);
    var same: usize = 0;
    for (outs) |o| same += @intFromBool(std.mem.eql(u32, o, want));
    const sec = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.of;
    const nf: f64 = @floatFromInt(n);
    const dec = all -| pre;
    std.debug.print("{s:<14} {s:<5} streams {d} lanes {d} chunk {d}: prefill {d} positions x {d} in {d:.1} ms = {d:.1} tok/s; decode {d} tokens x {d} in {d:.1} ms = {d:.1} tok/s ({d:.1} a stream); {d}/{d} sequences token-identical\n", .{
        @tagName(cfg.order),
        @tagName(cfg.level),
        n,
        half.stages.len,
        chunk,
        prompt.len,
        n,
        sec(pre) * 1e3,
        nf * @as(f64, @floatFromInt(prompt.len)) / sec(pre),
        want.len,
        n,
        sec(dec) * 1e3,
        nf * @as(f64, @floatFromInt(want.len)) / sec(dec),
        @as(f64, @floatFromInt(want.len)) / sec(dec),
        same,
        n,
    });
}

/// `key=value` options after the cut, as numbers; a missing one is `default`.
fn option(args: []const []const u8, key: []const u8, default: usize) !usize {
    for (args) |arg| {
        if (arg.len > key.len and arg[key.len] == '=' and std.mem.eql(u8, arg[0..key.len], key)) return std.fmt.parseInt(usize, arg[key.len + 1 ..], 10);
    }
    return default;
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) return error.MissingArgument;
    const dir = args[1];
    const ref = try split.reference.load(a, args[2]);
    defer ref.deinit(a);
    const seq = try ref.sequence(a);
    defer a.free(seq);
    const layers: usize = ref.layers;
    const gen_from = ref.prompt.len - 1;
    const unsplit = std.mem.eql(u8, args[3], "unsplit");
    var order: pl.Order = .external_only;
    var cut: usize = layers;
    var sock: []const u8 = "";
    var levels: []const pl.Level = &.{.a};
    if (!unsplit) {
        if (args.len < 6) return error.MissingArgument;
        sock = args[3];
        order = std.meta.stringToEnum(pl.Order, args[4]) orelse return error.UnknownOrder;
        cut = try std.fmt.parseInt(usize, args[5], 10);
        if (order == .metal_first) levels = &.{ .a, .b };
    }
    const range: qwen35.view.Range = switch (order) {
        .external_only => .whole,
        .metal_first => .{ .first = cut, .last = layers },
        .external_first => .{ .first = 0, .last = cut },
    };
    const first_option: usize = if (unsplit) 4 else 6;
    const extra: []const []const u8 = if (args.len > first_option) args[first_option..] else &.{};
    const n_streams = try option(extra, "streams", 1);
    const chunk = try option(extra, "chunk", 1);
    const repeat = try option(extra, "repeat", 1);
    const n_lanes = try option(extra, "lanes", n_streams);
    const piped = n_streams > 1 or chunk > 1 or repeat > 1 or try option(extra, "streams", 0) > 0 or try option(extra, "lanes", 0) > 0;
    if (n_streams == 0 or n_lanes == 0 or n_lanes > n_streams or n_lanes > split.streams.max_streams or chunk == 0 or repeat == 0) return error.BadOption;
    const per_lane = (n_streams + n_lanes - 1) / n_lanes;
    if (per_lane > pl.max_slots) return error.BadOption;
    // the lanes actually used: the sequences go to them `per_lane` at a time
    const used_lanes = (n_streams + per_lane - 1) / per_lane;
    const max_rows: usize = @max(chunk, per_lane);
    const half = try Half.init(a, io, dir, range, used_lanes, per_lane, max_rows);
    defer half.deinit();
    const dev = half.dev.external();
    const hidden_bytes: usize = @as(usize, ref.hidden) * 4;
    const window = try dev.allocWindow(pl.Pipeline.windowBytes(.{ .order = order, .hidden_bytes = hidden_bytes, .max_rows = @intCast(max_rows) }, used_lanes));
    defer dev.freeWindow(window);
    var client: split.remote.Client = undefined;
    if (!unsplit) {
        const ex = try half.dev.exportWindow(window);
        defer _ = std.c.close(ex.fd);
        client = try split.remote.Client.connect(sock, window.bytes(), ex.fd, ex.offset, ex.size, 600_000_000_000);
    }
    defer if (!unsplit) client.close();
    const cl: ?*split.remote.Client = if (unsplit) null else &client;
    if (piped) {
        const prompt = try a.alloc(u32, ref.prompt.len * repeat);
        defer a.free(prompt);
        for (0..repeat) |r| @memcpy(prompt[r * ref.prompt.len ..][0..ref.prompt.len], ref.prompt);
        if (prompt.len + ref.want.len > capacity) return error.PromptTooLong;
        // what the tokens must be: the reference's, or (a longer prompt) a run with one position a step
        const want = try a.dupe(u32, ref.want);
        defer a.free(want);
        for (levels) |level| {
            const cfg: pl.Config = .{ .order = order, .level = level, .hidden_bytes = hidden_bytes, .max_rows = @intCast(max_rows), .timeout_ns = 60_000_000_000 };
            if (repeat > 1 and level == levels[0]) {
                // the tokens a long prompt must give: the same run one position a step, one sequence
                const outs = [1][]u32{want};
                _ = try pipelined(a, half, window, cfg, cl, 1, prompt, &outs);
            }
            try pipelinedReport(a, half, window, cfg, cl, chunk, prompt, want, n_streams);
        }
        return;
    }
    std.debug.print("{s:<14} {s:<5} {s:>4} {s:>11} {s:>10} {s:>10} {s:>16} {s:>10}\n", .{ "order", "level", "cut", "mean us/pos", "p50 us", "p99 us", "top-1 vs metal", "free==ref" });
    const logits_path = try std.fmt.allocPrintSentinel(a, "{s}.logits", .{args[2]}, 0);
    defer a.free(logits_path);
    const ref_logits: ?[]f32 = split.reference.loadLogits(a, logits_path) catch null;
    defer if (ref_logits) |l| a.free(l);
    for (levels) |level| {
        const cfg: pl.Config = .{ .order = order, .level = level, .hidden_bytes = hidden_bytes, .timeout_ns = 60_000_000_000 };
        const t = try forced(a, half, window, cfg, cl, seq);
        defer a.free(t.preds);
        const got = try a.alloc(u32, ref.want.len);
        defer a.free(got);
        @memset(got, std.math.maxInt(u32));
        try free(a, half, window, cfg, cl, ref.prompt, got);
        const same = std.mem.eql(u32, got, ref.want);
        std.debug.print("{s:<14} {s:<5} {d:>4} {d:>11.1} {d:>10.1} {d:>10.1} {d:>8}/{d:<7} {s:>10}  (generated {d}/{d})\n", .{
            @tagName(order),
            @tagName(level),
            cut,
            t.mean_us,
            t.p50_us,
            t.p99_us,
            agree(t.preds, ref.preds),
            seq.len,
            if (same) "yes" else "NO",
            agree(t.preds[gen_from..], ref.preds[gen_from..]),
            seq.len - gen_from,
        });
        if (level == .b and order == .metal_first) std.debug.print("  stamped hand-off: status word after the last step {d} (0: every group's stamp was current)\n", .{try half.stage().stampStatus()});
        const n: f64 = @floatFromInt(seq.len);
        const ph = t.phases;
        std.debug.print("  us/position: record {d:.0}, host stage {d:.0}, finish {d:.0}, wait for the card {d:.0}\n", .{ @as(f64, @floatFromInt(ph.queue_ns)) / 1e3 / n, @as(f64, @floatFromInt(ph.peer_ns)) / 1e3 / n, @as(f64, @floatFromInt(ph.finish_ns)) / 1e3 / n, @as(f64, @floatFromInt(ph.wait_ns)) / 1e3 / n });
        if (ref_logits != null and order != .external_first) try fidelity(a, half, window, cfg, cl, seq, ref_logits.?, ref.preds);
    }
}
