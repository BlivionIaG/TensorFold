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
/// Lanes (pipelines, each with a stage a side) and sequences (slots of a stage) one run here serves at most.
pub const max_lanes = 4;
pub const max_sequences = peer.max_slots;

/// Where a layer range's model comes from: a checkpoint directory, or generated weights.
pub const Source = struct {
    ctx: *const anyopaque,
    load: *const fn (ctx: *const anyopaque, a: std.mem.Allocator, range: Range) anyerror!*Model,
    layers: usize,
};

/// One layer range: its model and a stage a lane (each holding the state of several sequences, all over the model's one
/// frame), warmed so that no pipeline deadline sees a shader compile.
pub const Half = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    serial: *peer.Serial,
    stages: []*peer.Stage,

    pub fn init(a: std.mem.Allocator, src: Source, range: Range) !Half {
        return initLanes(a, src, range, 1, 1);
    }

    pub fn initLanes(a: std.mem.Allocator, src: Source, range: Range, n: usize, slots: u32) !Half {
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
            stages[made] = try peer.Stage.init(a, m, capacity, slots, serial);
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
/// pipeline a lane over its own slots of the window.
const Rig = struct {
    a: std.mem.Allocator,
    mock: *split.mock.Mock,
    dev: split.device.ExternalDevice,
    window: split.device.Window,
    host: Half,
    lps: [max_lanes]peer.LayerPeer = undefined,
    exts: [max_lanes]peer.External = undefined,
    pipes: [max_lanes]pl.Pipeline = undefined,
    ptrs: [max_lanes]*pl.Pipeline = undefined,
    scratch: []u8,
    n: usize,

    /// `cfg.order` names who runs first; the halves are `low` = layers [0, k) and `high` = [k, n).
    fn create(a: std.mem.Allocator, low: Half, high: Half, cfg: pl.Config, mock: split.mock.Options, n: usize) !*Rig {
        std.debug.assert(n >= 1 and n <= max_lanes and low.stages.len >= n and high.stages.len >= n);
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

/// One sequence of a `matrix` run: a prompt and the tokens the unsplit run makes after it (`want.len` of them).
pub const Seq = struct { prompt: []const u32, want: []const u32 };

/// `seqs` through `lanes` pipelines at once (`split.streams.generateLanes`), sequence i in lane i % lanes: each sequence's
/// tokens are identical to its own unsplit run (`want`). Prompts go in chunks of `chunk_size`, then the lane's sequences
/// decode together in batches that shrink as they finish.
pub fn matrix(a: std.mem.Allocator, low: Half, high: Half, order: pl.Order, level: pl.Level, seqs: []const Seq, lanes: usize, k: usize) !void {
    std.debug.assert(lanes >= 1 and lanes <= max_lanes and seqs.len >= lanes and seqs.len <= lanes * max_sequences);
    try low.reset();
    try high.reset();
    const outs = try a.alloc(u32, seqs.len * 64);
    defer a.free(outs);
    var works: [max_lanes][max_sequences]split.streams.Work = undefined;
    var counts: [max_lanes]usize = @splat(0);
    for (seqs, 0..) |s, i| {
        const l = i % lanes;
        works[l][counts[l]] = .{ .prompt = s.prompt, .out = outs[i * 64 ..][0..s.want.len] };
        counts[l] += 1;
    }
    var most: usize = 0;
    for (counts[0..lanes]) |c| most = @max(most, c);
    const r = try Rig.create(a, low, high, config(low.model, order, level, @max(chunk_size, most)), .{}, lanes);
    defer r.destroy();
    var ls: [max_lanes]split.streams.Lane = undefined;
    for (ls[0..lanes], 0..) |*l, i| l.* = .{ .pipe = r.ptrs[i], .works = works[i][0..counts[i]] };
    try split.streams.generateLanes(ls[0..lanes], chunk_size);
    for (seqs, 0..) |s, i| {
        std.testing.expectEqualSlices(u32, s.want, outs[i * 64 ..][0..s.want.len]) catch |err| {
            std.debug.print("  split at {d} {s} level {s}, {d} sequences in {d} lanes: sequence {d} differs from its own run\n", .{ k, @tagName(order), @tagName(level), seqs.len, lanes, i });
            return err;
        };
    }
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

pub const Report = struct { unsplit_ns: u64 = 0, split_ns: u64 = 0, split_steps: usize = 0, runs: usize = 0, chunked_runs: usize = 0, matrix_runs: usize = 0 };

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
/// `cases[0].chunked` and with `cases[0].single`. Then `seqs` (up to eight, prompts of different lengths, different
/// numbers of tokens to make) through 1, 2 or 4 lanes: at the cut `full_at` with every count from 1 to `seqs.len`, at the
/// others with four sequences; each sequence identical to its own unsplit run. The timing of the decode runs is added to
/// `report`.
pub fn cuts(a: std.mem.Allocator, src: Source, points: []const usize, full_at: usize, c0: Case, seqs: []const Seq, report: *Report) !void {
    const got = try a.alloc(u32, c0.single.len);
    defer a.free(got);
    for (points) |k| {
        const low = try Half.initLanes(a, src, .{ .first = 0, .last = k }, max_lanes, max_sequences);
        defer low.deinit();
        const high = try Half.initLanes(a, src, .{ .first = k, .last = src.layers }, max_lanes, max_sequences);
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
            const from: usize = if (k == full_at) 1 else 4;
            const to: usize = if (k == full_at) seqs.len else 4;
            for (from..to + 1) |n| for ([_]usize{ 1, 2, 4 }) |lanes| {
                if (lanes > n or n > lanes * max_sequences) continue;
                try matrix(a, low, high, order, level, seqs[0..n], lanes, k);
                report.matrix_runs += 1;
            };
        };
        try failure(a, low, high, .external_first, c0.prompt, c0.single);
        try failure(a, low, high, .metal_first, c0.prompt, c0.single);
        std.debug.print("  cut at {d}: decode, chunked and {d}-sequence runs identical, both failure modes named\n", .{ k, if (k == full_at) seqs.len else 4 });
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

/// The eight sequences of a `matrix` run: prompts of 5 to 20 tokens cut from `source` (at least 27 tokens), and 10 to 32
/// tokens to make after each, so that a lane's decode batches shrink. The unsplit run of each is made on `m` with the
/// same chunking as the split runs. Free with `freeSeqs`.
pub fn mixed(a: std.mem.Allocator, m: *Model, source: []const u32) ![]Seq {
    const prompt_lens = [max_sequences]usize{ 20, 9, 13, 5, 17, 7, 15, 11 };
    const out_lens = [max_sequences]usize{ 24, 12, 32, 18, 28, 16, 10, 20 };
    std.debug.assert(source.len >= 27);
    const seqs = try a.alloc(Seq, max_sequences);
    var made: usize = 0;
    errdefer {
        for (seqs[0..made]) |s| a.free(s.want);
        a.free(seqs);
    }
    while (made < max_sequences) : (made += 1) {
        const want = try a.alloc(u32, out_lens[made]);
        errdefer a.free(want);
        const prompt = source[2 * made ..][0..prompt_lens[made]];
        _ = try unsplitChunked(a, m, prompt, want, chunk_size);
        seqs[made] = .{ .prompt = prompt, .want = want };
    }
    return seqs;
}

pub fn freeSeqs(a: std.mem.Allocator, seqs: []const Seq) void {
    for (seqs) |s| a.free(s.want);
    a.free(seqs);
}

/// What `batchBits` saw: logits rows compared, and how many differ from the same row computed alone in any bit.
pub const Bits = struct { rows: usize = 0, differing: usize = 0 };

/// A decode batch row against the same row alone, in the bits of the logits. One stage runs the first four of `seqs`
/// (prompts, then `steps` decode steps each over a varying subset and order of the four slots) as batches, another runs
/// each named row on its own, both over `m` (the whole model); the logits row of every batch row is compared with the
/// single row's. Tokens come from the batch run, so both stages see the same inputs.
pub fn batchBits(a: std.mem.Allocator, m: *Model, seqs: []const Seq, steps: usize) !Bits {
    const n = 4;
    const vocab = m.config.vocab;
    var serial: peer.Serial = .{};
    const batch = try peer.Stage.init(a, m, capacity, n, &serial);
    defer batch.deinit();
    const single = try peer.Stage.init(a, m, capacity, n, &serial);
    defer single.deinit();
    const logits = m.frame.get(.logits).buffer.slice(u16, n * vocab);
    var pending: [n]u32 = undefined;
    for (seqs[0..n], 0..) |s, i| for ([2]*peer.Stage{ batch, single }) |st| {
        const slot = [1]u8{@intCast(i)};
        try st.launch(.{ .tokens = s.prompt }, .own, &slot);
        try st.finish();
        pending[i] = try q.session.argmax(logits[0..vocab]);
    };
    var bits = Bits{};
    const subsets = [3][]const u8{ &.{ 0, 1, 2, 3 }, &.{ 3, 1 }, &.{ 2, 0, 3 } };
    const rows = try a.alloc(u16, n * vocab);
    defer a.free(rows);
    for (0..steps) |step| {
        const slots = subsets[step % subsets.len];
        var ids: [n]u32 = undefined;
        for (slots, ids[0..slots.len]) |s, *id| id.* = pending[s];
        try batch.launch(.{ .tokens = ids[0..slots.len] }, .own, slots);
        try batch.finish();
        @memcpy(rows[0 .. slots.len * vocab], logits[0 .. slots.len * vocab]);
        for (slots, 0..) |s, i| {
            try single.launch(.{ .tokens = ids[i..][0..1] }, .own, &[1]u8{s});
            try single.finish();
            bits.rows += 1;
            bits.differing += @intFromBool(!std.mem.eql(u16, rows[i * vocab ..][0..vocab], logits[0..vocab]));
            pending[s] = try q.session.argmax(rows[i * vocab ..][0..vocab]);
        }
    }
    return bits;
}
