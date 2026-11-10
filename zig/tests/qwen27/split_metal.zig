//! The Mac's side of a model split with a card in another process (TF_SPLIT_QWEN_DIR, the Qwen3.5-4B MLX 4-bit
//! directory). `ref FILE [TOKENS]` runs the model unsplit on this GPU and writes the tokens and predictions the card's
//! process compares with (split.reference). `serve SOCKET FIRST LAST [LANES [SEQS_PER_LANE]]` holds layers [FIRST, LAST), a
//! stage for each of LANES (default 1; one pipeline each, the peers of split.remote.Host) holding the state of SEQS_PER_LANE
//! sequences (default 1), and is the host stage of the pipeline that process runs (split.remote): the window is
//! the card's pinned memory, mapped here and wrapped for Metal. `parallel REFERENCE SEQUENCES [CHUNK [REPEAT]]` is the baseline the
//! split is compared with: the Mac alone serving SEQUENCES concurrent copies of the reference's prompt in one shared
//! decode round (one row a sequence, one engine), each prompt (REPEAT times over, default 1) prefilled in steps of CHUNK positions (default 24). With REPEAT above 1 the
//! sequences are compared with the first, not with the reference.
const std = @import("std");
const tokenizer = @import("tokenizer");
const split = @import("split");
const h = @import("split_harness.zig");
const tf = @import("tensorfold");
const q = tf.qwen27;

const peer = q.split_peer;

fn loadRange(ctx: *const anyopaque, a: std.mem.Allocator, range: q.range.Range) anyerror!*q.model.Model {
    const io: *const std.Io = @ptrCast(@alignCast(ctx));
    return q.model.Model.loadRange(a, io.*, std.mem.span(std.c.getenv("TF_SPLIT_QWEN_DIR").?), 128, range);
}

/// The stages as the remote host, one a sequence: the pipeline binds the window it maps, resets between runs.
const Served = struct {
    half: h.Half,
    lps: []peer.LayerPeer,
    peers: []split.pipeline.Peer,

    fn init(a: std.mem.Allocator, half: h.Half) !Served {
        const lps = try a.alloc(peer.LayerPeer, half.stages.len);
        errdefer a.free(lps);
        const peers = try a.alloc(split.pipeline.Peer, half.stages.len);
        for (lps, peers, half.stages) |*lp, *p, stage| {
            lp.* = .{ .stage = stage };
            p.* = lp.peer();
        }
        return .{ .half = half, .lps = lps, .peers = peers };
    }
    fn deinit(s: Served, a: std.mem.Allocator) void {
        a.free(s.peers);
        a.free(s.lps);
        s.half.deinit();
    }
    fn host(s: *Served) split.remote.Host {
        return .{ .ptr = s, .bind = bind, .unbind = unbind, .reset = reset, .peers = s.peers };
    }
    fn me(ptr: *anyopaque) *Served {
        return @ptrCast(@alignCast(ptr));
    }
    fn bind(ptr: *anyopaque, window: []u8) anyerror!void {
        for (me(ptr).half.stages) |stage| try stage.bindWindow(window);
    }
    fn unbind(ptr: *anyopaque) void {
        for (me(ptr).half.stages) |stage| stage.unbindWindow();
    }
    fn reset(ptr: *anyopaque) anyerror!void {
        try me(ptr).half.reset();
    }
};

/// The round's logits (bf16 bits on this GPU) as f32.
fn keepLogits(r: *q.decode_round.Runner, out: []f32) void {
    const bits = r.model.frame.get(.logits).buffer.slice(u16, out.len);
    for (out, bits) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
}

/// One shared decode round, one row a slot: `pending` the slots' next tokens in and out.
fn sharedRound(a: std.mem.Allocator, m: *q.model.Model, r: *q.decode_round.Runner, pending: []u32) !void {
    const most = q.attention_plan.max_streams;
    const n = pending.len;
    const vocab = m.config.vocab;
    var inputs: [most]q.round_plan.Input = undefined;
    var ids: [most][1]u32 = undefined;
    for (0..n) |k| {
        ids[k] = .{pending[k]};
        inputs[k] = .{ .slot = @intCast(k), .start = r.offsets[k], .capacity = r.capacity, .ids = &ids[k] };
    }
    var plan = try q.round_plan.Round.init(a, inputs[0..n], @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
    defer plan.deinit();
    try r.verifyHead(&plan, .all);
    const words = m.frame.get(.logits).buffer.slice(u16, n * vocab);
    for (0..n) |k| pending[k] = try q.session.argmax(words[k * vocab ..][0..vocab]);
    const zero = [_]u32{0};
    var paths: [most][]const u32 = @splat(&zero);
    try r.keep(&plan, paths[0..n]);
}

/// The Mac alone with `n` sequences: prefill each in chunks, then decode them together in shared rounds. Prints both
/// rates (decode is timed alone) and whether every sequence made the reference's tokens.
fn parallel(a: std.mem.Allocator, io: std.Io, dir: []const u8, ref: split.reference.Reference, n: usize, chunk: usize, repeat: usize) !void {
    const prompt = try a.alloc(u32, ref.prompt.len * repeat);
    defer a.free(prompt);
    for (0..repeat) |i| @memcpy(prompt[i * ref.prompt.len ..][0..ref.prompt.len], ref.prompt);
    const m = try q.model.Model.load(a, io, dir, 128, null);
    defer m.deinit();
    const out = try a.alloc(u32, n * ref.want.len);
    defer a.free(out);
    const pending = try a.alloc(u32, n);
    defer a.free(pending);
    var best: [2]u64 = .{ std.math.maxInt(u64), std.math.maxInt(u64) };
    // three passes, the first warms the pipelines; the best of the others is kept
    for (0..3) |pass| {
        var r = try q.decode_round.Runner.init(a, m, @intCast(n), h.capacity);
        defer r.deinit();
        const t0 = split.device.nowNs();
        for (pending, 0..) |*p, k| {
            const s = q.session.Session{ .runner = &r, .slot = @intCast(k) };
            try s.prefill(prompt, chunk);
            p.* = try s.greedy();
        }
        const t1 = split.device.nowNs();
        for (0..ref.want.len) |i| {
            for (pending, 0..) |p, k| out[k * ref.want.len + i] = p;
            try sharedRound(a, m, &r, pending);
        }
        const t2 = split.device.nowNs();
        if (pass > 0) {
            best[0] = @min(best[0], t1 - t0);
            best[1] = @min(best[1], t2 - t1);
        }
    }
    var same: usize = 0;
    const want = if (repeat > 1) out[0..ref.want.len] else ref.want;
    for (0..n) |k| same += @intFromBool(std.mem.eql(u32, out[k * ref.want.len ..][0..ref.want.len], want));
    const nf: f64 = @floatFromInt(n);
    const sec = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.of;
    std.debug.print("metal alone: {d} sequences, chunk {d}: prefill {d} positions x {d} in {d:.1} ms = {d:.1} tok/s; decode {d} tokens x {d} in {d:.1} ms = {d:.1} tok/s ({d:.1} a stream); {d}/{d} sequences token-identical\n", .{
        n,
        chunk,
        prompt.len,
        n,
        sec(best[0]) * 1e3,
        nf * @as(f64, @floatFromInt(prompt.len)) / sec(best[0]),
        ref.want.len,
        n,
        sec(best[1]) * 1e3,
        nf * @as(f64, @floatFromInt(ref.want.len)) / sec(best[1]),
        @as(f64, @floatFromInt(ref.want.len)) / sec(best[1]),
        same,
        n,
    });
}

fn ready() void {
    std.debug.print("split metal: serving\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const dir = std.mem.span(std.c.getenv("TF_SPLIT_QWEN_DIR") orelse {
        std.debug.print("split metal: skipped, TF_SPLIT_QWEN_DIR is not set\n", .{});
        return;
    });
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.MissingArgument;
    if (std.mem.eql(u8, args[1], "serve")) {
        if (args.len < 5 or args.len > 7) return error.MissingArgument;
        const range: q.range.Range = .{ .first = try std.fmt.parseInt(usize, args[3], 10), .last = try std.fmt.parseInt(usize, args[4], 10) };
        // one stage a lane the client runs (the optional fifth argument), a slot in it a sequence of the lane (the sixth)
        const lanes: usize = if (args.len >= 6) try std.fmt.parseInt(usize, args[5], 10) else 1;
        const per_lane: usize = if (args.len == 7) try std.fmt.parseInt(usize, args[6], 10) else 1;
        if (lanes == 0 or lanes > split.streams.max_streams) return error.BadLaneCount;
        if (per_lane == 0 or per_lane > h.max_sequences) return error.BadSequenceCount;
        const src: h.Source = .{ .ctx = &io, .load = loadRange, .layers = 0 };
        var s = try Served.init(a, try h.Half.initLanes(a, src, range, lanes, @intCast(per_lane)));
        defer s.deinit(a);
        try split.remote.serve(args[2], s.host(), ready);
        return;
    }
    if (std.mem.eql(u8, args[1], "parallel")) {
        if (args.len < 4) return error.MissingArgument;
        const ref = try split.reference.load(a, args[2]);
        defer ref.deinit(a);
        const n = try std.fmt.parseInt(usize, args[3], 10);
        const chunk: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 24;
        if (n == 0 or n > q.attention_plan.max_streams or chunk == 0) return error.BadSequenceCount;
        const repeat: usize = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 1;
        if (repeat == 0 or ref.prompt.len * repeat + ref.want.len > h.capacity) return error.BadSequenceCount;
        return parallel(a, io, dir, ref, n, chunk, repeat);
    }
    if (!std.mem.eql(u8, args[1], "ref")) return error.UnknownCommand;
    const n_tokens: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 40;
    var tok = try tokenizer.loadTokenizer(io, a, dir);
    defer tok.deinit();
    const prompt = try tok.encode(a, "The three primary colors are red, yellow and");
    defer a.free(prompt);
    const want = try a.alloc(u32, n_tokens);
    defer a.free(want);
    const seq_len = prompt.len + n_tokens - 1;
    const preds = try a.alloc(u32, seq_len);
    defer a.free(preds);
    const m = try q.model.Model.load(a, io, dir, 128, null);
    defer m.deinit();
    const t = try h.unsplit(a, m, prompt, want);
    std.debug.print("metal unsplit (token by token): {d:.2} ms/position over {d} positions\n", .{ @as(f64, @floatFromInt(t.ns)) / 1e6 / @as(f64, @floatFromInt(t.steps)), t.steps });
    // the same positions teacher forced: the prompt, then every generated token but the last
    var r = try q.decode_round.Runner.init(a, m, 1, h.capacity);
    defer r.deinit();
    const s = q.session.Session{ .runner = &r };
    const vocab = m.config.vocab;
    const logits = try a.alloc(f32, seq_len * vocab);
    defer a.free(logits);
    for (prompt, preds[0..prompt.len], 0..) |tk, *o, i| {
        try s.step(tk);
        o.* = try s.greedy();
        keepLogits(&r, logits[i * vocab ..][0..vocab]);
    }
    for (want[0 .. n_tokens - 1], preds[prompt.len..], prompt.len..) |tk, *o, i| {
        try s.step(tk);
        o.* = try s.greedy();
        keepLogits(&r, logits[i * vocab ..][0..vocab]);
    }
    const text = try tok.decode(a, want, true);
    defer a.free(text);
    std.debug.print("  \"{s}\"\n", .{text});
    const logits_path = try std.fmt.allocPrintSentinel(a, "{s}.logits", .{args[2]}, 0);
    defer a.free(logits_path);
    try split.reference.saveLogits(logits_path, logits);
    try split.reference.save(a, args[2], .{ .layers = @intCast(m.config.layers), .hidden = @intCast(m.config.hidden), .prompt = prompt, .want = want, .preds = preds });
}
