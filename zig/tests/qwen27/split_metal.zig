//! The Mac's side of a model split with a card in another process (TF_SPLIT_QWEN_DIR, the Qwen3.5-4B MLX 4-bit
//! directory). `ref FILE [TOKENS]` runs the model unsplit on this GPU and writes the tokens and predictions the card's
//! process compares with (split.reference). `serve SOCKET FIRST LAST [SEQUENCES]` holds layers [FIRST, LAST), a stage for
//! each of SEQUENCES (default 1), and is the host stage of the pipeline that process runs (split.remote): the window is
//! the card's pinned memory, mapped here and wrapped for Metal.
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
        if (args.len != 5 and args.len != 6) return error.MissingArgument;
        const range: q.range.Range = .{ .first = try std.fmt.parseInt(usize, args[3], 10), .last = try std.fmt.parseInt(usize, args[4], 10) };
        // one stage a sequence the client may run at once (the optional sixth argument, default 1)
        const count: usize = if (args.len == 6) try std.fmt.parseInt(usize, args[5], 10) else 1;
        if (count == 0 or count > split.streams.max_streams) return error.BadSequenceCount;
        const src: h.Source = .{ .ctx = &io, .load = loadRange, .layers = 0 };
        var s = try Served.init(a, try h.Half.initSequences(a, src, range, count));
        defer s.deinit(a);
        try split.remote.serve(args[2], s.host(), ready);
        return;
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
