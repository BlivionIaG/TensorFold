//! Real-checkpoint split gate for a dense Qwen model (the Qwen3.5-4B MLX 4-bit directory): generated text and unsplit
//! decode speed on the default engine, then the layer ranges cut at several layers over the mock window, token for token.
//! Needs TF_SPLIT_QWEN_DIR; without it the program says so and exits 0.
const std = @import("std");
const tf = @import("tensorfold");
const tokenizer = @import("tokenizer");
const h = @import("split_harness.zig");
const q = tf.qwen27;

const Real = struct { io: std.Io, dir: []const u8 };

fn load(ctx: *const anyopaque, a: std.mem.Allocator, range: q.range.Range) anyerror!*q.model.Model {
    const real: *const Real = @ptrCast(@alignCast(ctx));
    return q.model.Model.loadRange(a, real.io, real.dir, 128, range);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const dir = std.mem.span(std.c.getenv("TF_SPLIT_QWEN_DIR") orelse {
        std.debug.print("qwen27 split (real checkpoint): skipped, TF_SPLIT_QWEN_DIR is not set\n", .{});
        return;
    });
    var tok = try tokenizer.loadTokenizer(io, a, dir);
    defer tok.deinit();
    const text = "The three primary colors are red, yellow and";
    const prompt = try tok.encode(a, text);
    defer a.free(prompt);

    // The default engine on the whole model: chunked prompt, then decode rows.
    {
        const m = try q.model.Model.load(a, io, dir, 128, null);
        defer m.deinit();
        var r = try q.decode_round.Runner.init(a, m, 1, h.capacity);
        defer r.deinit();
        const s = q.session.Session{ .runner = &r };
        try s.prefill(prompt, 128);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);
        const n = 48;
        var t0: u64 = 0;
        for (0..n) |i| {
            if (i == 1) t0 = nsNow(); // the first token's head is prefill's; decode rows are timed from the second
            const t = try s.greedy();
            try ids.append(a, t);
            if (i + 1 < n) try s.step(t);
        }
        const ns = nsNow() - t0;
        const out = try tok.decode(a, ids.items, true);
        defer a.free(out);
        std.debug.print("qwen27 default engine, {d} layers hidden {d}: \"{s}\" -> \"{s}\"\n", .{ m.config.layers, m.config.hidden, text, out });
        std.debug.print("  unsplit decode {d:.1} tok/s ({d:.2} ms/token over {d} decode rows)\n", .{ 1e9 * @as(f64, n - 1) / @as(f64, @floatFromInt(ns)), @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, n - 1), n - 1 });
    }

    const real = Real{ .io = io, .dir = dir };
    const src = h.Source{ .ctx = &real, .load = load, .layers = 32 };
    // the gate's prompts: 12 or more tokens each, so that chunks of five make a full chunk and a short last one
    const texts = [2][]const u8{ "The three primary colors are red, yellow and blue, and the three secondary colors are", "Water boils at one hundred degrees Celsius at sea level, and the capital of France is" };
    const ids0 = try tok.encode(a, texts[0]);
    defer a.free(ids0);
    const ids1 = try tok.encode(a, texts[1]);
    defer a.free(ids1);
    std.debug.print("qwen27 split gate prompts: {d} and {d} tokens, chunks of {d}\n", .{ ids0.len, ids1.len, h.chunk_size });
    var report = h.Report{};
    const m = try load(&real, a, .{ .first = 0, .last = src.layers });
    const c0 = try h.reference(a, m, ids0, 32, &report);
    defer h.freeCase(a, c0);
    const c1 = try h.reference(a, m, ids1, 32, null);
    defer h.freeCase(a, c1);
    m.deinit();
    for ([2]h.Case{ c0, c1 }, texts) |c, label| {
        const out = try tok.decode(a, c.single, true);
        defer a.free(out);
        std.debug.print("qwen27 unsplit reference \"{s}\" -> \"{s}\" ({d:.2} ms/position token by token)\n", .{ label, out, @as(f64, @floatFromInt(report.unsplit_ns)) / 1e6 });
    }
    try h.cuts(a, src, &.{ 1, 8, 16, 31 }, .{ c0, c1 }, &report);
    const split_ns = report.split_ns / report.split_steps;
    std.debug.print("qwen27 split: {d} decode, {d} chunked and {d} two-sequence runs identical; {d:.2} ms/position split against {d:.2} unsplit\n", .{ report.runs, report.chunked_runs, report.stream_runs, @as(f64, @floatFromInt(split_ns)) / 1e6, @as(f64, @floatFromInt(report.unsplit_ns)) / 1e6 });
}

fn nsNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
