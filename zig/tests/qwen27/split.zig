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
    const want = try a.alloc(u32, 32);
    defer a.free(want);
    var report = h.Report{};
    {
        const m = try load(&real, a, .{ .first = 0, .last = src.layers });
        defer m.deinit();
        const t = try h.unsplit(a, m, prompt, want);
        report.unsplit_ns = t.ns / t.steps;
        const out = try tok.decode(a, want, true);
        defer a.free(out);
        std.debug.print("qwen27 unsplit reference (token-by-token prompt): {d:.2} ms/position, \"{s}\"\n", .{ @as(f64, @floatFromInt(report.unsplit_ns)) / 1e6, out });
    }
    try h.cuts(a, src, &.{ 1, 8, 16, 31 }, prompt, want, &report);
    const split_ns = report.split_ns / report.split_steps;
    std.debug.print("qwen27 split: {d} runs token-identical; {d:.2} ms/position split against {d:.2} unsplit\n", .{ report.runs, @as(f64, @floatFromInt(split_ns)) / 1e6, @as(f64, @floatFromInt(report.unsplit_ns)) / 1e6 });
}

fn nsNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
