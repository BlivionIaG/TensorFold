//! `prefill <model dir> <length>...`: cold prompts of each length (fixed synthetic ids) through the engine's prefill,
//! one warm-up then the median of three timed runs, as prompt tokens a second.

const std = @import("std");
const qwen35 = @import("qwen35");

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    var longest: usize = 0;
    for (args[1..]) |a| longest = @max(longest, try std.fmt.parseInt(usize, a, 10));
    const e = try qwen35.engine.Engine.open(gpa, io, args[0], .{ .capacity = longest + 64, .batch_rows = 32 });
    defer e.deinit();
    const prompt = try gpa.alloc(u32, longest);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 50000);
    for (args[1..]) |a| {
        const len = try std.fmt.parseInt(usize, a, 10);
        var times: [3]f64 = undefined;
        for (0..4) |rep| {
            var caches = try e.newCaches(len + 8);
            defer caches.deinit(gpa);
            const t0 = std.Io.Clock.awake.now(io);
            _ = try e.prefill(&caches, prompt[0..len], 0, null, .{ .sampling = null, .position = len }, null);
            const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
            if (rep > 0) times[rep - 1] = dt;
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        std.debug.print("RESULT prefill {d} tokens: {d:.3} s, {d:.0} tok/s\n", .{ len, times[1], @as(f64, @floatFromInt(len)) / times[1] });
    }
}
