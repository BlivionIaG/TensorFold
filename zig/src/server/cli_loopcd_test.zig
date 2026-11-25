//! LoopCD's flags: off by default, the strength reaches the engine, and only the Metal build lists or parses them.
const std = @import("std");
const cli = @import("cli.zig");
const Args = cli.Args;

test "LoopCD: off by default, the flag and strength reach the engine, and only the Metal build serves them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try cli.request(a, "d", "inkling2_loop", .{});
    try std.testing.expect(!plain.loopcd and plain.loopcd_w == null);
    var out: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&out.writer, .{ .version = "0" });
    const listed = std.mem.indexOf(u8, out.written(), "\"--loopcd\"") != null;
    try std.testing.expectEqual(@import("builtin").os.tag == .macos, listed);
    var u: cli.Usage = .{};
    if (!listed) {
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--loopcd" }, &u));
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--loopcd-w", "0.3" }, &u));
        return;
    }
    const on: Args = try cli.parse(a, &.{ "m", "--loopcd" }, &u);
    try std.testing.expect((try cli.request(a, "d", "inkling2_loop", on)).loopcd);
    const w = try cli.request(a, "d", "inkling2_loop", try cli.parse(a, &.{ "m", "--loopcd-w", "0.25" }, &u));
    try std.testing.expect(w.loopcd);
    try std.testing.expectEqual(@as(?f32, 0.25), w.loopcd_w);
}
