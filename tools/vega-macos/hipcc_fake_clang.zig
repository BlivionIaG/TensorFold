//! hipcc_fake_clang: a stand-in for Homebrew's clang / clang++ so the hipcc wrapper runs offline (zig build test-vega-toolchain).
//! Prints its argv, one argument a line (argv[0] included, as it was exec'd), to stdout and exits with the status in FAKE_CLANG_EXIT (default 0).
const std = @import("std");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    var text: std.Io.Writer.Allocating = .init(a);
    for (argv) |arg| try text.writer.print("{s}\n", .{arg});
    try std.Io.File.stdout().writeStreamingAll(init.io, text.written());
    return std.fmt.parseInt(u8, init.environ_map.get("FAKE_CLANG_EXIT") orelse "0", 10) catch 0;
}
