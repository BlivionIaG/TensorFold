//! The kernel sources a decode round launches from, for the test that checks the hazard table's read-only pointers against the kernels' `const`s
//! (`zig build test-round-branches-host`; nothing else imports this).

pub const File = struct { path: []const u8, text: []const u8 };

pub const files = [_]File{
    .{ .path = "ops/elementwise.hpp", .text = @embedFile("ops/elementwise.hpp") },
    .{ .path = "ops/norms.hpp", .text = @embedFile("ops/norms.hpp") },
    .{ .path = "ops/linear.hpp", .text = @embedFile("ops/linear.hpp") },
    .{ .path = "ops/act.hip", .text = @embedFile("ops/act.hip") },
    .{ .path = "ops/rope.hpp", .text = @embedFile("ops/rope.hpp") },
    .{ .path = "ops/attention.hpp", .text = @embedFile("ops/attention.hpp") },
    .{ .path = "ops/draw.hpp", .text = @embedFile("ops/draw.hpp") },
    .{ .path = "decode/decode.hip", .text = @embedFile("decode/decode.hip") },
    .{ .path = "decode/plan.hip", .text = @embedFile("decode/plan.hip") },
    .{ .path = "recurrence/gated_delta.hip", .text = @embedFile("recurrence/gated_delta.hip") },
    .{ .path = "attention/attention.hip", .text = @embedFile("attention/attention.hip") },
};
