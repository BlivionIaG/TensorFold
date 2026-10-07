//! The MLX affine 4-bit product (bf16 x, scales and biases, fp32 out) as one launch of zig/kernels/hip/affine.hip.

const std = @import("std");
const launch_ = @import("runtime/launch.zig");
const Function = @import("runtime/module.zig").Function;
const Stream = @import("runtime/stream.zig").Stream;
const Error = @import("runtime/driver.zig").Error;

pub const symbol = "tf_affine4_bf16";

/// Threads a block, one output column each.
const block = 64;

/// Device addresses and the shape: x (m, k), words (n, k / 8), scale and bias (n, k / group), out (m, n).
pub const Product = struct {
    x: u64,
    words: u64,
    scale: u64,
    bias: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    group: u32,

    /// MLX's group sizes over a whole number of groups; anything else is refused before HIP.
    pub fn validate(p: Product) Error!void {
        if (p.m == 0 or p.n == 0 or p.k == 0) return error.Invalid;
        if (p.group != 32 and p.group != 64 and p.group != 128) return error.Invalid;
        if (p.k % p.group != 0) return error.Invalid;
    }
};

pub fn launch(f: Function, stream: Stream, p: Product) Error!void {
    try p.validate();
    var args: launch_.Args = .{};
    inline for (.{ p.x, p.words, p.scale, p.bias, p.out, p.m, p.n, p.k, p.group }) |v| args.add(v);
    try launch_.launch(f, .{ .grid = .{ .x = (p.n + block - 1) / block, .y = p.m }, .block = .{ .x = block } }, stream, &args);
}

test "shapes outside MLX's groups are refused" {
    const ok: Product = .{ .x = 0, .words = 0, .scale = 0, .bias = 0, .out = 0, .m = 1, .n = 1, .k = 64, .group = 64 };
    try ok.validate();
    var bad = ok;
    bad.group = 48;
    try std.testing.expectError(error.Invalid, bad.validate());
    bad = ok;
    bad.k = 96;
    try std.testing.expectError(error.Invalid, bad.validate());
    bad = ok;
    bad.m = 0;
    try std.testing.expectError(error.Invalid, bad.validate());
}
