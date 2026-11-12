//! quant/mlx.hpp's GroupTable, Routing and Affine as the kernels take them by value.
const std = @import("std");
const kernels = @import("../kernels.zig");

pub const GroupTable = extern struct { p: u64, kind: c_int };

/// The expert pointer table (tiles/plan.hpp ExpertPtrs), the last member of `affine_tiles_ptr`'s tile arguments.
pub const ExpertPtrs = if (kernels.expert_ptrs) extern struct { ptab: u64 = 0, off: i64 = 0, delta: i64 = 0 } else extern struct {};

pub const Routing = extern struct { items: u64 = 0, members: u64 = 0, x_div: c_int = 1, first: c_int = 0 };

/// The table at `ptab` (u64 pairs: an expert's blob, its line or 0) and this projection's offset in a blob.
pub fn expertPtrs(ptab: u64, off: u64) ExpertPtrs {
    if (comptime !kernels.expert_ptrs) {
        std.debug.assert(ptab == 0);
        return .{};
    }
    return .{ .ptab = ptab, .off = @intCast(off) };
}

pub const Arg = extern struct {
    x: u64,
    words: u64,
    scale: GroupTable,
    bias: GroupTable,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
    bits: c_int,
    group: c_int,
    fp16: c_int,
    route: Routing = .{},
    out16: u64 = 0,
    ptrs: ExpertPtrs = .{},
};

/// `Arg` without `ptrs`: what the tiles of every code object but `affine_tiles_ptr` take.
pub const ArgBase = extern struct {
    x: u64,
    words: u64,
    scale: GroupTable,
    bias: GroupTable,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
    bits: c_int,
    group: c_int,
    fp16: c_int,
    route: Routing = .{},
    out16: u64 = 0,
};

/// `a` as the tiles without expert pointers take it: `ptrs` is the last member, so the bytes are a prefix.
pub fn plainArg(a: Arg) ArgBase {
    return @as(*const ArgBase, @ptrCast(&a)).*;
}

comptime {
    const ptrs: usize = if (kernels.expert_ptrs) 24 else 0;
    std.debug.assert(@sizeOf(GroupTable) == 16 and @sizeOf(Routing) == 24 and @sizeOf(ArgBase) == 112 and @sizeOf(Arg) == 112 + ptrs);
    std.debug.assert(@offsetOf(Arg, "out16") == @offsetOf(ArgBase, "out16") and @offsetOf(Arg, "ptrs") == @sizeOf(ArgBase));
}
