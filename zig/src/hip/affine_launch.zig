//! The MLX affine kernels launched from Zig: the host dispatch of rocm/affine_{dot2,tiles}.hip (the decode tiles, the
//! GEMM tile, the fp16 split, routed plans) over the family's code objects. Schedule 0 (auto) only: the one-thread
//! GEMV, WMMA and column-stream schedules stay in the library.

const std = @import("std");
const abi = @import("abi.zig");
const driver = @import("driver.zig");
const hl = @import("launch.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;

const Error = driver.Error;

/// rocm/affine.hpp's GroupTable, Routing and Affine as the kernels take them by value.
pub const GroupTable = extern struct { p: u64, kind: c_int };
pub const Routing = extern struct { items: u64 = 0, members: u64 = 0, x_div: c_int = 1, first: c_int = 0 };
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
};

comptime {
    std.debug.assert(@sizeOf(GroupTable) == 16 and @sizeOf(Routing) == 24 and @sizeOf(Arg) == 112);
}

const bit_widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
const row_counts = [_]u8{ 1, 2, 4, 8 };
const piece_counts = [_]u8{ 1, 2, 4 };

const lane_rows = 8; // kLaneRows
const fast_m = 8;
const fast_n = 256;
const block_rows = 64; // kBlockRows
const lane_group_max = 128;

pub const Kernels = struct {
    wmma: bool,
    lanes: [bit_widths.len][row_counts.len][piece_counts.len]Function,
    row: [bit_widths.len][piece_counts.len]Function,
    block: [bit_widths.len]Function,
    fast: [2]Function, // rows 1, 8
    span: [2]Function,
    fold: Function,
    wide: Function,
    tiled: Function,
    reference: Function,
    fill: Function,

    /// `tiles` holds affine_tiles.hip's kernels, `dot2` affine_dot2.hip's; one activation type a family: bf16 on the
    /// WMMA build (v_dot2_f32_bf16), fp16 on RDNA2.
    pub fn load(tiles_obj: Module, dot2_obj: Module, wmma: bool) Error!Kernels {
        var k: Kernels = undefined;
        k.wmma = wmma;
        if (wmma) try k.resolve("7DotBF16", tiles_obj) else try k.resolve("6DotF16", tiles_obj);
        const r = "_ZN2tf4rocm";
        k.fast = .{ try dot2_obj.function(r ++ "16affine_dot2_fastILi1EEEvNS0_6AffineE"), try dot2_obj.function(r ++ "16affine_dot2_fastILi8EEEvNS0_6AffineE") };
        k.span = .{ try dot2_obj.function(r ++ "16affine_dot2_spanILi1EEEvNS0_6AffineEPfi"), try dot2_obj.function(r ++ "16affine_dot2_spanILi8EEEvNS0_6AffineEPfi") };
        k.fold = try dot2_obj.function(r ++ "16affine_dot2_foldEPKfNS0_10GroupTableES3_Pfiii");
        k.wide = try dot2_obj.function(r ++ "16affine_dot2_wideENS0_6AffineE");
        k.tiled = try dot2_obj.function(r ++ "17affine_dot2_tiledENS0_6AffineE");
        k.reference = try dot2_obj.function(r ++ "18affine_dot2_kernelENS0_6AffineE");
        k.fill = try dot2_obj.function(r ++ "13fill_byte_lutEP6__half");
        return k;
    }

    fn resolve(k: *Kernels, comptime dot: []const u8, m: Module) Error!void {
        inline for (bit_widths, 0..) |bits, b| {
            k.block[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot, bits }));
            inline for (piece_counts, 0..) |pieces, p| {
                k.row[b][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_rowINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, pieces }));
                inline for (row_counts, 0..) |rows, r| {
                    k.lanes[b][r][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_lanesINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ dot, bits, rows, pieces }));
                }
            }
        }
    }

    /// fp16(bf16(code)) for every byte into the module's constant table: the RDNA2 fp16 tiles read it. Once a device.
    pub fn fillByteLut(k: *const Kernels, d: *const driver.Driver, dot2_obj: Module) Error!void {
        if (k.wmma) return;
        const lut = try dot2_obj.global("_ZN2tf4rocm8kByteF16E");
        var args: hl.Args = .{};
        args.add(lut.ptr);
        try hl.launch(k.fill, .{ .grid = .{}, .block = .{ .x = 256 } }, .{ .d = d, .handle = null }, &args);
        try d.check(d.api.hipStreamSynchronize(null), "hipStreamSynchronize");
    }

    fn go(d: *const driver.Driver, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .d = d, .handle = s }, args);
    }

    fn refuse(what: []const u8) Error {
        std.log.err("affine: {s}", .{what});
        return error.Invalid;
    }

    fn bitIndex(bits: c_int) ?usize {
        return std.mem.indexOfScalar(c_int, &bit_widths, bits);
    }

    fn pieceIndex(group: c_int) usize {
        return switch (group) {
            32 => 0,
            64 => 1,
            else => 2,
        };
    }

    fn cdiv(n: c_int, by: c_int) c_uint {
        return @intCast(@divTrunc(n + by - 1, by));
    }

    /// Groups a lane covers: the groups rounded up to a power of two, at most 32; none under four groups.
    fn rowLanes(groups: c_int) c_int {
        if (groups < 4) return 0;
        var lpc: c_int = 4;
        while (lpc < groups and lpc < 32) lpc <<= 1;
        return lpc;
    }

    fn lanesLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        if (a.m < 1 or a.m > lane_rows or a.n < 1 or @rem(a.group, 32) != 0 or a.group > lane_group_max or @rem(a.k, a.group) != 0) return refuse("lanes shape");
        const b = bitIndex(a.bits) orelse return refuse("bits");
        const p = pieceIndex(a.group);
        var args: hl.Args = .{};
        args.add(a);
        const z: c_uint = @intCast(items);
        // one row on the row tile unless it has under four groups
        const lpc = if (a.m == 1) rowLanes(@divExact(a.k, a.group)) else 0;
        if (lpc != 0) {
            args.add(lpc);
            return go(d, k.row[b][p], .{ .x = cdiv(a.n, 128), .z = z }, .{ .x = 128 }, s, &args);
        }
        const r: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else if (a.m <= 4) 2 else 3;
        try go(d, k.lanes[b][r][p], .{ .x = cdiv(a.n, 32), .z = z }, .{ .x = 256 }, s, &args);
    }

    fn blockLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        if (a.m < 1 or a.n < 1 or @rem(a.group, 32) != 0 or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0) return refuse("block shape");
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        try go(d, k.block[b], .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    fn groupOk(group: c_int) bool {
        return group == 32 or group == 64 or group == 128;
    }

    /// launch_affine on the auto schedule: the activation type's dot2 tiles.
    pub fn run(k: *const Kernels, d: *const driver.Driver, arg: Arg, schedule: c_int, s: abi.Stream, partial: u64, parts: c_int, out_half: bool) Error!void {
        var a = arg;
        if (out_half) {
            if (a.fp16 == 0 or schedule != 0 or parts > 1) return refuse("fp16 output is the decode tile");
            a.out16 = a.out;
            a.out = 0;
        }
        if (partial != 0 and parts > 1) return k.split(d, a, partial, parts, s);
        if (k.wmma) {
            if (schedule != 0 or a.fp16 != 0) return refuse("this schedule stays in the library");
            // BF16 x on a gfx11 / gfx12 build: the decode tiles up to 8 rows, the GEMM tile past them.
            if (a.m < 1 or a.n < 1 or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0 or !groupOk(a.group)) return refuse("bf16 shape");
            if (a.m <= lane_rows) return k.lanesLaunch(d, a, s, 1);
            return k.blockLaunch(d, a, s, 1);
        }
        if (a.fp16 == 0 or schedule == 2) return refuse("this schedule stays in the library");
        try k.dotLaunch(d, a, schedule == 1, s);
    }

    fn dotLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, reference: bool, s: abi.Stream) Error!void {
        if (a.m < 1 or a.n < 1 or a.k < 1 or @rem(a.group, 2) != 0 or @rem(a.k, a.group) != 0) return refuse("dot2 shape");
        var args: hl.Args = .{};
        args.add(a);
        if (!reference) {
            if (a.m <= lane_rows and groupOk(a.group)) return k.lanesLaunch(d, a, s, 1);
            if (a.out16 != 0) return refuse("fp16 output is the decode tile's");
            if (groupOk(a.group)) {
                if (a.m >= block_rows and @rem(a.k, 16) == 0) return k.blockLaunch(d, a, s, 1);
                if (a.m > fast_m) return go(d, k.wide, .{ .x = cdiv(a.n, 32), .y = cdiv(a.m, 128) }, .{ .x = 512 }, s, &args);
                return go(d, k.fast[if (a.m == 1) 0 else 1], .{ .x = cdiv(a.n, fast_n), .y = cdiv(a.m, fast_m) }, .{ .x = fast_n }, s, &args);
            }
            if (a.m > 1 and a.group <= 128) return go(d, k.tiled, .{ .x = cdiv(a.n, 16), .y = cdiv(a.m, 16) }, .{ .x = 256 }, s, &args);
        }
        try go(d, k.reference, .{ .x = cdiv(a.n, 32), .y = @intCast(a.m) }, .{ .x = 32 }, s, &args);
    }

    /// The decode tile over groups [z * per, (z + 1) * per) into `partial`, then the fold in group order.
    fn split(k: *const Kernels, d: *const driver.Driver, a: Arg, partial: u64, parts: c_int, s: abi.Stream) Error!void {
        const groups = @divTrunc(a.k, a.group);
        if (a.fp16 == 0 or parts < 2 or @rem(groups, parts) != 0) return refuse("split shape");
        var args: hl.Args = .{};
        args.add(a);
        args.add(partial);
        args.add(@divExact(groups, parts));
        try go(d, k.span[if (a.m == 1) 0 else 1], .{ .x = cdiv(a.n, fast_n), .y = cdiv(a.m, fast_m), .z = @intCast(parts) }, .{ .x = fast_n }, s, &args);
        var fold: hl.Args = .{};
        fold.add(partial);
        fold.add(a.scale);
        fold.add(a.bias);
        fold.add(a.out);
        fold.add(a.m);
        fold.add(a.n);
        fold.add(groups);
        try go(d, k.fold, .{ .x = cdiv(a.n, fast_n), .y = @intCast(a.m) }, .{ .x = fast_n }, s, &fold);
    }

    /// affine_routed_launch: every item of a plan in one launch; `arg.m` is the most rows an item holds.
    pub fn routed(k: *const Kernels, d: *const driver.Driver, arg: Arg, items: c_int, s: abi.Stream) Error!void {
        // the family's own activation type: bf16 on the WMMA build, fp16 on RDNA2
        if ((arg.fp16 != 0) == k.wmma or arg.route.items == 0 or arg.route.members == 0 or items < 1 or arg.route.x_div < 1) return refuse("routed plan");
        if (!groupOk(arg.group)) return refuse("routed group");
        if (arg.m <= lane_rows) return k.lanesLaunch(d, arg, s, items);
        try k.blockLaunch(d, arg, s, items);
    }

    /// The split count of a decode launch: 1 unless `mode` is 2 (the tests' forced comparison) and the groups divide.
    pub fn splitCount(m: c_int, n: c_int, k: c_int, group: c_int, mode: c_int) c_int {
        if (mode != 2 or m < 1 or n < 512) return 1;
        if (!groupOk(group) or k < group or @rem(k, group) != 0) return 1;
        const groups = @divTrunc(k, group);
        if (groups < 2) return 1;
        // A full row tile or a wide column grid already fills the card.
        const blocks = @divTrunc(n + fast_n - 1, fast_n) * @divTrunc(m + fast_m - 1, fast_m);
        if (blocks < 1) return 1;
        var want = @divTrunc(256, blocks);
        if (want < 2) want = 2;
        if (want > groups) want = groups;
        var count = want;
        while (count >= 2) : (count -= 1) {
            if (@rem(groups, count) == 0) return count;
        }
        return 1;
    }
};

test "the split count follows the C rule" {
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 4096, 4096, 64, 0));
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 256, 4096, 64, 2));
    try std.testing.expectEqual(@as(c_int, 8), Kernels.splitCount(1, 4096, 512, 64, 2));
}

test "the lane groups of a row tile" {
    try std.testing.expectEqual(@as(c_int, 0), Kernels.rowLanes(3));
    try std.testing.expectEqual(@as(c_int, 4), Kernels.rowLanes(4));
    try std.testing.expectEqual(@as(c_int, 32), Kernels.rowLanes(64));
}
