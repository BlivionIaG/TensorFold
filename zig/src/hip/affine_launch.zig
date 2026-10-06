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

/// rocm/affine_stream.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
pub const StreamSides = extern struct {
    words: [4]u64 = @splat(0),
    scale: [4]u64 = @splat(0),
    bias: [4]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    n: [4]c_int = @splat(0),
    first: [5]c_int = @splat(0),
    count: c_int = 0,
    out_half: c_int = 0,
    pair_cols: c_int = 0,
    limit: f32 = 0,
};

/// One side of a group: its matrix and where its (rows, n) output goes.
pub const Side = struct { words: u64, scale: u64, bias: u64, n: c_int, out: u64 };

comptime {
    std.debug.assert(@sizeOf(GroupTable) == 16 and @sizeOf(Routing) == 24 and @sizeOf(Arg) == 112 and @sizeOf(StreamSides) == 184);
}

const bit_widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
const row_counts = [_]u8{ 1, 2, 4, 8 };
const piece_counts = [_]u8{ 1, 2, 4 };

const lane_rows = 8; // kLaneRows
const fast_m = 8;
const fast_n = 256;
const block_rows = 64; // kBlockRows
const wmma_rows = 16; // the matrix-core GEMM tile from here (a routed item's rows too)
const lane_group_max = 128;
const stream_cbs = [_]c_int{ 2, 4, 8 }; // columns a lane carries in the stream tile
const stream_rows = [_]u8{ 1, 2, 4 }; // its row counts: columns * rows <= 16
const stream_max_rows = 16; // kStreamRows
const stream_waves = 4; // kStreamWaves
const streams_wanted = 3000; // kStreamWavesWanted
const stream_code_words = 32; // kStreamCodeWords

/// The GEMM tiles of the m >= 64 products: `gemm` (the default) or the previous `block`.
pub const Tile = enum { gemm, block };

/// TF_WMMA on gfx11: `off` runs every affine product on the dot2 tiles, `on` on the matrix cores wherever a tile exists
/// (decode through the library's WMMA tiles, the GEMM tile for the products of 16 rows or more, routed items too);
/// unset is `auto`, what measured fastest.
pub const WmmaMode = enum { auto, on, off };

pub fn wmmaMode() WmmaMode {
    const v = std.c.getenv("TF_WMMA") orelse return .auto;
    const text = std.mem.span(v);
    if (std.mem.eql(u8, text, "1")) return .on;
    if (std.mem.eql(u8, text, "0")) return .off;
    return .auto;
}

/// The gfx major and minor as 10 * major + minor of the current device.
fn capability(d: *const driver.Driver) Error!u32 {
    var dev: c_int = 0;
    try d.check(d.api.hipGetDevice(&dev), "hipGetDevice");
    var major: c_int = 0;
    var minor: c_int = 0;
    try d.check(d.api.hipDeviceGetAttribute(&major, .compute_capability_major, dev), "hipDeviceGetAttribute");
    try d.check(d.api.hipDeviceGetAttribute(&minor, .compute_capability_minor, dev), "hipDeviceGetAttribute");
    return @intCast(10 * major + minor);
}

pub const Kernels = struct {
    wmma: bool,
    /// gfx11: the matrix-core GEMM tile runs (gfx12's WMMA has other layouts).
    matrix: bool,
    mode: WmmaMode,
    /// Which GEMM tile the products take; TF_AFFINE_GEMM=old picks the previous one.
    tile: Tile,
    lanes: [bit_widths.len][row_counts.len][piece_counts.len]Function,
    row: [bit_widths.len][piece_counts.len]Function,
    /// The stream tile (rocm/affine_stream.hpp) by width, rows and columns a lane; null where columns * rows > 16.
    stream: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with the (gate | up) activation as its epilogue.
    pairs: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// TF_AFFINE_GEMV=old keeps the decode tiles before the stream tile.
    stream_on: bool,
    block: [bit_widths.len]Function,
    gemm: [bit_widths.len]Function,
    wmma_gemm: [bit_widths.len]Function,
    fast: [2]Function, // rows 1, 8
    span: [2]Function,
    fold: Function,
    wide: Function,
    tiled: Function,
    reference: Function,
    fill: Function,

    /// `tiles` holds affine_tiles.hip's kernels, `dot2` affine_dot2.hip's; one activation type a family: bf16 on the
    /// WMMA build (v_dot2_f32_bf16), fp16 on RDNA2.
    pub fn load(d: *const driver.Driver, tiles_obj: Module, dot2_obj: Module, wmma: bool) Error!Kernels {
        var k: Kernels = undefined;
        k.wmma = wmma;
        const cap = try capability(d);
        k.matrix = wmma and (cap == 110 or cap == 115);
        k.mode = wmmaMode();
        k.tile = .gemm;
        k.stream_on = true;
        if (std.c.getenv("TF_AFFINE_GEMV")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "old")) k.stream_on = false;
        }
        if (std.c.getenv("TF_AFFINE_GEMM")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "old")) k.tile = .block;
        }
        // the GEMM tile's rows of x a lane keeps (rocm/affine_gemm.hpp GemmShape): 8 for BF16, 16 for FP16
        if (wmma) try k.resolve("7DotBF16", "Li8E", tiles_obj) else try k.resolve("6DotF16", "Li16E", tiles_obj);
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

    fn resolve(k: *Kernels, comptime dot: []const u8, comptime shape: []const u8, m: Module) Error!void {
        @setEvalBranchQuota(200_000);
        inline for (bit_widths, 0..) |bits, b| {
            k.block[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot, bits }));
            if (k.matrix) k.wmma_gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_gemmILi{d}EEEvNS0_6AffineE", .{bits}));
            k.gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_gemm_blockINS0_{s}ELi{d}E{s}EEvNS0_6AffineE", .{ dot, bits, shape }));
            inline for (piece_counts, 0..) |pieces, p| {
                k.row[b][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_rowINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, pieces }));
                inline for (row_counts, 0..) |rows, r| {
                    k.lanes[b][r][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_lanesINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ dot, bits, rows, pieces }));
                }
            }
            inline for (stream_rows, 0..) |rows, r| {
                inline for (stream_cbs, 0..) |cb, c| {
                    k.stream[b][r][c] = if (cb * rows <= 16)
                        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb0EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                    else
                        null;
                    k.pairs[b][r][c] = if (cb * rows <= 16)
                        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                    else
                        null;
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

    /// Whether the stream tile takes this product (its rows, widths, alignment and tables).
    pub fn streamFits(k: *const Kernels, a: Arg) bool {
        return k.streamTakes(a.m, a.n, a.k, a.bits, a.group, a.scale.kind, a.bias.kind, a.x);
    }

    pub fn streamTakes(k: *const Kernels, m: c_int, n: c_int, kk: c_int, bits: c_int, group: c_int, scale_kind: c_int, bias_kind: c_int, x: u64) bool {
        // 16 rows or more of BF16 are the matrix tile's on gfx11
        const most: c_int = if (k.matrix and k.mode != .off) wmma_rows - 1 else stream_max_rows;
        if (!k.stream_on or m < 1 or m > most or n < 1 or @rem(group, 32) != 0 or group > lane_group_max) return false;
        return @rem(kk, group) == 0 and x % 16 == 0 and scale_kind != 0 and bias_kind == scale_kind and bitIndex(bits) != null;
    }

    /// The stream tile of 1 to 16 rows; false when the shape keeps the previous tiles.
    fn streamLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!bool {
        if (!k.streamFits(a)) return false;
        try k.streamGo(d, a, .{}, a.n, items, s);
        return true;
    }

    /// The stream tile on `total_n` columns (a group's sum); `sides` counts them off block by block when it has any.
    fn streamGo(k: *const Kernels, d: *const driver.Driver, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream) Error!void {
        const b = bitIndex(a.bits).?;
        var sides = sides_in;
        // a pair's lane carries half as many output columns: a gate column and its up column make one
        const pair_div: c_int = if (sides.pair_cols > 0) 2 else 1;
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const r: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else 2;
        // the widest column count that still gives the card enough waves, within the registers the lane's code words take
        var pick: usize = 0;
        var c: usize = stream_cbs.len;
        while (c > 0) {
            c -= 1;
            if (stream_cbs[c] * stream_rows[r] > 16 or stream_cbs[c] * a.bits > stream_code_words) continue;
            pick = c;
            const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[c], pair_div);
            if (@as(u64, cdiv(total_n, per_block)) * stream_waves * @as(u64, @intCast(items)) * cdiv(a.m, stream_rows[r]) >= streams_wanted) break;
        }
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[pick], pair_div);
        var blocks: c_uint = cdiv(total_n, per_block);
        if (sides.count > 0) {
            blocks = 0;
            for (0..@intCast(sides.count)) |i| {
                sides.first[i] = @intCast(blocks);
                blocks += cdiv(sides.n[i], per_block);
            }
            sides.first[@intCast(sides.count)] = @intCast(blocks);
        }
        var args: hl.Args = .{};
        args.add(a);
        args.add(sides);
        args.add(@as(c_int, lpc_log2));
        args.add(@as(c_int, switch (a.group) {
            32 => 0,
            64 => 1,
            else => 2,
        }));
        const f = (if (sides.pair_cols > 0) k.pairs[b][r][pick] else k.stream[b][r][pick]).?;
        try go(d, f, .{ .x = blocks, .y = cdiv(a.m, stream_rows[r]), .z = @intCast(items) }, .{ .x = 32 * stream_waves }, s, &args);
    }

    /// The stacked (gate | up) product of `arg` (n the stacked width, out16 the (rows, n / 2) activation, plain or routed
    /// over `items`) as silu(gate) * up in the activation type, clamped by `limit` first when it is above 0. False when
    /// the shape keeps the separate products.
    pub fn pairRun(k: *const Kernels, d: *const driver.Driver, arg: Arg, limit: f32, items: c_int, s: abi.Stream) Error!bool {
        if (arg.out16 == 0 or @rem(arg.n, 2) != 0 or !k.streamFits(arg)) return false;
        try k.streamGo(d, arg, .{ .pair_cols = @divExact(arg.n, 2), .limit = limit }, @divExact(arg.n, 2), items, s);
        return true;
    }

    /// Up to four products of `m` rows over the same x in one launch (`a` holds x, m, k, bits, group, fp16 and the tables'
    /// kind); each side's output is (m, n) fp32, or the activation type with `out_half`.
    pub fn groupRun(k: *const Kernels, d: *const driver.Driver, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        var a = arg;
        var sides: StreamSides = .{ .count = @intCast(group.len), .out_half = @intFromBool(out_half) };
        var total: c_int = 0;
        a.n = 0;
        for (group, 0..) |side, i| {
            sides.words[i] = side.words;
            sides.scale[i] = side.scale;
            sides.bias[i] = side.bias;
            sides.out[i] = side.out;
            sides.n[i] = side.n;
            total += side.n;
            a.n = @max(a.n, side.n);
        }
        a.words = group[0].words;
        a.scale.p = group[0].scale;
        a.bias.p = group[0].bias;
        if (group.len == 0 or group.len > 4 or !k.streamFits(a)) return refuse("group shape");
        try k.streamGo(d, a, sides, total, 1, s);
    }

    fn lanesLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        if (a.m < 1 or a.m > lane_rows or a.n < 1 or @rem(a.group, 32) != 0 or a.group > lane_group_max or @rem(a.k, a.group) != 0) return refuse("lanes shape");
        if (try k.streamLaunch(d, a, s, items)) return;
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

    /// The GEMM tile's piece loads read the code words 16, 8 or 4 bytes at a time by width; the tables are one kind.
    fn gemmFits(a: Arg) bool {
        const align_bytes: u64 = switch (a.bits) {
            4, 8 => 16,
            2, 6 => 8,
            else => 4,
        };
        return a.words % align_bytes == 0 and a.scale.kind == a.bias.kind;
    }

    /// The matrix-core GEMM tile takes this product: gfx11, BF16, not switched off, 16 rows or more (TF_WMMA=1 is the same as unset: decode stays on the dot2 tiles).
    fn wmmaTile(k: *const Kernels, a: Arg) bool {
        return k.matrix and k.mode != .off and a.fp16 == 0 and gemmFits(a) and a.m >= wmma_rows;
    }

    fn blockLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        return k.blockWith(d, a, s, items, k.tile);
    }

    /// The m >= 64 tile of `tile` (the GEMM tile falls back to the previous one for words it cannot load wide).
    pub fn blockWith(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int, tile: Tile) Error!void {
        if (a.m < 1 or a.n < 1 or @rem(a.group, 32) != 0 or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0) return refuse("block shape");
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        const matrix = tile == .gemm and k.wmmaTile(a);
        const f = if (matrix) k.wmma_gemm[b] else if (tile == .gemm and gemmFits(a)) k.gemm[b] else k.block[b];
        try go(d, f, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// Prefill's tile at any row count, so a prompt's rows have the same bits however it is cut: the matrix tile on
    /// gfx11 unless switched off, else the dot2 GEMM tile (the previous one for words it cannot load wide).
    pub fn prefillLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        if (a.m < 1 or a.n < 1 or !groupOk(a.group) or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0) return refuse("prefill shape");
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        const matrix = k.tile == .gemm and k.matrix and k.mode != .off and a.fp16 == 0 and gemmFits(a);
        const f = if (matrix) k.wmma_gemm[b] else if (k.tile == .gemm and gemmFits(a)) k.gemm[b] else k.block[b];
        try go(d, f, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    fn groupOk(group: c_int) bool {
        return group == 32 or group == 64 or group == 128;
    }

    /// launch_affine on the auto schedule: the activation type's dot2 tiles.
    pub fn run(k: *const Kernels, d: *const driver.Driver, arg: Arg, schedule: c_int, s: abi.Stream, partial: u64, parts: c_int, out_half: bool) Error!void {
        var a = arg;
        if (out_half) {
            // the activation type's output is the decode tile's: fp16 on RDNA2, and either type from the stream tile
            if ((a.fp16 == 0 and !k.streamFits(arg)) or schedule != 0 or parts > 1) return refuse("fp16 output is the decode tile");
            a.out16 = a.out;
            a.out = 0;
        }
        if (schedule == 3) return k.prefillLaunch(d, a, s, 1);
        if (partial != 0 and parts > 1) return k.split(d, a, partial, parts, s);
        if (k.wmma) {
            if (schedule != 0 or a.fp16 != 0) return refuse("this schedule stays in the library");
            // BF16 x on a gfx11 / gfx12 build: the decode tiles up to 8 rows, the GEMM tile past them.
            if (a.m < 1 or a.n < 1 or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0 or !groupOk(a.group)) return refuse("bf16 shape");
            if (a.m <= lane_rows) return k.lanesLaunch(d, a, s, 1);
            if (try k.streamLaunch(d, a, s, 1)) return;
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
            if (groupOk(a.group) and try k.streamLaunch(d, a, s, 1)) return;
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
        return k.routedWith(d, arg, items, s, k.tile);
    }

    pub fn routedWith(k: *const Kernels, d: *const driver.Driver, arg: Arg, items: c_int, s: abi.Stream, tile: Tile) Error!void {
        // the family's own activation type: bf16 on the WMMA build, fp16 on RDNA2
        if ((arg.fp16 != 0) == k.wmma or arg.route.items == 0 or arg.route.members == 0 or items < 1 or arg.route.x_div < 1) return refuse("routed plan");
        if (!groupOk(arg.group)) return refuse("routed group");
        if (arg.m <= lane_rows and !(k.tile == .gemm and k.wmmaTile(arg))) return k.lanesLaunch(d, arg, s, items);
        if (try k.streamLaunch(d, arg, s, items)) return;
        try k.blockWith(d, arg, s, items, tile);
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
