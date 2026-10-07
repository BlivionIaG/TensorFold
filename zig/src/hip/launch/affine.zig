//! The MLX affine kernels launched from Zig: the tiles of tiles/dot2_tiles.hip and tiles/dot2.hip over the family's code
//! objects, a launcher a tile. Which tile takes a product is the registry's (registry.zig); the entry points at the end
//! (`run`, `routed`, `prefillLaunch`, `groupRun`, `pairRun`) are what the Zig launches and the C launchers' names call.

const std = @import("std");
const abi = @import("../runtime/abi.zig");
const driver = @import("../runtime/driver.zig");
const hl = @import("../runtime/launch.zig");
const Module = @import("../runtime/module.zig").Module;
const Function = @import("../runtime/module.zig").Function;

const Policy = @import("../policy.zig").Policy;
const Caps = @import("../caps.zig").Caps;
const Choice = @import("../policy.zig").Choice;
const registry = @import("registry.zig");
const quant = @import("../quant/quant.zig");
const tuning = @import("../tuning/tuning.zig");
const mlx_entries = @import("mlx_entries.zig");

const Error = driver.Error;

/// quant/mlx.hpp's GroupTable, Routing and Affine as the kernels take them by value.
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

/// quant/mlx_tiles.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
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

/// Everything a launch needs beside the kernels.
pub const Call = struct {
    d: *const driver.Driver,
    s: abi.Stream,
    arg: Arg,
    items: c_int = 1,
    sides: []const Side = &.{},
    limit: f32 = 0,
    out_half: bool = false,
    partial: u64 = 0,
    parts: c_int = 1,
};

/// A kernel of the registry: its launch on this backend's kernels.
pub const Launch = *const fn (*const Kernels, Call) Error!void;
pub const Registry = registry.Registry(Launch);
pub const Entry = Registry.E;

pub const bit_widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
const row_counts = [_]u8{ 1, 2, 4, 8 };
const piece_counts = [_]u8{ 1, 2, 4 };

pub const lane_rows = 8; // kLaneRows
pub const lane_group_max = 128;
const fast_n = 256;
const stream_cbs = [_]c_int{ 2, 4, 8 }; // columns a lane carries in the stream tile
const stream_rows = [_]u8{ 1, 2, 4 }; // its row counts: columns * rows <= 16
const stream_waves = 4; // kStreamWaves
const streams_wanted = 3000; // kStreamWavesWanted
const stream_code_words = 32; // kStreamCodeWords

/// The K-parallel tiles (tiles/gemm_kp.hpp) of a short prompt: cb columns a lane set, r rows a pass over up to rb row
/// blocks, waves a block (`loop`: it takes more rows than rb * r by passes). Which of them takes a product, by rows
/// and by a routed plan's items, is the tuning table's.
pub const KpTile = struct { cb: c_int, r: c_int, waves: c_int, rb: c_int, loop: bool = true };
pub const kp_tiles = [_]KpTile{
    .{ .cb = 8, .r = 1, .waves = 2, .rb = 2, .loop = false },
    .{ .cb = 4, .r = 4, .waves = 2, .rb = 4 },
    .{ .cb = 4, .r = 8, .waves = 2, .rb = 4, .loop = false },
    .{ .cb = 4, .r = 2, .waves = 2, .rb = 4 },
};

/// The GEMM tiles of the m >= 64 products: `gemm` (the default) or the previous `block`.
pub const Tile = enum { gemm, block };

/// Which GEMM tile of the 128-row family a launch runs: the matrix cores, the dot2 tile, or the previous tile.
pub const BlockKind = enum { matrix, gemm, block };

pub const Kernels = struct {
    /// The block shapes prefill takes by rows: the K-parallel tiles, then the 128 x 128 one (the tests compare them).
    pub const tier_count = kp_tiles.len + 1;

    /// Whether block shape `tier` can take `m` rows: a tile that does not loop over its rows holds rb * r of them.
    pub fn tierTakes(tier: usize, m: c_int) bool {
        return tier >= kp_tiles.len or kp_tiles[tier].loop or m <= kp_tiles[tier].rb * kp_tiles[tier].r;
    }

    caps: Caps,
    wmma: bool,
    /// gfx11: the matrix-core GEMM tile runs (gfx12's WMMA has other layouts).
    matrix: bool,
    /// The policy's `matrix`: `off` runs every product on the dot2 tiles, `on` and `auto` on the matrix cores where a tile exists.
    mode: Choice,
    /// Which GEMM tile the products take; the policy's reference switch picks the previous one.
    tile: Tile,
    lanes: [bit_widths.len][row_counts.len][piece_counts.len]Function,
    row: [bit_widths.len][piece_counts.len]Function,
    /// The stream tile (tiles/stream.hpp) by width, rows and columns a lane; null where columns * rows > 16.
    stream: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with the (gate | up) activation as its epilogue.
    pairs: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// Off keeps the decode tiles before the stream tile (the policy's reference switch).
    stream_on: bool,
    block: [bit_widths.len]Function,
    gemm: [bit_widths.len]Function,
    wmma_gemm: [bit_widths.len]Function,
    kp: [bit_widths.len][kp_tiles.len]Function,
    span: [2]Function,
    fold: Function,
    wide: Function,
    tiled: Function,
    reference: Function,
    fill: Function,
    reg: Registry,

    /// `tiles` holds dot2_tiles.hip's kernels, `dot2` dot2.hip's; one activation type a family: bf16 on the
    /// WMMA build (v_dot2_f32_bf16), fp16 on RDNA2.
    pub fn load(tiles_obj: Module, dot2_obj: Module, caps: Caps, policy: Policy) Error!Kernels {
        var k: Kernels = undefined;
        const wmma = caps.act == .bf16;
        k.caps = caps;
        k.wmma = wmma;
        k.matrix = wmma and caps.matrix == .wmma11;
        k.mode = policy.matrix;
        k.tile = if (policy.gemmOn()) .gemm else .block;
        k.stream_on = policy.streamOn();
        k.reg = Registry.init(tableFor(caps), &mlx_entries.all);
        k.reg.verify(k.env(), .mlx) catch return error.Invalid;
        // the GEMM tile's rows of x a lane keeps (tiles/gemm.hpp GemmShape): 8 for BF16, 16 for FP16
        if (wmma) try k.resolve("7DotBF16", "Li8E", tiles_obj) else try k.resolve("6DotF16", "Li16E", tiles_obj);
        const r = "_ZN2tf4rocm";
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
            inline for (kp_tiles, 0..) |t, i| {
                k.kp[b][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm14affine_gemm_kpINS0_{s}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineE", .{ dot, bits, t.cb, t.r, t.waves, t.rb, @intFromBool(t.loop) }));
            }
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
    pub fn rowLanes(groups: c_int) c_int {
        if (groups < 4) return 0;
        var lpc: c_int = 4;
        while (lpc < groups and lpc < 32) lpc <<= 1;
        return lpc;
    }

    /// The lanes a column takes in the K-parallel tile: one a group, 32 at most (kp_lanes).
    fn kpLanes(groups: c_int) c_int {
        return if (groups > 16) 32 else if (groups > 8) 16 else 8;
    }

    /// The bytes the GEMM tile's piece loads need the words aligned to, by width.
    pub fn gemmAlign(bits: c_int) u64 {
        return switch (bits) {
            4, 8 => 16,
            2, 6 => 8,
            else => 4,
        };
    }

    // ---- what the choice sees of this build and of a call ----

    /// The GPU and the run's switches as the registry reads them.
    pub fn env(k: *const Kernels) registry.Env {
        return .{ .bf16 = k.wmma, .matrix = k.matrix, .matrix_on = k.matrix and k.mode != .off, .stream_on = k.stream_on, .gemm_on = k.tile == .gemm };
    }

    /// The MLX entry that takes a product, or null.
    pub fn choose(k: *const Kernels, op: registry.Op, path: registry.Path, shape: registry.Shape) ?*const Entry {
        return k.reg.select(k.env(), .mlx, op, path, shape);
    }

    /// The costs of the GPU's family: RDNA2, or the gfx11 family (RDNA4 takes it for its dot2 tiles).
    fn tableFor(caps: Caps) *const tuning.Table {
        return switch (caps.family) {
            .rdna2, .gcn5 => &tuning.gfx1030,
            .rdna3 => &tuning.gfx1100,
        };
    }

    /// The product of `a` as a shape: `items` is a routed plan's item count.
    pub fn shapeOf(a: Arg, items: c_int) registry.Shape {
        const kind: quant.Tables = switch (a.scale.kind) {
            1 => .bf16,
            2 => .f16,
            else => .f32,
        };
        return .{
            .m = @intCast(@max(a.m, 0)),
            .n = @intCast(@max(a.n, 0)),
            .k = @intCast(@max(a.k, 0)),
            .bits = std.math.cast(u8, a.bits) orelse 0,
            .group = std.math.cast(u16, a.group) orelse 0,
            .fp16 = a.fp16 != 0,
            .tables = kind,
            .tables_alike = a.scale.kind == a.bias.kind,
            .x_aligned = a.x % 16 == 0,
            .words_aligned = a.words % gemmAlign(a.bits) == 0,
            .items = if (a.route.items != 0) @intCast(@max(items, 0)) else 0,
        };
    }

    // ---- the tiles ----

    /// The stream tile on `total_n` columns (a group's sum); `sides` counts them off block by block when it has any.
    pub fn streamGo(k: *const Kernels, d: *const driver.Driver, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream) Error!void {
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

    /// One row on the row tile: a lane owns a group, a wave reads a column's row in one run.
    pub fn rowGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        args.add(rowLanes(@divExact(a.k, a.group)));
        try go(d, k.row[b][pieceIndex(a.group)], .{ .x = cdiv(a.n, 128), .z = @intCast(items) }, .{ .x = 128 }, s, &args);
    }

    /// Up to 8 rows on the column tile.
    pub fn lanesGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        const r: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else if (a.m <= 4) 2 else 3;
        try go(d, k.lanes[b][r][pieceIndex(a.group)], .{ .x = cdiv(a.n, 32), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// The wide tile of the RDNA2 decode products of 9 rows and more.
    pub fn wideGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        args.add(a);
        try go(d, k.wide, .{ .x = cdiv(a.n, 32), .y = cdiv(a.m, 128) }, .{ .x = 512 }, s, &args);
    }

    /// Any group width up to 128 that the others do not take, 16 columns by 16 rows a block.
    pub fn tiledGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        args.add(a);
        try go(d, k.tiled, .{ .x = cdiv(a.n, 16), .y = cdiv(a.m, 16) }, .{ .x = 256 }, s, &args);
    }

    /// The one-thread tile every product has: the reference of the others.
    pub fn referenceGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        args.add(a);
        try go(d, k.reference, .{ .x = cdiv(a.n, 32), .y = @intCast(a.m) }, .{ .x = 32 }, s, &args);
    }

    /// The decode tile over groups [z * per, (z + 1) * per) into `partial`, then the fold in group order.
    pub fn splitGo(k: *const Kernels, d: *const driver.Driver, a: Arg, partial: u64, parts: c_int, s: abi.Stream) Error!void {
        const groups = @divTrunc(a.k, a.group);
        var args: hl.Args = .{};
        args.add(a);
        args.add(partial);
        args.add(@divExact(groups, parts));
        try go(d, k.span[if (a.m == 1) 0 else 1], .{ .x = cdiv(a.n, fast_n), .y = cdiv(a.m, 8), .z = @intCast(parts) }, .{ .x = fast_n }, s, &args);
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

    /// The 128 x 128 tile of a kind over every item of a plan (or the plain product).
    pub fn blockGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int, kind: BlockKind) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        const f = switch (kind) {
            .matrix => k.wmma_gemm[b],
            .gemm => k.gemm[b],
            .block => k.block[b],
        };
        try go(d, f, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// K-parallel tile `tier` (an index of kp_tiles) of a short prompt.
    pub fn kpGo(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int, tier: usize) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        args.add(a);
        const t = kp_tiles[tier];
        const sets = @divExact(32, kpLanes(@divExact(a.k, a.group)));
        const blocks = @min(cdiv(a.m, t.r), @as(c_uint, @intCast(t.rb)));
        try go(d, k.kp[b][tier], .{ .x = cdiv(a.n, t.waves * t.cb * sets) * blocks, .z = @intCast(items) }, .{ .x = @intCast(32 * t.waves) }, s, &args);
    }

    // ---- the entry points ----

    /// Whether the matrix-core tile takes this product's 128-row blocks (gfx11, bf16, switched on, words it can load wide).
    fn matrixTakes(k: *const Kernels, a: Arg) bool {
        return k.matrix and k.mode != .off and a.fp16 == 0 and a.words % gemmAlign(a.bits) == 0 and a.scale.kind == a.bias.kind;
    }

    fn blockShapeOk(a: Arg) bool {
        return a.m >= 1 and a.n >= 1 and @rem(a.group, 32) == 0 and @rem(a.k, a.group) == 0 and @rem(a.k, 16) == 0;
    }

    /// The 128-row tile of `tile` over the plan (the GEMM tile falls back to the previous one for words it cannot load wide).
    pub fn blockWith(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int, tile: Tile) Error!void {
        if (!blockShapeOk(a)) return refuse("block shape");
        const fits = a.words % gemmAlign(a.bits) == 0 and a.scale.kind == a.bias.kind;
        const kind: BlockKind = if (tile == .gemm and k.matrixTakes(a) and a.m >= 16) .matrix else if (tile == .gemm and fits) .gemm else .block;
        try k.blockGo(d, a, s, items, kind);
    }

    /// Prefill's tile at any row count, so a prompt's rows have the same bits however it is cut: the matrix tile on
    /// gfx11 unless switched off, else the dot2 GEMM tile (the previous one for words it cannot load wide); a few rows
    /// take the K-parallel tile, which computes the same bits and streams the weights faster.
    pub fn prefillLaunch(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const op: registry.Op = if (a.route.items != 0) .routed else .project;
        const e = k.choose(op, .prefill, shapeOf(a, items)) orelse return refuse("prefill shape");
        try e.launch(k, .{ .d = d, .s = s, .arg = a, .items = items });
    }

    /// prefillLaunch in block shape `tier`: an index of kp_tiles, or `kp_tiles.len` for the 128 x 128 one.
    pub fn prefillTier(k: *const Kernels, d: *const driver.Driver, a: Arg, s: abi.Stream, items: c_int, tier: usize) Error!void {
        const group_ok = a.group == 32 or a.group == 64 or a.group == 128;
        if (a.m < 1 or a.n < 1 or !group_ok or @rem(a.k, a.group) != 0 or @rem(a.k, 16) != 0) return refuse("prefill shape");
        const fits = a.words % gemmAlign(a.bits) == 0 and a.scale.kind == a.bias.kind;
        const matrix = k.tile == .gemm and k.matrixTakes(a);
        const dot2 = k.tile == .gemm and fits;
        if (tier < kp_tiles.len and (matrix or dot2) and tierTakes(tier, a.m)) return k.kpGo(d, a, s, items, tier);
        try k.blockGo(d, a, s, items, if (matrix) .matrix else if (dot2) .gemm else .block);
    }

    /// launch_affine on the auto schedule: the activation type's tiles. `schedule` 3 is prefill's, 4 a lane round's;
    /// `partial` and `parts` split the groups over a scratch (the tests').
    pub fn run(k: *const Kernels, d: *const driver.Driver, arg: Arg, schedule: c_int, s: abi.Stream, partial: u64, parts: c_int, out_half: bool) Error!void {
        var a = arg;
        if (schedule == 1) {
            if (a.fp16 == 0 or k.wmma) return refuse("this schedule stays in the library");
            return k.referenceGo(d, a, s);
        }
        if (schedule == 2 or schedule > 4) return refuse("this schedule stays in the library");
        var shape = shapeOf(a, 1);
        shape.round = schedule == 4;
        shape.parts = if (partial != 0 and parts > 1) @intCast(parts) else 1;
        const e = k.choose(.project, if (schedule == 3) .prefill else .decode, shape) orelse return refuse("no tile takes this product");
        if (out_half) {
            // the activation type's output is the decode tile's: fp16 on RDNA2, and either type from the stream tile
            if (schedule == 3 or parts > 1 or !e.roundsAct(shape)) return refuse("fp16 output is the decode tile");
            a.out16 = a.out;
            a.out = 0;
        }
        try e.launch(k, .{ .d = d, .s = s, .arg = a, .partial = partial, .parts = parts, .out_half = out_half });
    }

    /// affine_routed_launch: every item of a plan in one launch; `arg.m` is the most rows an item holds.
    pub fn routed(k: *const Kernels, d: *const driver.Driver, arg: Arg, items: c_int, s: abi.Stream) Error!void {
        return k.routedWith(d, arg, items, s, k.tile);
    }

    pub fn routedWith(k: *const Kernels, d: *const driver.Driver, arg: Arg, items: c_int, s: abi.Stream, tile: Tile) Error!void {
        // the family's own activation type: bf16 on the WMMA build, fp16 on RDNA2
        if ((arg.fp16 != 0) == k.wmma or arg.route.items == 0 or arg.route.members == 0 or items < 1 or arg.route.x_div < 1) return refuse("routed plan");
        var environment = k.env();
        environment.gemm_on = tile == .gemm;
        const e = k.reg.select(environment, .mlx, .routed, .decode, shapeOf(arg, items)) orelse return refuse("no tile takes this plan");
        try e.launch(k, .{ .d = d, .s = s, .arg = arg, .items = items });
    }

    /// The stacked (gate | up) product of `arg` (n the stacked width, out16 the (rows, n / 2) activation, plain or routed
    /// over `items`) as silu(gate) * up in the activation type, clamped by `limit` first when it is above 0. False when
    /// the shape keeps the separate products.
    pub fn pairRun(k: *const Kernels, d: *const driver.Driver, arg: Arg, limit: f32, items: c_int, s: abi.Stream) Error!bool {
        var shape = shapeOf(arg, items);
        shape.pairs = arg.out16 != 0;
        const e = k.choose(.routed_act, .decode, shape) orelse return false;
        try e.launch(k, .{ .d = d, .s = s, .arg = arg, .items = items, .limit = limit });
        return true;
    }

    /// Up to four products of `m` rows over the same x in one launch (`a` holds x, m, k, bits, group, fp16 and the tables'
    /// kind); each side's output is (m, n) fp32, or the activation type with `out_half`.
    pub fn groupRun(k: *const Kernels, d: *const driver.Driver, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        if (group.len == 0 or group.len > 4) return refuse("group shape");
        var a = arg;
        a.n = 0;
        for (group) |side| a.n = @max(a.n, side.n);
        var shape = shapeOf(a, 1);
        shape.round = true;
        const e = k.choose(.group, .decode, shape) orelse return refuse("group shape");
        try e.launch(k, .{ .d = d, .s = s, .arg = a, .sides = group, .out_half = out_half });
    }

    /// The stream tile over a group: the sides' addresses and widths in one launch.
    pub fn groupGo(k: *const Kernels, d: *const driver.Driver, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
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
        try k.streamGo(d, a, sides, total, 1, s);
    }

    /// The split count of a decode launch: 1 unless `mode` is 2 (the tests' forced comparison) and the groups divide.
    pub fn splitCount(m: c_int, n: c_int, k: c_int, group: c_int, mode: c_int) c_int {
        if (mode != 2 or m < 1 or n < 512) return 1;
        if (!(group == 32 or group == 64 or group == 128) or k < group or @rem(k, group) != 0) return 1;
        const groups = @divTrunc(k, group);
        if (groups < 2) return 1;
        // A full row tile or a wide column grid already fills the card.
        const blocks = @divTrunc(n + fast_n - 1, fast_n) * @divTrunc(m + 8 - 1, 8);
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

test {
    _ = mlx_entries;
}
