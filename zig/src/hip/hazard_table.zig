//! What every kernel of a decode round reads and writes, per kernel parameter, for the hazard check of a captured round (`hazard.zig`).
const std = @import("std");

/// How a device pointer is used by the kernel.
pub const Mode = enum { r, w, rw };

/// One kernel parameter, in the order of the kernel's C signature and of the launcher's `Args.add` calls.
pub const Arg = union(enum) {
    /// A scalar, or a by-value struct holding no device pointer.
    val,
    /// One 8-byte device pointer (read-only: `const T*`; written: the kernel stores through it; `rw` both). A pointer the launcher may pass as 0 (null) is still `ptr`.
    ptr: Mode,
    /// A by-value struct that is a sequence of 8-byte words: one entry per word, null = not a device pointer (a size, two packed ints...), a Mode = a device pointer used that way.
    words: []const ?Mode,
};

/// State a kernel touches through the plan's descriptors (PlanArgs.desc / .snaps / .pool), never as an argument.
pub const Hidden = struct {
    pub const What = enum {
        /// the K and V pages of one layer of the slot's caches
        kv,
        /// the conv window state of one linear layer (slot cache)
        conv,
        /// the DeltaNet recurrent state of one linear layer (slot cache)
        gdn,
        /// the kept-input snapshots of one linear layer (`PlanArgs.snaps` -> per-layer table of conv inputs / k, v, gate, beta buffers)
        snaps,
        /// memory shared by launches that is not an argument: `__device__` globals, ticket/fence words, a scratch buffer the launcher owns (not the arena)
        scratch,
    };
    what: What,
    mode: Mode,
    /// Index (0-based) of the kernel parameter that holds the layer number (a c_int), for kv/conv/gdn/snaps; null for scratch.
    layer_arg: ?u8 = null,
    /// With `layer_arg`: the layer is not that parameter itself but the low half (a c_int) of this 8-byte word of it, a by-value struct of words.
    layer_word: ?u8 = null,
};

pub const Entry = struct {
    /// Exactly the string the launcher passes to `Module.function(...)` in `Launcher.load` (zig/src/hip/launches.zig and zig/src/hip/launches/*.zig, affine tiles in launches/affine.zig): mangled C++ names stay mangled.
    name: []const u8,
    /// One element per kernel parameter.
    args: []const Arg,
    hidden: []const Hidden = &.{},
    /// The identifier the kernel has in the .hip sources if it is a plain `extern "C" __global__ void <src>(...)` (the integrator cross-checks const-qualified pointers against `r`); null for templated/mangled kernels and macro-generated tiles.
    src: ?[]const u8 = null,
};

// ---- shorthand ----

const R: Arg = .{ .ptr = .r };
const W: Arg = .{ .ptr = .w };
const RW: Arg = .{ .ptr = .rw };
const V: Arg = .val;

/// plan.hpp's PlanArgs by value (one parameter, 56 bytes): pos, slot, first, count, desc, snaps are device arrays the kernels only read;
/// `pages` and `pool` share the last word. launches/plan.zig adds `p.args` whole.
const PLAN: Arg = .{ .words = &.{ .r, .r, .r, .r, .r, .r, null } };

/// quant/mlx.hpp's GroupTable by value (launches/affine.zig `GroupTable`, 16 bytes): the table pointer, the kind.
const GROUP_TABLE: Arg = .{ .words = &.{ .r, null } };

/// quant/mlx.hpp's Affine by value (launches/affine.zig `Arg`, 112 bytes; the launchers add the struct whole): x, words, scale{p, kind}, bias{p, kind}, out,
/// m n, k bits, group fp16, route{items, members, x_div first}, out16. `out` and `out16` are only stored through (tiles/epilogue.hpp, tiles/dot2*.hip,
/// quant/mlx_tiles.hpp: no tile reads them back); `route.items` and `route.members` are only read (quant/mlx.hpp take_item, tiles/plan.hpp).
const AFFINE: Arg = .{ .words = &.{ .r, .r, .r, null, .r, null, .w, null, null, null, .r, .r, null, .w } };

/// quant/mlx_tiles.hpp's StreamSides by value (launches/stream_layout.zig, 184 bytes): words[4], scale[4], bias[4] read; out[4] written (unused sides hold 0);
/// then n[4], first[5], count, out_half, pair_cols, limit: 7 words of ints and a float.
const STREAM_SIDES: Arg = .{ .words = &.{
    .r,   .r,   .r,   .r,
    .r,   .r,   .r,   .r,
    .r,   .r,   .r,   .r,
    .w,   .w,   .w,   .w,
    null, null, null, null,
    null, null, null,
} };

/// decode.hip's ConvArgs by value (launches/conv.zig `ConvArgs`, 232 bytes), 29 words:
///  0 x  1 kind  2 weight  3 state  4 states  5 qn  6 kn  7 v  8 channels,kernel  9 rows,kw  10 vw  11 qw  12 kw_w  13 eps,norm
///  14 ga  15 gb  16 a_log  17 dt_bias  18 gate  19 beta  20 gcount,heads  21..27 PlanArgs  28 layer,slots (layer is the low half).
/// `state` is read (history) and stored (window) by the non-plan path; the launcher passes 0 and the plan path takes it from the descriptor (see `conv`).
const CONV: Arg = .{ .words = &.{
    .r,   null, .r, .rw,  .w,   .w,   .w,   .w,
    null, null, null, .r, .r,   null, .r,   .r,
    .r,   .r,   .w, .w,   null, .r,   .r,   .r,
    .r,   .r,   .r, null, null,
} };

/// act.hip's AddRmsIn by value (launches/norms.zig `AddRmsIn`, 40 bytes): y (read in modes 0 and 1, and 2), mode, wts, shared, slots.
const ADD_RMS_IN: Arg = .{ .words = &.{ .r, null, .r, .r, null } };

fn plain(comptime name: []const u8, comptime args: []const Arg) Entry {
    return .{ .name = name, .args = args, .src = name };
}

fn planned(comptime name: []const u8, comptime args: []const Arg, comptime hidden: []const Hidden) Entry {
    return .{ .name = name, .args = args, .hidden = hidden, .src = name };
}

fn mangled(comptime name: []const u8, comptime args: []const Arg) Entry {
    return .{ .name = name, .args = args };
}

// ---- hidden state ----

/// plan.hip's kv write: both pages of the layer stored (desc[plan_first/second(layer)]).
const kv_write: []const Hidden = &.{.{ .what = .kv, .mode = .w, .layer_arg = 7 }};
/// plan.hip's score and apply bodies: the layer's keys / values loaded through the descriptor.
const kv_read_8: []const Hidden = &.{.{ .what = .kv, .mode = .r, .layer_arg = 8 }};
const kv_read_10: []const Hidden = &.{.{ .what = .kv, .mode = .r, .layer_arg = 10 }};
/// gated_delta.hip's plan scan: the state is loaded and stored (kept as found when the window is replayed; rw covers both).
const gdn_state: []const Hidden = &.{.{ .what = .gdn, .mode = .rw, .layer_arg = 10 }};
/// decode.hip's conv_split: the window loaded and stored (not stored when the window is replayed), the conv inputs of a replayed window stored into
/// snaps[2 * layer]. The layer is a c_int inside the by-value ConvArgs (words[28], low half) of parameter 0, not a parameter of its own.
const conv_state: []const Hidden = &.{
    .{ .what = .conv, .mode = .rw, .layer_arg = 0, .layer_word = 28 },
    .{ .what = .snaps, .mode = .w, .layer_arg = 0, .layer_word = 28 },
};
/// A host word the argmax pick is mirrored to (the device word `mirror_at` holds its address; ops/draw.hpp `mirror_pick`).
const host_mirror: []const Hidden = &.{.{ .what = .scratch, .mode = .w }};
/// plan.hip's tf_plan_tap: the drafter's tap ring, desc[plan.pages - 1], an address in the slot descriptor.
const tap_ring: []const Hidden = &.{.{ .what = .scratch, .mode = .w }};

// ---- the kernels launched by a decode round ----

const kernels = [_]Entry{
    // ---- embedding, casts, elementwise (launches/elementwise.zig; ops/elementwise.hpp) ----
    // tf_embed_rows: elementwise.zig tf_embed_rows (11 adds); kernel ops/elementwise.hpp:19. words, scale, bias, ids read; out stored.
    plain("tf_embed_rows", &.{ R, R, R, V, R, V, V, V, V, W, V }),
    // tf_embed_dense: elementwise.zig (7 adds); ops/elementwise.hpp:34.
    plain("tf_embed_dense", &.{ R, V, R, V, V, W, V }),
    // tf_cast: elementwise.zig (5 adds); ops/elementwise.hpp:44.
    plain("tf_cast", &.{ R, V, W, V, V }),
    // tf_silu_mul: elementwise.zig (5 adds); ops/elementwise.hpp:49.
    plain("tf_silu_mul", &.{ R, R, W, V, V }),
    // tf_add: elementwise.zig (5 adds); ops/elementwise.hpp:56. `reduce.residual` passes out == x (in place): x is only read, out only stored, one element a thread.
    plain("tf_add", &.{ R, R, W, V, V }),
    // tf_copy_cols: elementwise.zig (7 adds); ops/elementwise.hpp:66 (window.zig's torch_text branch).
    plain("tf_copy_cols", &.{ R, V, V, W, V, V, V }),

    // ---- norms (launches/norms.zig) ----
    // rms_kernel<float|__half|hip_bfloat16>: norms.zig tf_rms (x, weight, y, width, eps = 5 adds, width > 1024); ops/act.hip:28. The names are launches.zig `l.rms`.
    mangled("_ZN2tf4rocm10rms_kernelIfEEvPKT_PKfPS2_if", &.{ R, R, W, V, V }),
    mangled("_ZN2tf4rocm10rms_kernelI6__halfEEvPKT_PKfPS3_if", &.{ R, R, W, V, V }),
    mangled("_ZN2tf4rocm10rms_kernelI12hip_bfloat16EEvPKT_PKfPS3_if", &.{ R, R, W, V, V }),
    // tf_rms_rows: norms.zig tf_rms (7 adds, width <= 1024); ops/norms.hpp:39.
    plain("tf_rms_rows", &.{ R, V, R, W, V, V, V }),
    // tf_rms2: norms.zig tf_rms2 (9 adds); decode.hip:735. Two row sets, each with its own weight and output (the dk 16 q/k norms).
    plain("tf_rms2", &.{ R, R, W, R, R, W, V, V, V }),
    // tf_gnorm_out: norms.zig tf_gnorm_out (9 adds); decode.hip:741.
    plain("tf_gnorm_out", &.{ R, R, R, W, V, V, V, V, V }),
    // tf_gnorm_silu: recurrence.zig tf_gnorm_silu (6 adds); ops/linear.hpp:180 (gatedOut when value_dim > 1024).
    plain("tf_gnorm_silu", &.{ R, R, W, V, V, V }),
    // tf_tail: norms.zig tf_tail (9 adds), reached by `Ops.addRms` (x = x + y, normed = rms(x) * weight); decode.hip:727. x is loaded and stored in place.
    plain("tf_tail", &.{ RW, R, R, R, W, V, V, V, V }),
    // tf_add_rms_f16 / _bf16: norms.zig tf_add_rms (x, AddRmsIn, weight, normed, width, eps = 6 adds); ops/act.hip:149. x in place.
    plain("tf_add_rms_f16", &.{ RW, ADD_RMS_IN, R, W, V, V }),
    plain("tf_add_rms_bf16", &.{ RW, ADD_RMS_IN, R, W, V, V }),

    // ---- rope, attention gate (launches/rope.zig, launches/attention.zig) ----
    // tf_qk_rope: rope.zig tf_qk_rope (15 adds); ops/rope.hpp:35. `wide` (fp32 rows) and `cache` (kind rows) are only stored through, either may be 0.
    plain("tf_qk_rope", &.{ R, V, V, V, R, V, V, V, V, V, V, R, W, W, V }),
    // tf_attn_gate: attention.zig (8 adds); ops/attention.hpp:10.
    plain("tf_attn_gate", &.{ R, R, W, V, V, V, V, V }),
    // tf_qvh_text_rope: vision/qwen_hip_ops.zig textRope (12 adds), window.zig's torch_text branch; kernels/hip/qwen_vision.hip:189 (the QVH_TEXT macro). Module: ops/vision group.
    plain("tf_qvh_text_rope", &.{ R, V, W, V, V, V, V, V, V, V, R, V }),

    // ---- the DeltaNet gate when heads are not 128 (launches/recurrence.zig tf_gdn_gate, 8 adds; ops/act.hip:360) ----
    mangled("_ZN2tf4rocm15gdn_gate_kernelIfEEvPKT_S4_PKfS6_PfS7_ii", &.{ R, R, R, R, W, W, V, V }),
    mangled("_ZN2tf4rocm15gdn_gate_kernelI6__halfEEvPKT_S5_PKfS7_PfS8_ii", &.{ R, R, R, R, W, W, V, V }),
    mangled("_ZN2tf4rocm15gdn_gate_kernelI12hip_bfloat16EEvPKT_S5_PKfS7_PfS8_ii", &.{ R, R, R, R, W, W, V, V }),

    // ---- the conv and the recurrence over the plan (launches/conv.zig, launches/plan.zig) ----
    // tf_conv_split: conv.zig tf_conv_split (1 add: ConvArgs whole); decode.hip:733. See `CONV` and `conv_state`.
    planned("tf_conv_split", &.{CONV}, conv_state),
    // tf_gdn_plan_128 / _16: plan.zig planGatedDelta (11 adds); gated_delta.hip:539 / :555. q, k, v, gate, beta read; y stored; the slot's state loaded and stored.
    planned("tf_gdn_plan_128", &.{ R, R, R, R, R, W, V, V, V, PLAN, V }, gdn_state),
    planned("tf_gdn_plan_16", &.{ R, R, R, R, R, W, V, V, V, PLAN, V }, gdn_state),

    // ---- attention over the plan (launches/plan.zig; decode/plan.hip) ----
    // tf_plan_kv_write: plan.zig planKvWrite (8 adds); plan.hip:340. fp32 keys, 16-bit values read; the layer's pages stored.
    planned("tf_plan_kv_write", &.{ R, R, V, V, V, V, PLAN, V }, kv_write),
    // tf_plan_kv_write_act: plan.zig planKvWriteAct (8 adds); plan.hip:369. keys in the activation kind.
    planned("tf_plan_kv_write_act", &.{ R, R, V, V, V, V, PLAN, V }, kv_write),
    // tf_plan_score{,2}_{f16,bf16}: plan.zig planCausalAs, first launch (9 adds); plan.hip:345-367. q fp32, scores stored; keys read through the plan.
    planned("tf_plan_score_f16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score_bf16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score2_f16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score2_bf16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    // tf_plan_score{,2}_act_{f16,bf16}: the same with q in the activation kind (`act`); plan.hip:374-395.
    planned("tf_plan_score_act_f16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score_act_bf16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score2_act_f16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_score2_act_bf16", &.{ R, W, V, V, V, V, V, PLAN, V }, kv_read_8),
    // tf_softmax_stats: plan.zig planCausalAs second launch (5 adds: scores, stats, span, 0, p.args.pos); attention.hip:118. `pos` is the plan's int array.
    plain("tf_softmax_stats", &.{ R, W, V, V, R }),
    // tf_plan_apply_{f16,bf16}: plan.zig planCausalAs third launch (9 adds); plan.hip:427-435. partials stored; values read through the plan.
    planned("tf_plan_apply_f16", &.{ R, R, W, V, V, V, V, PLAN, V }, kv_read_8),
    planned("tf_plan_apply_bf16", &.{ R, R, W, V, V, V, V, PLAN, V }, kv_read_8),
    // tf_sum_partials: plan.zig planCausalAs last launch (4 adds); attention.hip:200.
    plain("tf_sum_partials", &.{ R, W, V, V }),
    // tf_sum_partials_act: the same with `kind` (5 adds, act); plan.hip:397.
    plain("tf_sum_partials_act", &.{ R, W, V, V, V }),
    // tf_plan_attend_{f16,bf16}: plan.zig planAttend, the one-block walk when `o.fused()` (11 adds); plan.hip:417-426. Only `causalAct` takes it; `causal` (fp32 q) never does.
    planned("tf_plan_attend_f16", &.{ R, V, W, V, V, V, V, V, V, PLAN, V }, kv_read_10),
    planned("tf_plan_attend_bf16", &.{ R, V, W, V, V, V, V, V, V, PLAN, V }, kv_read_10),
    // tf_plan_tap: plan.zig planTap (7 adds), forward's tap of drafter layers; plan.hip:494. Stores into the tap ring the descriptor names.
    planned("tf_plan_tap", &.{ R, V, V, V, V, V, PLAN }, tap_ring),

    // ---- argmax and top-k of the head's logits (launches/draw.zig; ops/draw.hpp) ----
    // tf_argmax: draw.zig tf_argmax_rows_to (5 adds); draw.hpp:37. `mirror_at` is a device word (read) holding a host address the pick is stored to.
    planned("tf_argmax", &.{ R, V, V, W, R }, host_mirror),
    // tf_argmax_part: draw.zig tf_argmax_parts_to first launch (6 adds); draw.hpp:106.
    plain("tf_argmax_part", &.{ R, V, V, V, W, W }),
    // tf_argmax_join: draw.zig second launch (5 adds); draw.hpp:141.
    planned("tf_argmax_join", &.{ R, R, V, W, R }, host_mirror),
    // tf_topk: draw.zig tf_topk_rows (6 adds); draw.hpp:180. ks read; ids and values stored.
    plain("tf_topk", &.{ R, V, R, V, W, W }),

    // ---- split-K affine tail and the other non-tile affine kernels (launches/affine.zig; tiles/dot2.hip) ----
    // affine_dot2_span<1|8>: affine.zig splitGo (3 adds: Affine, partial, per); dot2.hip:191. Decode on RDNA2 only. partial stored.
    mangled("_ZN2tf4rocm16affine_dot2_spanILi1EEEvNS0_6AffineEPfi", &.{ AFFINE, W, V }),
    mangled("_ZN2tf4rocm16affine_dot2_spanILi8EEEvNS0_6AffineEPfi", &.{ AFFINE, W, V }),
    // affine_dot2_fold: splitGo's second launch (7 adds: partial, scale, bias, out, m, n, groups); dot2.hip:271. GroupTable passed whole.
    mangled("_ZN2tf4rocm16affine_dot2_foldEPKfNS0_10GroupTableES3_Pfiii", &.{ R, GROUP_TABLE, GROUP_TABLE, W, V, V, V }),
    // affine_dot2_wide / tiled / kernel(reference): affine.zig wideGo, tiledGo, referenceGo (1 add: Affine); dot2.hip:326, :128, :16.
    mangled("_ZN2tf4rocm16affine_dot2_wideENS0_6AffineE", &.{AFFINE}),
    mangled("_ZN2tf4rocm17affine_dot2_tiledENS0_6AffineE", &.{AFFINE}),
    mangled("_ZN2tf4rocm18affine_dot2_kernelENS0_6AffineE", &.{AFFINE}),
};

// ---- the MLX affine tiles by name (launches/affine.zig `Kernels.resolve`, one symbol a dot kind, bit width and shape) ----
//
// Every tile takes `Affine` whole (see `AFFINE`) except the row tile (+ lpc), the stream tiles (+ StreamSides, lpc_log2 | stage flag, group shift) and
// affine_dot2_span (above). The dot kind is "6DotF16" with GemmShape "Li16E" (fp16 activations: gfx900, RDNA2) or "7DotBF16" with "Li8E" (bf16, the WMMA
// builds); `Launcher.load` takes one by `caps.act`, both are listed. The relu^2 stream tiles are Nemotron's and not listed.

const Dot = struct { name: []const u8, shape: []const u8 };
const dots = [_]Dot{
    .{ .name = "6DotF16", .shape = "Li16E" },
    .{ .name = "7DotBF16", .shape = "Li8E" },
};
const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };
const row_counts = [_]u8{ 1, 2, 4, 8 };
const piece_counts = [_]u8{ 1, 2, 4 };
const stream_rows = [_]u8{ 1, 2, 4 };
const stream_cbs = [_]u8{ 2, 4, 8 };
const Kp = struct { cb: u8, r: u8, waves: u8, rb: u8, loop: bool };
const kp_tiles = [_]Kp{
    .{ .cb = 8, .r = 1, .waves = 2, .rb = 2, .loop = false },
    .{ .cb = 4, .r = 4, .waves = 2, .rb = 4, .loop = true },
    .{ .cb = 4, .r = 8, .waves = 2, .rb = 4, .loop = false },
    .{ .cb = 4, .r = 2, .waves = 2, .rb = 4, .loop = true },
};

const one_tile: []const Arg = &.{AFFINE};
const row_tile: []const Arg = &.{ AFFINE, V };
const stream_tile: []const Arg = &.{ AFFINE, STREAM_SIDES, V, V };

fn streamShapes() usize {
    var n: usize = 0;
    for (stream_rows) |rows| {
        for (stream_cbs) |cb| {
            if (@as(usize, cb) * rows <= 16) n += 1;
        }
    }
    return n;
}

/// Per (dot kind, width): block, gemm, two gemm_rows, the K-parallel tiles, the row tile a piece count, the lane tile a row count and piece count, the stream tile
/// and its (gate | up) pair variant a shape.
const per_tile = 1 + 1 + 2 + kp_tiles.len + piece_counts.len + row_counts.len * piece_counts.len + 2 * streamShapes();
const tile_count = dots.len * bit_widths.len * per_tile;

fn affineTiles() [tile_count]Entry {
    @setEvalBranchQuota(1_000_000);
    var out: [tile_count]Entry = undefined;
    var n: usize = 0;
    for (dots) |dot| {
        for (bit_widths) |bits| {
            // affine.zig blockGo: the reference GEMM block, the dot2 GEMM tile, its 64- and 32-row forms (gcn5)
            out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot.name, bits }), one_tile);
            n += 1;
            out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm17affine_gemm_blockINS0_{s}ELi{d}E{s}EEvNS0_6AffineE", .{ dot.name, bits, dot.shape }), one_tile);
            n += 1;
            for ([_]u8{ 64, 32 }) |rows| {
                out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm16affine_gemm_rowsINS0_{s}ELi{d}E{s}Li{d}EEEvNS0_6AffineE", .{ dot.name, bits, dot.shape, rows }), one_tile);
                n += 1;
            }
            // affine.zig kpGo
            for (kp_tiles) |t| {
                out[n] = mangled(std.fmt.comptimePrint(
                    "_ZN2tf4rocm14affine_gemm_kpINS0_{s}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineE",
                    .{ dot.name, bits, t.cb, t.r, t.waves, t.rb, @intFromBool(t.loop) },
                ), one_tile);
                n += 1;
            }
            for (piece_counts) |pieces| {
                // affine.zig rowGo (2 adds: Affine, lpc)
                out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_rowINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot.name, bits, pieces }), row_tile);
                n += 1;
                // affine.zig lanesGo (1 add)
                for (row_counts) |rows| {
                    out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_lanesINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ dot.name, bits, rows, pieces }), one_tile);
                    n += 1;
                }
            }
            // affine.zig streamWith (4 adds: Affine, StreamSides, lpc_log2 | stage flag, group shift); the `b` flag is the pair (gate | up) epilogue
            for ([_]u8{ 0, 1 }) |pair| {
                for (stream_rows) |rows| {
                    for (stream_cbs) |cb| {
                        if (@as(usize, cb) * rows > 16) continue;
                        out[n] = mangled(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot.name, bits, rows, cb, pair }), stream_tile);
                        n += 1;
                    }
                }
            }
        }
    }
    if (n != tile_count) @compileError("affineTiles: count");
    return out;
}

const tiles = affineTiles();
const all = kernels ++ tiles;

pub const entries: []const Entry = &all;

/// The entry of the kernel named `name` (the string `Module.function` took), or null.
pub fn find(name: []const u8) ?*const Entry {
    for (entries) |*e| {
        if (std.mem.eql(u8, e.name, name)) return e;
    }
    return null;
}

test "hazard table: every kernel is listed once and found by name" {
    for (entries, 0..) |e, i| {
        try std.testing.expect(e.name.len > 0);
        for (entries[0..i]) |before| try std.testing.expect(!std.mem.eql(u8, before.name, e.name));
        try std.testing.expectEqual(@as(?*const Entry, &entries[i]), find(e.name));
        // a hidden layer is read from a parameter that exists, and a struct word from a struct parameter
        for (e.hidden) |h| {
            if (h.layer_arg) |la| {
                try std.testing.expect(la < e.args.len);
                if (h.layer_word) |w| try std.testing.expect(e.args[la] == .words and w < e.args[la].words.len);
            } else {
                try std.testing.expect(h.what == .scratch);
            }
        }
    }
    try std.testing.expect(find("not_a_kernel") == null);
    // the launches of the decode round
    for ([_][]const u8{ "tf_qk_rope", "tf_plan_kv_write", "tf_conv_split", "tf_gdn_plan_128", "tf_tail", "tf_argmax", "tf_softmax_stats" }) |name| try std.testing.expect(find(name) != null);
}
