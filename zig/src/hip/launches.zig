//! The ROCm kernels launched from Zig on the family's code objects, under the C launchers' names and parameters.

const std = @import("std");
const driver = @import("runtime/driver.zig");
const kernels = @import("kernels.zig");
const launch = @import("runtime/launch.zig");
const Module = @import("runtime/module.zig").Module;
const Function = @import("runtime/module.zig").Function;
const affine_launch = @import("launch/affine.zig");
const Policy = @import("policy.zig").Policy;
const Caps = @import("caps.zig").Caps;
const Affine = affine_launch.Kernels;

const util = @import("launch/util.zig");

pub const Error = util.Error;

const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const I = util.I;
const CI = util.CI;
const Dim3 = util.Dim3;
const Args = util.Args;
const Triple = util.Triple;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;

/// affine_dot2_splits: the split count of a decode launch (1 unless mode is 2).
pub const affineSplits = Affine.splitCount;


const norms = @import("launch/norms.zig");
const conv = @import("launch/conv.zig");
const rope = @import("launch/rope.zig");
const moe = @import("launch/moe.zig");
const attention = @import("launch/attention.zig");
const recurrence = @import("launch/recurrence.zig");
const plan = @import("launch/plan.zig");
const elementwise = @import("launch/elementwise.zig");

pub const ConvArgs = conv.ConvArgs;
pub const PlanArgs = plan.PlanArgs;
pub const PlanRef = plan.PlanRef;
pub const PlanKeep = plan.Keep;
pub const gdn_chunk = recurrence.gdn_chunk;
pub const GdnScratch = recurrence.GdnScratch;
pub const gdnScratch = recurrence.gdnScratch;

pub const Launcher = struct {
    d: *const driver.Driver,
    mods: [kernels.group_count]Module,
    rms: Triple,
    conv_decode: Function,
    conv_rows: Function,
    rope_decode: Function,
    moe_router: [2]Function, // fp16, bf16
    moe_select: Function,
    moe_act: [2]Function,
    moe_combine: Triple,
    gdn_gate: Triple,
    gated_delta_tile: [2]Function, // dk 128, 16
    gated_delta_wave: [2]Function,
    score_keys: Triple, // by cache kind
    apply_values: Triple,
    causal: Triple,
    fa_prefill: Triple,
    fa_wide: [2]Function, // the 64-row prefill tile, fp16 and bf16 caches
    gdn_chunked: [5]Function, // prep, kt, wy, h, o
    softmax_stats: Function,
    sum_partials: Function,
    op: Ops,
    /// decode.hip's kernels: a round's few rows, and the launches that merge several small ones.
    dec: Decode,
    /// plan.hip's kernels and the DeltaNet's over a round's plan.
    plan: PlanFns,
    /// The policy keeps the launches decode.hip replaces (`fuse` off).
    fuse: bool,
    /// The 64-row prefill attention tile is on.
    wide: bool,
    affine: Affine,

    const PlanFns = struct {
        kv_write: Function,
        score: [2]Function, // fp16, bf16
        apply: [2]Function,
        keep: Function,
        gather: Function,
        gdn: [2]Function, // dk 128, 16
    };

    const Decode = struct {
        router: [2]Function, // fp16, bf16
        tail: Function,
        conv_split: Function,
        rms2: Function,
        gnorm_out: Function,
        select: Function,
    };

    const Ops = struct {
        embed_rows: Function,
        cast: Function,
        silu_mul: Function,
        add: Function,
        attn_gate: Function,
        gnorm_silu: Function,
        copy_cols: Function,
        rope_prefill: Function,
        kv_write: Function,
        conv_prefill: Function,
        gdn_gate_prefill: Function,
        dense_rows: Function,
        moe_route: Function,
        router_rows: [2]Function, // fp16, bf16
        router_tile: [2]Function,
        router_small: [2]Function,
        rms_rows: Function,
        qk_rope: Function,
    };

    /// Loads the family's code objects on the current device and resolves every kernel the launchers use.
    pub fn load(d: *const driver.Driver, caps: Caps, policy: Policy, images: [kernels.group_count][]const u8) Error!Launcher {
        var l: Launcher = undefined;
        l.d = d;
        var loaded: usize = 0;
        errdefer for (l.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            l.mods[i] = try Module.load(d, img);
            loaded += 1;
        }
        const ops = l.mods[@backingInt(kernels.Group.ops)];
        const act = l.mods[@backingInt(kernels.Group.act)];
        const att = l.mods[@backingInt(kernels.Group.attention)];
        const gd = l.mods[@backingInt(kernels.Group.gated_delta)];
        const pre = l.mods[@backingInt(kernels.Group.prefill)];
        const dec = l.mods[@backingInt(kernels.Group.decode)];
        const pl = l.mods[@backingInt(kernels.Group.plan)];
        l.fuse = policy.fused();
        l.wide = policy.wideAttention();
        l.dec = .{ .router = .{ try dec.function("tf_router_decode_f16"), try dec.function("tf_router_decode_bf16") }, .tail = try dec.function("tf_tail"), .conv_split = try dec.function("tf_conv_split"), .rms2 = try dec.function("tf_rms2"), .gnorm_out = try dec.function("tf_gnorm_out"), .select = try dec.function("tf_select_decode") };
        l.plan = .{
            .kv_write = try pl.function("tf_plan_kv_write"),
            .score = .{ try pl.function("tf_plan_score_f16"), try pl.function("tf_plan_score_bf16") },
            .apply = .{ try pl.function("tf_plan_apply_f16"), try pl.function("tf_plan_apply_bf16") },
            .keep = try pl.function("tf_plan_keep"),
            .gather = try pl.function("tf_plan_gather"),
            .gdn = .{ try gd.function("tf_gdn_plan_128"), try gd.function("tf_gdn_plan_16") },
        };
        l.fa_wide = .{ try pre.function("tf_fa_wide_f16"), try pre.function("tf_fa_wide_bf16") };
        const gp = l.mods[@backingInt(kernels.Group.gdn_prefill)];
        l.gdn_chunked = .{ try gp.function("tf_gdn_prep"), try gp.function("tf_gdn_kt"), try gp.function("tf_gdn_wy"), try gp.function("tf_gdn_h"), try gp.function("tf_gdn_o") };
        const anon = "_ZN12_GLOBAL__N_1";
        l.op = .{
            .embed_rows = try ops.function(anon ++ "17embed_rows_kernelEPKjPKvS3_iPKiiiiiPvi"),
            .cast = try ops.function(anon ++ "11cast_kernelEPKviPvix"),
            .silu_mul = try ops.function(anon ++ "15silu_mul_kernelEPKvS1_Pvix"),
            .add = try ops.function(anon ++ "10add_kernelEPKvS1_Pvix"),
            .attn_gate = try ops.function(anon ++ "16attn_gate_kernelEPKfPKvPviiiii"),
            .gnorm_silu = try ops.function(anon ++ "17gnorm_silu_kernelEPKfPKvPvix"),
            .copy_cols = try ops.function(anon ++ "16copy_cols_kernelEPKvxiPviii"),
            .rope_prefill = try ops.function(anon ++ "19rope_prefill_kernelEPKviPvixxiiiiif"),
            .kv_write = try ops.function(anon ++ "15kv_write_kernelEPKvPviiiiii"),
            .conv_prefill = try ops.function(anon ++ "19conv_prefill_kernelEPKviPKfS3_PfS4_iii"),
            .gdn_gate_prefill = try ops.function(anon ++ "23gdn_gate_prefill_kernelEPKvS1_iPKfS3_PfS4_ii"),
            .dense_rows = try ops.function(anon ++ "17dense_rows_kernelEPKviPKfPviii"),
            .moe_route = try ops.function(anon ++ "16moe_route_kernelEPKiiiiPiS2_i"),
            .router_rows = .{ try ops.function("tf_router_rows_f16"), try ops.function("tf_router_rows_bf16") },
            .router_tile = .{ try ops.function("tf_router_tile_f16"), try ops.function("tf_router_tile_bf16") },
            .router_small = .{ try ops.function("tf_router_small_f16"), try ops.function("tf_router_small_bf16") },
            .rms_rows = try ops.function("tf_rms_rows"),
            .qk_rope = try ops.function(anon ++ "14qk_rope_kernelEPKvixiPKffiiiifPKiPfPvi"),
        };
        const r = "_ZN2tf4rocm";
        l.rms = .{
            try act.function(r ++ "10rms_kernelIfEEvPKT_PKfPS2_if"),
            try act.function(r ++ "10rms_kernelI6__halfEEvPKT_PKfPS3_if"),
            try act.function(r ++ "10rms_kernelI12hip_bfloat16EEvPKT_PKfPS3_if"),
        };
        l.conv_decode = try act.function(r ++ "18conv_decode_kernelEPKfS2_PfS3_ii");
        l.conv_rows = try act.function(r ++ "16conv_rows_kernelEPKfS2_PfS3_S3_iii");
        l.rope_decode = try act.function(r ++ "18rope_decode_kernelEPKfPfiiifPKii");
        l.moe_router = .{
            try act.function(r ++ "17moe_router_kernelI6__halfEEvPKT_PKfPfii"),
            try act.function(r ++ "17moe_router_kernelI12hip_bfloat16EEvPKT_PKfPfii"),
        };
        l.moe_select = try act.function(r ++ "17moe_select_kernelEPKfPiPfS3_S3_iii");
        l.moe_act = .{
            try act.function(r ++ "14moe_act_kernelI6__halfEEvPKfPT_ifx"),
            try act.function(r ++ "14moe_act_kernelI12hip_bfloat16EEvPKfPT_ifx"),
        };
        l.moe_combine = .{
            try act.function(r ++ "18moe_combine_kernelIfEEvPKfS3_PT_ii"),
            try act.function(r ++ "18moe_combine_kernelI6__halfEEvPKfS4_PT_ii"),
            try act.function(r ++ "18moe_combine_kernelI12hip_bfloat16EEvPKfS4_PT_ii"),
        };
        l.gdn_gate = .{
            try act.function(r ++ "15gdn_gate_kernelIfEEvPKT_S4_PKfS6_PfS7_ii"),
            try act.function(r ++ "15gdn_gate_kernelI6__halfEEvPKT_S5_PKfS7_PfS8_ii"),
            try act.function(r ++ "15gdn_gate_kernelI12hip_bfloat16EEvPKT_S5_PKfS7_PfS8_ii"),
        };
        l.gated_delta_tile = .{
            try gd.function(r ++ "16gated_delta_tileILi128ELi8ELi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
            try gd.function(r ++ "16gated_delta_tileILi16ELi8ELi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
        };
        l.gated_delta_wave = .{
            try gd.function(r ++ "18gated_delta_kernelILi128EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
            try gd.function(r ++ "18gated_delta_kernelILi16EEEvPKfS3_S3_S3_S3_PfS4_iiiiS4_"),
        };
        l.score_keys = .{
            try att.function("_Z10score_keysI6__halfEvPKfPKT_PfiiiifixxxPKi"),
            try att.function("_Z10score_keysI12hip_bfloat16EvPKfPKT_PfiiiifixxxPKi"),
            try att.function("_Z10score_keysIfEvPKfPKT_PfiiiifixxxPKi"),
        };
        l.apply_values = .{
            try att.function("_Z12apply_valuesI6__halfEvPKfS2_PKT_PfiiiiixxxPKi"),
            try att.function("_Z12apply_valuesI12hip_bfloat16EvPKfS2_PKT_PfiiiiixxxPKi"),
            try att.function("_Z12apply_valuesIfEvPKfS1_PKT_PfiiiiixxxPKi"),
        };
        l.causal = .{
            try att.function(r ++ "13causal_kernelI6__halfEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(r ++ "13causal_kernelI12hip_bfloat16EEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(r ++ "13causal_kernelIfEEvPKfPKT_S6_Pfiiiiifixxxxxx"),
        };
        l.fa_prefill = .{
            try att.function(r ++ "10fa_prefillI6__halfLi64EEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(r ++ "10fa_prefillI12hip_bfloat16Li64EEEvPKfPKT_S7_Pfiiiiifixxxxxx"),
            try att.function(r ++ "10fa_prefillIfLi32EEEvPKfPKT_S6_Pfiiiiifixxxxxx"),
        };
        l.softmax_stats = try att.function("_Z13softmax_statsPKfPfiiPKi");
        l.sum_partials = try att.function("_Z12sum_partialsPKfPfii");
        l.affine = try Affine.load(l.mods[@backingInt(kernels.Group.affine_tiles)], l.mods[@backingInt(kernels.Group.affine_dot2)], caps, policy);
        try l.affine.fillByteLut(d, l.mods[@backingInt(kernels.Group.affine_dot2)]);
        return l;
    }

    pub fn unload(l: *Launcher) void {
        for (&l.mods) |*m| m.unload();
    }

    pub const tf_rms = norms.tf_rms;
    pub const tf_rms2 = norms.tf_rms2;
    pub const tf_gnorm_out = norms.tf_gnorm_out;
    pub const tf_tail = norms.tf_tail;
    pub const tf_conv_decode = conv.tf_conv_decode;
    pub const tf_conv_rows = conv.tf_conv_rows;
    pub const tf_conv_split = conv.tf_conv_split;
    pub const tf_conv_prefill = conv.tf_conv_prefill;
    pub const tf_qk_rope = rope.tf_qk_rope;
    pub const tf_rope_decode = rope.tf_rope_decode;
    pub const tf_rope_prefill = rope.tf_rope_prefill;
    pub const routerTile = moe.routerTile;
    pub const routerWith = moe.routerWith;
    pub const routerWindow = moe.routerWindow;
    pub const tf_moe_router = moe.tf_moe_router;
    pub const tf_moe_select = moe.tf_moe_select;
    pub const tf_moe_act = moe.tf_moe_act;
    pub const tf_moe_combine = moe.tf_moe_combine;
    pub const tf_moe_route = moe.tf_moe_route;
    pub const tf_causal = attention.tf_causal;
    pub const tf_attn_gate = attention.tf_attn_gate;
    pub const tf_kv_write = attention.tf_kv_write;
    pub const tf_gdn_gate = recurrence.tf_gdn_gate;
    pub const tf_gated_delta = recurrence.tf_gated_delta;
    pub const planKvWrite = plan.planKvWrite;
    pub const planCausal = plan.planCausal;
    pub const planGatedDelta = plan.planGatedDelta;
    pub const planKeep = plan.planKeep;
    pub const planGather = plan.planGather;
    pub const gdnChunked = recurrence.gdnChunked;
    pub const tf_gdn_gate_prefill = recurrence.tf_gdn_gate_prefill;
    pub const tf_gnorm_silu = recurrence.tf_gnorm_silu;
    pub const tf_embed_rows = elementwise.tf_embed_rows;
    pub const tf_cast = elementwise.tf_cast;
    pub const tf_silu_mul = elementwise.tf_silu_mul;
    pub const tf_add = elementwise.tf_add;
    pub const tf_copy_cols = elementwise.tf_copy_cols;
    pub const tf_dense_rows = elementwise.tf_dense_rows;

    pub fn go(l: *const Launcher, f: Function, grid: Dim3, block: Dim3, shared: u32, s: S, args: *Args) Error!void {
        try launch.launch(f, .{ .grid = grid, .block = block, .shared = shared }, .{ .d = l.d, .handle = s }, args);
    }

    pub fn tf_affine(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, m: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, schedule: c_int, fp16: c_int, s: S, partial: F, splits: c_int, out_half: c_int) Error!void {
        try l.affine.run(l.d, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = m, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16 }, schedule, s, ad(partial), splits, out_half != 0);
    }

    pub fn tf_affine_routed(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, items: CI, count: c_int, members: CI, x_div: c_int, rows: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, fp16: c_int, s: S) Error!void {
        try l.affine.routed(l.d, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = rows, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16, .route = .{ .items = ad(items), .members = ad(members), .x_div = x_div } }, count, s);
    }

    /// The 256-thread, one-element-a-thread launch of an ops.hip kernel over `n` elements.
    pub fn flat(l: *const Launcher, f: Function, n: i64, s: S, a: *Args) Error!void {
        if (n == 0) return;
        try l.go(f, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, a);
    }
};
