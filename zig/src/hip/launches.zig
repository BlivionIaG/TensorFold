//! The ROCm kernels launched from Zig: each launcher of zig/kernels/hip (capi.hip, ops.hip, rocm/*.hip) as a
//! hipModuleLaunchKernel on the family's code objects, with the same kernel, grid, block, shared bytes and arguments.
//! Methods keep the C launchers' names and parameter lists, so a call site reads the same on either path.

const std = @import("std");
const abi = @import("abi.zig");
const driver = @import("driver.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Affine = @import("affine_launch.zig").Kernels;

pub const Error = driver.Error;

const S = abi.Stream;
const P = ?*anyopaque;
const C = ?*const anyopaque;
const F = ?[*]f32;
const CF = ?[*]const f32;
const I = ?[*]i32;
const CI = ?[*]const i32;

const Dim3 = launch.Dim3;
const Args = launch.Args;

/// A device address as the kernel argument it is.
fn ad(p: anytype) u64 {
    return @intFromPtr(p);
}

fn dim(x: anytype, y: anytype, z: anytype) Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
}

/// ceil(n / by) as a grid extent.
fn cdiv(n: anytype, by: anytype) usize {
    return (@as(usize, @intCast(n)) + by - 1) / by;
}

/// affine_dot2_splits: the split count of a decode launch (1 unless mode is 2).
pub const affineSplits = Affine.splitCount;

/// The kernel of a triple for an activation kind: fp16 and bf16 by number, anything else the first.
fn tri(kind: c_int) usize {
    return if (kind == 1 or kind == 2) @intCast(kind) else 0;
}

/// By kind (0 fp32, 1 fp16, 2 bf16) or by cache kind (0 fp16, 1 bf16, 2 fp32), the kernel one instantiation.
const Triple = [3]Function;

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
    softmax_stats: Function,
    sum_partials: Function,
    op: Ops,
    affine: Affine,

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
    };

    /// Loads the family's code objects on the current device and resolves every kernel the launchers use.
    pub fn load(d: *const driver.Driver, wmma: bool, images: [kernels.group_count][]const u8) Error!Launcher {
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
        l.fa_wide = .{ try pre.function("tf_fa_wide_f16"), try pre.function("tf_fa_wide_bf16") };
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
        l.affine = try Affine.load(l.mods[@backingInt(kernels.Group.affine_tiles)], l.mods[@backingInt(kernels.Group.affine_dot2)], wmma);
        try l.affine.fillByteLut(d, l.mods[@backingInt(kernels.Group.affine_dot2)]);
        return l;
    }

    pub fn unload(l: *Launcher) void {
        for (&l.mods) |*m| m.unload();
    }

    fn go(l: *const Launcher, f: Function, grid: Dim3, block: Dim3, shared: u32, s: S, args: *Args) Error!void {
        try launch.launch(f, .{ .grid = grid, .block = block, .shared = shared }, .{ .d = l.d, .handle = s }, args);
    }

    fn invalid(what: []const u8) Error {
        std.log.err("{s}: shape", .{what});
        return error.Invalid;
    }

    pub fn tf_rms(l: *const Launcher, x: C, weight: CF, y: P, kind: c_int, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(weight));
        a.add(ad(y));
        a.add(width);
        a.add(eps);
        try l.go(l.rms[tri(kind)], dim(rows, 1, 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_conv_decode(l: *const Launcher, x: CF, weight: CF, state: F, y: F, batch: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(weight));
        a.add(ad(state));
        a.add(ad(y));
        a.add(channels);
        a.add(kernel);
        try l.go(l.conv_decode, dim(cdiv(channels, 128), batch, 1), dim(128, 1, 1), 0, s, &a);
    }

    pub fn tf_conv_rows(l: *const Launcher, x: CF, weight: CF, state: F, y: F, states: F, rows: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(weight));
        a.add(ad(state));
        a.add(ad(y));
        a.add(ad(states));
        a.add(channels);
        a.add(kernel);
        a.add(rows);
        try l.go(l.conv_rows, dim(cdiv(channels, 128), 1, 1), dim(128, 1, 1), 0, s, &a);
    }

    pub fn tf_rope_decode(l: *const Launcher, x: CF, y: F, rows: c_int, width: c_int, rotary: c_int, pos: c_int, theta: f32, s: S, pos_dev: CI, per: c_int) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(y));
        a.add(width);
        a.add(rotary);
        a.add(pos);
        a.add(theta);
        a.add(ad(pos_dev));
        a.add(per);
        try l.go(l.rope_decode, dim(rows, 1, 1), dim(32, 1, 1), 0, s, &a);
    }

    pub fn tf_moe_router(l: *const Launcher, x: C, kind: c_int, rows: CF, logits: F, r: c_int, d: c_int, e: c_int, s: S) Error!void {
        // a wave an expert over 8 rows: each logit's sum as moe_router_kernel's, the expert row read once for 8 rows
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(rows));
        a.add(ad(logits));
        a.add(r);
        a.add(d);
        a.add(e);
        try l.go(l.op.router_rows[if (kind == 1) 0 else 1], dim(cdiv(e, 8), cdiv(r, 8), 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_moe_select(l: *const Launcher, logits: CF, pick: I, wts: F, items: I, members: I, capacity: c_int, r: c_int, experts: c_int, top_k: c_int, s: S) Error!void {
        if (experts > 1024 or top_k < 1 or top_k > 31) return invalid("moe select");
        var a: Args = .{};
        a.add(ad(logits));
        a.add(ad(pick));
        a.add(ad(wts));
        a.add(ad(items));
        a.add(ad(members));
        a.add(capacity);
        a.add(experts);
        a.add(top_k);
        try l.go(l.moe_select, dim(r, 1, 1), dim(32, 1, 1), 0, s, &a);
    }

    pub fn tf_moe_act(l: *const Launcher, both: CF, out: P, kind: c_int, pairs: c_int, width: c_int, limit: f32, s: S) Error!void {
        const total = @as(i64, pairs) * width;
        var a: Args = .{};
        a.add(ad(both));
        a.add(ad(out));
        a.add(width);
        a.add(limit);
        a.add(total);
        try l.go(l.moe_act[if (kind == 1) 0 else 1], dim(cdiv(total, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_moe_combine(l: *const Launcher, y: CF, wts: CF, out: P, kind: c_int, r: c_int, slots: c_int, d: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(y));
        a.add(ad(wts));
        a.add(ad(out));
        a.add(slots);
        a.add(d);
        try l.go(l.moe_combine[tri(kind)], dim(cdiv(d, 256), r, 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_gdn_gate(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(av));
        a.add(ad(bv));
        a.add(ad(a_log));
        a.add(ad(dt_bias));
        a.add(ad(gate));
        a.add(ad(beta));
        a.add(count);
        a.add(heads);
        try l.go(l.gdn_gate[tri(kind)], dim(cdiv(count, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_gated_delta(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, batch: c_int, length: c_int, key_heads: c_int, value_heads: c_int, dk: c_int, dv: c_int, s: S, states: F) Error!void {
        if ((dk != 16 and dk != 128) or dv < 1 or key_heads < 1 or @rem(value_heads, key_heads) != 0 or batch < 1 or length < 1) {
            return invalid("gated delta");
        }
        var a: Args = .{};
        a.add(ad(q));
        a.add(ad(k));
        a.add(ad(v));
        a.add(ad(gate));
        a.add(ad(beta));
        a.add(ad(state));
        a.add(ad(y));
        a.add(length);
        a.add(key_heads);
        a.add(value_heads);
        a.add(dv);
        a.add(ad(states));
        const wide: usize = if (dk == 128) 0 else 1;
        // Eight rows share one k/q tile; a short dv stays on the one-row wave, which is the same scan.
        if (@rem(dv, 8) == 0) {
            try l.go(l.gated_delta_tile[wide], dim(@divExact(dv, 8), value_heads, batch), dim(32, 8, 1), 0, s, &a);
        } else {
            try l.go(l.gated_delta_wave[wide], dim(dv, value_heads, batch), dim(32, 1, 1), 0, s, &a);
        }
    }

    pub fn tf_causal(l: *const Launcher, q: CF, k: C, v: C, out: F, batch: c_int, qlen: c_int, span: c_int, heads: c_int, kv_heads: c_int, d: c_int, scale: f32, q_pos0: c_int, k_sb: c_longlong, k_sh: c_longlong, k_ss: c_longlong, v_sb: c_longlong, v_sh: c_longlong, v_ss: c_longlong, cache_kind: c_int, scores: F, stats: F, partials: F, s: S, pos: CI) Error!void {
        if (d < 1 or d > 256 or heads < 1 or kv_heads < 1 or @rem(heads, kv_heads) != 0 or batch < 1 or qlen < 1 or span < 1 or q_pos0 < 0) {
            return invalid("causal attention");
        }
        if (cache_kind < 0 or cache_kind > 2) return invalid("causal attention cache kind");
        const kind: usize = @intCast(cache_kind);
        const visible: c_int = @min(q_pos0 + 1, span);
        if (scores != null) {
            if (qlen != 1 or stats == null or partials == null) return invalid("the decode walk");
            // With a device position the keys in use are read on the device and the grid covers the whole cache.
            const tiles: c_int = @intCast(if (pos != null) cdiv(span, 128) else cdiv(visible, 128));
            const head_grid = dim(heads, batch, 1);
            var a: Args = .{};
            a.add(ad(q));
            a.add(ad(k));
            a.add(ad(scores));
            a.add(span);
            a.add(heads);
            a.add(kv_heads);
            a.add(d);
            a.add(scale);
            a.add(visible);
            a.add(k_sb);
            a.add(k_sh);
            a.add(k_ss);
            a.add(ad(pos));
            // a warp scores a key: four warps a block
            try l.go(l.score_keys[kind], dim(cdiv(span, 4), heads, batch), dim(128, 1, 1), 0, s, &a);
            var b: Args = .{};
            b.add(ad(scores));
            b.add(ad(stats));
            b.add(span);
            b.add(visible);
            b.add(ad(pos));
            try l.go(l.softmax_stats, head_grid, dim(256, 1, 1), 0, s, &b);
            var c: Args = .{};
            c.add(ad(scores));
            c.add(ad(stats));
            c.add(ad(v));
            c.add(ad(partials));
            c.add(span);
            c.add(heads);
            c.add(kv_heads);
            c.add(d);
            c.add(visible);
            c.add(v_sb);
            c.add(v_sh);
            c.add(v_ss);
            c.add(ad(pos));
            try l.go(l.apply_values[kind], dim(tiles, heads, batch), dim(256, 1, 1), 0, s, &c);
            var e: Args = .{};
            e.add(ad(partials));
            e.add(ad(out));
            e.add(tiles);
            e.add(d);
            try l.go(l.sum_partials, head_grid, dim(256, 1, 1), 0, s, &e);
            return;
        }
        var a: Args = .{};
        a.add(ad(q));
        a.add(ad(k));
        a.add(ad(v));
        a.add(ad(out));
        a.add(qlen);
        a.add(span);
        a.add(heads);
        a.add(kv_heads);
        a.add(d);
        a.add(scale);
        a.add(q_pos0);
        a.add(k_sb);
        a.add(k_sh);
        a.add(k_ss);
        a.add(v_sb);
        a.add(v_sh);
        a.add(v_ss);
        // A 16-bit cache, whole 64-wide head chunks and 16-byte rows take the 64-row tile (dot2 on fp16, TF_FA_WIDE=0
        // keeps the 16-row fp32 one); other even head sizes the 16-row tile; odd ones stay on one wave per query.
        const wide_off = if (std.c.getenv("TF_FA_WIDE")) |text| text[0] == '0' else false;
        if (kind < 2 and @rem(d, 64) == 0 and d <= 256 and @rem(k_ss, 8) == 0 and @rem(v_ss, 8) == 0 and !wide_off) {
            try l.go(l.fa_wide[kind], dim(cdiv(qlen, 64), heads, batch), dim(256, 1, 1), 0, s, &a);
        } else if (d >= 2 and @rem(d, 2) == 0) {
            try l.go(l.fa_prefill[kind], dim(cdiv(qlen, 16), heads, batch), dim(256, 1, 1), 0, s, &a);
        } else {
            try l.go(l.causal[kind], dim(qlen, heads, batch), dim(32, 1, 1), 0, s, &a);
        }
    }

    pub fn tf_affine(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, m: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, schedule: c_int, fp16: c_int, s: S, partial: F, splits: c_int, out_half: c_int) Error!void {
        try l.affine.run(l.d, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = m, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16 }, schedule, s, ad(partial), splits, out_half != 0);
    }

    pub fn tf_affine_routed(l: *const Launcher, x: C, words: C, scale: C, bias: C, scale_kind: c_int, out: P, items: CI, count: c_int, members: CI, x_div: c_int, rows: c_int, n: c_int, k: c_int, bits: c_int, group: c_int, fp16: c_int, s: S) Error!void {
        try l.affine.routed(l.d, .{ .x = ad(x), .words = ad(words), .scale = .{ .p = ad(scale), .kind = scale_kind }, .bias = .{ .p = ad(bias), .kind = scale_kind }, .out = ad(out), .m = rows, .n = n, .k = k, .bits = bits, .group = group, .fp16 = fp16, .route = .{ .items = ad(items), .members = ad(members), .x_div = x_div } }, count, s);
    }

    /// The 256-thread, one-element-a-thread launch of an ops.hip kernel over `n` elements.
    fn flat(l: *const Launcher, f: Function, n: i64, s: S, a: *Args) Error!void {
        if (n == 0) return;
        try l.go(f, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, a);
    }

    pub fn tf_embed_rows(l: *const Launcher, words: C, scale: C, bias: C, scale_kind: c_int, ids: CI, n: c_int, bits: c_int, group: c_int, k: c_int, out: P, out_kind: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(words));
        a.add(ad(scale));
        a.add(ad(bias));
        a.add(scale_kind);
        a.add(ad(ids));
        a.add(n);
        a.add(bits);
        a.add(group);
        a.add(k);
        a.add(ad(out));
        a.add(out_kind);
        try l.flat(l.op.embed_rows, @as(i64, n) * k, s, &a);
    }

    pub fn tf_cast(l: *const Launcher, src: C, skind: c_int, dst: P, dkind: c_int, n: c_longlong, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(src));
        a.add(skind);
        a.add(ad(dst));
        a.add(dkind);
        a.add(n);
        try l.flat(l.op.cast, n, s, &a);
    }

    pub fn tf_silu_mul(l: *const Launcher, gate: C, up: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(gate));
        a.add(ad(up));
        a.add(ad(out));
        a.add(kind);
        a.add(n);
        try l.flat(l.op.silu_mul, n, s, &a);
    }

    pub fn tf_add(l: *const Launcher, x: C, y: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(ad(y));
        a.add(ad(out));
        a.add(kind);
        a.add(n);
        try l.flat(l.op.add, n, s, &a);
    }

    pub fn tf_attn_gate(l: *const Launcher, att: CF, qg: C, out: P, kind: c_int, len: c_int, heads: c_int, d: c_int, rows_major: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(att));
        a.add(ad(qg));
        a.add(ad(out));
        a.add(kind);
        a.add(len);
        a.add(heads);
        a.add(d);
        a.add(rows_major);
        try l.flat(l.op.attn_gate, @as(i64, len) * heads * d, s, &a);
    }

    pub fn tf_gnorm_silu(l: *const Launcher, y: CF, z: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(y));
        a.add(ad(z));
        a.add(ad(out));
        a.add(kind);
        a.add(n);
        try l.flat(l.op.gnorm_silu, n, s, &a);
    }

    pub fn tf_copy_cols(l: *const Launcher, src: C, stride: c_longlong, offset: c_int, dst: P, kind: c_int, rows: c_int, cols: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(src));
        a.add(stride);
        a.add(offset);
        a.add(ad(dst));
        a.add(kind);
        a.add(rows);
        a.add(cols);
        try l.flat(l.op.copy_cols, @as(i64, rows) * cols, s, &a);
    }

    pub fn tf_rope_prefill(l: *const Launcher, x: C, kind: c_int, out: P, out_kind: c_int, s_head: c_longlong, s_row: c_longlong, len: c_int, heads: c_int, d: c_int, rotary: c_int, pos0: c_int, theta: f32, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(kind);
        a.add(ad(out));
        a.add(out_kind);
        a.add(s_head);
        a.add(s_row);
        a.add(len);
        a.add(heads);
        a.add(d);
        a.add(rotary);
        a.add(pos0);
        a.add(theta);
        try l.flat(l.op.rope_prefill, @as(i64, len) * heads * d, s, &a);
    }

    pub fn tf_kv_write(l: *const Launcher, src: C, cache: P, kind: c_int, len: c_int, kv_heads: c_int, d: c_int, total: c_int, pos0: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(src));
        a.add(ad(cache));
        a.add(kind);
        a.add(len);
        a.add(kv_heads);
        a.add(d);
        a.add(total);
        a.add(pos0);
        try l.flat(l.op.kv_write, @as(i64, len) * kv_heads * d, s, &a);
    }

    pub fn tf_conv_prefill(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, out: F, new_state: F, len: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(kind);
        a.add(ad(weight));
        a.add(ad(state));
        a.add(ad(out));
        a.add(ad(new_state));
        a.add(len);
        a.add(channels);
        a.add(kernel);
        const n = @as(i64, @max(len, kernel - 1)) * channels;
        try l.go(l.op.conv_prefill, dim(cdiv(n, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    }

    pub fn tf_gdn_gate_prefill(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(av));
        a.add(ad(bv));
        a.add(kind);
        a.add(ad(a_log));
        a.add(ad(dt_bias));
        a.add(ad(gate));
        a.add(ad(beta));
        a.add(count);
        a.add(heads);
        try l.flat(l.op.gdn_gate_prefill, @as(i64, count) * heads, s, &a);
    }

    pub fn tf_dense_rows(l: *const Launcher, x: C, kind: c_int, w: CF, out: P, rows: c_int, n: c_int, k: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(x));
        a.add(kind);
        a.add(ad(w));
        a.add(ad(out));
        a.add(rows);
        a.add(n);
        a.add(k);
        try l.go(l.op.dense_rows, dim(n, rows, 1), dim(32, 1, 1), 0, s, &a);
    }

    pub fn tf_moe_route(l: *const Launcher, picks: CI, pairs: c_int, experts: c_int, tile: c_int, members: I, items: I, capacity: c_int, s: S) Error!void {
        var a: Args = .{};
        a.add(ad(picks));
        a.add(pairs);
        a.add(experts);
        a.add(tile);
        a.add(ad(members));
        a.add(ad(items));
        a.add(capacity);
        // every expert's count, start and end, and each pair segment's 16-bit counts, in dynamic shared memory
        const ex: usize = @intCast(experts);
        const segs: usize = if (ex <= 256) 64 else 16;
        if (ex > 1024) return invalid("moe_route");
        try l.go(l.op.moe_route, dim(1, 1, 1), dim(256, 1, 1), @intCast(3 * ex * 4 + segs * ex * 2), s, &a);
    }
};
