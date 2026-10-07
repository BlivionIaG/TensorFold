//! Typed launches of the ROCm kernel library on device addresses: shapes checked here (the C launchers trust them),
//! scratch from the forward's arena, the Python wrappers' choices (splits, fp16 decode output) made the same way.
//! The launches live by job in this folder; `Ops` is the one value the forward calls them through.

const std = @import("std");
const abi = @import("../runtime/abi.zig");
const rocm = @import("../rocm.zig");
const Arena = @import("../runtime/arena.zig").Arena;
const types = @import("types.zig");
const project = @import("project.zig");
const attention = @import("attention.zig");
const recurrence = @import("recurrence.zig");
const norms = @import("norms.zig");
const elementwise = @import("elementwise.zig");
const moe = @import("moe.zig");
const draw = @import("draw.zig");
const tp = @import("tp.zig");

pub const Kind = types.Kind;
pub const Error = types.Error;
pub const Tensor = types.Tensor;
pub const Affine = types.Affine;

pub const Ops = struct {
    lib: *const rocm.Library,
    stream: abi.Stream,
    arena: *Arena,
    /// A prompt's span: every product, the router and the recurrence take prefill's one kernel at any row count, so a
    /// prompt's rows have the same bits however it is cut (fresh, resumed or in steps).
    prefill: bool = false,
    /// A lane round's forward: the decode tiles at any row count (the stream tile over 4-row blocks, the router a wave an
    /// expert, 8-row routed items), so a row's bits do not depend on the rows it shares the round with.
    window: bool = false,

    /// The GPU's activations are bf16 (the matrix-core build); fp16 on RDNA2.
    pub fn wmma(o: Ops) bool {
        return o.lib.caps.act == .bf16;
    }

    /// Whether decode.hip's merged launches are on: Zig launches, and the policy does not switch them to the reference.
    pub fn fused(o: Ops) bool {
        return if (o.lib.zig) |z| z.fuse else false;
    }

    pub const affine = project.affine;
    pub const affineGroup = project.affineGroup;
    pub const affineRoutedAct = project.affineRoutedAct;
    pub const affineRouted = project.affineRouted;
    pub const embedRows = project.embedRows;
    pub const denseRows = project.denseRows;
    pub const attnGate = attention.attnGate;
    pub const ropePrefill = attention.ropePrefill;
    pub const qkRope = attention.qkRope;
    pub const ropeDecode = attention.ropeDecode;
    pub const kvWrite = attention.kvWrite;
    pub const kvWriteAt = attention.kvWriteAt;
    pub const Cache = attention.Cache;
    pub const Paged = attention.Paged;
    pub const pageWrite = attention.pageWrite;
    pub const causalPaged = attention.causalPaged;
    pub const causalPrefill = attention.causalPrefill;
    pub const causalAt = attention.causalAt;
    pub const convPrefill = recurrence.convPrefill;
    pub const convDecode = recurrence.convDecode;
    pub const convRows = recurrence.convRows;
    pub const gdnGatePrefill = recurrence.gdnGatePrefill;
    pub const gdnGate = recurrence.gdnGate;
    pub const gatedDelta = recurrence.gatedDelta;
    pub const rms = norms.rms;
    pub const rms2 = norms.rms2;
    pub const gnormOut = norms.gnormOut;
    pub const addRms = norms.addRms;
    pub const gnormSilu = norms.gnormSilu;
    pub const cast = elementwise.cast;
    pub const siluMul = elementwise.siluMul;
    pub const add = elementwise.add;
    pub const copyCols = elementwise.copyCols;
    pub const moeTail = moe.moeTail;
    pub const moeRouter = moe.moeRouter;
    pub const moeSelect = moe.moeSelect;
    pub const moeRoute = moe.moeRoute;
    pub const moeAct = moe.moeAct;
    pub const moeCombine = moe.moeCombine;
    pub const argmaxRows = draw.argmaxRows;
    pub const tokenProb = draw.tokenProb;
    pub const topkRows = draw.topkRows;
    pub const addWide = tp.addWide;
    pub const moeLocalize = tp.moeLocalize;
    pub const moeForeignItems = tp.moeForeignItems;
    pub const moeZeroForeign = tp.moeZeroForeign;
};

test "dtype numberings of the three kernel families" {
    try std.testing.expectEqual(@as(c_int, 1), @backingInt(Kind.f16));
    try std.testing.expectEqual(@as(c_int, 0), Kind.f16.cache());
    try std.testing.expectEqual(@as(c_int, 2), Kind.f16.table());
    try std.testing.expectEqual(@as(c_int, 1), Kind.bf16.table());
}

test "every launch compiles" {
    std.testing.refAllDecls(Ops);
}
