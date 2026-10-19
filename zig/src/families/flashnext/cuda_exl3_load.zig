//! EXL3 weights for the Flash Next CUDA family's own loaders (cuda_projection_load.zig, cuda_prompt_load.zig):
//!  - `Proj`: a projection made of column parts, each an EXL3 matrix in its stored layout (cuda.exl3.Weight, no
//!    strips reorder) or a dense BF16 lane weight (the F16 in_proj_a/b, widened exactly by the Store), written into
//!    one row of `n` columns: what Projection.exl3 points at and Pass.lane() runs (cuda_exl3_pass.zig);
//!  - `Experts`: one layer's 512 routed experts and the shared one as id 512 in the same 513-entry pointer and width
//!    tables (cuda.exl3_experts.Layer), suh/svh stacked [513, ·], resident;
//!  - `Ops`: the kernels and scratch every EXL3 projection and expert call of one engine shares (one exl3.Dense, one
//!    exl3_experts.Experts), counted so the last owner unloads them.
//! Widths come from each matrix's own trellis (16 * bits int16 words a tile), never from config.json's `bits`.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("cuda_prompt_wt.zig");
const projection = @import("cuda_projection.zig");

const Tensor = core.safetensors.Tensor;
const Store = wt.Store;

/// format.py's mul1 marker value (core.exl3_load.markers).
const mul1_magic: u32 = blk: {
    for (core.exl3_load.markers) |m| if (m.codebook == .mul1) break :blk m.value;
    unreachable;
};

/// The EXL3 kernels and scratch one engine's calls share, on the engine's one compute stream.
pub const Ops = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    refs: usize = 1,
    dense: ?cuda.exl3.Dense = null,
    experts: ?cuda.exl3_experts.Experts = null,
    /// A part's rows before its columns land in the stitched row (bf16 [rows, part n]).
    tmp: ?cuda.DeviceBuffer = null,
    /// The routed experts' scratch for `moe_rows` rows of 11 slots.
    moe: ?cuda.DeviceBuffer = null,
    moe_rows: usize = 0,
    /// fp16 rows split into bf16 hi and lo [2, rows, k], and their fp32 products [2, rows, n] (an fp16-input Proj).
    halves: ?cuda.DeviceBuffer = null,
    sums: ?cuda.DeviceBuffer = null,
    glue: ?cuda.Module = null,
    split_fn: cuda.Function = undefined,
    /// Expert parallel's masks, sums and kernels (cuda_exl3_tp.zig), made on first use.
    tp: ?*@import("cuda_exl3_tp.zig").Scratch = null,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Ops {
        const o = try gpa.create(Ops);
        o.* = .{ .gpa = gpa, .d = d };
        return o;
    }

    pub fn retain(o: *Ops) *Ops {
        o.refs += 1;
        return o;
    }

    pub fn release(o: *Ops) void {
        o.refs -= 1;
        if (o.refs != 0) return;
        if (o.dense) |*x| x.unload();
        if (o.experts) |*x| x.unload();
        if (o.tmp) |*b| b.free();
        if (o.moe) |*b| b.free();
        if (o.halves) |*b| b.free();
        if (o.sums) |*b| b.free();
        if (o.glue) |*m| m.unload();
        if (o.tp) |t| t.destroy();
        o.gpa.destroy(o);
    }

    /// The dense EXL3 linear, loaded on first use (no GPU work at checkpoint load).
    pub fn linear(o: *Ops) !*cuda.exl3.Dense {
        if (o.dense == null) o.dense = try cuda.exl3.Dense.load(o.d);
        return &o.dense.?;
    }

    /// fn_exl3_split_f16 (flashnext_exl3.cu), loaded on first use.
    pub fn split(o: *Ops) !cuda.Function {
        if (o.glue == null) {
            if (!cuda.kernels.available) return error.BuiltWithoutKernels;
            var m = try cuda.Module.load(o.d, cuda.kernels.flashnext_exl3);
            errdefer m.unload();
            o.split_fn = try m.function("fn_exl3_split_f16");
            o.glue = m;
        }
        return o.split_fn;
    }

    pub fn grouped(o: *Ops) !*const cuda.exl3_experts.Experts {
        if (o.experts == null) o.experts = try cuda.exl3_experts.Experts.load(o.d);
        return &o.experts.?;
    }

    /// At least `bytes` of part scratch; a growth waits for `s` first (an older call may still read the old buffer).
    pub fn scratch(o: *Ops, s: cuda.Stream, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.*) |b| if (b.len >= bytes) return b.ptr;
        try s.synchronize();
        if (slot.*) |*b| b.free();
        slot.* = null;
        slot.* = try cuda.DeviceBuffer.alloc(o.d, bytes);
        return slot.*.?.ptr;
    }
};

/// The store's shared Ops, made on first use; every holder retains it.
pub fn opsOf(store: *Store, d: *const cuda.Driver) !*Ops {
    if (store.exl3_ops == null) store.exl3_ops = try Ops.create(store.gpa, d);
    return store.exl3_ops.?.retain();
}

/// One EXL3 matrix read from the store: words as stored, its width and shape.
pub const Mat = struct {
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    k: usize,
    n: usize,
    halves: u32,
};

/// Whether `prefix` names an EXL3 matrix (its trellis is indexed).
pub fn has(store: *const Store, prefix: []const u8) bool {
    var buf: [256]u8 = undefined;
    return store.has(std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}) catch return false);
}

/// `prefix`'s trellis, scales and mul1 marker, checked: int16 [K/16, N/16, 16 * bits], fp16 suh [K] and svh [N].
pub fn mat(store: *Store, prefix: []const u8) !Mat {
    var buf: [256]u8 = undefined;
    const t = try store.hold(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 8 != 0) return error.UnexpectedTensor;
    const halves = t.dim(2) / 8;
    // the mul1 half steps are odd halves; 2..16 halves (1 to 8 bits) are what the kernels instantiate
    if (halves < 2 or halves > 16) return error.UnsupportedExl3Width;
    const k = t.dim(0) * 16;
    const n = t.dim(1) * 16;
    if (k % 128 != 0 or n % 128 != 0) return error.UnexpectedTensor;
    const marker = store.hold(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix})) catch return error.NotExl3Mul1;
    if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.NotExl3Mul1;
    const suh = try store.hold(try std.fmt.bufPrint(&buf, "{s}.suh", .{prefix}));
    const svh = try store.hold(try std.fmt.bufPrint(&buf, "{s}.svh", .{prefix}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedTensor;
    if (store.has(try std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}))) return error.UnexpectedTensor; // none in the text model
    return .{ .trellis = t.bytes, .suh = suh.bytes, .svh = svh.bytes, .k = k, .n = n, .halves = @intCast(halves) };
}

/// One column range of a projection's output row.
pub const Part = struct {
    col: usize,
    n: usize,
    kind: union(enum) { exl3: cuda.exl3.Weight, native: cuda.qmmf.Weight },
};

/// A projection's parts and the device buffers they own.
pub const Proj = struct {
    gpa: std.mem.Allocator,
    ops: *Ops,
    k: usize,
    n: usize,
    parts: []Part,
    /// The rows the pass hands in: bf16, or fp16 bit patterns (an EXL3 pack's n-gram rows, cuda_exl3_ple.zig).
    x: enum { bf16, f16 } = .bf16,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    natives: std.ArrayList(projection.Projection) = .empty,

    pub fn destroy(p: *Proj) void {
        for (p.bufs.items) |*b| b.free();
        p.bufs.deinit(p.gpa);
        for (p.natives.items) |*q| q.free();
        p.natives.deinit(p.gpa);
        p.gpa.free(p.parts);
        p.ops.release();
        p.gpa.destroy(p);
    }

    fn up(p: *Proj, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try p.bufs.ensureUnusedCapacity(p.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        p.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }

    /// An EXL3 matrix as the stored-layout weight (trellis [K/16][N/16][W]: strides(k, n, halves, false)).
    fn exl3Part(p: *Proj, d: *const cuda.Driver, m: Mat) !cuda.exl3.Weight {
        const st = cuda.exl3.strides(m.k, m.n, m.halves, false);
        return .{ .trellis = try p.up(d, m.trellis), .suh = try p.up(d, m.suh), .svh = try p.up(d, m.svh), .k = @intCast(m.k), .n = @intCast(m.n), .halves = m.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    }
};

/// A Projection whose rows come from `parts` (EXL3 matrices or dense BF16 rows), `n` columns of `k` inputs.
fn wrap(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !projection.Projection {
    const p = try gpa.create(Proj);
    p.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .k = k, .n = n, .parts = &.{} };
    errdefer p.destroy();
    var parts: std.ArrayList(Part) = .empty;
    errdefer parts.deinit(gpa);
    var buf: [256]u8 = undefined;
    var col: usize = 0;
    var i: usize = 0;
    while (i < names.len) {
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, names[i] });
        if (has(store, name)) {
            const m = try mat(store, name);
            if (m.k != k) return error.UnexpectedTensor;
            try parts.append(gpa, .{ .col = col, .n = m.n, .kind = .{ .exl3 = try p.exl3Part(d, m) } });
            col += m.n;
            i += 1;
            continue;
        }
        // a run of dense BF16 parts (after the Store's exact widening) packs as one lane weight
        var rows: usize = 0;
        var j = i;
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        while (j < names.len) : (j += 1) {
            var nb: [256]u8 = undefined;
            const dense_name = try std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, names[j] });
            if (has(store, dense_name)) break;
            const t = try store.tensor(try std.fmt.bufPrint(&nb, "{s}{s}.weight", .{ prefix, names[j] }));
            if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != k or t.bytes.len != t.dim(0) * k * 2) return error.UnexpectedTensor;
            try bytes.appendSlice(gpa, t.bytes);
            rows += t.dim(0);
        }
        try p.natives.ensureUnusedCapacity(gpa, 1);
        const q = try projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, k, 1, 1, 1 }, .bytes = bytes.items });
        p.natives.appendAssumeCapacity(q);
        try parts.append(gpa, .{ .col = col, .n = rows, .kind = .{ .native = q.native.? } });
        col += rows;
        i = j;
    }
    if (col != n) return error.UnexpectedTensor;
    p.parts = try parts.toOwnedSlice(gpa);
    var out = projection.Projection.empty(d);
    out.n = n;
    out.k = k;
    out.exl3 = p;
    return out;
}

/// cuda_projection_load.load's EXL3 branch: `prefix` one EXL3 matrix of n outputs and k inputs.
pub fn load(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, n: usize, k: usize) !projection.Projection {
    return wrap(gpa, d, store, prefix, &.{""}, n, k);
}

/// Whether any of `prefix`'s parts is an EXL3 matrix (cuda_projection_load.loadStitched's branch).
pub fn hasAny(store: *const Store, prefix: []const u8, names: []const []const u8) bool {
    var buf: [256]u8 = undefined;
    for (names) |name| if (has(store, std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }) catch return false)) return true;
    return false;
}

/// cuda_projection_load.loadStitched's EXL3 branch: parts side by side in one output row, EXL3 or dense BF16.
pub fn stitched(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !projection.Projection {
    return wrap(gpa, d, store, prefix, names, n, k);
}

/// cuda_prompt_load's PLE branch: an EXL3 pack's n-gram rows reach key_proj and value_proj as fp16 bit patterns
/// (cuda_exl3_ple.zig's default, as the reference multiplies fp16 rows by the F16 projections), so both dense
/// projections take fp16 rows: split exactly into two bf16 rows, two bf16 products summed in fp32
/// (cuda_exl3_pass.zig). TF_FLASHNEXT_PLE_EMB=bf16 keeps the bf16 lanes (the rows then come rounded to bf16).
pub fn pleRows(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, key: *projection.Projection, value: *projection.Projection) !void {
    if (store.exl3_ops == null) return; // not an EXL3 pack: the n-gram rows are bf16
    if (@import("cuda_exl3_ple.zig").optionsFromEnv().emb != .f16) return;
    for ([_]*projection.Projection{ key, value }) |q| {
        const native = q.native orelse return error.UnexpectedTensor;
        const p = try gpa.create(Proj);
        p.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .k = q.k, .n = q.n, .parts = &.{}, .x = .f16 };
        errdefer p.destroy();
        p.parts = try gpa.alloc(Part, 1);
        p.parts[0] = .{ .col = 0, .n = q.n, .kind = .{ .native = native } };
        try p.natives.append(gpa, q.*);
        var out = projection.Projection.empty(d);
        out.n = q.n;
        out.k = q.k;
        out.exl3 = p;
        q.* = out;
    }
}

/// One layer's routed experts and the shared expert (id `wt.experts_n`) for exl3_experts.Experts.
pub const Experts = struct {
    gpa: std.mem.Allocator,
    ops: *Ops,
    layer: cuda.exl3_experts.Layer,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Each expert's and projection's trellis words on the device, for a pool to rewrite (gate, up, down; 513 each).
    trellis: [3][]u64 = .{ &.{}, &.{}, &.{} },
    halves: [3][]i32 = .{ &.{}, &.{}, &.{} },
    /// The ids this rank holds under expert parallel (cuda_exl3_tp.zig); null: one rank holds all 513.
    tp: ?@import("cuda_exl3_tp.zig").Rank = null,

    pub fn destroy(x: *Experts) void {
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
        for (x.trellis) |t| x.gpa.free(t);
        for (x.halves) |h| x.gpa.free(h);
        x.ops.release();
        x.gpa.destroy(x);
    }

    fn up(x: *Experts, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try x.bufs.ensureUnusedCapacity(x.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        x.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }
};

/// Whether `mlp` (".mlp" of a layer) holds EXL3 experts.
pub fn hasExperts(store: *const Store, mlp: []const u8) bool {
    var buf: [256]u8 = undefined;
    return store.has(std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj.trellis", .{mlp}) catch return false);
}

/// Every routed expert of `mlp` resident, the shared expert as id 512 (the same 2560 x 640 x 2560 shape).
pub fn experts(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, mlp: []const u8, world: usize, rank: usize) !*Experts {
    if (world != 1 or rank != 0) return @import("cuda_exl3_tp.zig").experts(gpa, d, store, mlp, world, rank); // plan step 10
    const count = wt.experts_n + 1;
    const x = try gpa.create(Experts);
    x.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .layer = undefined };
    errdefer x.destroy();
    for (&x.trellis, &x.halves) |*t, *h| {
        t.* = try gpa.alloc(u64, count);
        @memset(t.*, 0);
        h.* = try gpa.alloc(i32, count);
        @memset(h.*, 0);
    }
    const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    const ks = [3]usize{ wt.dims, wt.dims, wt.moe_w };
    const ns = [3]usize{ wt.moe_w, wt.moe_w, wt.dims };
    var scales: [6]std.ArrayList(u8) = @splat(.empty);
    defer for (&scales) |*l| l.deinit(gpa);
    var buf: [256]u8 = undefined;
    for (0..count) |e| {
        for (projs, 0..) |proj, j| {
            const name = if (e == wt.experts_n)
                try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ mlp, proj })
            else
                try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ mlp, e, proj });
            const m = try mat(store, name);
            if (m.k != ks[j] or m.n != ns[j]) return error.UnexpectedTensor;
            x.trellis[j][e] = try x.up(d, m.trellis);
            x.halves[j][e] = @intCast(m.halves);
            try scales[2 * j].appendSlice(gpa, m.suh);
            try scales[2 * j + 1].appendSlice(gpa, m.svh);
        }
        // each expert's mapped pages are read once: release them so the 513 experts never pin a shard's worth
        store.release();
    }
    x.layer = .{
        .gate = try x.up(d, std.mem.sliceAsBytes(x.trellis[0])),
        .up = try x.up(d, std.mem.sliceAsBytes(x.trellis[1])),
        .down = try x.up(d, std.mem.sliceAsBytes(x.trellis[2])),
        .gate_k2 = try x.up(d, std.mem.sliceAsBytes(x.halves[0])),
        .up_k2 = try x.up(d, std.mem.sliceAsBytes(x.halves[1])),
        .down_k2 = try x.up(d, std.mem.sliceAsBytes(x.halves[2])),
        .suh_g = try x.up(d, scales[0].items),
        .svh_g = try x.up(d, scales[1].items),
        .suh_u = try x.up(d, scales[2].items),
        .svh_u = try x.up(d, scales[3].items),
        .suh_d = try x.up(d, scales[4].items),
        .svh_d = try x.up(d, scales[5].items),
        .experts = count,
        .dims = wt.dims,
        .width = wt.moe_w,
        .codebook = .mul1,
        // fp32 SwiGLU between the grouped GEMMs (GLM's mode 0 rounds to bf16 there)
        .act_mode = 1,
    };
    return x;
}

test "exl3: the mul1 marker value is format.py's" {
    try std.testing.expectEqual(@as(u32, 0x83DCD12D), mul1_magic);
}
