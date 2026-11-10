//! The device weights the forward reads: projections, fp32 norms, conv taps, DeltaNet tables, MLPs and the MTP head.

const std = @import("std");
const hip = @import("hip");

pub const Projection = hip.quant.Projection;
pub const Kind = hip.ops.Kind;

/// The text model's dimensions (Python's Spec), as the checkpoint's config gives them.
pub const Spec = @import("../../../weights/config.zig").Spec;

/// The conv's channels: q, k and v of the linear attention.
pub fn convChannels(s: Spec) usize {
    return 2 * s.keyWidth() + s.valueWidth();
}

/// A layer's experts and router rows (hip/moe.zig).
pub const Experts = hip.moe.Experts;
pub const Routed = hip.moe.Routed;

pub const Mlp = union(enum) {
    dense: struct { gate: Projection, up: Projection, down: Projection, gate_up: ?Projection = null },
    moe: Routed,
};

pub const Full = struct {
    input_norm: u64,
    post_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: u64,
    k_norm: u64,
    mlp: Mlp,
};

pub const Linear = struct {
    input_norm: u64,
    post_norm: u64,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    conv: u64,
    a_log: u64,
    dt_bias: u64,
    gnorm: u64,
    mlp: Mlp,
};

pub const Layer = union(enum) { full: Full, linear: Linear };

/// The MTP head: the Qwen3 shape (gated attention, an MLP or experts) with its own or the model's output head.
pub const Mtp = struct {
    fc_e_norm: u64,
    fc_h_norm: u64,
    fc_e: Projection,
    fc_h: Projection,
    q_norm: u64,
    k_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    final_norm: u64,
    head: ?Projection,
    input_norm: ?u64,
    post_norm: ?u64,
    mlp: ?Mlp,
    gated: bool,
};

/// normalize_qk's constant norm weights (key_dim ** -0.5 squared for q, once for k) and its eps.
pub const QkNorm = struct { q_weight: u64, k_weight: u64, eps: f32 };

pub const ImageRun = struct {
    at: usize,
    rows: usize,
    features: u64,
    pub const Clip = struct { first: usize, rows: usize, features: u64 };
    pub fn clip(run: ImageRun, first: usize, rows: usize, hidden: usize) ?Clip {
        const lo = @max(run.at, first);
        const hi = @min(run.at + run.rows, first + rows);
        if (lo >= hi) return null;
        return .{ .first = lo - first, .rows = hi - lo, .features = run.features + (lo - run.at) * hidden * 2 };
    }
};
pub const ImageFrame = struct { positions: u64, runs: []const ImageRun };

/// The layers `[first, last)` of the text model that one stage runs; `whole` is every layer (the unsplit model).
/// The stage holding layer 0 embeds, the stage holding the last layer applies the final norm, the head and the argmax.
pub const Range = struct {
    first: usize = 0,
    /// Past the last layer; at least `n_layers` means through the last.
    last: usize = std.math.maxInt(usize),

    pub const whole: Range = .{};

    pub const Error = error{ BadLayerRange, RangeWithVision, RangeWithTensorParallel, RangeWithDraft, RangeWithBf16Checkpoint };

    /// `r` over a model of `n_layers`: non-empty and inside it.
    pub fn check(r: Range, n_layers: usize) Error!void {
        if (r.first >= n_layers) return error.BadLayerRange;
        if (r.last != std.math.maxInt(usize) and (r.last > n_layers or r.last <= r.first)) return error.BadLayerRange;
    }

    /// What a stage's engine would also need, for `admit`.
    pub const Needs = struct { world: usize = 1, vision: bool = false, bf16: bool = false };

    /// `r` as a stage of a model of `n_layers`: inside it, and refused by name where a range cannot run (a vision
    /// tower, tensor parallelism, a bf16 checkpoint's rocBLAS products). The whole model is always admitted.
    pub fn admit(r: Range, n_layers: usize, needs: Needs) Error!void {
        try r.check(n_layers);
        if (r.isWhole(n_layers)) return;
        if (needs.vision) return error.RangeWithVision;
        if (needs.world > 1) return error.RangeWithTensorParallel;
        if (needs.bf16) return error.RangeWithBf16Checkpoint;
    }

    /// "first:last" (half-open), "first:" through the last layer, or "all".
    pub fn parse(text: []const u8) Error!Range {
        if (std.mem.eql(u8, text, "all")) return .whole;
        const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadLayerRange;
        const first = std.fmt.parseInt(usize, text[0..colon], 10) catch return error.BadLayerRange;
        const tail = text[colon + 1 ..];
        const last = if (tail.len == 0) std.math.maxInt(usize) else std.fmt.parseInt(usize, tail, 10) catch return error.BadLayerRange;
        if (last <= first) return error.BadLayerRange;
        return .{ .first = first, .last = last };
    }

    /// `r` with `last` clamped to `n_layers`.
    pub fn within(r: Range, n_layers: usize) Range {
        return .{ .first = r.first, .last = @min(r.last, n_layers) };
    }

    pub fn owns(r: Range, layer: usize) bool {
        return layer >= r.first and layer < r.last;
    }

    /// Layer 0 is in the range: the stage embeds token ids.
    pub fn embeds(r: Range) bool {
        return r.first == 0;
    }

    /// The model's last layer is in the range: the stage applies the final norm and the head.
    pub fn heads(r: Range, n_layers: usize) bool {
        return r.last >= n_layers;
    }

    /// Every layer: the unsplit model.
    pub fn isWhole(r: Range, n_layers: usize) bool {
        return r.first == 0 and r.last >= n_layers;
    }

    /// Layers the range runs of a model of `n_layers`.
    pub fn count(r: Range, n_layers: usize) usize {
        return @min(r.last, n_layers) - r.first;
    }
};

pub const Model = struct {
    /// Torch's interleaved M-RoPE operator, engine-owned: HF BF16 text, and image prompts' rows.
    rope: ?*const @import("qwen_hip_vision").Kernels = null,
    /// Every text row rotates with `rope` (an HF BF16 checkpoint), with or without --vision.
    torch_text: bool = false,
    /// [p, p, p] for each position: a text-only prompt's rows under `torch_text`.
    positions: u64 = 0,
    frame: ?ImageFrame = null,
    /// What this rank sees: under tp its heads and value heads.
    spec: Spec,
    /// The tensor-parallel communicator; null on one rank.
    tp: ?hip.collective.Comm = null,
    act: Kind,
    /// The layers this model runs (`layers[0]` is layer `range.first`); the whole model unless a stage's range says less.
    range: Range = .whole,
    /// Null when the range does not start at layer 0.
    embed: ?Projection,
    /// The layers of `range` only: `layer(i)` takes a global index.
    layers: []const Layer,
    /// Zero when the range does not end at the last layer.
    final_norm: u64,
    /// Null when the range does not end at the last layer.
    head: ?Projection,
    qk: QkNorm,
    mtp: ?Mtp = null,
    /// A drafter's tapped layers: after each, every row's residual lands in its stream's tap ring.
    taps: Taps = .{},

    /// Layer `index` (a global index inside the range).
    pub fn layer(m: *const Model, index: usize) Layer {
        return m.layers[index - m.range.first];
    }

    /// The range's end: through the last layer of the model.
    pub fn headed(m: *const Model) bool {
        return m.range.heads(m.spec.n_layers);
    }
};

/// The layers a drafter taps and the rows a stream's ring keeps (`Caches.tap`, row = position % rows).
pub const Taps = struct {
    layers: [8]u32 = undefined,
    n: usize = 0,
    rows: usize = 0,

    /// The tap index of layer `layer`, or null when it is not tapped.
    pub fn of(t: *const Taps, layer: usize) ?usize {
        for (t.layers[0..t.n], 0..) |l, i| if (l == layer) return i;
        return null;
    }

    /// Bytes of one stream's ring: `rows` rows of every tapped layer's `hidden` 16-bit values.
    pub fn ringBytes(t: *const Taps, hidden: usize) usize {
        return t.rows * t.n * hidden * 2;
    }
};

/// q and k's normalize_qk weights and eps exactly as Python makes them: double arithmetic, then fp32.
pub fn qkConstants(key_dim: usize, eps: f64) struct { q: f32, k: f32, eps: f32 } {
    const inv = std.math.pow(f64, @floatFromInt(key_dim), -0.5);
    return .{ .q = @floatCast(inv * inv), .k = @floatCast(inv), .eps = @floatCast(eps * inv * inv) };
}

test "a stage is refused with a vision tower, tensor parallelism or a bf16 checkpoint" {
    const half: Range = .{ .first = 0, .last = 16 };
    try half.admit(32, .{});
    try std.testing.expectError(error.RangeWithVision, half.admit(32, .{ .vision = true }));
    try std.testing.expectError(error.RangeWithTensorParallel, half.admit(32, .{ .world = 2 }));
    try std.testing.expectError(error.RangeWithBf16Checkpoint, half.admit(32, .{ .bf16 = true }));
    try std.testing.expectError(error.BadLayerRange, (Range{ .first = 16, .last = 40 }).admit(32, .{}));
    // the whole model keeps every mode it had
    try Range.whole.admit(32, .{ .world = 4, .vision = true, .bf16 = true });
    try (Range{ .first = 0, .last = 32 }).admit(32, .{ .world = 2 });
}

test "layer ranges parse as first:last, first: and all" {
    try std.testing.expectEqual(Range{ .first = 0, .last = 16 }, try Range.parse("0:16"));
    try std.testing.expectEqual(Range{ .first = 16, .last = std.math.maxInt(usize) }, try Range.parse("16:"));
    try std.testing.expectEqual(Range.whole, try Range.parse("all"));
    for ([_][]const u8{ "", "7", "4:4", "9:3", "a:3", "3:b", ":5", "-1:4" }) |bad| try std.testing.expectError(error.BadLayerRange, Range.parse(bad));
}

test "layer ranges validate against the model's layer count" {
    const r: Range = .{ .first = 8, .last = 20 };
    try r.check(32);
    try std.testing.expectError(error.BadLayerRange, r.check(16));
    try std.testing.expectError(error.BadLayerRange, (Range{ .first = 4, .last = 4 }).check(32));
    try std.testing.expectError(error.BadLayerRange, (Range{ .first = 32 }).check(32));
    try Range.whole.check(32);
    try std.testing.expect(Range.whole.isWhole(32) and !r.isWhole(32));
    try std.testing.expect(r.owns(8) and r.owns(19) and !r.owns(20) and !r.owns(7));
    try std.testing.expect((Range{ .first = 0, .last = 4 }).embeds() and !r.embeds());
    try std.testing.expect((Range{ .first = 20 }).heads(32) and !r.heads(32));
    try std.testing.expectEqual(@as(usize, 12), r.count(32));
    try std.testing.expectEqual(@as(usize, 32), Range.whole.count(32));
}

test "normalize_qk constants for a 128-wide key" {
    const c = qkConstants(128, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0078125), c.q);
    try std.testing.expect(@abs(c.k - 0.08838834764) < 1e-8);
}

test "Qwen HIP image features splice correctly across prompt chunks" {
    const run: ImageRun = .{ .at = 100, .rows = 300, .features = 0x10000 };
    try std.testing.expectEqual(ImageRun.Clip{ .first = 36, .rows = 28, .features = 0x10000 }, run.clip(64, 64, 1024).?);
    try std.testing.expectEqual(ImageRun.Clip{ .first = 0, .rows = 128, .features = 0x10000 + 28 * 2048 }, run.clip(128, 128, 1024).?);
    try std.testing.expectEqual(ImageRun.Clip{ .first = 0, .rows = 16, .features = 0x10000 + 284 * 2048 }, run.clip(384, 128, 1024).?);
    try std.testing.expect(run.clip(0, 100, 1024) == null);
    try std.testing.expect(run.clip(400, 100, 1024) == null);
}
