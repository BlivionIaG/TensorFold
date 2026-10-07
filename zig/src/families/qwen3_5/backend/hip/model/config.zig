//! Qwen3.5 / Qwen3.6 text configuration: the Python ROCm loader's Spec, defaults and checks, and the quantization map.

const std = @import("std");
const hip = @import("hip");

pub const Error = error{
    InvalidConfig,
    MissingField,
    UnsupportedModel,
    UnsupportedQuantization,
    InvalidMoe,
    InvalidRotary,
    InvalidLayerTypes,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// One affine width: MLX groups of `group` weights share a scale and a bias, `bits` per weight.
pub const Width = hip.quant.mlx.Width;

/// The Python `Spec`, plus the shared expert's width and the MTP layer count the config declares.
pub const Spec = struct {
    hidden: usize,
    intermediate: usize,
    n_layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    vocab: usize,
    eps: f64,
    rope_theta: f64,
    rotary_dim: usize,
    full_every: usize,
    bits: u8,
    group: u16,
    experts: usize,
    top_k: usize,
    moe_width: usize,
    shared_width: usize,
    mtp_layers: usize,

    /// Whether layer `index` is full attention (the others are gated delta-net).
    pub fn full(s: Spec, index: usize) bool {
        return (index + 1) % s.full_every == 0;
    }

    pub fn keyWidth(s: Spec) usize {
        return s.key_heads * s.key_dim;
    }

    pub fn valueWidth(s: Spec) usize {
        return s.value_heads * s.value_dim;
    }
};

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    spec: Spec,
    quant: hip.quant.Config,
    /// `tie_word_embeddings`: the output head is the embedding.
    tied: bool,

    pub fn deinit(c: *Config) void {
        c.arena.deinit();
        c.* = undefined;
    }

    /// The config of the checkpoint directory `dir`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// `load()`'s reading of config.json: top level or `text_config`, model_type qwen3_5 or qwen3_5_moe.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const root = try object(try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
        const kind = root.get("model_type") orelse return error.UnsupportedModel;
        if (kind != .string or !(eql(kind.string, "qwen3_5") or eql(kind.string, "qwen3_5_moe"))) return error.UnsupportedModel;
        const text = if (truthy(root.get("text_config"))) try object(root.get("text_config").?) else root;
        const quant = try hip.quant.detect(a, .{ .root = root, .quantization = try pick(root, text, "quantization") });
        const width = try quant.width("");

        const experts = try optionalInt(text, "num_experts", 0);
        const top_k = try optionalInt(text, "num_experts_per_tok", 0);
        const moe_width = try optionalInt(text, "moe_intermediate_size", 0);
        if (experts != 0) {
            if (top_k == 0 or top_k > experts or moe_width == 0) return error.InvalidMoe;
            if (text.get("norm_topk_prob")) |v| if (!truthyValue(v)) return error.InvalidMoe;
        }
        const tied = if (root.get("tie_word_embeddings")) |v| truthyValue(v) else if (text.get("tie_word_embeddings")) |v| truthyValue(v) else true;
        const hidden = try requiredInt(text, "hidden_size");
        const heads = try requiredInt(text, "num_attention_heads");
        const head_dim = if (truthy(text.get("head_dim"))) try toInt(text.get("head_dim").?) else hidden / heads;
        const rope = if (truthy(text.get("rope_parameters"))) try object(text.get("rope_parameters").?) else std.json.ObjectMap.empty;
        const partial = try number(rope.get("partial_rotary_factor") orelse text.get("partial_rotary_factor") orelse .{ .float = 0.25 });
        const product = @as(f64, @floatFromInt(head_dim)) * partial;
        if (!(product >= 0 and product <= @as(f64, @floatFromInt(head_dim)))) return error.InvalidRotary;
        const rotary: usize = @intFromFloat(product);
        const theta = try number(rope.get("rope_theta") orelse if (truthy(text.get("rope_theta"))) text.get("rope_theta").? else .{ .integer = 10_000_000 });
        const spec: Spec = .{
            .hidden = hidden,
            .intermediate = try optionalInt(text, "intermediate_size", 0),
            .n_layers = try requiredInt(text, "num_hidden_layers"),
            .heads = heads,
            .kv_heads = try requiredInt(text, "num_key_value_heads"),
            .head_dim = head_dim,
            .key_heads = try requiredInt(text, "linear_num_key_heads"),
            .value_heads = try requiredInt(text, "linear_num_value_heads"),
            .key_dim = try requiredInt(text, "linear_key_head_dim"),
            .value_dim = try requiredInt(text, "linear_value_head_dim"),
            .conv = try requiredInt(text, "linear_conv_kernel_dim"),
            .vocab = try requiredInt(text, "vocab_size"),
            .eps = try number(text.get("rms_norm_eps") orelse .{ .float = 1e-6 }),
            .rope_theta = theta,
            .rotary_dim = rotary,
            .full_every = try optionalInt(text, "full_attention_interval", 4),
            .bits = width.bits,
            .group = width.group,
            .experts = experts,
            .top_k = top_k,
            .moe_width = moe_width,
            .shared_width = try optionalInt(text, "shared_expert_intermediate_size", 0),
            .mtp_layers = try optionalInt(text, "mtp_num_hidden_layers", 0),
        };
        if (rotary % 2 != 0 or rotary == 0 or rotary > head_dim) return error.InvalidRotary;
        if (spec.full_every == 0 or heads == 0) return error.InvalidConfig;
        if (text.get("layer_types")) |kinds| if (kinds != .null) {
            if (kinds != .array) return error.InvalidLayerTypes;
            for (kinds.array.items, 0..) |item, i| {
                const want = if (spec.full(i)) "full_attention" else "linear_attention";
                if (item != .string or !eql(item.string, want)) return error.InvalidLayerTypes;
            }
        };
        return .{ .arena = arena, .spec = spec, .quant = quant, .tied = tied };
    }
};

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.InvalidConfig;
}

/// Python truthiness of an optional JSON value (`x.get(k) or default`).
fn truthy(v: ?std.json.Value) bool {
    return if (v) |value| truthyValue(value) else false;
}

fn truthyValue(v: std.json.Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .string => |s| s.len > 0,
        .array => |x| x.items.len > 0,
        .object => |o| o.count() > 0,
        else => true,
    };
}

/// Python `int(x)` of a JSON number.
fn toInt(v: std.json.Value) Error!usize {
    return switch (v) {
        .integer => |i| std.math.cast(usize, i) orelse error.InvalidConfig,
        .float => |f| if (f >= 0 and f < 1e15) @intFromFloat(f) else error.InvalidConfig,
        else => error.InvalidConfig,
    };
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.InvalidConfig,
    };
}

fn requiredInt(o: std.json.ObjectMap, name: []const u8) Error!usize {
    return toInt(o.get(name) orelse return error.MissingField);
}

/// `int(o.get(name, default) or 0)`: absent is `default`, null or zero is 0.
fn optionalInt(o: std.json.ObjectMap, name: []const u8, default: usize) Error!usize {
    const v = o.get(name) orelse return default;
    return if (truthyValue(v)) toInt(v) else 0;
}

/// `cfg.get(key) or text.get(key) or {}`.
fn pick(root: std.json.ObjectMap, text: std.json.ObjectMap, key: []const u8) Error!std.json.ObjectMap {
    for ([_]std.json.ObjectMap{ root, text }) |source| if (truthy(source.get(key))) return object(source.get(key).?);
    return std.json.ObjectMap.empty;
}

const sample =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
    \\ "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\   "language_model.model.layers.0.mlp.gate": {"group_size": 32, "bits": 8},
    \\   "vision_tower.x": {"group_size": 64, "bits": 7}},
    \\ "text_config": {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 8,
    \\   "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    \\   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    \\   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320,
    \\   "rms_norm_eps": 1e-06, "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
    \\   "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\     "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    \\   "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
;

test "text_config parses into the Python Spec" {
    var c = try Config.parse(std.testing.allocator, sample);
    defer c.deinit();
    const s = c.spec;
    try std.testing.expectEqual(@as(usize, 4096), s.hidden);
    try std.testing.expectEqual(@as(usize, 12288), s.intermediate);
    try std.testing.expectEqual(@as(usize, 64), s.rotary_dim);
    try std.testing.expectEqual(@as(usize, 256), s.head_dim);
    try std.testing.expectEqual(@as(usize, 2048), s.keyWidth());
    try std.testing.expectEqual(@as(usize, 4096), s.valueWidth());
    try std.testing.expectEqual(@as(f64, 10_000_000), s.rope_theta);
    try std.testing.expectEqual(@as(f64, 1e-6), s.eps);
    try std.testing.expectEqual(@as(u8, 4), s.bits);
    try std.testing.expectEqual(@as(u16, 64), s.group);
    try std.testing.expectEqual(@as(usize, 0), s.experts);
    try std.testing.expectEqual(@as(usize, 1), s.mtp_layers);
    try std.testing.expect(!c.tied);
    try std.testing.expect(s.full(3) and s.full(7) and !s.full(0) and !s.full(4));
}

test "per-tensor widths override the global one, and a bad entry fails only when used" {
    var c = try Config.parse(std.testing.allocator, sample);
    defer c.deinit();
    try std.testing.expectEqual(Width{ .bits = 8, .group = 32 }, try c.quant.width("language_model.model.layers.0.mlp.gate"));
    try std.testing.expectEqual(Width{ .bits = 4, .group = 64 }, try c.quant.width("language_model.model.layers.1.mlp.gate"));
    try std.testing.expectError(error.UnsupportedQuantization, c.quant.width("vision_tower.x"));
}

test "a top-level config, the partial rotary default and the MoE fields" {
    const text =
        \\{"model_type": "qwen3_5_moe", "quantization": {"group_size": 64, "bits": 8},
        \\ "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 2,
        \\ "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1000, "rope_theta": 5000000,
        \\ "partial_rotary_factor": 0.5, "num_experts": 256, "num_experts_per_tok": 8,
        \\ "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
    ;
    var c = try Config.parse(std.testing.allocator, text);
    defer c.deinit();
    try std.testing.expect(c.tied);
    try std.testing.expectEqual(@as(usize, 128), c.spec.head_dim);
    try std.testing.expectEqual(@as(usize, 64), c.spec.rotary_dim);
    try std.testing.expectEqual(@as(f64, 5_000_000), c.spec.rope_theta);
    try std.testing.expectEqual(@as(usize, 256), c.spec.experts);
    try std.testing.expectEqual(@as(usize, 8), c.spec.top_k);
    try std.testing.expectEqual(@as(usize, 512), c.spec.moe_width);
    try std.testing.expectEqual(@as(usize, 512), c.spec.shared_width);
    try std.testing.expectEqual(@as(usize, 0), c.spec.intermediate);
    try std.testing.expectEqual(@as(usize, 4), c.spec.full_every);
}

test "the Python loader's refusals" {
    const a = std.testing.allocator;
    const wrong = try std.mem.replaceOwned(u8, a, sample, "\"qwen3_5\"", "\"llama\"");
    defer a.free(wrong);
    try std.testing.expectError(error.UnsupportedModel, Config.parse(a, wrong));
    const types = try std.mem.replaceOwned(u8, a, sample, "\"full_attention\"],", "\"linear_attention\"],");
    defer a.free(types);
    try std.testing.expectError(error.InvalidLayerTypes, Config.parse(a, types));
    const bits = try std.mem.replaceOwned(u8, a, sample, "\"bits\": 4,", "\"bits\": 7,");
    defer a.free(bits);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, bits));
    const gptq = try std.mem.replaceOwned(u8, a, sample, "\"mode\": \"affine\"", "\"mode\": \"gptq\"");
    defer a.free(gptq);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, gptq));
    const moe = try std.mem.replaceOwned(u8, a, sample, "\"vocab_size\"", "\"num_experts\": 8, \"num_experts_per_tok\": 9, \"vocab_size\"");
    defer a.free(moe);
    try std.testing.expectError(error.InvalidMoe, Config.parse(a, moe));
}
