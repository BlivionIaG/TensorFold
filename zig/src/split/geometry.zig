//! Model geometry for the planner from a config.json: per-layer bytes, KV and state, over the MLX affine layout.
const std = @import("std");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;

pub const MiB: u64 = 1 << 20;
pub const GiB: u64 = 1 << 30;

/// Decode workspace beside the weights, in bytes (assumed, not yet verified on a card): activations, logits, sampling.
pub const scratch_moe: u64 = 400 * MiB;
/// The same for a dense model (assumed, not yet verified; the 4B recipe's workspace is smaller than the A3B's).
pub const scratch_dense: u64 = 300 * MiB;
/// Kernel launches per token and layer on a launch-bound device: dense Qwen3.5-4B (measured: 393 a token on the card).
pub const dispatches_dense: u64 = 12;
/// The same for a MoE layer: router, top-k, gather, three expert projections, combine, shared expert (assumed).
pub const dispatches_moe: u64 = 22;
/// KV is kept in fp16 (the spec's 20,480 B a token for the A3B).
const kv_bytes: u64 = 2;
/// An unquantized checkpoint stores 2-byte weights (bf16); a norm weight is always 2 bytes.
const dense_bytes: u64 = 2;
/// DeltaNet recurrent state is fp32; its convolution state is kept in the activation type (2 bytes).
const delta_state_bytes: u64 = 4;
/// The gated full-attention q projection emits query and gate, so twice the head width.
const q_gate_factor: u64 = 2;

pub const Error = error{
    InvalidJson,
    /// A required config field is absent; `Diag.field` names it.
    MissingField,
    /// A field has the wrong JSON type, is out of range or inconsistent; `Diag.field` names it.
    BadField,
    /// `model_type` is not `qwen3_5` or `qwen3_5_moe`.
    UnsupportedModel,
} || Allocator.Error;

/// Names the field behind a `MissingField` or `BadField` error (a string literal, valid for the program's life).
pub const Diag = struct { field: []const u8 = "" };

pub const Layer = struct {
    /// Weights that must be resident somewhere: mixer, shared expert, router, norms (and the MLP of a dense model).
    resident: u64,
    /// Mixer, shared-expert/MLP and router bytes read per token.
    dense: u64,
    /// Routed-expert bytes resident (0 for dense models).
    experts: u64,
    /// Routed-expert bytes read per decode token (top-k experts).
    experts_read: u64,
    /// KV bytes per context token, read per token and resident (0 for a DeltaNet layer).
    kv_per_tok: u64,
    /// Recurrent state bytes, resident and read plus written per token (0 for an attention layer).
    state: u64,
    /// Kernel launches per token on a launch-bound device.
    dispatches: u64,
};

pub const Model = struct {
    name: []const u8,
    layers: []Layer,
    embed: u64,
    /// Zero when the head is tied to the embedding: the head then reads the embedding table (see `headRead`).
    head: u64,
    scratch: u64,
    hidden: u32,

    /// Checkpoint bytes that must be resident: every layer, the embedding and the head.
    pub fn weights(m: Model) u64 {
        var sum: u64 = m.embed + m.head;
        for (m.layers) |l| sum += l.resident + l.experts;
        return sum;
    }

    /// Bytes the head reads per token (the embedding table when tied).
    pub fn headRead(m: Model) u64 {
        return if (m.head != 0) m.head else m.embed;
    }

    /// True when any layer holds routed experts.
    pub fn hasExperts(m: Model) bool {
        for (m.layers) |l| if (l.experts > 0) return true;
        return false;
    }

    pub fn deinit(m: Model, gpa: Allocator) void {
        gpa.free(m.layers);
        gpa.free(m.name);
    }
};

/// MLX affine quantization of one matrix class. `bits` 0 means unquantized (2-byte weights).
const Quant = struct {
    bits: u32 = 0,
    group: u32 = 64,

    /// Bytes of an n x k matrix: packed codes plus a bf16 scale and bf16 bias per group.
    fn bytes(q: Quant, n: u64, k: u64) u64 {
        if (q.bits == 0) return n * k * dense_bytes;
        return n * k * q.bits / 8 + n * (k / q.group) * 4;
    }
};

const Reader = struct {
    obj: ObjectMap,
    diag: *Diag,

    fn get(r: Reader, comptime name: []const u8) Error!Value {
        return r.obj.get(name) orelse {
            r.diag.field = name;
            return error.MissingField;
        };
    }

    fn uint(r: Reader, comptime name: []const u8) Error!u64 {
        return asUint(try r.get(name), r.diag, name);
    }

    fn optUint(r: Reader, comptime name: []const u8) Error!?u64 {
        const v = r.obj.get(name) orelse return null;
        if (v == .null) return null;
        return try asUint(v, r.diag, name);
    }
};

fn asUint(v: Value, diag: *Diag, comptime name: []const u8) Error!u64 {
    if (v == .integer and v.integer >= 0) return @intCast(v.integer);
    diag.field = name;
    return error.BadField;
}

fn quantOf(v: Value, diag: *Diag, base: Quant) Error!Quant {
    if (v != .object) {
        diag.field = "quantization";
        return error.BadField;
    }
    var q = base;
    if (v.object.get("bits")) |b| q.bits = @intCast(try asUint(b, diag, "quantization.bits"));
    if (v.object.get("group_size")) |g| q.group = @intCast(try asUint(g, diag, "quantization.group_size"));
    if (q.group == 0 or q.bits > 8) {
        diag.field = "quantization";
        return error.BadField;
    }
    return q;
}

/// The override of layer `i` ending in `suffix`, else the first one with that suffix on any layer, else `base`.
fn override(qobj: ?ObjectMap, i: usize, comptime suffix: []const u8, diag: *Diag, base: Quant) Error!Quant {
    const obj = qobj orelse return base;
    var buf: [96]u8 = undefined;
    const exact = std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, suffix }) catch unreachable;
    var any: ?Value = null;
    var it = obj.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .object) continue;
        if (std.mem.endsWith(u8, kv.key_ptr.*, exact)) return quantOf(kv.value_ptr.*, diag, base);
        if (any == null and std.mem.endsWith(u8, kv.key_ptr.*, "." ++ suffix)) any = kv.value_ptr.*;
    }
    return if (any) |v| quantOf(v, diag, base) else base;
}

fn isFull(types: ?Value, interval: u64, i: usize, diag: *Diag) Error!bool {
    if (types) |t| {
        if (t != .array or i >= t.array.items.len or t.array.items[i] != .string) {
            diag.field = "layer_types";
            return error.BadField;
        }
        const s = t.array.items[i].string;
        if (std.mem.eql(u8, s, "full_attention")) return true;
        if (std.mem.eql(u8, s, "linear_attention")) return false;
        diag.field = "layer_types";
        return error.BadField;
    }
    return (i + 1) % interval == 0;
}

/// Parse a Qwen3.5 / Qwen3.6 config.json (`model_type` `qwen3_5` or `qwen3_5_moe`, language model under `text_config`).
pub fn fromConfigJson(gpa: Allocator, json: []const u8) Error!Model {
    var diag: Diag = .{};
    return fromConfigJsonDiag(gpa, json, &diag);
}

/// As `fromConfigJson`; on `MissingField` or `BadField` `diag.field` names the field.
pub fn fromConfigJsonDiag(gpa: Allocator, json: []const u8, diag: *Diag) Error!Model {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const root = std.json.parseFromSliceLeaky(Value, arena.allocator(), json, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidJson,
    };
    if (root != .object) return error.InvalidJson;
    const mt = root.object.get("model_type") orelse {
        diag.field = "model_type";
        return error.MissingField;
    };
    if (mt != .string) {
        diag.field = "model_type";
        return error.BadField;
    }
    const moe = std.mem.eql(u8, mt.string, "qwen3_5_moe");
    if (!moe and !std.mem.eql(u8, mt.string, "qwen3_5")) return error.UnsupportedModel;

    const text = root.object.get("text_config") orelse root;
    if (text != .object) {
        diag.field = "text_config";
        return error.BadField;
    }
    const c: Reader = .{ .obj = text.object, .diag = diag };

    const hidden = try c.uint("hidden_size");
    const n_layers = try c.uint("num_hidden_layers");
    const vocab = try c.uint("vocab_size");
    const heads = try c.uint("num_attention_heads");
    const kv_heads = try c.uint("num_key_value_heads");
    const head_dim = try c.uint("head_dim");
    const interval = (try c.optUint("full_attention_interval")) orelse 4;
    const types = text.object.get("layer_types");
    if (hidden == 0 or n_layers == 0 or interval == 0 or vocab == 0) {
        diag.field = "hidden_size";
        return error.BadField;
    }
    const tied = if (text.object.get("tie_word_embeddings") orelse root.object.get("tie_word_embeddings")) |t| blk: {
        if (t != .bool) {
            diag.field = "tie_word_embeddings";
            return error.BadField;
        }
        break :blk t.bool;
    } else false;

    var qobj: ?ObjectMap = null;
    var base: Quant = .{};
    const qv = root.object.get("quantization") orelse root.object.get("quantization_config") orelse text.object.get("quantization");
    if (qv) |v| if (v == .object) {
        qobj = v.object;
        if (v.object.get("bits") != null) base = try quantOf(v, diag, .{});
    };

    // MoE or dense feed-forward geometry.
    var ff: u64 = 0;
    var shared: u64 = 0;
    var n_experts: u64 = 0;
    var top_k: u64 = 0;
    var expert_ff: u64 = 0;
    if (moe) {
        n_experts = try c.uint("num_experts");
        top_k = try c.uint("num_experts_per_tok");
        expert_ff = try c.uint("moe_intermediate_size");
        shared = try c.uint("shared_expert_intermediate_size");
        if (n_experts == 0 or top_k == 0 or top_k > n_experts) {
            diag.field = "num_experts_per_tok";
            return error.BadField;
        }
        if (text.object.get("mlp_only_layers")) |m| if (m != .array or m.array.items.len > 0) {
            diag.field = "mlp_only_layers";
            return error.BadField;
        };
    } else ff = try c.uint("intermediate_size");

    // Linear-attention (Gated DeltaNet) dimensions, read only when some layer is linear.
    var lk_heads: u64 = 0;
    var lv_heads: u64 = 0;
    var lk_dim: u64 = 0;
    var lv_dim: u64 = 0;
    var conv_k: u64 = 0;
    var any_linear = false;
    for (0..n_layers) |i| if (!try isFull(types, interval, i, diag)) {
        any_linear = true;
    };
    if (any_linear) {
        lk_heads = try c.uint("linear_num_key_heads");
        lv_heads = try c.uint("linear_num_value_heads");
        lk_dim = try c.uint("linear_key_head_dim");
        lv_dim = try c.uint("linear_value_head_dim");
        conv_k = try c.uint("linear_conv_kernel_dim");
    }
    const conv_dim = 2 * lk_heads * lk_dim + lv_heads * lv_dim;
    const z_dim = lv_heads * lv_dim;

    const layers = try gpa.alloc(Layer, n_layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| {
        const full = try isFull(types, interval, i, diag);
        const q = base;
        // two norms a layer (input and post-attention)
        const norms = 2 * hidden * dense_bytes;
        var mixer: u64 = 0;
        var kv: u64 = 0;
        var state: u64 = 0;
        if (full) {
            mixer = q.bytes(q_gate_factor * heads * head_dim, hidden) + 2 * q.bytes(kv_heads * head_dim, hidden) + q.bytes(hidden, heads * head_dim);
            kv = 2 * kv_heads * head_dim * kv_bytes;
        } else {
            // in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, depthwise conv (bf16, unquantized), out_proj
            mixer = q.bytes(conv_dim, hidden) + q.bytes(z_dim, hidden) + 2 * q.bytes(lv_heads, hidden) + q.bytes(hidden, z_dim) + conv_dim * conv_k * dense_bytes;
            state = lv_heads * lk_dim * lv_dim * delta_state_bytes + (conv_k - 1) * conv_dim * 2;
        }
        if (moe) {
            const gate_q = try override(qobj, i, "mlp.gate", diag, base);
            const sgate_q = try override(qobj, i, "mlp.shared_expert_gate", diag, base);
            const shared_bytes = 3 * q.bytes(shared, hidden) + sgate_q.bytes(1, hidden);
            const router = gate_q.bytes(n_experts, hidden);
            const expert = 3 * q.bytes(expert_ff, hidden);
            const dense = mixer + shared_bytes + router;
            l.* = .{ .resident = dense + norms, .dense = dense, .experts = n_experts * expert, .experts_read = top_k * expert, .kv_per_tok = kv, .state = state, .dispatches = dispatches_moe };
        } else {
            const dense = mixer + 3 * q.bytes(ff, hidden);
            l.* = .{ .resident = dense + norms, .dense = dense, .experts = 0, .experts_read = 0, .kv_per_tok = kv, .state = state, .dispatches = dispatches_dense };
        }
    }

    const name = try std.fmt.allocPrint(gpa, "{s} {d}x{d} {d}-bit", .{ mt.string, n_layers, hidden, if (base.bits == 0) 16 else base.bits });
    return .{
        .name = name,
        .layers = layers,
        .embed = base.bytes(vocab, hidden),
        .head = if (tied) 0 else base.bytes(vocab, hidden),
        .scratch = if (moe) scratch_moe else scratch_dense,
        .hidden = @intCast(hidden),
    };
}

const fx = @import("fixtures.zig");

fn mib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / @as(f64, @floatFromInt(MiB));
}

fn near(expected: f64, actual: f64, rel: f64) !void {
    try std.testing.expectApproxEqRel(expected, actual, rel);
}

test "A3B geometry reproduces the safetensors byte facts" {
    const m = try fromConfigJson(std.testing.allocator, fx.a3b_json);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 40), m.layers.len);
    // routed expert 1,769,472 B, 432 MiB a layer, 16.875 GiB total
    const per_expert: u64 = 1_769_472;
    try std.testing.expectEqual(per_expert * 256, m.layers[0].experts);
    try std.testing.expectEqual(@as(u64, 432 * MiB), m.layers[7].experts);
    try std.testing.expectEqual(per_expert * 8, m.layers[0].experts_read);
    var routed: u64 = 0;
    var kv: u64 = 0;
    var state: u64 = 0;
    var full: usize = 0;
    for (m.layers) |l| {
        routed += l.experts;
        kv += l.kv_per_tok;
        state += l.state;
        if (l.kv_per_tok > 0) full += 1;
    }
    try std.testing.expectEqual(@as(u64, 16 * GiB + 7 * GiB / 8), routed);
    try std.testing.expectEqual(@as(usize, 10), full);
    // non-expert bytes: 1,325 MiB (0.5% tolerance), embed = head = 272.8 MiB
    try near(1325.0, mib(m.weights() - routed), 0.005);
    try near(272.8, mib(m.embed), 0.001);
    try std.testing.expectEqual(m.embed, m.head);
    // KV 20,480 B a token; DeltaNet state 61.4 MiB a stream
    try std.testing.expectEqual(@as(u64, 20_480), kv);
    try near(61.4, mib(state), 0.005);
    // language-model total 18.17 GiB
    try near(18.17, @as(f64, @floatFromInt(m.weights())) / @as(f64, @floatFromInt(GiB)), 0.005);
    try std.testing.expect(m.hasExperts());
    try std.testing.expectEqual(m.head, m.headRead());
}

test "A3B per-class sizes match the spec table" {
    const m = try fromConfigJson(std.testing.allocator, fx.a3b_json);
    defer m.deinit(std.testing.allocator);
    // layer 0 is DeltaNet: mixer 18.13 MiB + shared expert and gate 1.69 MiB + 8-bit router 0.53 MiB
    try near(18.13 + 1.69 + 0.53, mib(m.layers[0].dense), 0.001);
    // layer 3 is full attention: 14.63 MiB
    try near(14.63 + 1.69 + 0.53, mib(m.layers[3].dense), 0.001);
    try std.testing.expectEqual(@as(u64, 0), m.layers[3].state);
    try std.testing.expectEqual(@as(u64, 2048), m.layers[3].kv_per_tok);
    // norms are two 2-byte rows of the hidden size
    try std.testing.expectEqual(m.layers[0].dense + 8192, m.layers[0].resident);
    try std.testing.expectEqual(@as(u64, 22), m.layers[0].dispatches);
}

test "dense 4B geometry: 2.20 GiB, tied head, 4096 B of KV a token" {
    const m = try fromConfigJson(std.testing.allocator, fx.d4_json);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 32), m.layers.len);
    try near(2.20, @as(f64, @floatFromInt(m.weights())) / @as(f64, @floatFromInt(GiB)), 0.005);
    try std.testing.expectEqual(@as(u64, 0), m.head);
    try std.testing.expectEqual(m.embed, m.headRead());
    try std.testing.expect(!m.hasExperts());
    var full: usize = 0;
    for (m.layers, 0..) |l, i| {
        try std.testing.expectEqual(@as(u64, 0), l.experts);
        try std.testing.expectEqual(@as(u64, 12), l.dispatches);
        const is_full = i % 4 == 3;
        try std.testing.expectEqual(is_full, l.kv_per_tok > 0);
        try std.testing.expectEqual(!is_full, l.state > 0);
        if (is_full) {
            full += 1;
            try std.testing.expectEqual(@as(u64, 4096), l.kv_per_tok);
        }
    }
    try std.testing.expectEqual(@as(usize, 8), full);
    try std.testing.expectEqual(scratch_dense, m.scratch);
}

test "layer_types overrides the interval and a missing quantization means 2-byte weights" {
    const json =
        \\{ "model_type": "qwen3_5", "text_config": { "hidden_size": 128, "num_hidden_layers": 3,
        \\  "layer_types": ["full_attention", "full_attention", "linear_attention"], "num_attention_heads": 2,
        \\  "num_key_value_heads": 1, "head_dim": 64, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
        \\  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 32,
        \\  "intermediate_size": 256, "vocab_size": 1000 } }
    ;
    const m = try fromConfigJson(std.testing.allocator, json);
    defer m.deinit(std.testing.allocator);
    try std.testing.expect(m.layers[0].kv_per_tok > 0 and m.layers[1].kv_per_tok > 0 and m.layers[2].kv_per_tok == 0);
    try std.testing.expectEqual(@as(u64, 1000 * 128 * 2), m.embed);
    try std.testing.expectEqual(m.embed, m.head);
    // q+gate 2*2*64=256, k and v 64 each, o 128x128; mlp 3 x 256x128; all 2 bytes
    try std.testing.expectEqual(@as(u64, (256 * 128 + 2 * 64 * 128 + 128 * 128 + 3 * 256 * 128) * 2), m.layers[0].dense);
}

test "planted-bad configs name the problem" {
    const gpa = std.testing.allocator;
    var diag: Diag = .{};
    try std.testing.expectError(error.InvalidJson, fromConfigJsonDiag(gpa, "{", &diag));
    try std.testing.expectError(error.InvalidJson, fromConfigJsonDiag(gpa, "[]", &diag));
    try std.testing.expectError(error.MissingField, fromConfigJsonDiag(gpa, "{}", &diag));
    try std.testing.expectEqualStrings("model_type", diag.field);
    try std.testing.expectError(error.UnsupportedModel, fromConfigJsonDiag(gpa, "{\"model_type\":\"llama\"}", &diag));
    try std.testing.expectError(error.MissingField, fromConfigJsonDiag(gpa, "{\"model_type\":\"qwen3_5\",\"text_config\":{}}", &diag));
    try std.testing.expectEqualStrings("hidden_size", diag.field);
    // wrong type
    const bad_type = try std.mem.replaceOwned(u8, gpa, fx.d4_json, "\"hidden_size\": 2560", "\"hidden_size\": \"2560\"");
    defer gpa.free(bad_type);
    try std.testing.expectError(error.BadField, fromConfigJsonDiag(gpa, bad_type, &diag));
    try std.testing.expectEqualStrings("hidden_size", diag.field);
    // missing MoE field
    const no_experts = try std.mem.replaceOwned(u8, gpa, fx.a3b_json, "\"num_experts\": 256, ", "");
    defer gpa.free(no_experts);
    try std.testing.expectError(error.MissingField, fromConfigJsonDiag(gpa, no_experts, &diag));
    try std.testing.expectEqualStrings("num_experts", diag.field);
    // layer_types of the wrong length or with an unknown kind
    const short = try std.mem.replaceOwned(u8, gpa, fx.a3b_json, "\"num_hidden_layers\": 40", "\"num_hidden_layers\": 41");
    defer gpa.free(short);
    try std.testing.expectError(error.BadField, fromConfigJsonDiag(gpa, short, &diag));
    try std.testing.expectEqualStrings("layer_types", diag.field);
    // top-k larger than the expert count
    const over = try std.mem.replaceOwned(u8, gpa, fx.a3b_json, "\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 300");
    defer gpa.free(over);
    try std.testing.expectError(error.BadField, fromConfigJsonDiag(gpa, over, &diag));
}
