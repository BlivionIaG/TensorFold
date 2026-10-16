//! Host admission for one EXL3 Qwen3.8-Flash-Next config.
//! Affine and MLX stay `NotExl3` and never become an affine spec. No weights are loaded.
//! `--ple-on-ssd` is `exl3_ple.Mode.ssd` (host preads on the existing PLE path).
//! `--ssd-experts` records host-resident routed experts. The n-gram table is not allocated.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const ple = @import("exl3_ple.zig");
const cache = @import("exl3_cache.zig");

pub const PleMode = ple.Mode;
pub const KvDtype = cache.Dtype;

const model_type_name = "qwen4_exp";

/// `0` means the loader sizes the expert lines with `cache.learnedBytes`
/// after the pinned weights and the KV table are known. A flat 16 GiB cache
/// on top of those weights would oversubscribe the card.

pub const Error = error{
    InvalidJson,
    NotObject,
    MissingField,
    BadInteger,
    NotExl3,
    NotFlashNext,
    ShapeMismatch,
    IllegalCut,
};

pub const Request = struct {
    /// `page_cache` unless `--ple-on-ssd`.
    ple: PleMode = .page_cache,
    /// `--kv-dtype`. bf16, int8, or int4. There is no `none`.
    kv: KvDtype = .bf16,
    /// `--tp`. Only 1 or 2.
    ranks: u8,
    /// 0 leaves the line count to `cache.learnedBytes`.
    cache_bytes: u64 = 0,
    /// `--ssd-experts`.
    ssd_experts: bool,
};

pub const Admitted = struct {
    model_type: []const u8,
    quant_method: []const u8,
    hidden: usize,
    layers: usize,
    experts: usize,
    topk: usize,
    expert_intermediate: usize,
    vocab: usize,
    ngram_shards: usize,
    ngram_bits: usize,
    ple: PleMode,
    kv: KvDtype,
    ranks: u8,
    cache_bytes: u64,
    ssd_experts: bool,
    /// Both PLE modes stay on the host path. Admission does not allocate the table.
    ngram_host: bool,
};

const hidden_keys = [_][]const u8{ "hidden_size", "hidden" };
const layer_keys = [_][]const u8{ "num_hidden_layers", "layers" };
const expert_keys = [_][]const u8{ "num_experts", "experts" };
const topk_keys = [_][]const u8{ "num_experts_per_tok", "topk" };
const moe_keys = [_][]const u8{ "moe_intermediate_size", "expert_intermediate" };
const vocab_keys = [_][]const u8{ "vocab_size", "vocab" };
const shard_keys = [_][]const u8{ "split_ngram_parts", "ngram_shards" };
const bit_keys = [_][]const u8{"ngram_bits"};

pub fn admit(json: []const u8, req: Request) Error!Admitted {
    if (req.ranks != 1 and req.ranks != 2) return error.IllegalCut;

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const root_value = std.json.parseFromSliceLeaky(std.json.Value, arena_state.allocator(), json, .{}) catch return error.InvalidJson;
    const root = if (root_value == .object) root_value.object else return error.NotObject;
    const text = textMap(root);

    try requireModel(root, text);
    var qbuf: [4]std.json.ObjectMap = undefined;
    const quants = try quantMaps(root, text, &qbuf);
    try requireExl3(root, text, quants);

    const hidden = try requireField(text, root, &hidden_keys, shape.hidden);
    const layers = try requireField(text, root, &layer_keys, shape.layers);
    const experts = try requireField(text, root, &expert_keys, shape.experts);
    const topk = try requireField(text, root, &topk_keys, shape.topk);
    const expert_intermediate = try requireField(text, root, &moe_keys, shape.expert_intermediate);
    const vocab = try requireField(text, root, &vocab_keys, shape.vocab);
    const ngram_shards = try requireField(text, root, &shard_keys, shape.ngram_shards);
    const ngram_bits = try requireBits(text, root, quants);

    const host = cache.placeWeights(.{
        .device_bytes = 0,
        .resident_bytes = 0,
        .kv_bytes = 0,
        .routed_expert_bytes = 0,
        .ngram_bytes = 0,
    }) == .host;

    return .{
        .model_type = model_type_name,
        .quant_method = shape.quant_method,
        .hidden = hidden,
        .layers = layers,
        .experts = experts,
        .topk = topk,
        .expert_intermediate = expert_intermediate,
        .vocab = vocab,
        .ngram_shards = ngram_shards,
        .ngram_bits = ngram_bits,
        .ple = req.ple,
        .kv = req.kv,
        .ranks = req.ranks,
        .cache_bytes = req.cache_bytes,
        .ssd_experts = req.ssd_experts,
        .ngram_host = host,
    };
}

fn textMap(root: std.json.ObjectMap) std.json.ObjectMap {
    if (root.get("text_config")) |v| {
        if (v == .object and v.object.count() > 0) return v.object;
    }
    return root;
}

fn requireModel(root: std.json.ObjectMap, text: std.json.ObjectMap) Error!void {
    const tag = root.get("model_type") orelse text.get("model_type") orelse return error.MissingField;
    if (tag != .string or !std.mem.eql(u8, tag.string, model_type_name)) return error.NotFlashNext;
}

fn quantMaps(root: std.json.ObjectMap, text: std.json.ObjectMap, buf: *[4]std.json.ObjectMap) Error![]std.json.ObjectMap {
    var n: usize = 0;
    try pushQuant(&n, buf, root);
    try pushQuant(&n, buf, text);
    return buf[0..n];
}

fn pushQuant(n: *usize, buf: *[4]std.json.ObjectMap, source: std.json.ObjectMap) Error!void {
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const v = source.get(key) orelse continue;
        if (v == .null) continue;
        if (v != .object) return error.NotExl3;
        if (n.* >= buf.len) return error.NotExl3;
        buf[n.*] = v.object;
        n.* += 1;
    }
}

fn requireExl3(root: std.json.ObjectMap, text: std.json.ObjectMap, quants: []const std.json.ObjectMap) Error!void {
    var saw = false;
    try note(&saw, root);
    try note(&saw, text);
    for (quants) |q| try note(&saw, q);
    if (!saw) return error.NotExl3;
}

fn note(saw: *bool, o: std.json.ObjectMap) Error!void {
    if (o.get("mode")) |v| switch (v) {
        .null => {},
        .string => |s| {
            if (s.len != 0 and !std.mem.eql(u8, s, shape.quant_method)) return error.NotExl3;
        },
        else => return error.NotExl3,
    };
    if (o.get("quant_method")) |v| switch (v) {
        .string => |s| {
            if (!std.mem.eql(u8, s, shape.quant_method)) return error.NotExl3;
            saw.* = true;
        },
        else => return error.NotExl3,
    };
}

fn requireBits(text: std.json.ObjectMap, root: std.json.ObjectMap, quants: []const std.json.ObjectMap) Error!usize {
    var saw = try found(text, &bit_keys, shape.ngram_bits);
    if (try found(root, &bit_keys, shape.ngram_bits)) saw = true;
    for (quants) |q| {
        if (try found(q, &bit_keys, shape.ngram_bits)) saw = true;
    }
    if (!saw) return error.MissingField;
    return shape.ngram_bits;
}

fn requireField(text: std.json.ObjectMap, root: std.json.ObjectMap, names: []const []const u8, want: usize) Error!usize {
    const in_text = try found(text, names, want);
    const in_root = try found(root, names, want);
    if (!in_text and !in_root) return error.MissingField;
    return want;
}

fn found(o: std.json.ObjectMap, names: []const []const u8, want: usize) Error!bool {
    var saw = false;
    for (names) |name| {
        const v = o.get(name) orelse continue;
        const got = try integer(v);
        if (got != want) return error.ShapeMismatch;
        saw = true;
    }
    return saw;
}

fn integer(v: std.json.Value) Error!usize {
    if (v != .integer or v.integer < 0) return error.BadInteger;
    return std.math.cast(usize, v.integer) orelse error.BadInteger;
}

const published =
    \\{
    \\  "model_type": "qwen4_exp",
    \\  "quantization_config": {"quant_method": "exl3"},
    \\  "text_config": {
    \\    "hidden_size": 2560,
    \\    "num_hidden_layers": 48,
    \\    "num_experts": 512,
    \\    "num_experts_per_tok": 10,
    \\    "moe_intermediate_size": 640,
    \\    "vocab_size": 248320,
    \\    "split_ngram_parts": 128,
    \\    "ngram_bits": 5
    \\  }
    \\}
;

const offload = Request{ .ranks = 1, .ssd_experts = true };

test "published qwen4_exp exl3 shape admits with host experts and an unsized expert cache" {
    const a = try admit(published, offload);
    try std.testing.expectEqualStrings(model_type_name, a.model_type);
    try std.testing.expectEqualStrings(shape.quant_method, a.quant_method);
    try std.testing.expectEqual(shape.hidden, a.hidden);
    try std.testing.expectEqual(shape.layers, a.layers);
    try std.testing.expectEqual(shape.experts, a.experts);
    try std.testing.expectEqual(shape.topk, a.topk);
    try std.testing.expectEqual(shape.expert_intermediate, a.expert_intermediate);
    try std.testing.expectEqual(shape.vocab, a.vocab);
    try std.testing.expectEqual(shape.ngram_shards, a.ngram_shards);
    try std.testing.expectEqual(shape.ngram_bits, a.ngram_bits);
    try std.testing.expectEqual(PleMode.page_cache, a.ple);
    try std.testing.expectEqual(KvDtype.bf16, a.kv);
    try std.testing.expectEqual(@as(u8, 1), a.ranks);
    try std.testing.expectEqual(@as(u64, 0), a.cache_bytes);
    try std.testing.expect(a.ssd_experts);
    try std.testing.expect(a.ngram_host);
}

test "ranks 2 admits and 4 or 0 are an illegal cut" {
    const two = try admit(published, .{ .ranks = 2, .ssd_experts = true });
    try std.testing.expectEqual(@as(u8, 2), two.ranks);
    try std.testing.expectError(error.IllegalCut, admit(published, .{ .ranks = 4, .ssd_experts = true }));
    try std.testing.expectError(error.IllegalCut, admit(published, .{ .ranks = 0, .ssd_experts = true }));
}

test "mlx, affine, and a non-flash model are refused" {
    const mlx =
        \\{"model_type":"qwen4_exp","quant_method":"mlx","hidden_size":2560,"num_hidden_layers":48,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    const affine =
        \\{"model_type":"qwen4_exp","quantization":{"mode":"affine","quant_method":"affine","bits":6},"hidden_size":2560,"num_hidden_layers":48,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    const llama =
        \\{"model_type":"llama","quant_method":"exl3","hidden_size":2560,"num_hidden_layers":48,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    const missing =
        \\{"model_type":"qwen4_exp","hidden_size":2560,"num_hidden_layers":48,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    try std.testing.expectError(error.NotExl3, admit(mlx, offload));
    try std.testing.expectError(error.NotExl3, admit(affine, offload));
    try std.testing.expectError(error.NotExl3, admit(missing, offload));
    try std.testing.expectError(error.NotFlashNext, admit(llama, offload));
}

test "ple on ssd is visible and the n-gram table stays host" {
    const a = try admit(published, .{ .ple = .ssd, .ranks = 1, .ssd_experts = true });
    try std.testing.expectEqual(PleMode.ssd, a.ple);
    try std.testing.expect(a.ngram_host);
    try std.testing.expect(a.ssd_experts);
}

test "kv dtype is only bf16, int8, and int4" {
    const names = std.meta.fieldNames(KvDtype);
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expectEqualStrings("bf16", names[0]);
    try std.testing.expectEqualStrings("int8", names[1]);
    try std.testing.expectEqualStrings("int4", names[2]);

    const as_int8 = try admit(published, .{ .ranks = 1, .kv = .int8, .ssd_experts = true });
    try std.testing.expectEqual(KvDtype.int8, as_int8.kv);
    const as_int4 = try admit(published, .{ .ranks = 1, .kv = .int4, .ssd_experts = true });
    try std.testing.expectEqual(KvDtype.int4, as_int4.kv);
}

test "hidden_size that is not the published width is refused" {
    const wide =
        \\{"model_type":"qwen4_exp","quant_method":"exl3","hidden_size":9999,"num_hidden_layers":48,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    const experts =
        \\{"model_type":"qwen4_exp","quant_method":"exl3","hidden_size":2560,"num_hidden_layers":48,"num_experts":8,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    const layers =
        \\{"model_type":"qwen4_exp","quant_method":"exl3","hidden_size":2560,"num_hidden_layers":12,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"vocab_size":248320,"split_ngram_parts":128,"ngram_bits":5}
    ;
    try std.testing.expectError(error.ShapeMismatch, admit(wide, offload));
    try std.testing.expectError(error.ShapeMismatch, admit(experts, offload));
    try std.testing.expectError(error.ShapeMismatch, admit(layers, offload));
}

test "ssd_experts is recorded only when the request asks for it" {
    const on = try admit(published, .{ .ranks = 1, .ssd_experts = true });
    try std.testing.expect(on.ssd_experts);
    const off = try admit(published, .{ .ranks = 1, .ssd_experts = false });
    try std.testing.expect(!off.ssd_experts);
    try std.testing.expectEqual(shape.experts, off.experts);
    try std.testing.expectEqual(PleMode.page_cache, off.ple);
}
