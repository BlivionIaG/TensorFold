//! Public index admission proves required tensor names and safe shard basenames before payload headers are available.
const std = @import("std");
const Config = @import("config.zig").Config;
const tensors = @import("tensors.zig");

const Check = struct {
    map: std.json.ObjectMap,
    names: usize = 0,
    runtime: ?*const Config = null,

    fn take(c: *Check, name: []const u8) !void {
        if (c.map.get(name) == null) return error.MissingFlashTensor;
        c.names += 1;
    }

    fn part(c: *Check, stem: []const u8, suffix: []const u8) !void {
        var buf: [256]u8 = undefined;
        try c.take(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    fn matrix(c: *Check, stem: []const u8) !void {
        try c.part(stem, "weight");
        var buf: [256]u8 = undefined;
        const scales = c.map.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{stem})) != null;
        const biases = c.map.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{stem})) != null;
        if (scales != biases) return error.IncompleteFlashAffine;
        if (scales) {
            if (c.runtime) |cfg| {
                const spec = (try cfg.quantization(stem)) orelse return error.UnsupportedFlashAffineKernel;
                try spec.checkFlashKernel();
            }
            try c.part(stem, "scales");
            try c.part(stem, "biases");
        }
    }

    fn proj(c: *Check, stem: []const u8, suffix: []const u8) !void {
        var buf: [256]u8 = undefined;
        try c.matrix(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    fn hc(c: *Check, stem: []const u8, inject: bool) !void {
        try c.part(stem, tensors.hc.norm);
        for ([_][]const u8{ tensors.hc.down, tensors.hc.up }) |suffix| try c.proj(stem, suffix);
        if (inject) try c.proj(stem, tensors.hc.inject);
    }

    fn layer(c: *Check, stem: []const u8, kind: @import("config.zig").Kind) !void {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ tensors.hc.attn, tensors.hc.mlp }) |suffix| try c.hc(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }), true);
        for ([_][]const u8{ tensors.moe.gate, tensors.moe.shared_gate } ++ tensors.moe.shared ++ tensors.moe.routed) |suffix| try c.proj(stem, suffix);
        if (kind == .linear_attention) {
            for (tensors.linear.inputs ++ [_][]const u8{tensors.linear.out}) |suffix| try c.proj(stem, suffix);
            for ([_][]const u8{ tensors.linear.a_log, tensors.linear.dt_bias, tensors.linear.conv, tensors.linear.norm }) |suffix| try c.part(stem, suffix);
        } else {
            for (tensors.sparse.inputs ++ [_][]const u8{tensors.sparse.o}) |suffix| try c.proj(stem, suffix);
            for ([_][]const u8{ tensors.sparse.q_norm, tensors.sparse.k_norm, tensors.sparse.iq_norm, tensors.sparse.ik_norm }) |suffix| try c.part(stem, suffix);
        }
    }

    fn ple(c: *Check, stem: []const u8, shards: usize, spelling: []const u8) !void {
        for (tensors.ple.vectors) |suffix| try c.part(stem, suffix);
        try c.proj(stem, tensors.ple.key);
        try c.proj(stem, tensors.ple.value);
        var buf: [256]u8 = undefined;
        for (0..shards) |i| try c.matrix(try tensors.ple.table(&buf, stem, spelling, i));
    }
};

/// The index's spelling of its n-gram table shards, `shard_N` when a `shard_0` is listed, else oMLX's `shards.N`.
pub fn ngramSpelling(map: std.json.ObjectMap) []const u8 {
    for (map.keys()) |name| if (std.mem.endsWith(u8, name, ".ngram_embedding.shard_0.weight")) return "shard_";
    return "shards.";
}

pub const Inventory = struct { tensor_names: usize, required_names: usize, shards: usize, headers_verified: bool = false };

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config) !Inventory {
    return admitWith(gpa, bytes, config, false);
}

/// Runtime kernels consume only six-bit/group-32 text matrices, unlike the generic pack metadata reader.
pub fn admitRuntime(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config) !Inventory {
    try config.global_affine.checkFlashKernel();
    return admitWith(gpa, bytes, config, true);
}

fn admitWith(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config, runtime: bool) !Inventory {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer p.deinit();
    if (p.value != .object) return error.InvalidFlashIndex;
    const map = p.value.object.get("weight_map") orelse return error.InvalidFlashIndex;
    if (map != .object or map.object.count() == 0) return error.InvalidFlashIndex;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = map.object.iterator();
    while (it.next()) |item| {
        const file = item.value_ptr.*;
        if (file != .string or file.string.len == 0 or file.string[0] == '.' or std.mem.indexOfAny(u8, file.string, "/\\") != null or !std.mem.endsWith(u8, file.string, ".safetensors")) return error.UnsafeFlashShard;
        try files.put(gpa, file.string, {});
    }
    var c = Check{ .map = map.object, .runtime = if (runtime) config else null };
    const spelling = ngramSpelling(map.object); // every shard is checked under this one, so an index mixing the two is refused
    try c.matrix(tensors.embed);
    try c.matrix(tensors.lm_head);
    try c.hc(tensors.mixer, false);
    var stem: [128]u8 = undefined;
    for (0..config.layers) |i| {
        const layer = try tensors.layer(&stem, i);
        try c.layer(layer, try config.kind(i));
        if (config.ple[i]) try c.ple(layer, config.ngram_shards, spelling);
    }
    if (config.mtp.layers > 0) {
        for ([_][]const u8{ tensors.mtp.fc_embedding, tensors.mtp.fc_hidden }) |name| try c.matrix(name);
        for ([_][]const u8{ tensors.mtp.norm_embedding, tensors.mtp.norm_hidden }) |name| try c.take(name);
        try c.hc(tensors.mtp.mixer, false);
        for (0..config.mtp.layers) |i| try c.layer(try std.fmt.bufPrint(&stem, "{s}.layers.{d}", .{ tensors.mtp.stem, i }), config.mtp.kinds[i]);
    }
    return .{ .tensor_names = map.object.count(), .required_names = c.names, .shards = files.count() };
}
