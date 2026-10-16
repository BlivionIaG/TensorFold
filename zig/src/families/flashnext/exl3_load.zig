//! Host placement of one EXL3 safetensors shard.
//! The header is walked; tensor payloads are not copied, so an n-gram table or an expert bank is not allocated.
//! Homes come from `place.home`, which calls `scan.classifyName`. This file does not restate those rules.
//! `centred_norm` is that tag alone. The fp32 add is `scan.centredOffset`, applied when the gamma bytes are read.

const std = @import("std");
const scan = @import("exl3_scan.zig");
const place = @import("exl3_place.zig");
const st = @import("safetensors.zig");
const cache = @import("exl3_cache.zig");
const tp = @import("exl3_tp.zig");
const shape = @import("exl3_shape.zig");

pub const Tensor = struct {
    name: []const u8,
    home: place.Home,
    dtype: st.DType,
    /// Byte range from the start of the data region, as the header stored it.
    begin: u64,
    end: u64,
};

pub const Placement = struct {
    tensors: []Tensor,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Placement) void {
        self.arena.deinit();
    }

    pub fn get(self: *const Placement, name: []const u8) ?Tensor {
        for (self.tensors) |tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
        return null;
    }
};

/// One home per header entry, in header order. Names alias `header`.
/// `file` may be a path; the MTP mixer check uses its basename.
pub fn classify(gpa: std.mem.Allocator, header: st.Header, file: []const u8) ![]Tensor {
    const out = try gpa.alloc(Tensor, header.count());
    const base = std.fs.path.basename(file);
    for (header.keys(), header.values(), 0..) |name, entry, i| {
        const rank: usize = entry.rank;
        // `place.home` is the decision. It reads this kind; the call keeps that dependency here.
        _ = scan.classifyName(name, rank);
        out[i] = .{
            .name = name,
            .home = place.home(.{ .name = name, .rank = rank, .file = base }),
            .dtype = entry.dtype,
            .begin = @intCast(entry.begin),
            .end = @intCast(entry.end),
        };
    }
    return out;
}

/// Parse a safetensors image (u64 LE header length, JSON, data) and place each tensor.
/// Data bytes are not read and not copied. A bad header is the safetensors error.
pub fn loadHeader(gpa: std.mem.Allocator, bytes: []const u8, file: []const u8) !Placement {
    if (bytes.len < 8) return error.BadSafetensors;
    const header_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    if (header_len > bytes.len - 8) return error.BadSafetensors;
    var placement: Placement = .{ .tensors = &.{}, .arena = .init(gpa) };
    errdefer placement.deinit();
    const a = placement.arena.allocator();
    const json = try a.dupe(u8, bytes[8..][0..header_len]);
    const header = try st.parseHeader(a, json, bytes.len - 8 - header_len);
    placement.tensors = try owned(a, header, file);
    return placement;
}

/// Map `path`, record homes and data-region offsets, then drop the map.
/// The tensor bodies are not copied into the placement.
pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Placement {
    var file = try st.File.open(gpa, io, path);
    defer file.close(io);
    var placement: Placement = .{ .tensors = &.{}, .arena = .init(gpa) };
    errdefer placement.deinit();
    placement.tensors = try owned(placement.arena.allocator(), file.names, path);
    return placement;
}

/// Bytes that stay on the device for the life of the process.
/// Routed experts, the n-gram table, and the vision encoder are left out.
/// The VRAM expert cache is not in this sum. Add it with `deviceWeights`.
pub fn residentBytes(tensors: []const Tensor) u64 {
    var n: u64 = 0;
    for (tensors) |t| {
        switch (t.home) {
            .routed_host, .ngram_host, .vision_skip, .plain => {},
            .shared_resident, .resident, .centred_norm, .mtp_patch => {
                if (t.end > t.begin) n += t.end - t.begin;
            },
        }
    }
    return n;
}

/// What CUDA admission should treat as this rank's weights.
/// `cache_bytes` is the VRAM line budget `schedule` fills, not the routed bank.
/// Zero does not add a flat cache; `learnedLines` sizes that budget from `learnedBytes`.
pub fn deviceWeights(tensors: []const Tensor, cache_bytes: u64) error{ExceedsDevice}!u64 {
    const pinned = residentBytes(tensors);
    // A flat 16 GiB cache on top of a pin is the whole card again.
    if (pinned != 0 and cache_bytes == cache.card_bytes) return error.ExceedsDevice;
    const sum = std.math.add(u64, pinned, cache_bytes) catch return error.ExceedsDevice;
    if (sum > cache.card_bytes) return error.ExceedsDevice;
    return sum;
}

pub const TensorHeader = struct {
    name: []const u8,
    /// Safetensors shape rank. `stayOf` uses it to separate an n-gram row from a plain weight.
    ndims: usize,
    shape: [st.max_rank]usize = @splat(0),
    begin: u64 = 0,
    end: u64 = 0,
    file: []const u8 = "",
};

/// Bytes this rank pins. Homes other than `vram_pinned` contribute nothing.
/// The mixer-patch file is included only when a header names it; its size is the span.
pub fn pinRank(headers: []const TensorHeader, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!u64 {
    var n: u64 = 0;
    for (headers) |h| {
        const base = std.fs.path.basename(h.file);
        const where = try place.stayOf(.{ .name = h.name, .rank = h.ndims, .file = base }, rank, ranks);
        if (where != .vram_pinned) continue;
        n +|= try pinnedPart(h, rank, ranks);
    }
    return n;
}

/// A layer that will run the router needs its own `mlp.shared_expert_gate` header.
pub fn requireSharedGate(headers: []const TensorHeader) error{MissingSharedGate}!void {
    for (headers) |h| {
        if (!isRouterGate(h.name)) continue;
        var found = false;
        for (headers) |other| {
            if (contains(other.name, "mlp.shared_expert_gate") and sameLayer(h.name, other.name)) {
                found = true;
                break;
            }
        }
        if (!found) return error.MissingSharedGate;
    }
}

/// `cache_bytes == 0` is `learnedBytes` divided by the header stride.
/// The stride stays `full_expert_bytes` only when every owned span is at most that.
pub fn learnedLines(pinned: u64, kv: u64, cache_bytes: u64, owned_spans: []const u64) error{ExceedsDevice}!u64 {
    if (pinned != 0 and cache_bytes == cache.card_bytes) return error.ExceedsDevice;
    if (cache_bytes > cache.card_bytes) return error.ExceedsDevice;
    const sum = std.math.add(u64, pinned, cache_bytes) catch return error.ExceedsDevice;
    if (cache_bytes != 0 and sum > cache.card_bytes) return error.ExceedsDevice;
    const room = if (cache_bytes == 0) cache.learnedBytes(cache.card_bytes, pinned, kv) else cache_bytes;
    return cache.expertLinesOf(room, lineStride(owned_spans));
}

fn lineStride(spans: []const u64) u64 {
    var max: u64 = 0;
    for (spans) |s| max = @max(max, s);
    if (max <= cache.full_expert_bytes) return cache.full_expert_bytes;
    const boundary: u64 = 256;
    return (max + (boundary - 1)) & ~(boundary - 1);
}

fn spanOf(h: TensorHeader) u64 {
    return if (h.end > h.begin) h.end - h.begin else 0;
}

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

fn mixerPatch(file: []const u8) bool {
    return std.mem.eql(u8, std.fs.path.basename(file), "mtp_hyper_connection_mixer_patch.safetensors");
}

fn fullReplica(h: TensorHeader) bool {
    if (mixerPatch(h.file)) return true;
    if (place.isCentred(h.name)) return true;
    // Not cut(.gate) or cut(.shared): those fix n at expert_intermediate.
    if (contains(h.name, "mlp.shared_expert_gate")) return true;
    if (contains(h.name, "mlp.gate") and !contains(h.name, "gate_proj")) return true;
    if (contains(h.name, "key_proj") or contains(h.name, "value_proj")) return true;
    if (contains(h.name, "conv1d") or contains(h.name, "A_log") or contains(h.name, "dt_bias")) return true;
    return false;
}

fn pinnedPart(h: TensorHeader, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!u64 {
    const span = spanOf(h);
    if (span == 0 or fullReplica(h)) return span;
    if (gdnSplitK(h.name)) |split_k| {
        const axis = axisLen(h, split_k);
        if (ranks > 1 and axis != 0 and tp.legal(axis, ranks)) return span / @as(u64, ranks);
        return span;
    }
    if (roleOf(h.name)) |role| return tp.share(role, span, axisFor(h, role), rank, ranks);
    return span;
}

/// `false` is an N-split (`in_proj`, indexer). `true` is a K-split (`out_proj`).
fn gdnSplitK(name: []const u8) ?bool {
    if (contains(name, "in_proj") or contains(name, "indexer")) return false;
    if (contains(name, "out_proj")) return true;
    return null;
}

fn roleOf(name: []const u8) ?tp.Role {
    if (contains(name, "shared_expert")) return .shared;
    if (contains(name, "hyper_connection")) return .hc_lowrank;
    if (contains(name, "q_proj")) return .q;
    if (contains(name, "k_proj")) return .k;
    if (contains(name, "v_proj")) return .v;
    if (contains(name, "o_proj")) return .o;
    if (contains(name, "lm_head")) return .head;
    return null;
}

fn axisFor(h: TensorHeader, role: tp.Role) usize {
    if (role == .hc_lowrank) {
        var i: usize = 0;
        while (i < h.ndims and i < h.shape.len) : (i += 1) {
            if (h.shape[i] == shape.hc_lowrank) return shape.hc_lowrank;
        }
        return 0;
    }
    // o is split along K. q, k, v, and lm_head are split along N.
    return axisLen(h, role == .o);
}

fn suffix(name: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return name;
    return name[dot + 1 ..];
}

fn axisLen(h: TensorHeader, split_k: bool) usize {
    const suf = suffix(h.name);
    if (std.mem.eql(u8, suf, "trellis") and h.ndims >= 2) {
        return if (split_k) h.shape[0] * shape.tile else h.shape[1] * shape.tile;
    }
    if (h.ndims >= 2) return if (split_k) h.shape[1] else h.shape[0];
    if (h.ndims == 1) {
        const along_k = std.mem.eql(u8, suf, "suh") or std.mem.eql(u8, suf, "su");
        const along_n = std.mem.eql(u8, suf, "svh") or std.mem.eql(u8, suf, "sv") or std.mem.eql(u8, suf, "bias");
        if (split_k and along_k) return if (std.mem.eql(u8, suf, "su")) h.shape[0] * shape.tile else h.shape[0];
        if (!split_k and along_n) return if (std.mem.eql(u8, suf, "sv")) h.shape[0] * shape.tile else h.shape[0];
    }
    return 0;
}

fn isRouterGate(name: []const u8) bool {
    return contains(name, "mlp.gate") and !contains(name, "gate_proj");
}

fn layerIndex(name: []const u8) ?usize {
    const key = "layers.";
    const at = std.mem.indexOf(u8, name, key) orelse return null;
    var i = at + key.len;
    if (i >= name.len or name[i] < '0' or name[i] > '9') return null;
    var n: usize = 0;
    while (i < name.len and name[i] >= '0' and name[i] <= '9') : (i += 1) {
        n = n * 10 + (name[i] - '0');
    }
    return n;
}

/// Prefix `mtp.`, infix `.mtp.`, or a final `.mtp`. `notmtp` is not that component.
fn sameLayer(a: []const u8, b: []const u8) bool {
    const a_mtp = std.mem.startsWith(u8, a, "mtp.") or std.mem.endsWith(u8, a, ".mtp") or std.mem.indexOf(u8, a, ".mtp.") != null;
    const b_mtp = std.mem.startsWith(u8, b, "mtp.") or std.mem.endsWith(u8, b, ".mtp") or std.mem.indexOf(u8, b, ".mtp.") != null;
    if (a_mtp != b_mtp) return false;
    return layerIndex(a) == layerIndex(b);
}

fn owned(a: std.mem.Allocator, header: st.Header, file: []const u8) ![]Tensor {
    const listed = try classify(a, header, file);
    for (listed, 0..) |tensor, i| listed[i].name = try a.dupe(u8, tensor.name);
    return listed;
}

const sample_json =
    \\{"__metadata__":{"format":"exl3"},
    \\ "model.layers.0.mlp.experts.0.gate.trellis":{"dtype":"I16","shape":[2,4],"data_offsets":[0,16]},
    \\ "model.layers.0.mlp.shared_expert.512.up_proj.trellis":{"dtype":"I16","shape":[2,4],"data_offsets":[16,32]},
    \\ "ple.ngram_embedding.shard_0.trellis":{"dtype":"I16","shape":[4,2],"data_offsets":[32,48]},
    \\ "ple.ngram_embedding.shard_1.trellis":{"dtype":"I16","shape":[2,2,2],"data_offsets":[48,64]},
    \\ "model.layers.0.self_attn.q_norm.weight":{"dtype":"F16","shape":[8],"data_offsets":[64,80]},
    \\ "model.layers.0.self_attn.k_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[80,88]},
    \\ "model.layers.0.attn_hyper_connection.hc_norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[88,96]},
    \\ "model.visual.blocks.0.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[96,112]}}
;
const sample_data: usize = 112;

fn synthetic(gpa: std.mem.Allocator, json: []const u8, data_len: usize) ![]u8 {
    const bytes = try gpa.alloc(u8, 8 + json.len + data_len);
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    @memset(bytes[8 + json.len ..], 0);
    return bytes;
}

fn loadSample(gpa: std.mem.Allocator) !Placement {
    const bytes = try synthetic(gpa, sample_json, sample_data);
    defer gpa.free(bytes);
    return loadHeader(gpa, bytes, "model.safetensors");
}

fn require(p: *const Placement, name: []const u8, home: place.Home, dtype: st.DType, begin: u64, end: u64) !void {
    const tensor = p.get(name) orelse return error.MissingTensor;
    try std.testing.expectEqual(home, tensor.home);
    try std.testing.expectEqual(dtype, tensor.dtype);
    try std.testing.expectEqual(begin, tensor.begin);
    try std.testing.expectEqual(end, tensor.end);
}

test "rank-2 routed expert trellis is routed_host, not resident" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    const name = "model.layers.0.mlp.experts.0.gate.trellis";
    try require(&placed, name, .routed_host, .i16, 0, 16);
    try std.testing.expect(placed.get(name).?.home != .resident);
}

test "shared expert 512 is shared_resident" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    try require(&placed, "model.layers.0.mlp.shared_expert.512.up_proj.trellis", .shared_resident, .i16, 16, 32);
}

test "rank-2 n-gram trellis is ngram_host and rank-3 is not" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    try require(&placed, "ple.ngram_embedding.shard_0.trellis", .ngram_host, .i16, 32, 48);
    const wide = placed.get("ple.ngram_embedding.shard_1.trellis").?;
    try std.testing.expect(wide.home != .ngram_host);
    try std.testing.expectEqual(place.Home.plain, wide.home);
    try std.testing.expectEqual(@as(u64, 48), wide.begin);
    try std.testing.expectEqual(@as(u64, 64), wide.end);
}

test "centred RMSNorm weights are centred_norm" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    // place.home returns the tag. The offset stays scan.centredOffset; it is not stored here.
    try require(&placed, "model.layers.0.self_attn.q_norm.weight", .centred_norm, .f16, 64, 80);
    try require(&placed, "model.layers.0.self_attn.k_norm.weight", .centred_norm, .f16, 80, 88);
    try require(&placed, "model.layers.0.attn_hyper_connection.hc_norm.weight", .centred_norm, .bf16, 88, 96);
}

test "mtp.pre_fc_norm_embedding in the mixer patch is mtp_patch" {
    const gpa = std.testing.allocator;
    const json = "{\"mtp.pre_fc_norm_embedding\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    const bytes = try synthetic(gpa, json, 8);
    defer gpa.free(bytes);
    const path = "/tmp/mtp_hyper_connection_mixer_patch.safetensors";
    var from_header = try loadHeader(gpa, bytes, path);
    defer from_header.deinit();
    try require(&from_header, "mtp.pre_fc_norm_embedding", .mtp_patch, .f16, 0, 8);

    const io = std.testing.io;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    var from_file = try open(gpa, io, path);
    defer from_file.deinit();
    try require(&from_file, "mtp.pre_fc_norm_embedding", .mtp_patch, .f16, 0, 8);
}

test "a vision encoder tensor is vision_skip" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    try require(&placed, "model.visual.blocks.0.weight", .vision_skip, .bf16, 96, 112);
    try std.testing.expect(placed.get("__metadata__") == null);
}

test "a bad header returns the safetensors error, not a placement" {
    const gpa = std.testing.allocator;
    const past = try synthetic(gpa, "{\"t\":{\"dtype\":\"I16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}", 4);
    defer gpa.free(past);
    try std.testing.expectError(error.BadSafetensors, loadHeader(gpa, past, "model.safetensors"));

    const mismatch = try synthetic(gpa, "{\"t\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,4]}}", 8);
    defer gpa.free(mismatch);
    try std.testing.expectError(error.BadSafetensors, loadHeader(gpa, mismatch, "model.safetensors"));

    var short: [8]u8 = undefined;
    std.mem.writeInt(u64, &short, 64, .little);
    try std.testing.expectError(error.BadSafetensors, loadHeader(gpa, &short, "model.safetensors"));
}

test "device weights skip the routed bank, the n-gram table, and vision" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    // shared 16 and three centred norms 16+8+8. The rank-3 n-gram name is plain and stays off the card.
    try std.testing.expectEqual(@as(u64, 48), residentBytes(placed.tensors));
    try std.testing.expectEqual(@as(u64, 48 + 32), try deviceWeights(placed.tensors, 32));
    try std.testing.expect(residentBytes(placed.tensors) < sample_data);
}

test "opening records offsets and does not allocate tensor payloads" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try synthetic(gpa, sample_json, sample_data);
    defer gpa.free(bytes);
    const path = "/tmp/tensorfold-exl3-flashnext-load.safetensors";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var from_header = try loadHeader(gpa, bytes, path);
    defer from_header.deinit();
    var from_file = try open(gpa, io, path);
    defer from_file.deinit();

    try std.testing.expectEqual(from_header.tensors.len, from_file.tensors.len);
    try std.testing.expectEqual(@as(usize, 8), from_file.tensors.len);
    for (from_header.tensors, from_file.tensors) |header_tensor, file_tensor| {
        try std.testing.expectEqualStrings(header_tensor.name, file_tensor.name);
        try std.testing.expectEqual(header_tensor.home, file_tensor.home);
        try std.testing.expectEqual(header_tensor.dtype, file_tensor.dtype);
        try std.testing.expectEqual(header_tensor.begin, file_tensor.begin);
        try std.testing.expectEqual(header_tensor.end, file_tensor.end);
    }
    // Names and offsets only. A copied expert bank or n-gram table would not fit in a megabyte.
    try std.testing.expect(from_header.arena.queryCapacity() < (1 << 20));
    try std.testing.expect(from_file.arena.queryCapacity() < (1 << 20));
}

fn hdr(name: []const u8, dims: []const usize, end: u64) TensorHeader {
    var h: TensorHeader = .{ .name = name, .ndims = dims.len, .begin = 0, .end = end };
    for (dims, 0..) |d, i| {
        if (i < h.shape.len) h.shape[i] = d;
    }
    return h;
}

test "pinRank splits legal q and hc axes and replicates the router, ple pair, and shared gate" {
    const q = hdr("model.layers.0.self_attn.q_proj.weight", &.{ 6144, 2560 }, 400);
    const q_suh = hdr("model.layers.0.self_attn.q_proj.suh", &.{2560}, 80);
    const q_svh = hdr("model.layers.0.self_attn.q_proj.svh", &.{6144}, 80);
    const q_odd = hdr("model.layers.0.self_attn.q_proj.weight", &.{ 100, 2560 }, 400);
    const gate = hdr("model.language_model.layers.0.mlp.gate.weight", &.{ 512, 2560 }, 2_621_440);
    const shared_gate = hdr("model.language_model.layers.0.mlp.shared_expert_gate.weight", &.{ 1, 2560 }, 7);
    const shared_up = hdr("model.layers.0.mlp.shared_expert.up_proj.weight", &.{ 640, 2560 }, 400);
    const key = hdr("model.language_model.layers.1.ple.key_proj.weight", &.{ 10240, 2560 }, 500);
    const value = hdr("model.language_model.layers.1.ple.value_proj.weight", &.{ 2560, 2560 }, 300);
    const conv = hdr("model.layers.0.linear_attn.conv1d.weight", &.{ 10240, 4 }, 200);
    const a_log = hdr("model.layers.0.linear_attn.A_log", &.{48}, 96);
    const dt = hdr("model.layers.0.linear_attn.dt_bias", &.{48}, 96);
    const norm = hdr("model.layers.0.self_attn.q_norm.weight", &.{256}, 64);
    const hc = hdr("model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", &.{ 320, 10240 }, 400);
    const inject = hdr("model.layers.0.mlp_hyper_connection.block_inject_weight.weight", &.{ 4, 10240 }, 80);
    const in_legal = hdr("model.layers.0.linear_attn.in_proj_qkv.weight", &.{ 10240, 2560 }, 200);
    const in_full = hdr("model.layers.0.linear_attn.in_proj_a.weight", &.{ 48, 2560 }, 200);
    const out_legal = hdr("model.layers.0.linear_attn.out_proj.weight", &.{ 2560, 6144 }, 200);
    const out_full = hdr("model.layers.0.linear_attn.out_proj.weight", &.{ 2560, 48 }, 200);
    const index_full = hdr("model.layers.3.self_attn.indexer.index_qk_proj.weight", &.{ 640, 2560 }, 200);
    const index_split = hdr("model.layers.3.self_attn.indexer.index_qk_proj.weight", &.{ 512, 2560 }, 200);
    var mix = hdr("weight", &.{4}, 24);
    mix.file = "mtp_hyper_connection_mixer_patch.safetensors";

    const headers = [_]TensorHeader{
        q, q_suh, q_svh, gate, shared_gate, shared_up, key, value, conv, a_log, dt, norm, hc, inject, in_legal, in_full, out_legal, out_full, index_full, index_split, mix,
    };
    // q 200 + suh 80 + svh 40 + gate full + shared gate 7 + shared up full + ple full + conv/a/dt/norm full + hc half + inject full + legal gdn/indexer halves + the mixer span.
    const expect: u64 = 200 + 80 + 40 + 2_621_440 + 7 + 400 + 500 + 300 + 200 + 96 + 96 + 64 + 200 + 80 + 100 + 200 + 100 + 200 + 200 + 100 + 24;
    try std.testing.expectEqual(expect, try pinRank(&headers, 0, 2));
    try std.testing.expectEqual(expect, try pinRank(&headers, 1, 2));
    try std.testing.expectEqual(@as(u64, 400), try pinRank(&.{q_odd}, 0, 2));
    try std.testing.expectEqual(@as(u64, 400), try pinRank(&.{q}, 0, 1));
    // No mixer header: do not invent a size, and do not halve a class total.
    try std.testing.expectEqual(@as(u64, 200), try pinRank(&.{q}, 0, 2));
    try std.testing.expect((try pinRank(&.{q}, 0, 2)) != @as(u64, 3_060_807_864) / 2);

    const routed = hdr("model.layers.0.mlp.experts.0.gate_proj.trellis", &.{ 160, 40, 48 }, 999);
    const ngram = hdr("ple.ngram_embedding.shard_0.trellis", &.{ 4, 2 }, 999);
    const vision = hdr("model.visual.blocks.0.weight", &.{ 2, 4 }, 16);
    const embed = hdr("model.embed_tokens.weight", &.{ 248320, 2560 }, 999);
    try std.testing.expectEqual(@as(u64, 0), try pinRank(&.{ routed, ngram, vision, embed }, 0, 2));

    try std.testing.expectError(error.IllegalCut, pinRank(&.{q}, 0, 0));
    try std.testing.expectError(error.IllegalCut, pinRank(&.{q}, 0, 4));
}

test "a router layer without mlp.shared_expert_gate fails the launch" {
    const gate = hdr("model.language_model.layers.0.mlp.gate.weight", &.{ 512, 2560 }, 2_621_440);
    try std.testing.expectError(error.MissingSharedGate, requireSharedGate(&.{gate}));
    const shared = hdr("model.language_model.layers.0.mlp.shared_expert_gate.weight", &.{ 1, 2560 }, 7);
    try requireSharedGate(&.{ gate, shared });
    const other = hdr("model.language_model.layers.1.mlp.gate.weight", &.{ 512, 2560 }, 2_621_440);
    try std.testing.expectError(error.MissingSharedGate, requireSharedGate(&.{ gate, shared, other }));
    try std.testing.expectEqual(@as(u64, 7), try pinRank(&.{shared}, 0, 2));
    try std.testing.expectEqual(@as(u64, 2_621_440), try pinRank(&.{gate}, 1, 2));
}

test "an mtp. prefix does not pair with the decoder shared gate" {
    const mtp_gate = hdr("mtp.layers.0.mlp.gate", &.{ 512, 2560 }, 2_621_440);
    const dec_shared = hdr("model.layers.0.mlp.shared_expert_gate", &.{ 1, 2560 }, 7);
    try std.testing.expectError(error.MissingSharedGate, requireSharedGate(&.{ mtp_gate, dec_shared }));

    const dec_gate = hdr("model.layers.0.mlp.gate", &.{ 512, 2560 }, 2_621_440);
    const mtp_shared = hdr("mtp.layers.0.mlp.shared_expert_gate", &.{ 1, 2560 }, 7);
    try std.testing.expectError(error.MissingSharedGate, requireSharedGate(&.{ dec_gate, mtp_shared }));

    try requireSharedGate(&.{ mtp_gate, mtp_shared });
    const lm_gate = hdr("language_model.mtp.layers.0.mlp.gate", &.{ 512, 2560 }, 2_621_440);
    const lm_shared = hdr("language_model.mtp.layers.0.mlp.shared_expert_gate", &.{ 1, 2560 }, 7);
    try requireSharedGate(&.{ lm_gate, lm_shared });
    try std.testing.expectEqual(@as(u64, 2_621_440), try pinRank(&.{mtp_gate}, 0, 2));

    const switch_gate = hdr("model.layers.0.mlp.switch_mlp.gate_proj.weight", &.{ 640, 2560 }, 400);
    try std.testing.expectEqual(@as(u64, 0), try pinRank(&.{switch_gate}, 0, 2));
}

test "the routed bank is not a device weight and a flat 16 GiB cache on a pin is refused" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    try std.testing.expectError(error.ExceedsDevice, deviceWeights(&.{}, 46_724_192_256));
    try std.testing.expectError(error.ExceedsDevice, deviceWeights(placed.tensors, 16 << 30));
    const pin = residentBytes(placed.tensors);
    try std.testing.expect(pin != 0);
    try std.testing.expectError(error.ExceedsDevice, learnedLines(pin, 0, 16 << 30, &.{cache.full_expert_bytes}));
}

test "cache_bytes 0 uses learnedBytes and the header stride" {
    const pin: u64 = 1_000_000;
    const kv: u64 = 100_663_296;
    const room = cache.learnedBytes(cache.card_bytes, pin, kv);
    const small = [_]u64{ 100, cache.full_expert_bytes };
    try std.testing.expectEqual(cache.expertLinesOf(room, cache.full_expert_bytes), try learnedLines(pin, kv, 0, &small));
    try std.testing.expectEqual(cache.expertLinesOf(room, cache.full_expert_bytes), try learnedLines(pin, kv, 0, &.{}));
    const big = cache.full_expert_bytes + 22;
    const stride = (big + 255) & ~@as(u64, 255);
    try std.testing.expect(stride > cache.full_expert_bytes);
    try std.testing.expect(stride != cache.expertBytes(2));
    try std.testing.expectEqual(cache.expertLinesOf(room, stride), try learnedLines(pin, kv, 0, &.{ big, 10 }));
    const wider = cache.full_expert_bytes * 2;
    const wide_stride = (wider + 255) & ~@as(u64, 255);
    const lines = try learnedLines(pin, kv, 0, &.{wider});
    try std.testing.expectEqual(cache.expertLinesOf(room, wide_stride), lines);
    try std.testing.expect(lines < cache.expertLinesOf(room, cache.full_expert_bytes));
    try std.testing.expect(lines != cache.expertLinesOf(room, cache.expertBytes(2)));
}

test "vision tensors are classified and not mapped" {
    const gpa = std.testing.allocator;
    var placed = try loadSample(gpa);
    defer placed.deinit();
    const vision = placed.get("model.visual.blocks.0.weight").?;
    try std.testing.expectEqual(place.Home.vision_skip, vision.home);
    const headers = [_]TensorHeader{.{
        .name = vision.name,
        .ndims = 2,
        .shape = .{ 2, 4, 0, 0, 0 },
        .begin = vision.begin,
        .end = vision.end,
    }};
    try std.testing.expectEqual(@as(u64, 0), try pinRank(&headers, 0, 2));
    try std.testing.expect(residentBytes(placed.tensors) + (vision.end - vision.begin) > residentBytes(placed.tensors));
    // The placement records the name and the offsets. It does not copy the payload.
    try std.testing.expect(placed.arena.queryCapacity() < (1 << 20));
}
