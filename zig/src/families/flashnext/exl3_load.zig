//! Host placement of one EXL3 safetensors shard.
//! The header is walked; tensor payloads are not copied, so an n-gram table or an expert bank is not allocated.
//! Homes come from `place.home`, which calls `scan.classifyName`. This file does not restate those rules.
//! `centred_norm` is that tag alone. The fp32 add is `scan.centredOffset`, applied when the gamma bytes are read.

const std = @import("std");
const scan = @import("exl3_scan.zig");
const place = @import("exl3_place.zig");
const st = @import("safetensors.zig");

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
pub fn deviceWeights(tensors: []const Tensor, cache_bytes: u64) u64 {
    return residentBytes(tensors) + cache_bytes;
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
    try std.testing.expectEqual(@as(u64, 48 + 32), deviceWeights(placed.tensors, 32));
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
