//! What two CUDA ranks must agree on before they replay each other's operations.
const std = @import("std");
const kvl = @import("cuda_kv_lanes.zig");

pub const Identity = extern struct {
    config: [32]u8,
    index: [32]u8,
    program: [32]u8,
    checkpoint: [32]u8,
    context: u64,
    chunk: u64,
    decode_graph: u64 = 0,
    mtp: u64 = 0,
    head_sharded: u64 = 0,
    image_tokens: u64 = 0,
    max_images: u64 = 0,
    /// Positions a page of the core KV pool holds on this rank (0: no pool), and whether every stream is on pages.
    kv_pages: u64 = 0,
    kv_pages_all: u64 = 0,
    /// The elastic K/V arena's knob (cuda_kv_elastic.knob): 0 off, else 1 and the minimum pages named; ranks match.
    kv_elastic: u64 = 0,
};

fn fileHash(a: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    defer a.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

/// The checkpoint's bytes as both ranks see them: TENSORFOLD_CHECKPOINT_SHA256 when set, else the cached full digest.
pub fn identity(a: std.mem.Allocator, io: std.Io, dir: []const u8, context: usize, chunk: usize) !Identity {
    var checkpoint: [32]u8 = undefined;
    if (std.c.getenv("TENSORFOLD_CHECKPOINT_SHA256")) |given| {
        const digest = std.mem.span(given);
        if (digest.len != 64) return error.NoCheckpointIdentity;
        _ = std.fmt.hexToBytes(&checkpoint, digest) catch return error.NoCheckpointIdentity;
    } else checkpoint = try @import("core").checkpoint_identity.fullIdentity(a, io, dir);
    const config = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(config);
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(index);
    const program = try std.process.executablePathAlloc(io, a);
    defer a.free(program);
    return .{ .config = try fileHash(a, io, config), .index = try fileHash(a, io, index), .program = try fileHash(a, io, program), .checkpoint = checkpoint, .context = context, .chunk = chunk, .decode_graph = @intFromBool(@import("cuda_round_graph.zig").enabled()), .head_sharded = @intFromBool(try @import("cuda_head.zig").enabled()), .kv_pages = try kvl.paging(), .kv_pages_all = @intFromBool(kvl.pagedOnly()), .kv_elastic = try @import("cuda_kv_elastic.zig").knob() };
}
