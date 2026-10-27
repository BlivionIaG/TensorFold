//! Host tests of the KV accounting: the pieces of a sequence's bytes.

const std = @import("std");
const hub = @import("cuda_exl3_kv.zig");

const kvRowBytes = hub.kvRowBytes;
const seqBytes = hub.seqBytes;

test "exl3: kv the default policy is upstream's seqBytes, split" {
    const prompt = @import("../../cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap, .{});
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap)), s.device());
    }
}

test "exl3: kv a token's K and V are upstream's row twice; a head split halves K/V, not the indexer" {
    const prompt = @import("../../cuda_prompt.zig");
    try std.testing.expectEqual(@as(usize, 2048), kvRowBytes(2));
    try std.testing.expectEqual(2 * prompt.kv_row_bytes, kvRowBytes(2));
    try std.testing.expectEqual(@as(usize, 1024), kvRowBytes(1));
    const cap = 200_000;
    const a = seqBytes(cap, .{});
    const b = seqBytes(cap, .{ .heads = 1 });
    try std.testing.expectEqual(a.kv, 2 * b.kv);
    try std.testing.expectEqual(a.ikc, b.ikc);
}
