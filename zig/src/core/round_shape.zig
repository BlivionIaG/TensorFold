//! What a lane round's launches depend on, for a backend that replays rounds from captured graphs: its rows padded to a
//! bucket, its slots and the keys the attention walk covers. A graph captured for a shape serves any streams of it.

const std = @import("std");

/// Rows a round runs are padded up to one of these (and the limit), so the shapes stay few.
pub const buckets = [_]usize{ 1, 2, 4, 8, 16, 32, 64 };

/// Shortest span the attention walk covers; spans double from here.
pub const min_span = 128;

/// The rows a round of `rows` real rows runs: the smallest bucket that holds them, the limit past the last bucket.
pub fn bucketOf(rows: usize, limit: usize) usize {
    for (buckets) |b| if (b >= rows) return @min(b, limit);
    return limit;
}

/// The keys the attention walk covers when the longest row sees `visible`: min_span doubled until it holds them.
pub fn spanOf(visible: usize) usize {
    var span: usize = min_span;
    while (span < visible) span *= 2;
    return span;
}

/// What a round's launches depend on: the rows it runs (padding included), its slots (the streams', empty ones up to a
/// bucket, then the scratch slot that holds the padding rows) and the keys the walk covers.
pub const Shape = struct {
    rows: u32,
    slots: u32,
    span: u32,
};

test "buckets round up and the span doubles" {
    try std.testing.expectEqual(@as(usize, 4), bucketOf(3, 64));
    try std.testing.expectEqual(@as(usize, 32), bucketOf(17, 32));
    try std.testing.expectEqual(@as(usize, 6), bucketOf(5, 6));
    try std.testing.expectEqual(@as(usize, 128), spanOf(1));
    try std.testing.expectEqual(@as(usize, 512), spanOf(300));
}
