//! A contiguous run of a model's layers, as one stage of a layer-split model runs it. Half open: `[first, last)`.
const std = @import("std");

pub const Range = struct {
    first: usize,
    last: usize,

    pub fn whole(layers: usize) Range {
        return .{ .first = 0, .last = layers };
    }

    pub fn len(r: Range) usize {
        return r.last - r.first;
    }

    /// The side that holds the first layer embeds: it reads token ids, the others read a hidden row.
    pub fn embed(r: Range) bool {
        return r.first == 0;
    }

    /// The side that holds the last layer applies the final norm, the head and the draw.
    pub fn head(r: Range, layers: usize) bool {
        return r.last == layers;
    }

    pub fn owns(r: Range, layer: usize) bool {
        return layer >= r.first and layer < r.last;
    }

    /// A range is non-empty and inside a model of `layers` layers.
    pub fn check(r: Range, layers: usize) !void {
        if (r.first >= r.last or r.last > layers) return error.BadLayerRange;
    }
};

test "a range's ends decide who embeds and who draws" {
    const r: Range = .{ .first = 0, .last = 5 };
    try std.testing.expect(r.embed() and !r.head(32) and r.owns(4) and !r.owns(5) and r.len() == 5);
    try std.testing.expect(!(Range{ .first = 5, .last = 32 }).embed() and (Range{ .first = 5, .last = 32 }).head(32));
    const all = Range.whole(32);
    try std.testing.expect(all.embed() and all.head(32));
    try all.check(32);
    try std.testing.expectError(error.BadLayerRange, (Range{ .first = 3, .last = 3 }).check(32));
    try std.testing.expectError(error.BadLayerRange, (Range{ .first = 3, .last = 33 }).check(32));
}
