const std = @import("std");

pub const experts = 512;

pub const Demand = struct {
    counts: []const u32,
    line_cost: u64,
};

/// Allocate a fixed byte budget to per-layer expert pools. Each extra line is
/// assigned to the layer whose next hottest expert yields the most historical
/// requests per byte. Ties prefer the layer with fewer lines, then lower index.
pub fn allocate(gpa: std.mem.Allocator, demands: []const Demand, budget: u64, minimum: usize) ![]usize {
    if (demands.len == 0 or minimum == 0 or minimum > experts) return error.Invalid;
    var lines = try gpa.alloc(usize, demands.len);
    errdefer gpa.free(lines);
    @memset(lines, minimum);
    var ordered = try gpa.alloc(u16, demands.len * experts);
    defer gpa.free(ordered);
    var spent: u64 = 0;
    for (demands, 0..) |demand, i| {
        if (demand.counts.len != experts or demand.line_cost == 0) return error.Invalid;
        spent = try std.math.add(u64, spent, try std.math.mul(u64, minimum, demand.line_cost));
        const ids = ordered[i * experts ..][0..experts];
        for (ids, 0..) |*id, expert| id.* = @intCast(expert);
        std.mem.sort(u16, ids, demand.counts, struct {
            fn lessThan(counts: []const u32, a: u16, b: u16) bool {
                if (counts[a] != counts[b]) return counts[a] > counts[b];
                return a < b;
            }
        }.lessThan);
    }
    if (spent > budget) return error.InsufficientBudget;

    while (true) {
        var best: ?usize = null;
        for (demands, 0..) |demand, i| {
            if (lines[i] == experts) continue;
            if (demand.line_cost > budget - spent) continue;
            if (best) |current| {
                const candidate_count = demands[i].counts[ordered[i * experts + lines[i]]];
                const current_count = demands[current].counts[ordered[current * experts + lines[current]]];
                const candidate_score = @as(u64, candidate_count) * demands[current].line_cost;
                const current_score = @as(u64, current_count) * demand.line_cost;
                if (candidate_score > current_score or
                    (candidate_score == current_score and lines[i] < lines[current])) best = i;
            } else best = i;
        }
        const i = best orelse break;
        const cost = demands[i].line_cost;
        lines[i] += 1;
        spent += cost;
    }
    return lines;
}

test "exl3: line allocation puts additional capacity where profile predicts hits" {
    var hot: [experts]u32 = @splat(0);
    var even: [experts]u32 = @splat(100);
    for (0..10) |expert| hot[expert] = @intCast(20_000 - expert * 1_000);
    hot[10] = 9_000;
    hot[11] = 8_000;
    hot[12] = 7_000;
    hot[13] = 6_000;
    even[0] = 101;
    const demands = [_]Demand{ .{ .counts = &hot, .line_cost = 10 }, .{ .counts = &even, .line_cost = 10 } };
    const lines = try allocate(std.testing.allocator, &demands, 24 * 10, 10);
    defer std.testing.allocator.free(lines);
    try std.testing.expectEqual(@as(usize, 14), lines[0]);
    try std.testing.expectEqual(@as(usize, 10), lines[1]);
}

test "exl3: line allocation accounts for pool-specific line costs" {
    const counts: [experts]u32 = @splat(1);
    const demands = [_]Demand{ .{ .counts = &counts, .line_cost = 10 }, .{ .counts = &counts, .line_cost = 20 } };
    const lines = try allocate(std.testing.allocator, &demands, 2 * 10 + 2 * 20 + 30, 2);
    defer std.testing.allocator.free(lines);
    try std.testing.expectEqual(@as(usize, 5), lines[0]);
    try std.testing.expectEqual(@as(usize, 2), lines[1]);
}

test "exl3: line allocation rejects a budget below the minimum floor" {
    const counts: [experts]u32 = @splat(0);
    const demands = [_]Demand{.{ .counts = &counts, .line_cost = 20 }};
    try std.testing.expectError(error.InsufficientBudget, allocate(std.testing.allocator, &demands, 19, 1));
}
