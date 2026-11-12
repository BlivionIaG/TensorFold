//! The Sparks' boot budget at --context 1048576: the long-lane window's rank 1 figures and what a 1M pool costs.
const std = @import("std");
const budget = @import("budget.zig");
const layout_mod = @import("layout.zig");

const GiB = budget.GiB;

test "a 1M-token pool is 1.0 GiB a rank split over two ranks, and four slots of 1M share 4 GiB, not 4 x 1M resident" {
    var sl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const spl = sl.pool();
    const one_m = @as(f64, @floatFromInt(budget.poolBytes(spl, 2, 1 << 20))) / GiB;
    try std.testing.expectApproxEqAbs(@as(f64, 1060.0 * (1 << 20) / GiB), one_m, 1e-9);
    try std.testing.expect(one_m > 1.03 and one_m < 1.04);
    // the old 503,296-token cap is 0.50 GiB, the window's 1.1M pool 1.09 GiB: 0.59 GiB more
    const old = @as(f64, @floatFromInt(budget.poolBytes(spl, 2, 503_296))) / GiB;
    const wide = @as(f64, @floatFromInt(budget.poolBytes(spl, 2, 1_100_032))) / GiB;
    try std.testing.expectApproxEqAbs(@as(f64, 0.50), old, 0.005);
    try std.testing.expectApproxEqAbs(@as(f64, 1.09), wide, 0.005);
    try std.testing.expectApproxEqAbs(@as(f64, 0.59), wide - old, 0.01);
    // four live slots of 1M each, all resident, would be 4.2 GiB a rank: one shared pool of 1.1M is the plan
    const four = @as(f64, @floatFromInt(budget.poolBytes(spl, 2, 4 << 20))) / GiB;
    try std.testing.expectApproxEqAbs(@as(f64, 4.14), four, 0.01);
}

test "rank 1's long-lane figures: 3.41 GiB at a 1.1M pool refuses, 4.29 GiB at 503k passes, and the pool needs 0.59 GiB more room" {
    var sl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const spl = sl.pool();
    const k: budget.Knobs = .{};
    // MemAvailable at the check 11.31 (1.1M run) / 11.59 (503k run), boot growth 3.82 / 3.81, the 3.0 GiB dip
    const wide = budget.afterBoot(11.31, 1.09, 11.31 - 1.09 - 3.82, 0, budget.zig_dip_gib, k.hard_gib, 1_100_032, budget.tokenBytes(spl, 2));
    try std.testing.expect(!wide.ok);
    try std.testing.expectApproxEqAbs(@as(f64, 3.41), wide.worst, 0.015);
    try std.testing.expectApproxEqAbs(@as(f64, 3.82), wide.growth, 0.015);
    const old = budget.afterBoot(11.59, 0.50, 11.59 - 0.50 - 3.81, 0, budget.zig_dip_gib, k.hard_gib, 503_296, budget.tokenBytes(spl, 2));
    try std.testing.expect(old.ok);
    try std.testing.expectApproxEqAbs(@as(f64, 4.29), old.worst, 0.015);
    // the planner's cap the window met: the pool the room allows is ~503k tokens, 0.59 GiB short of 1.1M
    try std.testing.expect(wide.largest > 480_000 and wide.largest < 540_000);
    const short = k.hard_gib - wide.worst;
    try std.testing.expectApproxEqAbs(@as(f64, 0.59), short, 0.015);
    // every GiB the boot gives back counts one for one: 0.6 GiB lifts the 1.1M pool over the floor
    const need = short + 0.01;
    try std.testing.expect(budget.afterBoot(11.31, 1.09, 11.31 - 1.09 - 3.82 + need, 0, budget.zig_dip_gib, k.hard_gib, 1_100_032, budget.tokenBytes(spl, 2)).ok);
}

test "the two Sparks' headroom gap: rank 1 sees 2.0 GiB less MemTotal than rank 0 before anything of ours runs" {
    // MemTotal 127,535,336 / 125,442,400 kB, kernel reserved 6,390,136K / 8,483,072K: firmware, not the engine
    const head_kb: f64 = 127_535_336;
    const worker_kb: f64 = 125_442_400;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), (head_kb - worker_kb) * 1024 / GiB, 0.01);
    const reserved_gap = (8_483_072.0 - 6_390_136.0) * 1024 / GiB;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), reserved_gap, 0.01);
    // the window's MemAvailable at the check: 13.22 (rank 0) and 11.59 (rank 1), and the boot growth 3.05 / 3.81
    try std.testing.expectApproxEqAbs(@as(f64, 1.63), 13.22 - 11.59, 0.005);
    try std.testing.expectApproxEqAbs(@as(f64, 0.76), 3.81 - 3.05, 0.005);
    // so the worst points differ by the sum: 6.66 - 4.29 = 2.37 GiB (rank 0 pool 0.50 GiB, rank 1 pool 0.50 GiB)
    try std.testing.expectApproxEqAbs(@as(f64, 2.37), (13.22 - 0.50 - 3.05 - 3.0) - (11.59 - 0.50 - 3.81 - 3.0), 0.03);
}
