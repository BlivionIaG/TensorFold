const std = @import("std");
const family = @import("family");
const affine = family.affine;
const c = family.spec;
const rows = family.rows;

fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / 32 + 8;
}

test "fixture2 full width group seed matrix" {
    const a = std.testing.allocator;
    var cases: usize = 0;
    var minimum = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| {
        for ([_]usize{ 32, 64, 128 }) |group| {
            const k: usize = if (bits == 2) 2048 else if (bits == 4 and group == 64) 2688 else 512;
            // Two-bit columns are identical by construction, so one column proves all 4096 served columns.
            const n: usize = if (bits == 2) 1 else if (bits == 4 and group == 64) 1856 else 64;
            // Cover every actual runner position (indices 10..49) plus 128 unrelated broad-range seeds.
            for (0..168) |seed_index| {
                const seed: u64 = if (seed_index < 40) 0x6b63 + 0x9e37 * (10 + seed_index)
                    else 0x100000 + 0x12345 * (seed_index - 40);
                var f = try affine.Affine.init(a, .{ .rows = 1, .n = n, .k = k }, bits, group);
                defer f.deinit();
                try f.fill(.stress, seed);
                if (bits == 2) for (0..k) |i| {
                    const nudge = i >= k / 2 and i < k / 2 + 8;
                    f.x[i] = c.toBf16(if (nudge) 256 else 255);
                };
                const want = try a.alloc(f64, n);
                defer a.free(want);
                const mag = try a.alloc(f64, n);
                defer a.free(mag);
                const got = try a.alloc(f64, n);
                defer a.free(got);
                try f.reference(want, mag);
                if (bits == 2) {
                    try std.testing.expectEqual(@as(f64, -8), want[0]);
                    for (f.codes, 0..) |q, i| try std.testing.expectEqual(@as(u8, if (i < k / 2) 3 else 1), q);
                }
                const spec: c.Spec = .{ .name = "stream", .accum = .f32, .depth = depth, .out = .f32, .served = &.{} };
                var margin = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
                for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
                    try f.simulate(.f32, order, got);
                    try std.testing.expectEqualSlices(f64, want, got);
                    for ([_]c.Accum{ .f16, .bf16 }, &margin) |acc, *m| {
                        try f.simulate(acc, order, got);
                        var most: f64 = 0;
                        for (got, want) |v, w| most = @max(most, if (std.math.isFinite(v))
                            @abs(v - w) / c.boundExact(spec, w) else std.math.inf(f64));
                        m.* = @min(m.*, most);
                    }
                }
                if (!(margin[0] >= 100 and margin[1] >= 100)) {
                    std.debug.print("FAIL bits {d} group {d} seed {x}: {d} {d}\n", .{ bits, group, seed, margin[0], margin[1] });
                    return error.WeakStressFixture;
                }
                minimum[0] = @min(minimum[0], margin[0]);
                minimum[1] = @min(minimum[1], margin[1]);
                cases += 1;
            }
            std.debug.print("PASS bits {d} group {d} K{d}: 168 seeds, all f32 replays exact\n", .{ bits, group, k });
        }
    }
    std.debug.print("FIXTURE2_PASS cases={d} min_f16={d} min_bf16={d}\n", .{ cases, minimum[0], minimum[1] });
}
