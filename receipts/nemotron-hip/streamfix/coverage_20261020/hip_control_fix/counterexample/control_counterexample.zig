const std = @import("std");
const records = @import("hip_records.zig");
const limits = @import("limits.zig");
test "one chip loses bound plant: aggregate passes but per-chip gate rejects" {
    const a = std.testing.allocator;
    var changed = records.controls[0];
    changed.usage[0].bound = changed.launch.threads;
    const aggregate = try records.check(a, &.{changed}, true);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var green: usize = 0;
    for (limits.rdna, changed.usage) |chip, usage| {
        const count = try limits.hip(changed.launch, usage, chip, &out, a);
        if (count == 0) green += 1;
        std.debug.print("{s}: {d} planted limit violations\n", .{chip.name, count});
    }
    try std.testing.expectEqual(@as(usize, 3), aggregate);
    try std.testing.expectEqual(@as(usize, 1), green);
    const shipped_table_accepts_control = aggregate > 0;
    const shipped_objects_accepts_control = aggregate != 0;
    const proposed_per_chip_accepts_control = green == 0;
    try std.testing.expect(shipped_table_accepts_control and shipped_objects_accepts_control);
    try std.testing.expect(!proposed_per_chip_accepts_control);
    std.debug.print("COUNTEREXAMPLE: shipped aggregate accepts; one gfx1030 plant green; per-chip requirement rejects\n", .{});
}
