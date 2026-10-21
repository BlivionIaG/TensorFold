const std = @import("std");
const records = @import("hip_records.zig");
test "a control must fail when any chip fits" {
    var changed = records.controls[0];
    changed.usage[0].bound = changed.launch.threads;
    const accepted = try records.check(std.testing.allocator, &.{changed}, true) > 0;
    try std.testing.expect(!accepted);
}
