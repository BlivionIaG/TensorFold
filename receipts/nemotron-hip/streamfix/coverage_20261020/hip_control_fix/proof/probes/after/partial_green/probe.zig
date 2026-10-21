const std = @import("std");
const gate = @import("contract_limits.zig");
test "table enforces every control chip" { const n = try gate.table(std.testing.allocator); try std.testing.expect(n > 0); }
test "objects enforces control chip, usage and metadata" { const n = try gate.objects(std.testing.allocator, std.testing.io, "<PROOF_ROOT>/fixtures/partial_green"); try std.testing.expect(n > 0); }
