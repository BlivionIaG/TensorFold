//! The expert offload's host tests, device-free: `zig build test-expert-offload-host` runs this file alone.
test {
    _ = @import("expert_offload.zig");
    _ = @import("expert_offload_test.zig");
    _ = @import("expert_window.zig");
    _ = @import("expert_stats.zig");
    _ = @import("expert_tier_test.zig");
    _ = @import("expert_service_test.zig");
}
