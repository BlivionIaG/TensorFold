//! Copies on an integrated GPU: blit kernels, as its SDMA queues stall (gfx1033: host-to-device copies stop retiring).

const std = @import("std");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// The value of `key` in a KFD node's properties (one `key value` a line), or null.
fn property(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const sp = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
        if (!std.mem.eql(u8, line[0..sp], key)) continue;
        return std.fmt.parseInt(u64, std.mem.trim(u8, line[sp + 1 ..], " \r"), 10) catch null;
    }
    return null;
}

/// Whether a KFD node is a GPU (a gfx target) with no local memory of its own: an APU's, sharing the host's.
pub fn integratedNode(text: []const u8) bool {
    const gfx = property(text, "gfx_target_version") orelse return false;
    return gfx != 0 and (property(text, "local_mem_size") orelse return false) == 0;
}

/// HSA_ENABLE_SDMA=0 before HIP loads when the KFD topology has an integrated GPU, unless the variable is set.
pub fn preferBlit() void {
    if (std.c.getenv("HSA_ENABLE_SDMA") != null) return;
    var path_buf: [96]u8 = undefined;
    var text: [4096]u8 = undefined;
    for (0..64) |node| {
        const path = std.mem.printSentinel(&path_buf, "/sys/class/kfd/kfd/topology/nodes/{d}/properties", .{node}, 0) catch return;
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return;
        const n = std.c.read(fd, &text, text.len);
        _ = std.c.close(fd);
        if (n > 0 and integratedNode(text[0..@intCast(n)])) {
            _ = setenv("HSA_ENABLE_SDMA", "0", 0);
            return;
        }
    }
}

test "a GPU node without local memory is integrated; a CPU node and a card with VRAM are not" {
    const deck = "cpu_cores_count 0\nsimd_count 16\ngfx_target_version 100303\nlocal_mem_size 0\nnum_sdma_engines 1\n";
    const card = "cpu_cores_count 0\nsimd_count 120\ngfx_target_version 110000\nlocal_mem_size 51522830336\n";
    const cpu = "cpu_cores_count 8\nsimd_count 0\ngfx_target_version 0\nlocal_mem_size 0\n";
    try std.testing.expect(integratedNode(deck));
    try std.testing.expect(!integratedNode(card));
    try std.testing.expect(!integratedNode(cpu));
    try std.testing.expect(!integratedNode("gfx_target_version 100303\n"));
}
