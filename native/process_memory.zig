const std = @import("std");

// Darwin sys/resource.h rusage_info_v4, queried through libproc.
const Usage = extern struct {
    uuid: [16]u8,
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
    cpu_time_qos_default: u64,
    cpu_time_qos_maintenance: u64,
    cpu_time_qos_background: u64,
    cpu_time_qos_utility: u64,
    cpu_time_qos_legacy: u64,
    cpu_time_qos_user_initiated: u64,
    cpu_time_qos_user_interactive: u64,
    billed_system_time: u64,
    serviced_system_time: u64,
    logical_writes: u64,
    lifetime_max_phys_footprint: u64,
    instructions: u64,
    cycles: u64,
    billed_energy: u64,
    serviced_energy: u64,
    interval_max_phys_footprint: u64,
    runnable_time: u64,
};

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *Usage) c_int;

pub const Snapshot = struct {
    rss_bytes: u64,
    footprint_bytes: u64,
    peak_footprint_bytes: u64,
    peak_rss_bytes: ?u64 = null,

    pub fn read(pid: c_int) !Snapshot {
        var usage: Usage = undefined;
        if (proc_pid_rusage(pid, 4, &usage) != 0) return error.ProcessMemoryUnavailable;
        return .{ .rss_bytes = usage.resident_size, .footprint_bytes = usage.phys_footprint, .peak_footprint_bytes = usage.lifetime_max_phys_footprint };
    }

    pub fn current() !Snapshot {
        var snapshot = try read(std.c.getpid());
        snapshot.peak_rss_bytes = @intCast(@max(0, std.posix.getrusage(std.posix.rusage.SELF).maxrss));
        return snapshot;
    }
};

test "Darwin process counters include resident pages and lifetime footprint" {
    const usage = try Snapshot.current();
    try std.testing.expect(usage.rss_bytes > 0);
    try std.testing.expect(usage.footprint_bytes > 0);
    try std.testing.expect(usage.peak_footprint_bytes >= usage.footprint_bytes);
    try std.testing.expect(usage.peak_rss_bytes.? > 0);
}
