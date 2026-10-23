const std = @import("std");

pub const Phase = enum { prefill, decode, verify, unknown };
pub const Rank = struct {
    rank: usize,
    world: usize,

    pub fn validate(owner: Rank) !void {
        if (owner.world == 0 or owner.rank >= owner.world) return error.TelemetryRank;
    }
};

var next_identity: std.atomic.Value(u64) = .init(0);

pub fn identity() !u64 {
    var previous = next_identity.load(.monotonic);
    while (true) {
        const next = try std.math.add(u64, previous, 1);
        previous = next_identity.cmpxchgWeak(previous, next, .monotonic, .monotonic) orelse return next;
    }
}

pub const Device = struct {
    plans: u64 = 0,
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    copies: u64 = 0,
    evictions: u64 = 0,
    inplace_experts: u64 = 0,
    overflow_plans: u64 = 0,

    pub fn pending(current: anytype, seen: @TypeOf(current)) Device {
        return .{ .plans = current.rounds -% seen.rounds, .hit_picks = current.hit_slots -% seen.hit_slots, .miss_picks = current.miss_slots -% seen.miss_slots, .copies = current.copies -% seen.copies, .evictions = current.evictions -% seen.evictions, .inplace_experts = current.inplace -% seen.inplace, .overflow_plans = current.overflow -% seen.overflow };
    }

    pub fn pendingBounded(current: anytype, seen: @TypeOf(current), max_picks: u64) !Device {
        const delta = pending(current, seen);
        const bound = try std.math.mul(u64, delta.plans, max_picks);
        if (max_picks == 0 or bound >= @as(u64, 1) << 32) return error.TelemetryExpertIntervalTooLong;
        if (try std.math.add(u64, delta.hit_picks, delta.miss_picks) > bound or try std.math.add(u64, delta.copies, delta.inplace_experts) > bound or delta.overflow_plans > delta.plans) return error.TelemetryInconsistentCounters;
        return delta;
    }

    pub fn record(total: *Device, valid: *bool, current: anytype, seen: @TypeOf(current), max_picks: u64) void {
        if (!valid.*) return;
        const delta = pendingBounded(current, seen, max_picks) catch {
            valid.* = false;
            return;
        };
        add(Device, total, delta) catch {
            valid.* = false;
        };
    }
};

pub fn add(comptime Counters: type, total: *Counters, contribution: Counters) !void {
    var result = total.*;
    inline for (@typeInfo(Counters).@"struct".field_names) |field| @field(result, field) = try std.math.add(u64, @field(result, field), @field(contribution, field));
    total.* = result;
}

pub fn difference(comptime Counters: type, before: Counters, after: Counters) !Counters {
    var result: Counters = undefined;
    inline for (@typeInfo(Counters).@"struct".field_names) |field| {
        if (@field(after, field) < @field(before, field)) return error.TelemetryCounterRegression;
        @field(result, field) = @field(after, field) - @field(before, field);
    }
    return result;
}

pub const Experts = struct {
    host_plans: u64 = 0,
    host_hit_picks: u64 = 0,
    host_miss_picks: u64 = 0,
    host_mandatory_copies: u64 = 0,
    host_prefetch_copies: u64 = 0,
    pool_dma_copies: u64 = 0,
    pool_dma_requested_bytes: u64 = 0,
    device_plans: u64 = 0,
    device_hit_picks: u64 = 0,
    device_miss_picks: u64 = 0,
    device_copies: u64 = 0,
    device_inplace_experts: u64 = 0,
    device_overflow_plans: u64 = 0,
    mapped_copy_payload_min_bytes: u64 = 0,
    mapped_copy_payload_max_bytes: u64 = 0,
    ring_dma_copies: u64 = 0,
    ring_dma_requested_bytes: u64 = 0,
    ring_unused_copies: u64 = 0,
    ring_hit_picks: u64 = 0,
    ring_miss_picks: u64 = 0,
    ring_hit_expert_requests: u64 = 0,
    ring_miss_expert_requests: u64 = 0,
    ring_demand_copies: u64 = 0,
    ring_demand_requested_bytes: u64 = 0,
    ring_demand_unused_copies: u64 = 0,
    ring_demand_unused_requested_bytes: u64 = 0,
    ring_initial_prefetch_copies: u64 = 0,
    ring_initial_prefetch_requested_bytes: u64 = 0,
    ring_initial_prefetch_unused_copies: u64 = 0,
    ring_initial_prefetch_unused_requested_bytes: u64 = 0,
    ring_cross_layer_prefetch_copies: u64 = 0,
    ring_cross_layer_prefetch_requested_bytes: u64 = 0,
    ring_cross_layer_prefetch_unused_copies: u64 = 0,
    ring_cross_layer_prefetch_unused_requested_bytes: u64 = 0,
};

pub const Kv = struct {
    block_requests: u64 = 0,
    block_misses: u64 = 0,
    sparse_calls: u64 = 0,
    logical_fetch_bytes: u64 = 0,
    rebases: u64 = 0,
    written_rows_per_side: u64 = 0,
};

pub const KvSample = struct {
    owner: u64,
    revision: u64,
    layer: usize,
    empty: bool,
    epoch: u32,
    clock: u32,
    requests: u32,
    misses: u32,
    calls: u32,
    overflow: bool,
    max_blocks: u64,
    block_bytes: u64,
    rebases: u64,
    written: u64,

    pub fn delta(before: KvSample, after: KvSample, allow_initial_reset: bool) !Kv {
        if (before.owner == 0 or before.owner != after.owner or before.layer != after.layer or before.max_blocks != after.max_blocks or before.block_bytes != after.block_bytes) return error.TelemetryOwnerChanged;
        if (before.overflow or after.overflow) return error.TelemetryKvOverflow;
        if (before.revision != after.revision) {
            if (!allow_initial_reset or !before.empty or before.epoch != 0 or before.requests != 0 or before.misses != 0 or before.calls != 0 or after.revision != try std.math.add(u64, before.revision, 1)) return error.TelemetryKvReset;
        }
        if (after.clock < before.clock or after.rebases < before.rebases or after.written < before.written) return error.TelemetryCounterRegression;
        const clock_delta = after.clock - before.clock;
        const bound = try std.math.mul(u64, clock_delta, after.max_blocks);
        if (bound >= @as(u64, 1) << 32) return error.TelemetryKvIntervalTooLong;
        const requests: u64 = after.requests -% before.requests;
        const misses: u64 = after.misses -% before.misses;
        const calls: u64 = after.calls -% before.calls;
        if (requests > bound or misses > requests or calls > clock_delta) return error.TelemetryInconsistentCounters;
        return .{ .block_requests = requests, .block_misses = misses, .sparse_calls = calls, .logical_fetch_bytes = try std.math.mul(u64, misses, after.block_bytes), .rebases = after.rebases - before.rebases, .written_rows_per_side = after.written - before.written };
    }
};

pub const RoutedPicks = struct {
    host_hit_picks: u64 = 0,
    host_miss_picks: u64 = 0,
    device_hit_picks: u64 = 0,
    device_miss_picks: u64 = 0,
};

pub const ExpertLayer = struct {
    layer: usize,
    owner: u64,
    picks: RoutedPicks,

    pub fn delta(before: ExpertLayer, after: ExpertLayer) !RoutedPicks {
        if (before.owner == 0 or before.owner != after.owner or before.layer != after.layer) return error.TelemetryOwnerChanged;
        const picks = try difference(RoutedPicks, before.picks, after.picks);
        _ = try std.math.add(u64, picks.host_hit_picks, picks.device_hit_picks);
        _ = try std.math.add(u64, picks.host_miss_picks, picks.device_miss_picks);
        return picks;
    }
};

pub const Snapshot = struct {
    rank: Rank,
    scope: u64,
    ring_owner: u64 = 0,
    experts: Experts = .{},
    kv: []KvSample,
    expert_pools: usize = 0,
    expert_layers: []ExpertLayer = &.{},

    pub fn deinit(snapshot: *Snapshot, allocator: std.mem.Allocator) void {
        allocator.free(snapshot.kv);
        allocator.free(snapshot.expert_layers);
        snapshot.* = undefined;
    }
};

pub const Interval = struct {
    phase: Phase,
    model_rows: u64,
    steps: u64,
    allow_initial_kv_reset: bool = false,
};

pub fn fraction(misses: u64, hits: u64) !?f64 {
    const denominator = try std.math.add(u64, misses, hits);
    return if (denominator == 0) null else @as(f64, @floatFromInt(misses)) / @as(f64, @floatFromInt(denominator));
}

pub const Report = struct {
    rank: Rank,
    interval: Interval,
    experts: Experts,
    kv: Kv,
    expert_pools: usize = 0,
    kv_layers: usize = 0,
    expert_layers_before: []const ExpertLayer = &.{},
    expert_layers_after: []const ExpertLayer = &.{},

    pub fn write(report: Report, writer: *std.Io.Writer) !void {
        try report.header(writer, "expert");
        inline for (@typeInfo(Experts).@"struct".field_names) |field| try writer.print(" {s}={d}", .{ field, @field(report.experts, field) });
        try writer.writeByte('\n');
        try report.header(writer, "bf16-kv");
        inline for (@typeInfo(Kv).@"struct".field_names) |field| try writer.print(" {s}={d}", .{ field, @field(report.kv, field) });
        try writer.writeByte('\n');
        for (report.expert_layers_before, report.expert_layers_after) |before, after| {
            const picks = try before.delta(after);
            try writer.print("EXL3-EXPERT-LAYER v=1 rank={d} world={d} phase={s} rows={d} steps={d} layer={d} hit_picks={d} miss_picks={d}", .{ report.rank.rank, report.rank.world, @tagName(report.interval.phase), report.interval.model_rows, report.interval.steps, after.layer, try std.math.add(u64, picks.host_hit_picks, picks.device_hit_picks), try std.math.add(u64, picks.host_miss_picks, picks.device_miss_picks) });
            inline for (@typeInfo(RoutedPicks).@"struct".field_names) |field| try writer.print(" {s}={d}", .{ field, @field(picks, field) });
            try writer.writeByte('\n');
        }
    }

    fn header(report: Report, writer: *std.Io.Writer, cache: []const u8) !void {
        try writer.print("EXL3-CACHE v=1 rank={d} world={d} phase={s} rows={d} steps={d} expert_pools={d} kv_layers={d} cache={s}", .{ report.rank.rank, report.rank.world, @tagName(report.interval.phase), report.interval.model_rows, report.interval.steps, report.expert_pools, report.kv_layers, cache });
    }

    pub fn formatAlloc(report: Report, allocator: std.mem.Allocator) ![]u8 {
        var writer: std.Io.Writer.Allocating = .init(allocator);
        errdefer writer.deinit();
        try report.write(&writer.writer);
        return writer.toOwnedSlice();
    }

    pub fn between(before: Snapshot, after: Snapshot, interval: Interval) !Report {
        try before.rank.validate();
        try after.rank.validate();
        if (!std.meta.eql(before.rank, after.rank) or before.scope != after.scope or before.kv.len != after.kv.len or before.expert_pools != after.expert_pools) return error.TelemetryOwnerChanged;
        if (before.ring_owner != 0 and before.ring_owner != after.ring_owner) return error.TelemetryOwnerChanged;
        if (interval.model_rows == 0 or interval.steps == 0) return error.TelemetryDenominator;
        if (interval.allow_initial_kv_reset and interval.phase != .prefill) return error.TelemetryKvReset;
        if (before.expert_layers.len != after.expert_layers.len) return error.TelemetryOwnerChanged;
        for (before.expert_layers, after.expert_layers) |first, last| _ = try first.delta(last);
        const experts = try difference(Experts, before.experts, after.experts);
        if (experts.device_overflow_plans != 0) return error.TelemetryExpertOverflow;
        var kv: Kv = .{};
        for (before.kv, after.kv) |first, last| try add(Kv, &kv, try first.delta(last, interval.allow_initial_kv_reset));
        return .{ .rank = after.rank, .interval = interval, .experts = experts, .kv = kv, .expert_pools = after.expert_pools, .kv_layers = after.kv.len, .expert_layers_before = before.expert_layers, .expert_layers_after = after.expert_layers };
    }
};

test "exl3: telemetry host contracts" {
    _ = @import("exl3_telemetry_test.zig");
}
