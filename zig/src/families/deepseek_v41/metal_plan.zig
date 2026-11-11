//! The Metal resident memory plan: weights, shared workspaces and a full-context KV lane a stream (0.94 GB at 1M).

const std = @import("std");
const prefix = @import("kv_prefix.zig");

/// One lane's bytes: its packed pool for `capacity` rows of the ratio-1 owner, two mutable blocks, the descriptors.
pub fn laneBytes(layout: prefix.Layout, capacity: u32, mutable: u64, descriptors: u64) u64 {
    return layout.storageBytes(capacity) + 2 * mutable + descriptors;
}

pub const Inputs = struct {
    /// the checkpoint's resident bytes (load.Plan.bytes)
    weights: u64,
    /// the work plan and the attention workspace: one set, shared by every lane
    workspace: u64,
    /// the rotary tables for the capacity
    rotary: u64,
    /// the prompt-row int8 twin's extra, when on
    extra: u64 = 0,
    lanes: u32,
    /// one lane at this capacity (`laneBytes`) and what a position adds to it (890 B at 1M)
    lane: u64,
    lane_per_position: f64,
    capacity: u32,
    /// the load limit (GLM_LOAD_LIMIT_GB) and what Metal admission allows after its margin, bytes
    limit: ?u64 = null,
    available: u64,
};

pub const Plan = struct {
    in: Inputs,
    /// the lanes together, and every byte the load needs
    lanes_bytes: u64,
    needed: u64,
    needed_all: u64,

    pub fn of(in: Inputs) Plan {
        const lanes_bytes = in.lane * in.lanes;
        const needed = in.weights + in.workspace + lanes_bytes + in.rotary;
        return .{ .in = in, .lanes_bytes = lanes_bytes, .needed = needed, .needed_all = needed + in.extra };
    }

    /// The tighter of the load limit and Metal's admission.
    pub fn ceiling(p: Plan) u64 {
        return if (p.in.limit) |l| @min(l, p.in.available) else p.in.available;
    }

    pub fn fits(p: Plan) bool {
        return p.needed_all <= p.ceiling();
    }

    /// What is left under the ceiling once loaded (0 when it does not fit).
    pub fn headroom(p: Plan) u64 {
        return p.ceiling() -| p.needed_all;
    }

    /// The most lanes this capacity admits under the ceiling (0: not even one).
    pub fn maxLanes(p: Plan) u32 {
        const fixed = p.in.weights + p.in.workspace + p.in.rotary + p.in.extra;
        const room = p.ceiling() -| fixed;
        return @intCast(@min(room / @max(p.in.lane, 1), 256));
    }

    /// The largest `--context` `lanes` lanes admit, in positions (0: the lanes' fixed bytes alone do not fit).
    pub fn maxContext(p: Plan, lanes: u32, lane_fixed: u64) u64 {
        const fixed = p.in.weights + p.in.workspace + p.in.rotary + p.in.extra;
        const room = p.ceiling() -| fixed;
        const each = room / @max(lanes, 1);
        if (each <= lane_fixed) return 0;
        return @intFromFloat(@as(f64, @floatFromInt(each - lane_fixed)) / p.in.lane_per_position);
    }

    /// The boot's plan line: every term, the lanes at this capacity, the headroom.
    pub fn line(p: Plan, buf: []u8) []const u8 {
        const gb = struct {
            fn of(b: u64) f64 {
                return @as(f64, @floatFromInt(b)) / 1e9;
            }
        }.of;
        const in = p.in;
        return std.fmt.bufPrint(buf, "memory plan: weights {d:.2} GB + workspaces {d:.2} + rotary {d:.2} + {d} lanes x {d:.2} GB (context {d}, {d:.0} B a position) = {d:.2} GB; ceiling {d:.2} GB, headroom {d:.2} GB, at most {d} lanes at this context", .{
            gb(in.weights), gb(in.workspace), gb(in.rotary), in.lanes, gb(in.lane), in.capacity, in.lane_per_position, gb(p.needed_all), gb(p.ceiling()), gb(p.headroom()), p.maxLanes(),
        }) catch buf[0..0];
    }

    /// The refusal: what the load needs, what it may take, and the `--parallel` / `--context` that fit.
    pub fn refusal(p: Plan, buf: []u8, lane_fixed: u64) []const u8 {
        const gb = struct {
            fn of(b: u64) f64 {
                return @as(f64, @floatFromInt(b)) / 1e9;
            }
        }.of;
        return std.fmt.bufPrint(buf, "{d} lanes at context {d} need {d:.2} GB resident ({d:.2} GB a lane) but the ceiling is {d:.2} GB (GLM_LOAD_LIMIT_GB, Metal admission): --parallel {d} fits at this --context, or --context {d} fits {d} lanes", .{
            p.in.lanes, p.in.capacity, gb(p.needed_all), gb(p.in.lane), gb(p.ceiling()), p.maxLanes(), p.maxContext(p.in.lanes, lane_fixed), p.in.lanes,
        }) catch buf[0..0];
    }
};
