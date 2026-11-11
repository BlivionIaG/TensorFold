//! The policy configurations of expert_cache_sim.py (`build_configs`: same labels, same order) and the policy a configuration builds.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const pol = @import("expert_cache_sim_policy.zig");
const freq = @import("expert_cache_sim_freq.zig");

/// How a prompt's replay starts: empty cache, seeded with other categories' hottest keys, warm from its own prefill's top counts (pf), both
/// (pf+seed), or its prefill replayed token by token through the policy.
pub const Init = enum { cold, seed, pf, pf_seed, pfreplay };

pub const Kind = union(enum) {
    lru,
    per_layer,
    slru: f64,
    twoq: f64,
    tinylfu: struct { w: u64, door: bool, ge: bool },
    wtinylfu: struct { wfrac: f64, w: u64, door: bool },
    lfu: struct { w: u64, hist: bool },
    lrufreq: struct { k: usize, w: u64 },
    arc,
};

pub const Config = struct {
    label: []const u8,
    kind: Kind,
    init: Init,
};

pub const Policy = union(enum) {
    lru: pol.Lru,
    per_layer: pol.PerLayer,
    slru: pol.Slru,
    twoq: pol.TwoQ,
    tinylfu: freq.TinyLfu,
    wtinylfu: freq.WTinyLfu,
    lfu: freq.Lfu,
    lrufreq: freq.LruFreq,
    arc: pol.Arc,

    /// The policy for `kind` with `cap` lines; its memory comes from `a` (an arena the caller drops with the job).
    pub fn create(a: std.mem.Allocator, kind: Kind, cap: usize) !Policy {
        return switch (kind) {
            .lru => .{ .lru = try .init(a, cap) },
            .per_layer => .{ .per_layer = try .init(a, cap) },
            .slru => |pf| .{ .slru = try .init(a, cap, pf) },
            .twoq => |kin| .{ .twoq = try .init(a, cap, kin, 0.5) },
            .tinylfu => |t| .{ .tinylfu = try .init(a, cap, t.w, t.door, t.ge) },
            .wtinylfu => |t| .{ .wtinylfu = try .init(a, cap, t.wfrac, t.w, t.door) },
            .lfu => |t| .{ .lfu = try .init(a, cap, t.w, t.hist) },
            .lrufreq => |t| .{ .lrufreq = try .init(a, cap, t.k, t.w) },
            .arc => .{ .arc = try .init(a, cap) },
        };
    }

    /// One (token, layer) call: the number of misses.
    pub fn step(self: *Policy, layer: usize, keys: []const u16) u32 {
        return switch (self.*) {
            inline else => |*p| p.step(layer, keys),
        };
    }

    pub fn adm(self: *const Policy) u64 {
        return switch (self.*) {
            inline else => |p| p.adm,
        };
    }

    pub fn evi(self: *const Policy) u64 {
        return switch (self.*) {
            inline else => |p| p.evi,
        };
    }

    pub fn resetCounters(self: *Policy) void {
        switch (self.*) {
            inline else => |*p| {
                p.adm = 0;
                p.evi = 0;
            },
        }
    }

    /// Only the LRU configurations are seeded.
    pub fn preload(self: *Policy, keys_oldest_first: []const u16) void {
        self.lru.preload(keys_oldest_first);
    }
};

/// `f"{W / 1000:g}"` for a whole number of lines: 2500 -> "2.5", 1250 -> "1.25", 40000 -> "40".
fn kilo(a: std.mem.Allocator, w: u64) ![]const u8 {
    var frac: [3]u8 = undefined;
    _ = std.fmt.bufPrint(&frac, "{d:0>3}", .{w % 1000}) catch unreachable;
    const digits = std.mem.trimEnd(u8, &frac, "0");
    if (digits.len == 0) return std.fmt.allocPrint(a, "{d}", .{w / 1000});
    return std.fmt.allocPrint(a, "{d}.{s}", .{ w / 1000, digits });
}

/// `int(x * 100)` in double arithmetic, as the labels' `int(pf * 100)`.
fn percent(x: f64) u64 {
    return @intFromFloat(x * 100);
}

/// Every configuration of the study, in the original's order; labels are allocated from `a`.
pub fn build(a: std.mem.Allocator) ![]Config {
    var out: std.ArrayList(Config) = .empty;
    const yn = struct {
        fn f(door: bool) []const u8 {
            return if (door) "y" else "n";
        }
    }.f;
    try out.append(a, .{ .label = "LRU global (baseline)", .kind = .lru, .init = .cold });
    try out.append(a, .{ .label = "LRU per layer, equal split", .kind = .per_layer, .init = .cold });
    const pfs = [_]f64{ 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9 };
    for (pfs) |pf| try out.append(a, .{ .label = try std.fmt.allocPrint(a, "SLRU protected {d}%", .{percent(pf)}), .kind = .{ .slru = pf }, .init = .cold });
    const kins = [_]f64{ 0.1, 0.25, 0.5 };
    for (kins) |kin| try out.append(a, .{ .label = try std.fmt.allocPrint(a, "2Q A1in {d}% ghost 50%", .{percent(kin)}), .kind = .{ .twoq = kin }, .init = .cold });
    for ([_]u64{ 2_500, 5_000, 10_000, 40_000, 160_000, 640_000 }) |w| {
        for ([_]bool{ false, true }) |door| {
            try out.append(a, .{ .label = try std.fmt.allocPrint(a, "TinyLFU W={s}k door={s}", .{ try kilo(a, w), yn(door) }), .kind = .{ .tinylfu = .{ .w = w, .door = door, .ge = false } }, .init = .cold });
        }
    }
    for ([_]bool{ false, true }) |door| {
        try out.append(a, .{ .label = try std.fmt.allocPrint(a, "TinyLFU W=40k door={s} admit if >=", .{yn(door)}), .kind = .{ .tinylfu = .{ .w = 40_000, .door = door, .ge = true } }, .init = .cold });
    }
    const wfs = [_]f64{ 0.01, 0.05, 0.2 };
    for (wfs) |wf| try out.append(a, .{ .label = try std.fmt.allocPrint(a, "W-TinyLFU window {d}% W=160k door=y", .{percent(wf)}), .kind = .{ .wtinylfu = .{ .wfrac = wf, .w = 160_000, .door = true } }, .init = .cold });
    for ([_]u64{ 1_250, 2_500, 5_000, 10_000, 20_000, 40_000, 160_000, 640_000 }) |w| {
        try out.append(a, .{ .label = try std.fmt.allocPrint(a, "LFU aged W={s}k (history kept)", .{try kilo(a, w)}), .kind = .{ .lfu = .{ .w = w, .hist = true } }, .init = .cold });
    }
    for ([_]u64{ 40_000, 160_000 }) |w| {
        try out.append(a, .{ .label = try std.fmt.allocPrint(a, "LFU aged W={s}k (count restarts on evict)", .{try kilo(a, w)}), .kind = .{ .lfu = .{ .w = w, .hist = false } }, .init = .cold });
    }
    for ([_]u64{ 5_000, 10_000, 40_000, 160_000 }) |w| {
        for ([_]usize{ 2, 4, 8, 16, 32 }) |k| {
            try out.append(a, .{ .label = try std.fmt.allocPrint(a, "LRU, evict lowest-freq of K={d} oldest, W={s}k", .{ k, try kilo(a, w) }), .kind = .{ .lrufreq = .{ .k = k, .w = w } }, .init = .cold });
        }
    }
    try out.append(a, .{ .label = "ARC global", .kind = .arc, .init = .cold });
    try out.append(a, .{ .label = "LRU, seeded with other categories' hottest keys", .kind = .lru, .init = .seed });
    try out.append(a, .{ .label = "LRU, warm from own prefill top counts", .kind = .lru, .init = .pf });
    try out.append(a, .{ .label = "LRU, own prefill top counts + others' hottest fill", .kind = .lru, .init = .pf_seed });
    try out.append(a, .{ .label = "LRU, prefill replayed token by token through LRU", .kind = .lru, .init = .pfreplay });
    return out.toOwnedSlice(a);
}
