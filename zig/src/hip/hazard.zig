//! The hazard check of a recorded round: which memory every kernel read and wrote, which kernels must therefore follow which, and whether the graph's
//! edges say so.
//!
//! Input is what `capture_trace.zig` recorded from the production forward: every kernel with its argument pack and the dependency edges the capture
//! gave it, plus the arena's allocation log (`arena.Log`) and the plan buffer's address range. `hazard_table.zig` says what each kernel parameter is
//! (scalar, pointer read / written / both, a by-value struct of pointers) and which per-layer state the kernel reaches through the plan's
//! descriptors without an argument naming it (the layer's K/V pages, conv window, DeltaNet state, kept-input snapshots, shared scratch).
//!
//! Memory is tracked in intervals of addresses:
//!   * a pointer into the arena stands for the whole live allocation that holds it (the arena's log says which one was live at that launch: the
//!     scratch of one layer is handed back and reused by the next, so the same address is a different buffer at different times, and a write-after-read
//!     between layers is a hazard like any other);
//!   * a pointer into the plan buffer stands for the whole plan buffer;
//!   * per-layer state reached through the plan is a one-address interval of its own (`virtual_base` and up), the kept-input snapshots are the arena
//!     allocations the round's `inputs` table names;
//!   * any other pointer (weights, a pinned host word) stands for the eight bytes at it, which is only ever compared with the same address.
//! Two accesses conflict when their intervals overlap and one of them writes. Granularity is deliberately coarse: a pointer into an allocation
//! counts as the allocation, so the check may refuse a pair that touches disjoint halves of one buffer, never accept one that races.
//!
//! A kernel the table does not know is reported (`Analysis.unknown`) and counts as reading and writing everything, so it is ordered against every
//! other node or the check fails; a table entry whose arity differs from the launcher's is an error.

const std = @import("std");
/// What every kernel's parameters are (the checks of the table against the kernels' sources use it).
pub const table = @import("hazard_table.zig");
const arena = @import("arena.zig");
const capture_trace = @import("capture_trace.zig");

pub const Mode = table.Mode;

/// Where the round's memory is.
pub const Layout = struct {
    arena_base: u64,
    arena_len: usize,
    arena_log: []const arena.Log.Item,
    /// The plan buffer's addresses, [lo, hi).
    plan_lo: u64,
    plan_hi: u64,
    /// Each layer's kept-input snapshot buffers (`win.Round.inputs`): arena addresses, 0 where the layer keeps none.
    snaps: []const [2]u64,
};

/// Addresses at and above this are the per-layer state a kernel reaches through the plan, not memory.
pub const virtual_base: u64 = 0xFFFF_0000_0000_0000;

fn virtualOf(what: table.Hidden.What, layer: u32) u64 {
    return virtual_base + (@as(u64, @backingInt(what)) << 32) + layer;
}

pub const Access = struct {
    lo: u64,
    /// Exclusive.
    hi: u64,
    mode: Mode,
    /// The kernel parameter it came from (-1: hidden state).
    arg: i32,
};

pub const Kind = packed struct(u3) { raw: bool = false, war: bool = false, waw: bool = false };

/// Two nodes (`first` launched before `second`) whose accesses overlap with a write among them.
pub const Conflict = struct {
    first: u32,
    second: u32,
    kind: Kind,
    /// One overlapping address.
    at: u64,
};

pub const Pair = struct { first: u32, second: u32 };

pub const ExternalWrite = struct { node: u32, arg: i32, address: u64 };

pub const Analysis = struct {
    gpa: std.mem.Allocator,
    accesses: [][]Access,
    conflicts: []Conflict,
    /// Nodes of kernels the table lacks (they count as touching everything).
    unknown: []u32,
    /// Writes to memory that is neither the arena, nor the plan, nor per-layer state.
    external_writes: []ExternalWrite,
    /// Pointers into the arena that no live allocation holds.
    dangling: []ExternalWrite,

    pub fn deinit(a: *Analysis) void {
        for (a.accesses) |x| a.gpa.free(x);
        a.gpa.free(a.accesses);
        a.gpa.free(a.conflicts);
        a.gpa.free(a.unknown);
        a.gpa.free(a.external_writes);
        a.gpa.free(a.dangling);
        a.* = undefined;
    }

    /// The conflicts the edges of `closure` do not order (`later` does not follow `earlier`).
    pub fn uncovered(a: *const Analysis, gpa: std.mem.Allocator, closure: capture_trace.Closure) ![]Conflict {
        var out: std.ArrayList(Conflict) = .empty;
        errdefer out.deinit(gpa);
        for (a.conflicts) |c| if (!closure.follows(c.second, c.first)) try out.append(gpa, c);
        return out.toOwnedSlice(gpa);
    }

    /// Pairs of nodes no chain of conflicts orders: the pairs that could run at once without a hazard. Pairs that are only ordered because
    /// something between them conflicts with both are not listed.
    pub fn independent(a: *const Analysis, gpa: std.mem.Allocator) ![]Pair {
        const n = a.accesses.len;
        const words = (n + 63) / 64;
        const bits = try gpa.alloc(u64, @max(n * words, 1));
        defer gpa.free(bits);
        @memset(bits, 0);
        for (a.conflicts) |c| {
            const row = bits[c.second * words ..][0..words];
            row[c.first / 64] |= @as(u64, 1) << @intCast(c.first % 64);
        }
        // conflicts are listed by (second, first): rows are final before the transitive step reads them
        for (0..n) |j| {
            const row = bits[j * words ..][0..words];
            for (0..j) |i| {
                if (row[i / 64] >> @intCast(i % 64) & 1 == 0) continue;
                const from = bits[i * words ..][0..words];
                for (row, from) |*r, f| r.* |= f;
            }
        }
        var out: std.ArrayList(Pair) = .empty;
        errdefer out.deinit(gpa);
        for (0..n) |j| for (0..j) |i| {
            if (bits[j * words + i / 64] >> @intCast(i % 64) & 1 == 0) try out.append(gpa, .{ .first = @intCast(i), .second = @intCast(j) });
        };
        return out.toOwnedSlice(gpa);
    }
};

pub const Error = error{ ArityMismatch, BadLayer, OutOfMemory };

const Live = struct { lo: u64, hi: u64, off: u64 };

/// The accesses of every node of `trace` and the conflicts between them.
pub fn analyze(gpa: std.mem.Allocator, trace: *const capture_trace.Trace, layout: Layout) Error!Analysis {
    const n = trace.nodes.items.len;
    const accesses = try gpa.alloc([]Access, n);
    var made: usize = 0;
    errdefer {
        for (accesses[0..made]) |x| gpa.free(x);
        gpa.free(accesses);
    }
    var unknown: std.ArrayList(u32) = .empty;
    errdefer unknown.deinit(gpa);
    var external: std.ArrayList(ExternalWrite) = .empty;
    errdefer external.deinit(gpa);
    var dangling: std.ArrayList(ExternalWrite) = .empty;
    errdefer dangling.deinit(gpa);
    var live: std.ArrayList(Live) = .empty;
    defer live.deinit(gpa);
    var applied: usize = 0;

    for (trace.nodes.items, 0..) |node, i| {
        // the arena as it stood at this launch
        while (applied < node.arena_items) : (applied += 1) switch (layout.arena_log[applied]) {
            .take => |t| try live.append(gpa, .{ .lo = t.at, .hi = t.at + t.bytes, .off = t.at - layout.arena_base }),
            .release => |at| while (live.items.len > 0 and live.items[live.items.len - 1].off >= at) {
                _ = live.pop();
            },
        };
        var list: std.ArrayList(Access) = .empty;
        errdefer list.deinit(gpa);
        const entry = table.find(node.name);
        if (entry) |e| {
            if (e.args.len != node.offsets.len) {
                std.log.warn("hazard table: {s} has {d} parameters, the launch passed {d}", .{ node.name, e.args.len, node.offsets.len });
                return error.ArityMismatch;
            }
            for (e.args, 0..) |a, ai| {
                const at: usize = node.offsets[ai];
                switch (a) {
                    .val => {},
                    .ptr => |m| try pointer(gpa, &list, &external, &dangling, live.items, layout, @intCast(i), @intCast(ai), word(node.bytes, at), m),
                    .words => |ws| for (ws, 0..) |wm, k| if (wm) |m| {
                        try pointer(gpa, &list, &external, &dangling, live.items, layout, @intCast(i), @intCast(ai), word(node.bytes, at + 8 * k), m);
                    },
                }
            }
            for (e.hidden) |h| {
                const layer: u32 = if (h.layer_arg) |la| blk: {
                    const at: usize = node.offsets[la] + if (h.layer_word) |w| 8 * @as(usize, w) else 0;
                    const raw = std.mem.bytesToValue(i32, node.bytes[at..][0..4]);
                    if (raw < 0) return error.BadLayer;
                    break :blk @intCast(raw);
                } else 0;
                if (h.what == .snaps) {
                    if (layer >= layout.snaps.len) return error.BadLayer;
                    for (layout.snaps[layer]) |p| if (p != 0) {
                        const iv = containing(live.items, p) orelse continue;
                        try list.append(gpa, .{ .lo = iv.lo, .hi = iv.hi, .mode = h.mode, .arg = -1 });
                    };
                } else {
                    const v = virtualOf(h.what, layer);
                    try list.append(gpa, .{ .lo = v, .hi = v + 1, .mode = h.mode, .arg = -1 });
                }
            }
        } else {
            try unknown.append(gpa, @intCast(i));
            try list.append(gpa, .{ .lo = 0, .hi = std.math.maxInt(u64), .mode = .rw, .arg = -1 });
        }
        accesses[i] = try list.toOwnedSlice(gpa);
        made += 1;
    }

    var conflicts: std.ArrayList(Conflict) = .empty;
    errdefer conflicts.deinit(gpa);
    for (1..n) |j| for (0..j) |i| {
        var kind: Kind = .{};
        var at: u64 = 0;
        for (accesses[i]) |a| for (accesses[j]) |b| {
            const lo = @max(a.lo, b.lo);
            if (lo >= @min(a.hi, b.hi)) continue;
            const aw = a.mode != .r;
            const bw = b.mode != .r;
            const ar = a.mode != .w;
            const br = b.mode != .w;
            if (!(aw or bw)) continue;
            if (aw and br) kind.raw = true;
            if (ar and bw) kind.war = true;
            if (aw and bw) kind.waw = true;
            at = lo;
        };
        if (kind.raw or kind.war or kind.waw) try conflicts.append(gpa, .{ .first = @intCast(i), .second = @intCast(j), .kind = kind, .at = at });
    };
    return .{
        .gpa = gpa,
        .accesses = accesses,
        .conflicts = try conflicts.toOwnedSlice(gpa),
        .unknown = try unknown.toOwnedSlice(gpa),
        .external_writes = try external.toOwnedSlice(gpa),
        .dangling = try dangling.toOwnedSlice(gpa),
    };
}

fn word(bytes: []const u8, at: usize) u64 {
    return std.mem.bytesToValue(u64, bytes[at..][0..8]);
}

fn containing(live: []const Live, p: u64) ?Live {
    var k = live.len;
    while (k > 0) {
        k -= 1;
        if (p >= live[k].lo and p < live[k].hi) return live[k];
    }
    return null;
}

fn pointer(gpa: std.mem.Allocator, list: *std.ArrayList(Access), external: *std.ArrayList(ExternalWrite), dangling: *std.ArrayList(ExternalWrite), live: []const Live, layout: Layout, node: u32, arg: i32, p: u64, mode: Mode) Error!void {
    if (p == 0) return;
    if (p >= layout.arena_base and p - layout.arena_base < layout.arena_len) {
        if (containing(live, p)) |iv| {
            try list.append(gpa, .{ .lo = iv.lo, .hi = iv.hi, .mode = mode, .arg = arg });
        } else {
            try dangling.append(gpa, .{ .node = node, .arg = arg, .address = p });
            try list.append(gpa, .{ .lo = p, .hi = p + 1, .mode = mode, .arg = arg });
        }
        return;
    }
    if (p >= layout.plan_lo and p < layout.plan_hi) {
        try list.append(gpa, .{ .lo = layout.plan_lo, .hi = layout.plan_hi, .mode = mode, .arg = arg });
        return;
    }
    if (mode != .r) try external.append(gpa, .{ .node = node, .arg = arg, .address = p });
    try list.append(gpa, .{ .lo = p, .hi = p + 8, .mode = mode, .arg = arg });
}

// ---- tests ----

/// A launch of `name` whose arguments are the 8-byte words `args`, as the arena held `arena_items` log items at that moment.
fn push(t: *capture_trace.Trace, name: []const u8, args: []const u64, arena_items: usize) !void {
    const gpa = t.gpa;
    const bytes = try gpa.alloc(u8, 8 * args.len);
    const offsets = try gpa.alloc(u16, args.len);
    for (args, 0..) |a, i| {
        @memcpy(bytes[8 * i ..][0..8], std.mem.asBytes(&a));
        offsets[i] = @intCast(8 * i);
    }
    try t.nodes.append(gpa, .{ .name = name, .grid = .{ 1, 1, 1 }, .block = .{ 1, 1, 1 }, .shared = 0, .stream = 0, .bytes = bytes, .offsets = offsets, .arena_items = arena_items, .deps = try gpa.alloc(u32, 0) });
}

fn conflictOf(a: *const Analysis, first: u32, second: u32) ?Conflict {
    for (a.conflicts) |c| if (c.first == first and c.second == second) return c;
    return null;
}

const base: u64 = 0x1000_0000;
const plan_lo: u64 = 0x5000_0000;

fn layoutOf(log: []const arena.Log.Item) Layout {
    return .{ .arena_base = base, .arena_len = 1 << 20, .arena_log = log, .plan_lo = plan_lo, .plan_hi = plan_lo + 4096, .snaps = &.{} };
}

test "hazard: reads and writes of arena buffers conflict as RAW, WAR and WAW" {
    const gpa = std.testing.allocator;
    var t = capture_trace.Trace.init(gpa);
    defer t.deinit();
    const a = base;
    const b = base + 1024;
    const c = base + 2048;
    const log = [_]arena.Log.Item{
        .{ .take = .{ .at = a, .bytes = 1024 } },
        .{ .take = .{ .at = b, .bytes = 1024 } },
        .{ .take = .{ .at = c, .bytes = 1024 } },
    };
    // tf_cast(src, kind, dst, n, ...): the first read, the third written
    try push(&t, "tf_cast", &.{ a, 0, b, 0, 0 }, 3); // 0: a -> b
    try push(&t, "tf_cast", &.{ b, 0, c, 0, 0 }, 3); // 1: b -> c
    try push(&t, "tf_cast", &.{ c, 0, a, 0, 0 }, 3); // 2: c -> a
    try push(&t, "tf_cast", &.{ a, 0, b, 0, 0 }, 3); // 3: a -> b again
    var an = try analyze(gpa, &t, layoutOf(&log));
    defer an.deinit();
    try std.testing.expectEqual(@as(usize, 0), an.unknown.len);
    try std.testing.expectEqual(@as(usize, 0), an.dangling.len);
    // 1 reads what 0 wrote
    try std.testing.expectEqual(Kind{ .raw = true }, conflictOf(&an, 0, 1).?.kind);
    // 2 reads what 1 wrote, and writes what 0 read
    try std.testing.expectEqual(Kind{ .raw = true }, conflictOf(&an, 1, 2).?.kind);
    try std.testing.expectEqual(Kind{ .war = true }, conflictOf(&an, 0, 2).?.kind);
    // 3 reads what 2 wrote, writes what 0 wrote and what 1 read
    try std.testing.expectEqual(Kind{ .raw = true }, conflictOf(&an, 2, 3).?.kind);
    try std.testing.expectEqual(Kind{ .waw = true }, conflictOf(&an, 0, 3).?.kind);
    try std.testing.expectEqual(Kind{ .war = true }, conflictOf(&an, 1, 3).?.kind);
    // 0 and 3 read the same buffer: that alone is no conflict, and nothing is independent here: every pair is ordered through a chain
    const free = try an.independent(gpa);
    defer gpa.free(free);
    try std.testing.expectEqual(@as(usize, 0), free.len);
}

test "hazard: two launches that only read, or touch different buffers, are independent" {
    const gpa = std.testing.allocator;
    var t = capture_trace.Trace.init(gpa);
    defer t.deinit();
    const log = [_]arena.Log.Item{
        .{ .take = .{ .at = base, .bytes = 1024 } },
        .{ .take = .{ .at = base + 1024, .bytes = 1024 } },
        .{ .take = .{ .at = base + 2048, .bytes = 1024 } },
    };
    const weight: u64 = 0x7000_0000_0000;
    try push(&t, "tf_cast", &.{ weight, 0, base, 0, 0 }, 3); // 0: weight -> a
    try push(&t, "tf_cast", &.{ weight, 0, base + 1024, 0, 0 }, 3); // 1: the same weight -> b: shared reads
    try push(&t, "tf_cast", &.{ plan_lo + 8, 0, base + 2048, 0, 0 }, 3); // 2: a plan word -> c
    var an = try analyze(gpa, &t, layoutOf(&log));
    defer an.deinit();
    try std.testing.expectEqual(@as(usize, 0), an.conflicts.len);
    const free = try an.independent(gpa);
    defer gpa.free(free);
    try std.testing.expectEqual(@as(usize, 3), free.len);
}

test "hazard: scratch handed back and taken again is the same memory" {
    const gpa = std.testing.allocator;
    var t = capture_trace.Trace.init(gpa);
    defer t.deinit();
    const log = [_]arena.Log.Item{
        .{ .take = .{ .at = base, .bytes = 1024 } }, // 0
        .{ .release = 0 }, // 1
        .{ .take = .{ .at = base, .bytes = 512 } }, // 2: another buffer, at the same address
        .{ .take = .{ .at = base + 512, .bytes = 512 } }, // 3
    };
    const weight: u64 = 0x7000_0000_0000;
    try push(&t, "tf_cast", &.{ weight, 0, base, 0, 0 }, 1); // 0: writes the first buffer
    try push(&t, "tf_cast", &.{ weight, 0, base, 0, 0 }, 3); // 1: writes the second, at the same address
    try push(&t, "tf_cast", &.{ weight, 0, base + 512, 0, 0 }, 4); // 2: writes the one next to it
    var an = try analyze(gpa, &t, layoutOf(&log));
    defer an.deinit();
    try std.testing.expectEqual(Kind{ .waw = true }, conflictOf(&an, 0, 1).?.kind);
    // after the release the first buffer is not live: the next buffer at base + 512 stays disjoint from the one at base
    try std.testing.expect(conflictOf(&an, 1, 2) == null);
}

test "hazard: a kernel the table lacks touches everything, and stray pointers are reported" {
    const gpa = std.testing.allocator;
    var t = capture_trace.Trace.init(gpa);
    defer t.deinit();
    const log = [_]arena.Log.Item{.{ .take = .{ .at = base, .bytes = 1024 } }};
    const outside: u64 = 0x9000_0000_0000;
    try push(&t, "tf_cast", &.{ outside, 0, base, 0, 0 }, 1); // 0: reads outside memory
    try push(&t, "tf_cast", &.{ base + 4096, 0, outside, 0, 0 }, 1); // 1: reads where no buffer lives, writes outside memory
    try push(&t, "tf_not_in_the_table", &.{0}, 1); // 2
    var an = try analyze(gpa, &t, layoutOf(&log));
    defer an.deinit();
    try std.testing.expectEqual(@as(usize, 1), an.external_writes.len);
    try std.testing.expectEqual(@as(u32, 1), an.external_writes[0].node);
    try std.testing.expectEqual(@as(usize, 1), an.dangling.len);
    try std.testing.expectEqual(@as(u32, 1), an.dangling[0].node);
    try std.testing.expectEqualSlices(u32, &.{2}, an.unknown);
    // the unknown kernel conflicts with both: it may read and write anything
    try std.testing.expect(conflictOf(&an, 0, 2) != null);
    try std.testing.expect(conflictOf(&an, 1, 2) != null);
    // and with the arity check: a launch with the wrong number of arguments is refused
    try push(&t, "tf_cast", &.{ base, 0 }, 1);
    try std.testing.expectError(error.ArityMismatch, analyze(gpa, &t, layoutOf(&log)));
}
