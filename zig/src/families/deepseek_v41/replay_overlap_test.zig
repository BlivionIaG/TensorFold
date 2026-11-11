//! Host tests of the exact replay beside the next prompt's encoder segment: shared scratch and ordering schedules.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const ced = @import("ced.zig");
const fp = @import("forward_prefill.zig");
const Config = @import("config.zig").Config;

const big_limit: i64 = 1 << 20;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

/// What a prompt's calls need: config, layers, the forward's (2,048-row) and the workspace's (4,096-row) options.
const Fx = struct {
    a: std.mem.Allocator,
    cfg: Config = .{},
    w: block.Widths,
    layers: [40]u32 = undefined,
    d: u32,
    o: block.Options,
    o4: block.Options,

    fn init(a: std.mem.Allocator) !Fx {
        var fx: Fx = .{ .a = a, .w = try widths(a), .d = try ced.decoderStart(&(Config{})), .o = undefined, .o4 = undefined };
        for (&fx.layers, 0..) |*l, i| l.* = @intCast(i);
        const pages = @divExact(big_limit, block.Pool.page);
        fx.o = .{ .limit = big_limit, .rope_rows = big_limit + 2048, .index_budget = 64 << 20, .taps = true };
        fx.o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
        fx.o4 = fp.k4Options(fx.o);
        return fx;
    }

    /// The exact replay's programs for a prompt ending at `end`, as pf_cone.finish emits them, in issue order.
    fn replay(fx: *const Fx, o: block.Options, end: u64, ns: Ns) ![]const calls.Call {
        const dec = fx.layers[fx.d..];
        const nd: u32 = @intCast(dec.len);
        var gbuf: [32]ced.Group = undefined;
        const gs = ced.groups(&fx.cfg, dec, 4, &gbuf);
        const n0 = ced.groupRows(fx.cfg.window, nd, 0, end);
        var keep: bp.Keep = .empty;
        try keep.put(fx.a, "w.taps", {});
        var out: std.ArrayList(calls.Call) = .empty;
        for (gs, 0..) |g, gi| {
            const n = ced.groupRows(fx.cfg.window, nd, g.first, end);
            const last = gi + 1 == gs.len;
            const lay = dec[g.first .. g.first + g.count + @intFromBool(!last)];
            const cand_at: ?i64 = if (gi == 0) null else @intCast(n0 - n);
            const plain = try bp.emitCone(fx.a, &fx.cfg, &fx.w, o, lay, n, @intCast(end - n), if (last) .cone_last else .cone_mid, cand_at);
            try out.appendSlice(fx.a, try ns.apply(fx.a, plain, &keep));
        }
        return out.items;
    }

    /// The next prompt's first encoder segment of `n` rows at position 0.
    fn encoder(fx: *const Fx, o: block.Options, n: i64, ns: Ns) ![]const calls.Call {
        const plain = try bp.emitReplay(fx.a, &fx.cfg, &fx.w, o, fx.layers[0 .. fx.d + 1], n, 0, .encoder_cone);
        return ns.apply(fx.a, plain, &.empty);
    }
};

/// Where a program's scratch lives: the forward's roles, the 4K workspace's, or a replay namespace of its own.
const Ns = enum {
    forward,
    workspace,
    private,

    fn apply(ns: Ns, a: std.mem.Allocator, cs: []const calls.Call, keep: *const bp.Keep) ![]const calls.Call {
        return switch (ns) {
            .forward => cs,
            .workspace => bp.ownCallsKeep(a, cs, fp.K4.tag, keep),
            .private => bp.ownCallsKeep(a, cs, "rp~", keep),
        };
    }
};

/// Tables both programs only read: weight forms, pointer tables, rope, page tables.
fn constant(r: []const u8) bool {
    if (std.mem.eql(u8, r, "s.pf.had") or std.mem.eql(u8, r, "s.kv.pt") or std.mem.eql(u8, r, "s.kv.ct")) return true;
    if (std.mem.startsWith(u8, r, "s.rope.")) return true;
    for ([_][]const u8{ ".lanes", ".T", ".fn16" }) |s| if (std.mem.endsWith(u8, r, s)) return true;
    for ([_][]const u8{ ".gm.", ".ex.tp_", ".pf." }) |s| if (std.mem.startsWith(u8, r, "s.L") and std.mem.indexOf(u8, r, s) != null) return true;
    return false;
}

/// A slot's own state: two prompts are two slots (distinct memory, bound to the roles at issue), never one buffer.
fn slotState(r: []const u8) bool {
    return std.mem.startsWith(u8, r, "s.L") or std.mem.startsWith(u8, r, "s.kv.") or std.mem.startsWith(u8, r, "s.ced.");
}

fn scratch(r: []const u8) bool {
    return !constant(r) and !slotState(r);
}

fn collect(a: std.mem.Allocator, x: calls.Arg, out: *std.ArrayList([]const u8)) !void {
    switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |b| if (scratch(b)) try out.append(a, b),
            else => {},
        },
        .list => |l| for (l) |y| try collect(a, y, out),
        else => {},
    }
}

/// Every call's scratch roles.
fn accesses(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const []const u8 {
    const out = try a.alloc([]const []const u8, cs.len);
    for (cs, out) |c, *x| {
        var rs: std.ArrayList([]const u8) = .empty;
        for (c.args) |arg| try collect(a, arg.arg, &rs);
        x.* = rs.items;
    }
    return out;
}

/// One step a stream takes in issue order: a call, an event record, or a wait for an event.
const Op = union(enum) {
    run: struct { s: u1, roles: []const []const u8 },
    record: struct { s: u1, ev: u8 },
    wait: struct { s: u1, ev: u8 },
};

/// The scratch roles two calls on different streams touch with no event between them (none: the schedule is ordered).
fn races(a: std.mem.Allocator, ops: []const Op) ![]const []const u8 {
    const Seen = struct { s: u1, t: u32, roles: []const []const u8 };
    var clock: [2][2]u32 = @splat(@splat(0)); // what each stream has seen of each stream's position
    var events: [4][2]u32 = undefined;
    var seen: std.ArrayList(Seen) = .empty;
    var out: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (ops) |op| switch (op) {
        .record => |r| {
            clock[r.s][r.s] += 1;
            events[r.ev] = clock[r.s];
        },
        .wait => |x| for (0..2) |i| {
            clock[x.s][i] = @max(clock[x.s][i], events[x.ev][i]);
        },
        .run => |c| {
            clock[c.s][c.s] += 1;
            const v = clock[c.s];
            for (seen.items) |p| {
                if (p.s == c.s or v[p.s] >= p.t) continue;
                for (c.roles) |r| for (p.roles) |q| if (std.mem.eql(u8, r, q)) try out.put(a, r, {});
            }
            try seen.append(a, .{ .s = c.s, .t = clock[c.s][c.s], .roles = c.roles });
        },
    };
    return out.keys();
}

fn stream(a: std.mem.Allocator, ops: *std.ArrayList(Op), s: u1, acc: []const []const []const u8) !void {
    for (acc) |roles| try ops.append(a, .{ .run = .{ .s = s, .roles = roles } });
}

/// The calls at the front of `b` that touch none of `a`'s scratch: what a second stream could start without waiting.
fn leadingFree(a: []const []const []const u8, b: []const []const []const u8) usize {
    for (b, 0..) |roles, i| for (roles) |r| for (a) |ar| for (ar) |q| if (std.mem.eql(u8, r, q)) return i;
    return b.len;
}

fn has(rs: []const []const u8, name: []const u8) bool {
    for (rs) |r| if (std.mem.eql(u8, r, name)) return true;
    return false;
}

/// Races of `ra` on stream 0 and `rb` on stream 1 (`two`, else 0), `rb` held by one event after `ra` at `fence_at`.
fn schedule(a: std.mem.Allocator, ra: []const []const []const u8, rb: []const []const []const u8, two: bool, fence_at: ?usize) ![]const []const u8 {
    var ops: std.ArrayList(Op) = .empty;
    try stream(a, &ops, 0, ra);
    if (fence_at) |at| {
        try ops.append(a, .{ .record = .{ .s = 0, .ev = 0 } });
        try stream(a, &ops, if (two) 1 else 0, rb[0..at]);
        if (two) try ops.append(a, .{ .wait = .{ .s = 1, .ev = 0 } });
        try stream(a, &ops, if (two) 1 else 0, rb[at..]);
    } else try stream(a, &ops, if (two) 1 else 0, rb);
    return races(a, ops.items);
}

test "the replay and the next prompt's encoder segment share scratch in the forward's buffers and in the workspace" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = try Fx.init(a);
    for ([_]struct { Ns, u64, i64, []const u8 }{ .{ .forward, 1500, 2048, "w.x0" }, .{ .workspace, 11292, 4096, "w.k4~x0" } }) |c| {
        const o = if (c[0] == .forward) fx.o else fx.o4;
        const ra = try accesses(a, try fx.replay(o, c[1], c[0]));
        const rb = try accesses(a, try fx.encoder(o, c[2], c[0]));
        try testing.expect(ra.len > 1000 and rb.len > 1000);
        // two streams with no event: the window's scratch, the exchange and expert scratch, the layer's roles
        const open = try schedule(a, ra, rb, true, null);
        try testing.expect(open.len > 50);
        try testing.expect(has(open, c[3]));
        // the first call of the next segment already writes a role the replay's programs use: no call starts free
        try testing.expectEqual(@as(usize, 0), leadingFree(ra, rb));
        // the engine's schedule, one stream: the replay in flight, the next segment behind it
        try testing.expectEqual(@as(usize, 0), (try schedule(a, ra, rb, false, null)).len);
        // a second stream held by one event after the replay: ordered, and nothing of the segment ran beside it
        try testing.expectEqual(@as(usize, 0), (try schedule(a, ra, rb, true, 0)).len);
    }
}

test "a replay namespace of its own leaves the cone ring the one scratch both touch, for over a GiB a rank" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = try Fx.init(a);
    const mine = try fx.replay(fx.o4, 11292, .private);
    const next = try fx.encoder(fx.o4, 4096, .workspace);
    const ra = try accesses(a, mine);
    const rb = try accesses(a, next);
    // the whole next segment but its ring stash touches nothing of the replay's
    const open = try schedule(a, ra, rb, true, null);
    try testing.expectEqual(ced.cone_roles.len, open.len);
    for (ced.cone_roles) |r| try testing.expect(has(open, r));
    const free = leadingFree(ra, rb);
    try testing.expectEqual(rb.len - 1, free);
    try testing.expectEqualStrings("glue.cone_stash", next[free].name);
    // one event orders the ring's two ends: the stash waits for the replay's last program
    try testing.expectEqual(@as(usize, 0), (try schedule(a, ra, rb, true, free)).len);
    // the namespace's bytes beside the workspace's: the programs' window and layer roles, the expert and router scratch
    var plan: buffers.Plan = .{ .a = a };
    var own: u64 = 0;
    try plan.add(mine);
    for (plan.sizes.keys(), plan.sizes.values()) |k, v| if (std.mem.indexOf(u8, k, "rp~") != null) {
        own += std.mem.alignForward(u64, v, 256);
    };
    try testing.expect(own >= 1 << 30);
}
