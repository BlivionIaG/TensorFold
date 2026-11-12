//! The branches of a captured Qwen3.5-4B decode round, proven against the buffers the round's kernels read and write.
//!
//! The production forward (`round.forwardOn`) runs over a fake HIP runtime that follows hipmac's capture rules and records every launch with its complete
//! argument pack and the edges the capture gave it (`hip.capture_trace`). `hip.hazard` turns that into the memory every kernel read and wrote, from
//! the table of what each kernel parameter is (`hazard_table.zig`) and the arena's log of what was live where. Three things follow, none of them an
//! assumption about the code in `attentionRows`:
//!   * which pairs of launches have no data dependence at all (the only ones a branch may separate),
//!   * that every read-after-write, write-after-read and write-after-write between launches is ordered by an edge of the captured graph,
//!   * that the branches change no launch: the kernels, grids and arguments are those of the round without them.
//! `zig build test-round-branches-host` runs this (the kernel library names gfx900 there, which the launcher's tile registry needs).

const std = @import("std");
const hip = @import("hip");
const sources = @import("hip_sources");
const view = @import("../model/view.zig");
const win = @import("window.zig");
const plan = @import("plan.zig");
const state = @import("state.zig");
const round = @import("../engine/round.zig");
const draw = @import("hip_serve").draw;

const testing = std.testing;

const n_layers = 32;

/// Device addresses of the weights: far from the arena (the fake runtime's `hipMalloc` hands out low ones) and from the plan buffer.
const Weights = struct {
    next: u64 = 0x7000_0000_0000,

    fn take(w: *Weights, bytes: u64) u64 {
        const at = w.next;
        w.next += (bytes + 4095) / 4096 * 4096 + 4096;
        return at;
    }

    fn vec(w: *Weights, n: u64) u64 {
        return w.take(4 * n);
    }

    /// A 4-bit, group 64 MLX projection of `n` rows over `k` columns.
    fn proj(w: *Weights, n: u32, k: u32) hip.quant.Projection {
        const bytes = @as(u64, n) * k;
        return .{ .n = n, .k = k, .handle = .{ .mlx = .{
            .words = w.take(bytes / 2),
            .scale = w.take(bytes / 32),
            .bias = w.take(bytes / 32),
            .tables = .bf16,
            .bits = 4,
            .group = 64,
        } } };
    }
};

/// Qwen3.5-4B: 32 layers, every fourth full attention (8), the others gated delta-net (24); a tied 4-bit head.
fn model4b(gpa: std.mem.Allocator, w: *Weights, act: view.Kind, gate_up: bool) !view.Model {
    var spec: view.Spec = .{
        .hidden = 2560,
        .intermediate = 9216,
        .n_layers = n_layers,
        .heads = 16,
        .kv_heads = 4,
        .head_dim = 256,
        .key_heads = 16,
        .value_heads = 32,
        .key_dim = 128,
        .value_dim = 128,
        .conv = 4,
        .vocab = 248320,
        .eps = 1e-6,
        .rope_theta = 1e7,
        .rotary_dim = 64,
        .bits = 4,
        .group = 64,
        .experts = 0,
        .top_k = 0,
        .moe_width = 0,
        .shared_width = 0,
        .mtp_layers = 0,
    };
    for (0..n_layers) |i| spec.full_layers[i] = (i + 1) % 4 == 0;
    const h: u32 = @intCast(spec.hidden);
    const inter: u32 = @intCast(spec.intermediate);
    const ls = try gpa.alloc(view.Layer, n_layers);
    for (ls, 0..) |*l, i| {
        const mlp: view.Mlp = .{ .dense = .{
            .gate = w.proj(inter, h),
            .up = w.proj(inter, h),
            .down = w.proj(h, inter),
            .gate_up = if (gate_up) w.proj(2 * inter, h) else null,
        } };
        if (spec.full(i)) {
            l.* = .{ .full = .{
                .input_norm = w.vec(h),
                .post_norm = w.vec(h),
                .q = w.proj(@intCast(2 * spec.heads * spec.head_dim), h),
                .k = w.proj(@intCast(spec.kv_heads * spec.head_dim), h),
                .v = w.proj(@intCast(spec.kv_heads * spec.head_dim), h),
                .o = w.proj(h, @intCast(spec.heads * spec.head_dim)),
                .q_norm = w.vec(spec.head_dim),
                .k_norm = w.vec(spec.head_dim),
                .mlp = mlp,
            } };
        } else {
            const ch = view.convChannels(spec);
            l.* = .{ .linear = .{
                .input_norm = w.vec(h),
                .post_norm = w.vec(h),
                .qkv = w.proj(@intCast(ch), h),
                .z = w.proj(@intCast(spec.valueWidth()), h),
                .a = w.proj(@intCast(spec.value_heads), h),
                .b = w.proj(@intCast(spec.value_heads), h),
                .out = w.proj(h, @intCast(spec.valueWidth())),
                .conv = w.vec(ch * spec.conv),
                .a_log = w.vec(spec.value_heads),
                .dt_bias = w.vec(spec.value_heads),
                .gnorm = w.vec(spec.value_dim),
                .mlp = mlp,
            } };
        }
    }
    const head = w.proj(@intCast(spec.vocab), h);
    return .{
        .spec = spec,
        .act = act,
        .embed = head,
        .layers = ls,
        .final_norm = w.vec(h),
        .head = head,
        .qk = .{ .q_weight = w.vec(spec.key_dim), .k_weight = w.vec(spec.key_dim), .eps = 1e-6 },
    };
}

/// What differs between the rounds of the tests: the shape of the plan and of the draws, and the MLP's form.
const Shape = struct {
    rows: usize = 1,
    slots: usize = 2,
    span: usize = 512,
    /// The top-k launch's stride (0: none).
    stride: usize = 0,
    argmax: bool = true,
    /// gfx900's loader stacks gate and up (one launch); without, the products are two launches.
    gate_up: bool = true,
};

const How = enum {
    /// no capture, no side stream: the launches of an eager round on one stream
    eager,
    /// captured, with no side stream (`branches=off`, tensor parallel)
    plain,
    /// captured with the side stream, as the engine records it
    forked,
};

const plan_at: u64 = 0x6000_0000_0000;
/// Where a verify round's draws land (the drawer's own buffers).
const drawer_at: u64 = 0x5000_0000_0000;

/// The runtime, the launcher over it, the arena and the model a round needs. Built in place: the launcher and the arena point into it.
const Rig = struct {
    gpa: std.mem.Allocator,
    trace: hip.capture_trace.Trace,
    rt: hip.Runtime,
    launcher: hip.Launcher,
    stream: hip.Stream,
    fork: hip.Fork,
    log: hip.arena.Log,
    arena: hip.Arena,
    model: view.Model,
    snaps: [n_layers][2]u64,
    shape: Shape,
    layout: plan.Layout,

    fn create(gpa: std.mem.Allocator, shape: Shape) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.gpa = gpa;
        r.shape = shape;
        r.trace = .init(gpa);
        errdefer r.trace.deinit();
        r.trace.install();
        errdefer r.trace.uninstall();
        r.rt = r.trace.runtimeOf();
        r.launcher = try hip.Launcher.load(&r.rt, .{}, hip.kernels.images);
        errdefer r.launcher.unload();
        r.stream = try hip.Stream.init(&r.rt);
        errdefer r.stream.deinit();
        r.fork = try hip.Fork.init(&r.rt);
        errdefer r.fork.deinit();
        r.log = .{ .gpa = gpa };
        errdefer r.log.deinit();
        r.arena = try hip.Arena.init(&r.rt, 1 << 30);
        errdefer r.arena.deinit();
        r.arena.log = &r.log;
        r.trace.arena_log = &r.log;
        const chip = hip.Caps.of(hip.kernels.arch).?;
        var w: Weights = .{};
        r.model = try model4b(gpa, &w, if (chip.act == .f16) .f16 else .bf16, shape.gate_up);
        r.layout = plan.Layout.of(shape.rows, shape.slots, n_layers);
        return r;
    }

    fn destroy(r: *Rig) void {
        const gpa = r.gpa;
        gpa.free(r.model.layers);
        r.arena.deinit();
        r.log.deinit();
        r.fork.deinit();
        r.stream.deinit();
        r.launcher.unload();
        r.trace.uninstall();
        r.trace.deinit();
        gpa.destroy(r);
    }

    fn ops(r: *Rig, how: How) hip.ops.Ops {
        const chip = hip.Caps.of(hip.kernels.arch).?;
        var o: hip.ops.Ops = .{ .l = &r.launcher, .bf16 = chip.act != .f16, .stream = r.stream.handle, .arena = &r.arena };
        if (how == .forked) o.fork = &r.fork;
        return o;
    }

    /// The window's plan and draws of `shape` and the round's forward on `o`, as the engine's `verify` and `body` set them up.
    fn forward(r: *Rig, o: hip.ops.Ops) !void {
        const s = r.shape;
        const l = r.layout;
        try win.kept(o, &r.model, s.rows, &r.snaps);
        const p: hip.plan_ops.Plan = .{
            .args = .{
                .pos = plan_at + 4 * l.pos,
                .slot = plan_at + 4 * l.slot,
                .first = plan_at + 4 * l.first,
                .count = plan_at + 4 * l.count,
                .desc = plan_at + 4 * l.desc,
                .snaps = plan_at + 4 * l.snaps,
                .pages = @intCast(state.tableWord(n_layers)),
                .pool = 64,
            },
            .rows = s.rows,
            .slots = s.slots,
        };
        const rd: win.Round = .{
            .plan = p,
            .tokens = plan_at + 4 * l.tokens,
            .span = s.span,
            .inputs = &r.snaps,
            .positions3 = plan_at + 4 * l.rope,
            .rotary = plan_at + 4 * l.rotary,
        };
        // a queued round's pick replaces the plan's first word; the host mirror is read from the plan's `mirror` words
        const dl: draw.Launches = .{
            .argmax = s.argmax,
            .argmax_at = plan_at + 4 * l.tokens,
            .mirror_at = plan_at + 4 * l.mirror,
            .rows = s.rows,
            .stride = s.stride,
            .ks = plan_at + 4 * l.ks,
            .ids = drawer_at,
            .vals = drawer_at + (1 << 20),
        };
        _ = try round.forwardOn(o, &r.model, rd, null, dl);
    }

    /// Records one round, eagerly on the stream or captured (with the side stream when `how` is `forked`).
    fn record(r: *Rig, how: How) !void {
        r.trace.clear();
        r.log.items.clearRetainingCapacity();
        r.arena.used = 0;
        r.arena.reset();
        const o = r.ops(how);
        if (how == .eager) return r.forward(o);
        try hip.graph.beginCapture(r.stream, .thread_local);
        r.forward(o) catch |e| {
            if (hip.graph.endCapture(r.stream)) |g| {
                var dead = g;
                dead.deinit();
            } else |_| {}
            return e;
        };
        var g = try hip.graph.endCapture(r.stream);
        g.deinit();
    }

    fn hazards(r: *Rig) !hip.hazard.Analysis {
        return hip.hazard.analyze(r.gpa, &r.trace, .{
            .arena_base = r.arena.base,
            .arena_len = r.arena.len,
            .arena_log = r.log.items.items,
            .plan_lo = plan_at,
            .plan_hi = plan_at + 4 * r.layout.words,
            .snaps = &r.snaps,
        });
    }
};

/// How many launches of `name` the recorded round holds.
fn count(r: *const Rig, name: []const u8) usize {
    var n: usize = 0;
    for (r.trace.nodes.items) |node| {
        if (std.mem.eql(u8, node.name, name)) n += 1;
    }
    return n;
}

/// The launches whose symbol starts with `prefix`, in launch order.
fn nodesOf(gpa: std.mem.Allocator, r: *const Rig, prefix: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    for (r.trace.nodes.items, 0..) |node, i| {
        if (std.mem.startsWith(u8, node.name, prefix)) try out.append(gpa, @intCast(i));
    }
    return out.toOwnedSlice(gpa);
}

/// The pairs of launches the captured graph leaves unordered (neither follows the other through its edges), by (second, first).
fn unordered(gpa: std.mem.Allocator, r: *const Rig) ![]hip.hazard.Pair {
    var c = try r.trace.ancestors(gpa);
    defer c.deinit(gpa);
    var out: std.ArrayList(hip.hazard.Pair) = .empty;
    errdefer out.deinit(gpa);
    for (0..r.trace.nodes.items.len) |j| for (0..j) |i| {
        if (!c.follows(j, i)) try out.append(gpa, .{ .first = @intCast(i), .second = @intCast(j) });
    };
    return out.toOwnedSlice(gpa);
}

fn contains(set: []const u32, n: u32) bool {
    return std.mem.indexOfScalar(u32, set, n) != null;
}

test "round branches: the 4B round's launches" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.forked);
    try testing.expectEqual(@as(usize, 0), rig.trace.stray);
    try testing.expectEqual(@as(usize, 0), rig.trace.unobserved);
    try testing.expect(!rig.trace.invalidated);
    try testing.expectEqual(@as(usize, 333), rig.trace.nodes.items.len);
    // per full layer: two rope launches, one cache write, the causal chain of four launches and the gate
    try testing.expectEqual(@as(usize, 16), count(rig, "tf_qk_rope"));
    try testing.expectEqual(@as(usize, 8), count(rig, "tf_plan_kv_write"));
    try testing.expectEqual(@as(usize, 8), count(rig, "tf_softmax_stats"));
    try testing.expectEqual(@as(usize, 8), count(rig, "tf_sum_partials"));
    try testing.expectEqual(@as(usize, 8), count(rig, "tf_attn_gate"));
    // per linear layer: the conv with its split, norms and gates, the delta rule, the gated norm
    try testing.expectEqual(@as(usize, 24), count(rig, "tf_conv_split"));
    try testing.expectEqual(@as(usize, 24), count(rig, "tf_gdn_plan_128"));
    try testing.expectEqual(@as(usize, 24), count(rig, "tf_gnorm_out"));
}

test "round branches: the table knows every launch and every pointer is a weight, the arena or the plan" {
    // the argmax pick goes to a plan word, so nothing is written outside the arena and the plan
    {
        const rig = try Rig.create(testing.allocator, .{});
        defer rig.destroy();
        for ([_]How{ .eager, .forked }) |how| {
            try rig.record(how);
            var a = try rig.hazards();
            defer a.deinit();
            try testing.expectEqual(@as(usize, 0), a.unknown.len);
            try testing.expectEqual(@as(usize, 0), a.dangling.len);
            try testing.expectEqual(@as(usize, 0), a.external_writes.len);
        }
    }
    // the top-k launch stores its candidates in the drawer's buffers: those two writes, and only those
    {
        const rig = try Rig.create(testing.allocator, .{ .stride = 32, .argmax = false });
        defer rig.destroy();
        try rig.record(.forked);
        var a = try rig.hazards();
        defer a.deinit();
        try testing.expectEqual(@as(usize, 0), a.unknown.len);
        try testing.expectEqual(@as(usize, 0), a.dangling.len);
        try testing.expectEqual(@as(usize, 2), a.external_writes.len);
        for (a.external_writes) |w| try testing.expectEqualStrings("tf_topk", rig.trace.nameOf(w.node));
    }
}

test "round branches: only the K rope and the cache write of a full layer are independent of its Q rope" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.eager);
    var a = try rig.hazards();
    defer a.deinit();
    const independent = try a.independent(testing.allocator);
    defer testing.allocator.free(independent);
    const ropes = try nodesOf(testing.allocator, rig, "tf_qk_rope");
    defer testing.allocator.free(ropes);
    const writes = try nodesOf(testing.allocator, rig, "tf_plan_kv_write");
    defer testing.allocator.free(writes);
    try testing.expectEqual(@as(usize, 16), ropes.len);
    try testing.expectEqual(@as(usize, 8), writes.len);
    // by (second, first): per layer, (q, k) then (q, write) for the layer's launches q < k < write
    var expected: std.ArrayList(hip.hazard.Pair) = .empty;
    defer expected.deinit(testing.allocator);
    for (0..8) |layer| {
        const q = ropes[2 * layer];
        const k = ropes[2 * layer + 1];
        const w = writes[layer];
        try testing.expect(q < k and k < w);
        try expected.append(testing.allocator, .{ .first = q, .second = k });
        try expected.append(testing.allocator, .{ .first = q, .second = w });
    }
    try testing.expectEqualSlices(hip.hazard.Pair, expected.items, independent);
}

test "round branches: every hazard of the forked graph is ordered by an edge, and what it leaves unordered is what the buffers allow" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.forked);
    try testing.expectEqual(@as(usize, 0), rig.trace.stray);
    var a = try rig.hazards();
    defer a.deinit();
    var closure = try rig.trace.ancestors(testing.allocator);
    defer closure.deinit(testing.allocator);
    const open = try a.uncovered(testing.allocator, closure);
    defer testing.allocator.free(open);
    try testing.expectEqual(@as(usize, 0), open.len);
    // the graph orders every pair the memory orders; the pairs it leaves open are exactly the independent ones (the fork takes all that there are)
    const independent = try a.independent(testing.allocator);
    defer testing.allocator.free(independent);
    const left = try unordered(testing.allocator, rig);
    defer testing.allocator.free(left);
    try testing.expectEqualSlices(hip.hazard.Pair, independent, left);
    // and the conflicts are many: most of the round is a chain through the buffers
    try testing.expect(a.conflicts.len > 1000);
}

test "round branches: a forked layer is a diamond around the K rope and the cache write" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.forked);
    var closure = try rig.trace.ancestors(testing.allocator);
    defer closure.deinit(testing.allocator);
    const ropes = try nodesOf(testing.allocator, rig, "tf_qk_rope");
    defer testing.allocator.free(ropes);
    const writes = try nodesOf(testing.allocator, rig, "tf_plan_kv_write");
    defer testing.allocator.free(writes);
    const nodes = rig.trace.nodes.items;
    for (0..8) |layer| {
        const q = ropes[2 * layer];
        const k = ropes[2 * layer + 1];
        const w = writes[layer];
        // Q on the main stream, K and the write on the side stream, both after the projections
        try testing.expectEqual(@as(u32, 0), nodes[q].stream);
        try testing.expectEqual(@as(u32, 1), nodes[k].stream);
        try testing.expectEqual(@as(u32, 1), nodes[w].stream);
        try testing.expectEqual(w, k + 1);
        try testing.expectEqual(@as(usize, 1), nodes[q].deps.len);
        // K starts after the same launch as Q; the side stream also still names its own last launch of the layer before (which that launch follows)
        const start = nodes[q].deps[0];
        try testing.expect(contains(nodes[k].deps, start));
        for (nodes[k].deps) |d| try testing.expect(d == start or closure.follows(start, d));
        try testing.expectEqualSlices(u32, &.{k}, nodes[w].deps);
        // the first launch after the branches waits for both
        const next = w + 1;
        try testing.expect(std.mem.startsWith(u8, nodes[next].name, "tf_plan_score"));
        try testing.expectEqual(@as(usize, 2), nodes[next].deps.len);
        try testing.expect(contains(nodes[next].deps, q) and contains(nodes[next].deps, w));
    }
    // nothing else left the main stream
    var side: usize = 0;
    for (nodes) |n| side += @intFromBool(n.stream != 0);
    try testing.expectEqual(@as(usize, 16), side);
}

/// A hash of everything a launch is: symbol, geometry, argument pack with its layout, and the arena as it stood.
fn fingerprint(gpa: std.mem.Allocator, r: *const Rig) ![]u64 {
    const out = try gpa.alloc(u64, r.trace.nodes.items.len);
    for (r.trace.nodes.items, out) |n, *h| {
        var w = std.hash.Wyhash.init(0);
        w.update(n.name);
        w.update(std.mem.asBytes(&n.grid));
        w.update(std.mem.asBytes(&n.block));
        w.update(std.mem.asBytes(&n.shared));
        w.update(n.bytes);
        w.update(std.mem.sliceAsBytes(n.offsets));
        w.update(std.mem.asBytes(&n.arena_items));
        h.* = w.final();
    }
    return out;
}

test "round branches: forking changes no launch" {
    // one rig at a time: the fake runtime is the thread's active trace
    var prints: [3][]u64 = undefined;
    var logs: [3][]hip.arena.Log.Item = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        testing.allocator.free(prints[i]);
        testing.allocator.free(logs[i]);
    };
    for ([_]How{ .eager, .plain, .forked }, 0..) |how, i| {
        const rig = try Rig.create(testing.allocator, .{});
        defer rig.destroy();
        try rig.record(how);
        prints[i] = try fingerprint(testing.allocator, rig);
        logs[i] = try testing.allocator.dupe(hip.arena.Log.Item, rig.log.items.items);
        made += 1;
        if (how == .plain) {
            // without the side stream a captured round is one chain
            for (rig.trace.nodes.items, 0..) |n, at| {
                try testing.expectEqual(@as(u32, 0), n.stream);
                try testing.expectEqual(@as(usize, @intFromBool(at > 0)), n.deps.len);
            }
            const left = try unordered(testing.allocator, rig);
            defer testing.allocator.free(left);
            try testing.expectEqual(@as(usize, 0), left.len);
        }
    }
    try testing.expectEqual(@as(usize, 333), prints[0].len);
    try testing.expectEqualSlices(u64, prints[0], prints[1]);
    try testing.expectEqualSlices(u64, prints[0], prints[2]);
    try testing.expectEqualSlices(hip.arena.Log.Item, logs[0], logs[1]);
    try testing.expectEqualSlices(hip.arena.Log.Item, logs[0], logs[2]);
}

test "round branches: a counting round never forks" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.eager);
    const events = rig.trace.events.items.len;
    const streams = rig.trace.streams.items.len;
    rig.trace.clear();
    var arena = hip.Arena.counting();
    var o = rig.ops(.forked);
    o.stream = hip.counting;
    o.arena = &arena;
    try rig.forward(o);
    try testing.expectEqual(@as(usize, 0), rig.trace.nodes.items.len);
    try testing.expectEqual(events, rig.trace.events.items.len);
    try testing.expectEqual(streams, rig.trace.streams.items.len);
    try testing.expect(arena.peak > 0);
}

test "round branches: the two products of an unstacked MLP are recorded and ordered too" {
    const rig = try Rig.create(testing.allocator, .{ .gate_up = false });
    defer rig.destroy();
    try rig.record(.forked);
    var a = try rig.hazards();
    defer a.deinit();
    try testing.expectEqual(@as(usize, 0), a.unknown.len);
    var closure = try rig.trace.ancestors(testing.allocator);
    defer closure.deinit(testing.allocator);
    const open = try a.uncovered(testing.allocator, closure);
    defer testing.allocator.free(open);
    try testing.expectEqual(@as(usize, 0), open.len);
    const independent = try a.independent(testing.allocator);
    defer testing.allocator.free(independent);
    const left = try unordered(testing.allocator, rig);
    defer testing.allocator.free(left);
    // the graph never leaves a pair open that the buffers do not allow
    for (left) |p| {
        const found = for (independent) |q| {
            if (q.first == p.first and q.second == p.second) break true;
        } else false;
        try testing.expect(found);
    }
}

/// Removes node `from` from the dependencies of node `to`.
fn dropEdge(r: *Rig, from: u32, to: u32) !void {
    const node = &r.trace.nodes.items[to];
    var keep: std.ArrayList(u32) = .empty;
    defer keep.deinit(r.gpa);
    for (node.deps) |d| if (d != from) try keep.append(r.gpa, d);
    try testing.expect(keep.items.len < node.deps.len);
    const owned = try r.gpa.dupe(u32, keep.items);
    r.gpa.free(node.deps);
    node.deps = owned;
}

test "round branches: the check finds a join that is missing" {
    const rig = try Rig.create(testing.allocator, .{});
    defer rig.destroy();
    try rig.record(.forked);
    const ropes = try nodesOf(testing.allocator, rig, "tf_qk_rope");
    defer testing.allocator.free(ropes);
    const writes = try nodesOf(testing.allocator, rig, "tf_plan_kv_write");
    defer testing.allocator.free(writes);
    const q = ropes[0];
    const k = ropes[1];
    const w = writes[0];
    const next = w + 1;
    // the scores no longer wait for the cache write: a read of the layer's K/V after its write
    try dropEdge(rig, w, next);
    {
        var a = try rig.hazards();
        defer a.deinit();
        var closure = try rig.trace.ancestors(testing.allocator);
        defer closure.deinit(testing.allocator);
        const open = try a.uncovered(testing.allocator, closure);
        defer testing.allocator.free(open);
        try testing.expect(open.len >= 1);
        var seen = false;
        for (open) |c| {
            if (c.first == w and c.second == next) {
                seen = true;
                try testing.expect(c.kind.raw);
            }
        }
        try testing.expect(seen);
    }
    // the cache write loses its dependence on the K rope that makes its input: a read of k32 after its write
    try rig.record(.forked);
    try dropEdge(rig, k, w);
    {
        var a = try rig.hazards();
        defer a.deinit();
        var closure = try rig.trace.ancestors(testing.allocator);
        defer closure.deinit(testing.allocator);
        const open = try a.uncovered(testing.allocator, closure);
        defer testing.allocator.free(open);
        var seen = false;
        for (open) |c| {
            if (c.first == k and c.second == w) {
                seen = true;
                try testing.expect(c.kind.raw);
            }
        }
        try testing.expect(seen);
    }
    // and the Q rope that the scores read
    try rig.record(.forked);
    try dropEdge(rig, q, next);
    {
        var a = try rig.hazards();
        defer a.deinit();
        var closure = try rig.trace.ancestors(testing.allocator);
        defer closure.deinit(testing.allocator);
        const open = try a.uncovered(testing.allocator, closure);
        defer testing.allocator.free(open);
        var seen = false;
        for (open) |c| {
            if (c.first == q and c.second == next) seen = true;
        }
        try testing.expect(seen);
    }
}

// ---- the table of what each kernel reads and writes, against the kernels' sources ----

fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// The parameter list of the next definition of `name` at or after `at` in `text` (a call, a declaration or a longer name is not one), and where its body starts.
fn definition(text: []const u8, name: []const u8, at: usize) ?struct { params: []const u8, body: usize } {
    var i = at;
    while (std.mem.indexOfPos(u8, text, i, name)) |p| {
        i = p + name.len;
        if (p > 0 and isIdent(text[p - 1])) continue;
        var j = i;
        while (j < text.len and std.ascii.isWhitespace(text[j])) j += 1;
        if (j >= text.len or text[j] != '(') continue;
        var depth: usize = 0;
        var k = j;
        while (k < text.len) : (k += 1) {
            if (text[k] == '(') depth += 1;
            if (text[k] == ')') {
                depth -= 1;
                if (depth == 0) break;
            }
        }
        if (k >= text.len) continue;
        var m = k + 1;
        while (m < text.len and std.ascii.isWhitespace(text[m])) m += 1;
        if (m >= text.len or text[m] != '{') continue;
        return .{ .params = text[j + 1 .. k], .body = m };
    }
    return null;
}

/// `params` without its `//` comments, split at its top-level commas.
fn parameters(a: std.mem.Allocator, params: []const u8) ![][]const u8 {
    var clean: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, params, '\n');
    while (lines.next()) |line| {
        const cut = std.mem.indexOf(u8, line, "//") orelse line.len;
        try clean.appendSlice(a, line[0..cut]);
        try clean.append(a, ' ');
    }
    var out: std.ArrayList([]const u8) = .empty;
    const text = clean.items;
    var depth: usize = 0;
    var start: usize = 0;
    for (text, 0..) |c, i| {
        switch (c) {
            '(', '<', '[', '{' => depth += 1,
            ')', '>', ']', '}' => depth -|= 1,
            ',' => if (depth == 0) {
                try out.append(a, std.mem.trim(u8, text[start..i], " \t\r\n"));
                start = i + 1;
            },
            else => {},
        }
    }
    const last = std.mem.trim(u8, text[start..], " \t\r\n");
    if (last.len > 0) try out.append(a, last);
    return out.items;
}

/// What is wrong with `arg` as kernel parameter `param`, or null.
fn disagrees(arg: hip.hazard.table.Arg, param: []const u8) ?[]const u8 {
    // the outermost pointer is the last star: what it points to is what comes before it (`int* const* p` points to a const pointer)
    const star = std.mem.lastIndexOfScalar(u8, param, '*');
    switch (arg) {
        .val, .words => return if (star != null) "the table has a scalar or a struct, the kernel a pointer" else null,
        .ptr => |m| {
            const s = star orelse return "the table has a pointer, the kernel a value";
            const is_const = std.mem.indexOf(u8, param[0..s], "const") != null;
            if (m == .r and !is_const) return "the table says read-only, the kernel's pointer is not const";
            if (m != .r and is_const) return "the table says written, the kernel's pointer is const";
            return null;
        },
    }
}

test "round branches: the table's pointers agree with the kernels' signatures" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var checked: usize = 0;
    var params_checked: usize = 0;
    var failures: usize = 0;
    var missing: std.ArrayList([]const u8) = .empty;
    defer missing.deinit(gpa);
    for (hip.hazard.table.entries) |e| {
        const name = e.src orelse continue;
        var found = false;
        for (sources.files) |file| {
            var at: usize = 0;
            while (definition(file.text, name, at)) |d| : (at = d.body) {
                found = true;
                const ps = try parameters(arena.allocator(), d.params);
                if (ps.len != e.args.len) {
                    std.debug.print("round branches: {s} ({s}) has {d} parameters, the table {d}\n", .{ name, file.path, ps.len, e.args.len });
                    failures += 1;
                    continue;
                }
                for (e.args, ps, 0..) |arg, param, i| {
                    params_checked += 1;
                    if (disagrees(arg, param)) |why| {
                        std.debug.print("round branches: {s} ({s}) parameter {d} `{s}`: {s}\n", .{ name, file.path, i, param, why });
                        failures += 1;
                    }
                }
            }
        }
        if (found) checked += 1 else try missing.append(gpa, name);
    }
    // defined by a macro in the vision sources, not a plain definition (the text-rope of the `torch_text` path; no decode round launches it)
    for (missing.items) |name| {
        std.debug.print("round branches: no plain definition of {s} in the sources\n", .{name});
        try testing.expectEqualStrings("tf_qvh_text_rope", name);
    }
    try testing.expectEqual(@as(usize, 0), failures);
    try testing.expect(checked >= 40);
    try testing.expect(params_checked >= 200);
}
