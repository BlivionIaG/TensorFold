//! flashnext-exl3-graph plan-elastic: one planner graph replayed across active-line resizes, checked by trace replay.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const X = flash.exl3;
const dplan = X.dplan;
const fpool = flash.exl3_pool;

// a layer a tenth of the real one: lines of a few hundred KiB, so the capacity spans several 2 MiB chunks
const syn_dims = 1280;
const syn_width = 320;
const capacity = 40;
const rows = 4;
const phases = [_]usize{ 40, 28, 17, 10, 40, 12, 33 };
const rounds_per_phase = 40;

const Bank = struct {
    gpa: std.mem.Allocator,
    buf: cuda.HostBuffer,
    routed: [][3][]const u8,

    fn init(gpu: Gpu, seed: u64) !Bank {
        const tiles = (syn_dims / 16) * (syn_width / 16);
        var sizes: [512][3]usize = undefined;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var total: usize = 0;
        for (&sizes) |*e| for (e) |*b| {
            b.* = tiles * 16 * (2 + rnd.uintLessThan(usize, 4));
            total += (b.* + 255) / 256 * 256;
        };
        var buf = try cuda.HostBuffer.alloc(gpu.d, total);
        errdefer buf.free();
        rnd.bytes(buf.bytes);
        const routed = try gpu.gpa.alloc([3][]const u8, 512);
        var at: usize = 0;
        for (routed, sizes) |*r, sz| for (0..3) |w| {
            r[w] = buf.bytes[at..][0..sz[w]];
            at += (sz[w] + 255) / 256 * 256;
        };
        return .{ .gpa = gpu.gpa, .buf = buf, .routed = routed };
    }

    fn deinit(b: *Bank) void {
        b.gpa.free(b.routed);
        b.buf.free();
    }
};

fn elasticPool(gpu: Gpu, routed: [][3][]const u8) !fpool.Pool {
    return fpool.Pool.init(gpu.gpa, gpu.io, gpu.d, .{ .routed = routed }, .{ .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0 }, .{
        .layer = 0,
        .lines = capacity,
        .dims = syn_dims,
        .width = syn_width,
        .max_rows = rows,
        .slots = 11,
        .direct = true,
        .double_buffer = false,
        .max_passes = fpool.passesFor(fpool.min_lines),
        .elastic = true,
    }, null);
}

/// Skewed routing: a hot set of 24 experts takes most picks, so rounds hit, miss and evict.
fn fillPicks(rnd: std.Random, picks: []i32, round: usize) void {
    for (0..rows) |r| {
        var k: usize = 0;
        while (k < 10) {
            const id: i32 = if (rnd.uintLessThan(u32, 4) != 0) @intCast(rnd.uintLessThan(u32, 24) * 7) else @intCast(rnd.uintLessThan(u32, 512));
            var dup = false;
            for (picks[r * 11 ..][0..k]) |x| dup = dup or x == id;
            if (dup) continue;
            picks[r * 11 + k] = id;
            k += 1;
        }
        picks[r * 11 + 10] = if (round % 3 == 0) -1 else 512;
    }
}

fn tailEmpty(p: *const fpool.Pool, from: usize) !void {
    for (p.slots_tab[from..], p.line_tab[from..], from..) |id, state, ln| {
        try check.expect(state == .empty and id == fpool.offload.Cache.empty_id, "line {d} at or above {d} holds expert {d}", .{ ln, from, id });
    }
}

/// The host's view after a pull: nothing above the active count, every resident line's bytes and table entries right.
fn checkPool(gpu: Gpu, p: *fpool.Pool, routed: [][3][]const u8, active: usize) !void {
    const a = gpu.gpa;
    try check.expect(p.active == active, "pool active {d}, expected {d}", .{ p.active, active });
    try tailEmpty(p, active);
    const lb = p.lines.layout.line_bytes;
    const got = try a.alloc(u8, lb);
    defer a.free(got);
    var resident: [512]?usize = @splat(null);
    for (p.slots_tab[0..active], p.line_tab[0..active], 0..) |id, state, ln| {
        if (state == .empty) continue;
        resident[id] = ln;
        try p.lines.mem.download(ln * lb, got);
        for (0..3) |w| {
            const want = routed[id][w];
            try check.sameBytes("resident line bytes", got[p.lines.layout.off[w]..][0..want.len], want);
        }
    }
    const tl = p.lines.tlay;
    const tab = try a.alloc(u8, tl.bytes);
    defer a.free(tab);
    try p.lines.tables.download(0, tab);
    for (0..512) |e| for (0..3) |w| {
        const ptr = std.mem.readInt(u64, tab[tl.ptr[w] + e * 8 ..][0..8], .little);
        const k2 = std.mem.readInt(i32, tab[tl.k2[w] + e * 4 ..][0..4], .little);
        if (resident[e]) |ln| {
            try check.expect(ptr == p.lines.at(ln, w) and k2 == p.k2[e][w], "expert {d} projection {d}: table {x} width {d}, line {d} is at {x}", .{ e, w, ptr, k2, ln, p.lines.at(ln, w) });
        } else {
            try check.expect(ptr == 0 and k2 == 0, "absent expert {d} has a table entry", .{e});
        }
    };
}

fn freeBytes(gpu: Gpu) !usize {
    var free: usize = 0;
    var total: usize = 0;
    try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

pub fn elasticMode(gpu: Gpu) !void {
    const a = gpu.gpa;
    var bank = try Bank.init(gpu, 0xe1a5);
    defer bank.deinit();
    var p = try elasticPool(gpu, bank.routed);
    defer p.deinit();
    const base = p.lines.mem.ptr;
    const lb = p.lines.layout.line_bytes;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    dplan.demand_trace.requested = true; // the planner keeps its demand trace: the exact replay is the oracle
    defer dplan.demand_trace.requested = false;
    var pools = [_]*fpool.Pool{&p};
    const dp = try dplan.DevPlan.init(a, gpu.d, &pools);
    defer dp.deinit();
    var picks_dev = try cuda.DeviceBuffer.alloc(gpu.d, rows * 11 * 4);
    defer picks_dev.free();
    var picks: [rows * 11]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x51ab);
    const rnd = prng.random();
    fillPicks(rnd, &picks, 1);
    try picks_dev.upload(0, std.mem.sliceAsBytes(&picks));
    try dp.toDevice(s);
    // captured once at full capacity: the planner, its copies on the side stream, the in-place restore
    try s.synchronize();
    try cuda.graph.beginCapture(s, .thread_local);
    try dp.plan(s, 0, picks_dev.ptr, rows);
    try dp.join(s);
    try dp.restore(s, 0);
    var g = try cuda.graph.endCapture(s);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.upload(s);
    std.debug.print("  [elastic] planner graph captured once: {d} nodes, lines at {x}, {d} KiB a line, {d} lines\n", .{ try g.nodeCount(), base, lb / 1024, capacity });

    var total_rounds: usize = 0;
    var previous: usize = capacity;
    for (phases, 0..) |n, phase| {
        if (phase != 0) {
            try s.synchronize();
            const free_before = try freeBytes(gpu);
            const chunks_before = p.lines.window.?.mapped;
            try dp.setActive(0, n);
            const w = p.lines.window.?;
            try check.expect(p.lines.mem.ptr == base, "the lines moved: {x} -> {x}", .{ base, p.lines.mem.ptr });
            try check.expect(w.mapped == w.win.chunksFor(n) and p.lines.mem.len == w.mappedBytes(), "{d} lines: {d} chunks mapped, {d} bytes", .{ n, w.mapped, p.lines.mem.len });
            try check.expect(p.lines.mem.len >= n * lb, "{d} lines but {d} bytes backed", .{ n, p.lines.mem.len });
            if (n > previous) try tailEmpty(&p, previous) else try tailEmpty(&p, n);
            const free_after = try freeBytes(gpu);
            if (w.mapped < chunks_before) {
                const returned = (chunks_before - w.mapped) * w.win.granularity;
                try check.expect(free_after + returned / 4 >= free_before + returned / 2 and free_after > free_before, "a shrink to {d} lines returned {d} KiB of {d} KiB", .{ n, (free_after -| free_before) / 1024, returned / 1024 });
            }
            std.debug.print("  [elastic] {d} -> {d} lines: {d} -> {d} chunks, {d} KiB backed, graph kept\n", .{ previous, n, chunks_before, w.mapped, p.lines.mem.len / 1024 });
            try dp.toDevice(s);
            previous = n;
        }
        for (0..rounds_per_phase) |round| {
            fillPicks(rnd, &picks, round);
            try picks_dev.upload(0, std.mem.sliceAsBytes(&picks));
            try exec.launchOn(s); // the one executable graph: no recapture, no update
            try s.synchronize();
            total_rounds += 1;
        }
        try dp.toHost();
        try checkPool(gpu, &p, bank.routed, n);
    }

    var snapshot = try dp.downloadTraceAfterConsumers(a, 0);
    defer snapshot.deinit();
    const view = try snapshot.view();
    const terminal = try a.alloc(u32, dp.pool_states[0].words);
    defer a.free(terminal);
    try dp.state.download(dp.pool_states[0].byteOffset(), std.mem.sliceAsBytes(terminal));
    const replayed = try dplan.replay.verifyTraceWithFinal(a, view, terminal);
    try check.expect(replayed.rounds == total_rounds and view.dropped() == 0, "replayed {d} of {d} rounds, {d} dropped", .{ replayed.rounds, total_rounds, view.dropped() });
    try check.expect(view.checkpoints() == phases.len, "{d} ownership checkpoints, expected {d}", .{ view.checkpoints(), phases.len });
    try check.expect(replayed.delta.copies > 0 and replayed.delta.evictions > 0 and replayed.delta.inplace > 0, "a path was not exercised: copies {d} evictions {d} in place {d}", .{ replayed.delta.copies, replayed.delta.evictions, replayed.delta.inplace });
    check.pass("elastic planner: one captured graph replayed {d} rounds over {d} resizes ({d}..{d} lines of {d}); lines at {x} throughout; the kernel's decisions equal the exact replay (copies {d}, evictions {d}, in place {d})", .{ total_rounds, phases.len - 1, 10, capacity, capacity, base, replayed.delta.copies, replayed.delta.evictions, replayed.delta.inplace });
}
