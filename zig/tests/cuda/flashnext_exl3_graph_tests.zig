//! flashnext-exl3-graph: the EXL3 offload pool planned on the device (cuda_exl3_dplan.zig) and the decode step as one
//! CUDA graph (cuda_exl3_graph.zig).
//!
//!   flashnext-exl3-graph plan                               synthetic pools: the device planner's lines, stamps,
//!                                                          counts and tables equal the host planner's after random
//!                                                          rounds, every resident line holds its expert's bytes;
//!                                                          the mapped-bank copy kernel's bandwidth vs the copy engine
//!   flashnext-exl3-graph RANK WORLD PORT DIR eq GOLDEN [STEPS] [MODES]
//!                                                          the real pack at tp=1/2: each golden prompt prefilled,
//!                                                          then STEPS greedy decode rounds and the same continuation
//!                                                          as verify windows (rows 3,1,4,2,...) under each mode
//!                                                          (off = host cut, dplan = device plan eager, graph):
//!                                                          every step's logits bit-equal to off's, ms/token, hit rate
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const dplan = X.dplan;
const graph = X.graph;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;
const cap_default: usize = 32768;
const out_rows: usize = 64;

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

// -- synthetic pools: device planner == host planner -----------------------------------------------------------------

const syn_dims = 256;
const syn_width = 64;

fn synPool(gpu: Gpu, routed: [][3][]const u8, lines: usize, max_rows: usize) !fpool.Pool {
    return fpool.Pool.init(gpu.gpa, gpu.io, gpu.d, .{ .routed = routed }, .{ .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0 }, .{
        .layer = 0,
        .lines = lines,
        .dims = syn_dims,
        .width = syn_width,
        .max_rows = max_rows,
        .slots = 11,
        .direct = true,
        .double_buffer = false,
    }, null);
}

fn planCase(gpu: Gpu, lines: usize, max_rows: usize, rounds: usize, seed: u64) !void {
    const a = gpu.gpa;
    const tiles = (syn_dims / 16) * (syn_width / 16);
    // a pinned bank of 512 experts, widths 2..5 half-bits, random bytes
    var sizes: [512][3]usize = undefined;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var total: usize = 0;
    for (&sizes) |*e| for (e) |*b| {
        b.* = tiles * 16 * (2 + rnd.uintLessThan(usize, 4));
        total += (b.* + 255) / 256 * 256;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    rnd.bytes(bank.bytes);
    const routed = try a.alloc([3][]const u8, 512);
    defer a.free(routed);
    var at: usize = 0;
    for (routed, sizes) |*r, sz| for (0..3) |w| {
        r[w] = bank.bytes[at..][0..sz[w]];
        at += (sz[w] + 255) / 256 * 256;
    };
    var host = try synPool(gpu, routed, lines, max_rows);
    defer host.deinit();
    var dev = try synPool(gpu, routed, lines, max_rows);
    defer dev.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var pools = [_]*fpool.Pool{&dev};
    const dp = try dplan.DevPlan.init(a, gpu.d, &pools);
    defer dp.deinit();
    var picks_dev = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * 11 * 4);
    defer picks_dev.free();
    const picks = try a.alloc(i32, max_rows * 11);
    defer a.free(picks);
    // skewed routing: a hot set of 24 experts takes most picks, so rounds hit, miss and evict
    var host_ns: u64 = 0;
    for (0..rounds) |round| {
        const rows = 1 + rnd.uintLessThan(usize, max_rows);
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
        const n = rows * 11;
        // host
        const t0 = check.now(gpu.io);
        const rp = try host.plan(picks[0..n], rows);
        host_ns += @intCast(check.now(gpu.io) - t0);
        try check.expect(rp.n_passes == 1, "a synthetic round split into {d} passes", .{rp.n_passes});
        try host.stage(rp, 0, s);
        try host.release(rp, 0, s);
        // device
        try dp.toDevice(s);
        try picks_dev.upload(0, std.mem.sliceAsBytes(picks[0..n]));
        try dp.plan(s, 0, picks_dev.ptr, rows);
        if (round % 37 == 36) try dp.toHost(); // owner switches mid-stream keep both views consistent
    }
    try s.synchronize();
    try host.lines.drain();
    try dp.toHost();
    // residency, stamps, counts
    var same_lines: usize = 0;
    for (host.slots_tab, dev.slots_tab, host.stamp_tab, dev.stamp_tab) |hs, ds, ht, dt| same_lines += @intFromBool(hs == ds and ht == dt);
    const same_counts = std.mem.eql(u32, &host.counts[0], &dev.counts[0]);
    std.debug.print("RESULT plan lines={d} max_rows={d} rounds={d}: lines+stamps equal {d}/{d}, counts equal {}, host hits {d}/{d} copies {d} evictions {d}; device hits {d}/{d} copies {d} evictions {d}; host plan {d:.1} us a round\n", .{ lines, max_rows, rounds, same_lines, lines, same_counts, host.stats.hit_slots, host.stats.hit_slots + host.stats.miss_slots, host.stats.misses, host.stats.evictions, dev.stats.hit_slots, dev.stats.hit_slots + dev.stats.miss_slots, dev.stats.misses, dev.stats.evictions, @as(f64, @floatFromInt(host_ns)) / 1e3 / @as(f64, @floatFromInt(rounds)) });
    try check.expect(same_lines == lines and same_counts, "device planner diverged from the host planner", .{});
    try check.expect(host.stats.hit_slots == dev.stats.hit_slots and host.stats.misses == dev.stats.misses and host.stats.evictions == dev.stats.evictions, "device planner statistics differ", .{});
    // tables: the same line (relative to each pool's lines) and width for every expert
    const tb = host.lines.tables.len;
    const th = try a.alloc(u8, tb);
    defer a.free(th);
    const td = try a.alloc(u8, tb);
    defer a.free(td);
    try host.lines.tables.download(0, th);
    try dev.lines.tables.download(0, td);
    var bad_tab: usize = 0;
    for (0..512) |id| for (0..3) |w| {
        const ph = std.mem.readInt(u64, th[host.lines.tlay.ptr[w] + id * 8 ..][0..8], .little);
        const pd = std.mem.readInt(u64, td[dev.lines.tlay.ptr[w] + id * 8 ..][0..8], .little);
        const kh = std.mem.readInt(i32, th[host.lines.tlay.k2[w] + id * 4 ..][0..4], .little);
        const kd = std.mem.readInt(i32, td[dev.lines.tlay.k2[w] + id * 4 ..][0..4], .little);
        const rh = if (ph == 0) 0 else ph - host.lines.mem.ptr + 1;
        const rd = if (pd == 0) 0 else pd - dev.lines.mem.ptr + 1;
        bad_tab += @intFromBool(rh != rd or kh != kd);
    };
    // every resident line holds its expert's bytes (the copy kernel read the mapped bank)
    const mem = try a.alloc(u8, dev.lines.mem.len);
    defer a.free(mem);
    try dev.lines.mem.download(0, mem);
    var bad_lines: usize = 0;
    var resident: usize = 0;
    for (dev.slots_tab, 0..) |id, ln| {
        if (id == fpool.offload.Cache.empty_id) continue;
        resident += 1;
        for (0..3) |w| {
            const want = routed[id][w];
            const got = mem[ln * dev.lines.layout.line_bytes + dev.lines.layout.off[w] ..][0..want.len];
            if (!std.mem.eql(u8, want, got)) bad_lines += 1;
        }
    }
    std.debug.print("RESULT plan tables: {d} of 1536 entries differ; {d} resident lines, {d} projections with wrong bytes\n", .{ bad_tab, resident, bad_lines });
    try check.expect(bad_tab == 0 and bad_lines == 0, "device tables or line bytes wrong", .{});
    check.pass("device planner == host planner over {d} rounds of up to {d} rows on {d} lines (tables, lines, stamps, counts, line bytes)", .{ rounds, max_rows, lines });
}

fn copyBench(gpu: Gpu) !void {
    // Flash Next line: gate/up/down 3-bit, K 2560 x N 640: 160 x 40 tiles of 48 bytes = 614,400 bytes each
    const seg: usize = 614_400;
    const n_lines: usize = 30;
    var bank = try cuda.HostBuffer.alloc(gpu.d, n_lines * 3 * seg);
    defer bank.free();
    @memset(bank.bytes, 0x5a);
    var dst = try cuda.DeviceBuffer.alloc(gpu.d, n_lines * 3 * seg);
    defer dst.free();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_dplan);
    defer mod.unload();
    const f = try mod.function("fn_exl3_dcopy");
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    for ([_]usize{ 1, 3, 10 }) |k| {
        // work list: k lines of three segments
        var work: [10 * 9]i64 = undefined;
        for (0..k) |i| for (0..3) |w| {
            const off = (i * 3 + w) * seg;
            work[i * 9 + w * 3 + 0] = @intCast(@intFromPtr(bank.bytes.ptr) + off);
            work[i * 9 + w * 3 + 1] = @intCast(dst.ptr + off);
            work[i * 9 + w * 3 + 2] = @intCast(seg);
        };
        var wbuf = try cuda.DeviceBuffer.alloc(gpu.d, 10 * 9 * 8 + 16);
        defer wbuf.free();
        try wbuf.upload(0, std.mem.sliceAsBytes(work[0 .. k * 9]));
        const cnt = [1]i32{@intCast(k)};
        try wbuf.upload(10 * 9 * 8, std.mem.sliceAsBytes(&cnt));
        for ([_]u32{ 36, 72, 144, 288, 576 }) |blocks| {
            const reps = 20;
            var args: cuda.Args = .{};
            args.add(wbuf.ptr);
            args.add(wbuf.ptr + 10 * 9 * 8);
            try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, s, &args);
            try e0.record(s);
            for (0..reps) |_| try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, s, &args);
            try e1.record(s);
            try s.synchronize();
            const t = (try e0.elapsedMs(e1)) / reps;
            std.debug.print("BENCH dcopy {d} lines ({d:.2} MiB) with {d} blocks: {d:.3} ms = {d:.1} GB/s\n", .{ k, @as(f64, @floatFromInt(k * 3 * seg)) / (1 << 20), blocks, t, @as(f64, @floatFromInt(k * 3 * seg)) / (t * 1e6) });
        }
        // the copy engine, one H2D a segment
        try e0.record(s);
        for (0..20) |_| for (0..k * 3) |j| try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst.ptr + j * seg, bank.bytes[j * seg ..].ptr, seg, s.handle), "cuMemcpyHtoDAsync");
        try e1.record(s);
        try s.synchronize();
        const t = (try e0.elapsedMs(e1)) / 20;
        std.debug.print("BENCH dma   {d} lines ({d:.2} MiB): {d:.3} ms = {d:.1} GB/s\n", .{ k, @as(f64, @floatFromInt(k * 3 * seg)) / (1 << 20), t, @as(f64, @floatFromInt(k * 3 * seg)) / (t * 1e6) });
    }
}

fn planMode(gpu: Gpu) !void {
    try copyBench(gpu);
    try planCase(gpu, 10, 1, 400, 1);
    try planCase(gpu, 47, 4, 600, 2);
    try planCase(gpu, 63, 6, 600, 3);
}

// -- the engine at tp=1/2 -------------------------------------------------------------------------------------------

const Ranks = struct {
    rank: usize,
    link: wire.Link,
    collective: Collective,
    seq: u64 = 10,

    fn open(gpu: Gpu, rank: usize, port: u16) !*Ranks {
        const r = try gpu.gpa.create(Ranks);
        errdefer gpu.gpa.destroy(r);
        const link = try wire.Link.open(gpu.io, "127.0.0.1", port, rank, 600_000);
        errdefer link.close();
        r.* = .{ .rank = rank, .link = link, .collective = try Collective.init(link, rank, 3_600_000) };
        return r;
    }

    fn close(r: *Ranks, gpa: std.mem.Allocator) void {
        r.collective.deinit();
        r.link.close();
        gpa.destroy(r);
    }

    fn swap(r: *Ranks, value: u64) !u64 {
        r.seq += 1;
        const until = wire.deadline(3_600_000);
        if (r.rank == 0) {
            const got = try r.link.receiveHeader(until);
            if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
            try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
            return got.arg;
        }
        try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
        const got = try r.link.receiveHeader(until);
        if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
        return got.arg;
    }
};

const Eng = struct {
    gpu: Gpu,
    rank: usize,
    world: usize,
    ranks: ?*Ranks,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    out: cuda.DeviceBuffer,
    attached: xpass.Attached,

    fn init(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.rank = rank;
        g.world = world;
        g.ranks = if (world == 2) try Ranks.open(gpu, rank, port) else null;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, world, rank);
        if (g.ranks) |r| g.e.collective = &r.collective;
        _ = try g.e.reserve(@min(cap_default, g.e.chunk));
        _ = try g.e.reserveScores(cap_default, @min(cap_default, g.e.chunk));
        g.seq = try g.e.newSeq(cap_default);
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, xpass.budgetFromEnv(0));
        std.debug.print("RESULT rank {d}/{d} load: {d:.1} s; {d} lines a layer x {d} layers ({d:.2} GiB of pool)\n", .{ rank, world, ms(t0, check.now(gpu.io)) / 1e3, g.attached.lines, g.attached.layers, gib(g.attached.device_bytes) });
        return g;
    }

    fn deinit(g: *Eng) void {
        g.out.free();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        if (g.ranks) |r| r.close(g.gpu.gpa);
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    fn swap(g: *Eng, v: u64) !u64 {
        return if (g.ranks) |r| r.swap(v) else v;
    }

    fn forward(g: *Eng, ids: []const u32, hooks: prompt.batch.Hooks) !void {
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, hooks);
        if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    }

    fn round(g: *Eng, tok: u32) !u32 {
        const q = [_]*prompt.Seq{&g.seq};
        var t = tok;
        try g.e.round(flash.Tri, g.tri(), &q, (&t)[0..1], g.out.ptr);
        if (g.ranks) |r| try r.collective.finish(g.stream);
        return prompt.argmax(g.seq.logits);
    }

    fn stats(g: *Eng) !fpool.Stats {
        try dplan.toHost();
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

const modes = [_]graph.Mode{ .off, .dplan, .graph };

const Run = struct {
    ids: []u32,
    /// Per decode step, the hash of the step's whole bf16 logits row; then per window row the same.
    round_h: []u64,
    window_h: []u64,
    ms_token: f64 = 0,
    ms_window_row: f64 = 0,
    hit: f64 = 0,
    copies: f64 = 0,
};

const win_sizes = [_]usize{ 3, 1, 4, 2 };

fn runMode(g: *Eng, ids: []const u32, steps: usize, r: *Run) !void {
    const a = g.gpu.gpa;
    // decode rounds
    try dplan.toHost();
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var tok = prompt.argmax(g.seq.logits);
    r.ids[0] = tok;
    _ = try g.swap(1);
    const before = try g.stats();
    const t0 = check.now(g.gpu.io);
    for (0..steps) |i| {
        tok = try g.round(tok);
        r.ids[i + 1] = tok;
        r.round_h[i] = std.hash.Wyhash.hash(0, g.seq.logits);
    }
    const t1 = check.now(g.gpu.io);
    const after = try g.stats();
    r.ms_token = ms(t0, t1) / @as(f64, @floatFromInt(steps));
    const hits = after.hit_slots - before.hit_slots;
    const all = hits + after.miss_slots - before.miss_slots;
    r.hit = 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1)));
    r.copies = @as(f64, @floatFromInt(after.misses - before.misses)) / @as(f64, @floatFromInt(steps));
    // the same continuation teacher-forced in windows of 3, 1, 4, 2 rows (every row's logits)
    try dplan.toHost();
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var at: usize = 0;
    var k: usize = 0;
    const host = try a.alloc(u16, 4 * vocab);
    defer a.free(host);
    const tw = check.now(g.gpu.io);
    while (at < steps) : (k += 1) {
        const n = @min(win_sizes[k % win_sizes.len], steps - at);
        try g.forward(r.ids[at..][0..n], .{ .logits = .all, .logits_out = g.out.ptr });
        try g.out.download(0, std.mem.sliceAsBytes(host[0 .. n * vocab]));
        for (0..n) |j| r.window_h[at + j] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(host[j * vocab ..][0..vocab]));
        at += n;
    }
    r.ms_window_row = ms(tw, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps));
}

fn eqMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, steps: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, golden_path, a, .limited(1 << 26));
    defer a.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    var all_ok = true;
    for (parsed.value.object.get("prompts").?.array.items) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        var runs: [modes.len]Run = undefined;
        for (&runs) |*r| r.* = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
        defer for (&runs) |*r| {
            a.free(r.ids);
            a.free(r.round_h);
            a.free(r.window_h);
        };
        // warm the pools once (first-touch copies are not any mode's cost)
        graph.force = .off;
        try runMode(g, ids, steps, &runs[0]);
        for (modes, 0..) |m, mi| {
            if (std.mem.indexOf(u8, which, @tagName(m)) == null and m != .off) continue;
            graph.force = m;
            const g0 = graph.stats;
            try runMode(g, ids, steps, &runs[mi]);
            const g1 = graph.stats;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(runs[mi].round_h)) ^ std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(runs[mi].window_h));
            const other = try g.swap(h);
            if (world == 2) try check.expect(other == h, "rank {d}: the two ranks' logits differ ({s})", .{ rank, @tagName(m) });
            const rs = runs[mi];
            var same_r: usize = 0;
            var same_w: usize = 0;
            for (rs.round_h, runs[0].round_h) |x, y| same_r += @intFromBool(x == y);
            for (rs.window_h, runs[0].window_h) |x, y| same_w += @intFromBool(x == y);
            const ok = same_r == steps and same_w == steps and std.mem.eql(u32, rs.ids, runs[0].ids);
            if (mi != 0) all_ok = all_ok and ok;
            if (rank == 0) std.debug.print("RESULT tp={d} {s} {s}: decode {d:.2} ms/token (hit {d:.1}%, {d:.1} copies a token), windows {d:.2} ms/row; logits bit-equal to off: rounds {d}/{d}, window rows {d}/{d}; greedy ids {s}; graph captures {d} replays {d}, eager-dplan steps {d}, declined {d}, capture {d:.1} ms\n", .{ world, p.object.get("name").?.string, @tagName(m), rs.ms_token, rs.hit, rs.copies, rs.ms_window_row, same_r, steps, same_w, steps, if (std.mem.eql(u32, rs.ids, runs[0].ids)) "equal" else "DIFFER", g1.captures - g0.captures, g1.replays - g0.replays, g1.eager_dplan - g0.eager_dplan, g1.declined - g0.declined, @as(f64, @floatFromInt(g1.capture_ns - g0.capture_ns)) / 1e6 });
        }
        graph.force = null;
    }
    if (dplan.current()) |dp| if (rank == 0) std.debug.print("RESULT dplan owner switches: {d} pushes, {d} pulls\n", .{ dp.pushes, dp.pulls });
    try check.expect(all_ok, "rank {d}: a device-plan or graph step's logits differ from the host cut's", .{rank});
    check.pass("rank {d}/{d}: decode rounds and verify windows on the device plan and in graphs are bit-identical to the host cut ({d} steps a prompt)", .{ rank, world, steps });
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    if (std.mem.eql(u8, rest[0], "plan")) return planMode(gpu);
    if (rest.len < 6) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const mode = rest[4];
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const steps = if (rest.len > 6) try std.fmt.parseInt(usize, rest[6], 10) else 48;
    const which: []const u8 = if (rest.len > 7) rest[7] else "dplan,graph";
    if (std.mem.eql(u8, mode, "eq")) return eqMode(gpu, rank, world, port, dir, rest[5], steps, which);
    return error.UnknownMode;
}
