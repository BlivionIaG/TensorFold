//! exl3-kvpage: dynamic KV paging for long contexts (docs/plans/exl3-kv-paging.md) on the family's Engine with the
//! EXL3 offload pool, as serve attaches it (one card).
//!
//!   exl3-kvpage DIR trace IDS N[,N..] DECODE OUT [prompt]
//!       the first N ids of IDS (u32 little-endian, real text) prefilled, then DECODE greedy rounds; every attention
//!       row's selected blocks recorded (cuda_exl3_kvpage.Trace) into OUT-N.bin (prompt rows only with `prompt`)
//!   exl3-kvpage none kernel [SLOTS]
//!       the paging kernels on synthetic key lists: every remapped key reads its own host row, misses against an
//!       LRU model on the host, the plan's cost a call
//!   exl3-kvpage DIR cmp IDS N[,N..] DECODE WINDOW SLOTS [NEEDLE_AT]
//!       each prompt through the all-on-device cache, then through the paged one (window WINDOW tokens a layer, SLOTS
//!       4-key block slots shared by the 12 layers): the prefill's last logits and the DECODE greedy tokens must be
//!       bit-equal, and every K/V row of every attention layer equal after the decode (paged rows flushed to host);
//!       ms, copies and device bytes.
//!       NEEDLE_AT (a fraction, e.g. 0.25): a sentence with a 5-digit code inserted that far into the text and a
//!       question about it appended; the greedy answer must contain the code (both caches)
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;

const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const xpre = flash.exl3.prefill;
const kvp = flash.exl3.kvpage;
const vocab = prompt.vocabulary;
const core = @import("core");

const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    attached: xpass.Attached,

    fn init(gpu: Gpu, dir: []const u8, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        g.seq = try g.e.newSeq(cap);
        if (deep_rows > 0) try xpre.reserve(gpu.gpa, gpu.d, g.e, deep_rows);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, xpass.budgetFromEnv(0));
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.debug.print("engine: loaded in {d:.1} s; cap {d}; pool {d} lines a layer; {d:.2} GiB free\n", .{ @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, cap, g.attached.lines, @as(f64, @floatFromInt(free)) / (1 << 30) });
        return g;
    }

    fn deinit(g: *Eng) void {
        xpre.release();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
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

    fn prefill(g: *Eng, ids: []const u32) !f64 {
        try g.reset();
        const t0 = check.now(g.gpu.io);
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    }

    /// `n` greedy rounds from the prefill's logits; the tokens into `out`; ms a round.
    fn decode(g: *Eng, out: []u32) !f64 {
        var dev = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
        defer dev.free();
        var tok = prompt.argmax(g.seq.logits);
        const t0 = check.now(g.gpu.io);
        for (out) |*o| {
            o.* = tok;
            const q = [_]*prompt.Seq{&g.seq};
            try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], dev.ptr);
            tok = prompt.argmax(g.seq.logits);
        }
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6 / @as(f64, @floatFromInt(@max(out.len, 1)));
    }
};

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
    defer a.free(text);
    const ids = try a.alloc(u32, text.len / 4);
    for (ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    return ids;
}

fn parseList(s: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |t| {
        out[n] = try std.fmt.parseInt(usize, t, 10);
        n += 1;
    }
    return n;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "trace")) return traceRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "cmp")) return cmpRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "kernel")) return kernelRun(gpu, if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 6144);
    return error.UnknownMode;
}

fn traceRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all = try readIds(gpu, rest[0]);
    defer a.free(all);
    var ns_buf: [16]usize = undefined;
    const ns = ns_buf[0..try parseList(rest[1], &ns_buf)];
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    const out_prefix = rest[3];
    const with_prompt = rest.len > 4 and std.mem.eql(u8, rest[4], "prompt");
    var max_n: usize = 0;
    for (ns) |n| max_n = @max(max_n, n);
    if (max_n > all.len) return error.NotEnoughIds;
    const o = xpre.optionsFromEnv();
    const g = try Eng.init(gpu, dir, max_n + steps + 64, if (o.enabled) @min(o.rows, (max_n + 255) / 256 * 256) else 0);
    defer g.deinit();
    const toks = try a.alloc(u32, steps);
    defer a.free(toks);
    for (ns) |n| {
        var t = try kvp.Trace.init(a, with_prompt);
        defer t.deinit();
        kvp.trace = &t;
        defer kvp.trace = null;
        const ms_p = try g.prefill(all[0..n]);
        const prompt_rows = t.rows;
        const ms_d = try g.decode(toks);
        std.debug.print("TRACE {d} tokens: prefill {d:.0} ms (traced, {d} rows recorded), {d} decode rounds {d:.1} ms/round (traced: one sync a layer); {d} rows recorded; {d:.1} MiB\n", .{ n, ms_p, prompt_rows, steps, ms_d, t.rows, @as(f64, @floatFromInt(t.out.items.len)) / (1 << 20) });
        std.debug.print("  greedy: {any}\n", .{toks[0..@min(steps, 48)]});
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}-{d}.bin", .{ out_prefix, n });
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = t.out.items });
    }
    check.pass("exl3-kvpage trace written", .{});
}

fn freeMem(gpu: Gpu) !usize {
    var free: usize = 0;
    var total: usize = 0;
    try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

fn gib(x: usize) f64 {
    return @as(f64, @floatFromInt(x)) / (1 << 30);
}

const Run = struct {
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    logits: []u8 = &.{},
    toks: []u32 = &.{},
    /// [layer][K|V] rows 0..fed (all-on-device: downloaded; paged: its host copy after a flush)
    kv: [][]u8 = &.{},
    stats_prefill: kvp.Stats = .{},
    stats_decode: kvp.Stats = .{},
};

fn oneRun(g: *Eng, ids: []const u32, steps: usize, keep_kv: bool) !Run {
    const a = g.gpu.gpa;
    var r: Run = .{};
    r.prefill_ms = try g.prefill(ids);
    r.logits = try a.dupe(u8, g.seq.logits);
    if (g.seq.paged) |pg| r.stats_prefill = try pg.total();
    r.toks = try a.alloc(u32, steps);
    r.decode_ms = try g.decode(r.toks);
    if (g.seq.paged) |pg| {
        const t = try pg.total();
        r.stats_decode = .{ .misses = t.misses - r.stats_prefill.misses, .needed = t.needed - r.stats_prefill.needed, .calls = t.calls - r.stats_prefill.calls, .overflow = t.overflow, .rebases = t.rebases - r.stats_prefill.rebases, .written_rows = t.written_rows - r.stats_prefill.written_rows };
    }
    if (!keep_kv) return r;
    const fed = g.seq.fed;
    const row = kvp.row_bytes;
    r.kv = try a.alloc([]u8, 24);
    for (0..12) |ai| for (0..2) |v| {
        const dst = try a.alloc(u8, fed * row);
        if (g.seq.paged) |pg| {
            if (v == 0 and ai == 0) {
                try pg.flush(g.stream, fed);
                try g.stream.synchronize();
            }
            @memcpy(dst, pg.hostRows(ai, v == 1)[0 .. fed * row]);
        } else {
            const src = if (v == 0) g.seq.kc[ai].ptr else g.seq.vc[ai].ptr;
            try g.gpu.d.check(g.gpu.d.api.cuMemcpyDtoH_v2(dst.ptr, src, fed * row), "cuMemcpyDtoH");
        }
        r.kv[2 * ai + v] = dst;
    };
    return r;
}

fn freeRun(a: std.mem.Allocator, r: *Run) void {
    a.free(r.logits);
    a.free(r.toks);
    for (r.kv) |x| a.free(x);
    if (r.kv.len > 0) a.free(r.kv);
}

/// The first n ids, or (needle) the text with a code sentence at `at` and a question about it at the end.
fn promptOf(a: std.mem.Allocator, tok: *const core.tokenizer.Tokenizer, all: []const u32, n: usize, needle_at: ?f64, code: []const u8) ![]u32 {
    if (needle_at == null) return a.dupe(u32, all[0..n]);
    var buf: [256]u8 = undefined;
    const fact = try tok.encodePlain(a, try std.fmt.bufPrint(&buf, "\n\nIMPORTANT NOTE: the secret access code for the blue vault is {s}. Remember this code.\n\n", .{code}));
    defer a.free(fact);
    const q = try tok.encodePlain(a, "\n\nQuestion: what is the secret access code for the blue vault, given in the important note above?\nAnswer: the secret access code for the blue vault is");
    defer a.free(q);
    const body = n - fact.len - q.len;
    const at: usize = @intFromFloat(@as(f64, @floatFromInt(body)) * needle_at.?);
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(a, all[0..at]);
    try out.appendSlice(a, fact);
    try out.appendSlice(a, all[at..body]);
    try out.appendSlice(a, q);
    return out.toOwnedSlice(a);
}

fn cmpRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 5) return error.MissingArgument;
    const all = try readIds(gpu, rest[0]);
    defer a.free(all);
    var ns_buf: [16]usize = undefined;
    const ns = ns_buf[0..try parseList(rest[1], &ns_buf)];
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    const window = try std.fmt.parseInt(usize, rest[3], 10);
    const slots = try std.fmt.parseInt(usize, rest[4], 10);
    const needle_at: ?f64 = if (rest.len > 5) try std.fmt.parseFloat(f64, rest[5]) else null;
    var max_n: usize = 0;
    for (ns) |n| max_n = @max(max_n, n);
    if (max_n > all.len) return error.NotEnoughIds;
    var path_buf: [512]u8 = undefined;
    var tok = try core.tokenizer.loadTokenizer(gpu.io, a, try std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{dir}));
    defer tok.deinit();
    const o = xpre.optionsFromEnv();
    const cap = max_n + steps + 64;
    kvp.force_off = true;
    const g = try Eng.init(gpu, dir, cap, if (o.enabled) @min(o.rows, (max_n + 255) / 256 * 256) else 0);
    defer g.deinit();
    kvp.force_off = false;
    const cfg: kvp.Config = .{ .window = window, .slots = slots, .rows = g.e.chunk };
    var all_ok = true;
    const codes = [_][]const u8{ "48613", "70259", "31846", "95072", "62418", "27530" };
    for (ns, 0..) |n, ni| {
        const code = codes[ni % codes.len];
        const ids = try promptOf(a, &tok, all, n, needle_at, code);
        defer a.free(ids);
        // all on the device (the engine's own sequence)
        var d = try oneRun(g, ids, steps, true);
        defer freeRun(a, &d);
        const free_dense = try freeMem(gpu);
        g.seq.deinit(a);
        const free_none = try freeMem(gpu);
        kvp.force = cfg;
        g.seq = try g.e.newSeq(cap);
        kvp.force = null;
        const free_paged = try freeMem(gpu);
        const pg = g.seq.paged orelse return error.KvPageOff;
        const pg_dev = pg.deviceBytes();
        const pg_host = pg.host.bytes.len;
        var p = try oneRun(g, ids, steps, true);
        defer freeRun(a, &p);
        // back to an all-on-device sequence for the next prompt
        g.seq.deinit(a);
        kvp.force_off = true;
        g.seq = try g.e.newSeq(cap);
        kvp.force_off = false;
        const logits_eq = std.mem.eql(u8, d.logits, p.logits);
        const toks_eq = std.mem.eql(u32, d.toks, p.toks);
        var kv_eq: usize = 0;
        for (d.kv, p.kv) |x, y| kv_eq += @intFromBool(std.mem.eql(u8, x, y));
        const dense_seq = free_none - free_dense;
        const paged_seq = free_none - free_paged;
        std.debug.print("CMP {d} tokens{s}: prefill logits bit-equal {}; {d} greedy tokens equal {}; K/V rows 0..{d} equal in {d}/24 layer-sides\n", .{ ids.len, if (needle_at != null) " (needle)" else "", logits_eq, steps, toks_eq, ids.len + steps, kv_eq });
        std.debug.print("    device bytes of the sequence: all-on-device {d:.3} GiB, paged {d:.3} GiB (window {d} tokens a layer, {d} block slots shared by the layers: paged side {d:.3} GiB), freed {d:.3} GiB; host {d:.2} GiB pinned\n", .{ gib(dense_seq), gib(paged_seq), window, slots, gib(pg_dev), gib(dense_seq -| paged_seq), gib(pg_host) });
        std.debug.print("    prefill: all-on-device {d:.0} ms, paged {d:.0} ms; decode: all-on-device {d:.2} ms/token, paged {d:.2} ms/token\n", .{ d.prefill_ms, p.prefill_ms, d.decode_ms, p.decode_ms });
        const sp = p.stats_prefill;
        const sd = p.stats_decode;
        std.debug.print("    paged prefill: {d} calls, {d} blocks needed outside the window, {d} misses ({d:.1}% hit, {d:.1} MiB fetched, {d:.0} B a row a layer), {d} window moves, {d} rows written out; overflow {}\n", .{ sp.calls, sp.needed, sp.misses, 100 * (1 - @as(f64, @floatFromInt(sp.misses)) / @as(f64, @floatFromInt(@max(sp.needed, 1)))), @as(f64, @floatFromInt(sp.misses * 8192)) / (1 << 20), @as(f64, @floatFromInt(sp.misses * 8192)) / @as(f64, @floatFromInt(ids.len * 12)), sp.rebases, sp.written_rows, sp.overflow });
        std.debug.print("    paged decode: {d} calls, {d} needed outside the window, {d} misses ({d:.1}% hit), {d:.1} KiB fetched a token ({d:.2} KiB a layer); overflow {}\n", .{ sd.calls, sd.needed, sd.misses, 100 * (1 - @as(f64, @floatFromInt(sd.misses)) / @as(f64, @floatFromInt(@max(sd.needed, 1)))), @as(f64, @floatFromInt(sd.misses * 8192)) / 1024 / @as(f64, @floatFromInt(@max(steps, 1))), @as(f64, @floatFromInt(sd.misses * 8192)) / 1024 / @as(f64, @floatFromInt(@max(steps, 1) * 12)), sd.overflow });
        const td = try tok.decode(a, d.toks, false);
        defer a.free(td);
        const tp = try tok.decode(a, p.toks, false);
        defer a.free(tp);
        std.debug.print("    greedy (all-on-device): {any}\n    text: \"{f}\"\n", .{ d.toks[0..@min(steps, 32)], std.zig.fmtString(td[0..@min(td.len, 240)]) });
        if (!toks_eq) std.debug.print("    greedy (paged): {any}\n    text: \"{f}\"\n", .{ p.toks[0..@min(steps, 32)], std.zig.fmtString(tp[0..@min(tp.len, 240)]) });
        if (needle_at != null) {
            const hit_d = std.mem.indexOf(u8, td, code) != null;
            const hit_p = std.mem.indexOf(u8, tp, code) != null;
            std.debug.print("    NEEDLE code {s} at {d:.2} of the text: all-on-device {s}, paged {s}\n", .{ code, needle_at.?, if (hit_d) "FOUND" else "missed", if (hit_p) "FOUND" else "missed" });
            if (!hit_p) all_ok = false;
        }
        if (!logits_eq or !toks_eq or kv_eq != 24 or sp.overflow or sd.overflow) all_ok = false;
    }
    try check.expect(all_ok, "paged K/V bit-equal to all-on-device (logits, greedy tokens, rows){s}", .{if (needle_at != null) " and the needle found" else ""});
    check.pass("exl3-kvpage cmp", .{});
}

/// The host's model of kvp_plan: exact LRU at call granularity, ties to the lower slot.
const Model = struct {
    stamp: []u32,
    slot_of: std.AutoHashMap(u32, u32),
    blk: []i64,
    epoch: u32 = 0,

    fn use(mo: *Model, a: std.mem.Allocator, need: []const u32) !usize {
        mo.epoch += 1;
        var misses: std.ArrayList(u32) = .empty;
        defer misses.deinit(a);
        for (need) |b| {
            if (mo.slot_of.get(b)) |sl| mo.stamp[sl] = mo.epoch else try misses.append(a, b);
        }
        const order = try a.alloc(u32, mo.stamp.len);
        defer a.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            st: []const u32,
            ep: u32,
            fn key(c: @This(), i: u32) u64 {
                const t = c.st[i];
                return if (t == c.ep) std.math.maxInt(u64) else (@as(u64, t) << 32) | i;
            }
            fn less(c: @This(), x: u32, y: u32) bool {
                return c.key(x) < c.key(y);
            }
        };
        std.mem.sort(u32, order, Ctx{ .st = mo.stamp, .ep = mo.epoch }, Ctx.less);
        for (misses.items, 0..) |b, i| {
            const sl = order[i];
            if (mo.stamp[sl] == mo.epoch) return error.ModelOverflow;
            if (mo.blk[sl] >= 0) _ = mo.slot_of.remove(@intCast(mo.blk[sl]));
            mo.blk[sl] = b;
            try mo.slot_of.put(b, sl);
            mo.stamp[sl] = mo.epoch;
        }
        return misses.items.len;
    }
};

fn kernelRun(gpu: Gpu, slots: usize) !void {
    const a = gpu.gpa;
    const idw = flash.exl3.prompt_wt.index_idw;
    const row = kvp.row_bytes;
    const cap: usize = 60_000;
    const w0: usize = 40_000;
    const nb_before = w0 / 4;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const pg = try kvp.Paged.init(a, gpu.d, cap, .{ .window = 2052, .slots = slots, .rows = 256 });
    defer pg.destroy();
    pg.setWindow(0, w0);
    // each host row's 256 words unique to (position, K or V)
    for (0..2) |v| {
        const h = std.mem.bytesAsSlice(u32, pg.hostMut(0, v == 1));
        for (0..cap) |p| for (0..row / 4) |i| {
            h[p * (row / 4) + i] = @intCast((p *% 2654435761 +% i *% 40503 +% v *% 977) & 0xFFFF_FFFF);
        };
    }
    var ids_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * idw * 4);
    defer ids_d.free();
    var nk_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer nk_d.free();
    var sp_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer sp_d.free();
    const ids = try a.alloc(i32, 256 * idw);
    defer a.free(ids);
    const back = try a.alloc(i32, 256 * idw);
    defer a.free(back);
    const nk = try a.alloc(i32, 256);
    defer a.free(nk);
    const sp = try a.alloc(i32, 256);
    defer a.free(sp);
    @memset(sp, 1);
    const lay = pg.slotRows(0);
    const dk = try a.alloc(u8, lay[0] * row);
    defer a.free(dk);
    const dv = try a.alloc(u8, lay[0] * row);
    defer a.free(dv);
    var model: Model = .{ .stamp = try a.alloc(u32, slots), .slot_of = .init(a), .blk = try a.alloc(i64, slots) };
    defer {
        a.free(model.stamp);
        a.free(model.blk);
        model.slot_of.deinit();
    }
    @memset(model.stamp, 0);
    @memset(model.blk, -1);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    // a drifting selection: 512 distinct blocks of [0, nb_before), 15% replaced a call
    var base: [512]u32 = undefined;
    var used = std.AutoHashMap(u32, void).init(a);
    defer used.deinit();
    var k: usize = 0;
    while (k < 512) {
        const b = rng.uintLessThan(u32, @intCast(nb_before));
        if ((try used.getOrPut(b)).found_existing) continue;
        base[k] = b;
        k += 1;
    }
    var gpu_miss: usize = 0;
    var model_miss: usize = 0;
    var bad: usize = 0;
    var keys_checked: usize = 0;
    var plan_ns: i96 = 0;
    const calls: usize = 60;
    for (0..calls) |c| {
        const m: usize = if (c % 10 == 9) 64 else 1;
        var need = std.AutoHashMap(u32, void).init(a);
        defer need.deinit();
        for (0..m) |r| {
            // drift: replace ~15% of the base blocks
            for (0..77) |_| {
                const j = rng.uintLessThan(usize, 512);
                while (true) {
                    const b = rng.uintLessThan(u32, @intCast(nb_before));
                    if ((try used.getOrPut(b)).found_existing) continue;
                    _ = used.remove(base[j]);
                    base[j] = b;
                    break;
                }
            }
            var sorted = base;
            std.mem.sort(u32, &sorted, {}, std.sort.asc(u32));
            const out = ids[r * idw ..][0..idw];
            for (sorted, 0..) |b, j| for (0..4) |t| {
                out[j * 4 + t] = @intCast(b * 4 + t);
            };
            // 3 tail keys inside the window
            for (0..3) |t| out[2048 + t] = @intCast(w0 + 100 + c * 4 + t);
            nk[r] = 2051;
            for (sorted) |b| try need.put(b, {});
        }
        var need_list: std.ArrayList(u32) = .empty;
        defer need_list.deinit(a);
        var it = need.keyIterator();
        while (it.next()) |b| try need_list.append(a, b.*);
        model_miss += try model.use(a, need_list.items);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(ids_d.ptr, ids.ptr, m * idw * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(nk_d.ptr, nk.ptr, m * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(sp_d.ptr, sp.ptr, m * 4), "cuMemcpyHtoD");
        const before = try pg.stats(0);
        const t0 = check.now(gpu.io);
        try pg.page(stream, 0, ids_d.ptr, nk_d.ptr, sp_d.ptr, m);
        try stream.synchronize();
        plan_ns += check.now(gpu.io) - t0;
        const after = try pg.stats(0);
        gpu_miss += after.misses - before.misses;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(back.ptr, ids_d.ptr, m * idw * 4), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dk.ptr, pg.devRows(0, false), dk.len), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dv.ptr, pg.devRows(0, true), dv.len), "cuMemcpyDtoH");
        const hk = pg.hostRows(0, false);
        const hv = pg.hostRows(0, true);
        for (0..m) |r| for (0..2051) |ki| {
            const p: usize = @intCast(ids[r * idw + ki]);
            const q: usize = @intCast(back[r * idw + ki]);
            keys_checked += 1;
            if (p >= w0) {
                bad += @intFromBool(q != p - w0);
                continue;
            }
            if (q < lay[1] or q >= lay[0]) {
                bad += 1;
                continue;
            }
            bad += @intFromBool(!std.mem.eql(u8, dk[q * row ..][0..row], hk[p * row ..][0..row]) or !std.mem.eql(u8, dv[q * row ..][0..row], hv[p * row ..][0..row]));
        };
    }
    const st = try pg.stats(0);
    std.debug.print("kernel: {d} calls ({d} keys checked): bad {d}; misses GPU {d}, host LRU model {d}; needed {d}; overflow {}; plan+fetch+remap {d:.1} us a call (incl. launch + sync)\n", .{ calls, keys_checked, bad, gpu_miss, model_miss, st.needed, st.overflow, @as(f64, @floatFromInt(plan_ns)) / 1e3 / @as(f64, @floatFromInt(calls)) });
    try check.expect(bad == 0 and !st.overflow, "every remapped key reads its own host row", .{});
    const diff = @as(f64, @floatFromInt(@max(gpu_miss, model_miss) - @min(gpu_miss, model_miss))) / @as(f64, @floatFromInt(@max(model_miss, 1)));
    try check.expect(diff < 0.05, "misses within 5% of the LRU model ({d:.3})", .{diff});
    check.pass("exl3-kvpage kernel", .{});
}
