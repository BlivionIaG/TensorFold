//! Research harness, not for merge: LoopCD variants on the released checkpoint (scored items, state norms, generation).
const std = @import("std");
const Model = @import("families/inkling2/metal/model.zig").Model;
const fwd = @import("families/inkling2/metal/forward.zig");
const st = @import("families/inkling2/metal/state.zig");
const mtl = @import("metal");

const gpa = std.heap.page_allocator;
const io = std.testing.io;
const sys = @import("sys");

/// A GPU configuration: off; h:W:D:PL[:N[:WB]] (Hidden: strength, reference depth in core blocks, planes 1 residual
/// 2 B12 3 both, norm-matched 0/1, B12 strength); r:D[:PL[:N]] (the reference's own logits: strength -1).
const Cfg = struct {
    w: ?f32 = null,
    wb: ?f32 = null,
    depth: u32 = 2,
    planes: u8 = 3,
    norm: bool = false,
    ref: bool = false,
    name: []const u8 = "off",
};

fn parseCfg(item: []const u8) !Cfg {
    var f = std.mem.splitScalar(u8, item, ':');
    const k = f.next().?;
    var c: Cfg = .{ .name = item };
    if (std.mem.eql(u8, k, "off")) return c;
    if (std.mem.eql(u8, k, "h")) {
        c.w = try std.fmt.parseFloat(f32, f.next().?);
        c.depth = try std.fmt.parseInt(u32, f.next().?, 10);
        c.planes = try std.fmt.parseInt(u8, f.next().?, 10);
        if (f.next()) |n| c.norm = n[0] == '1';
        if (f.next()) |wb| c.wb = try std.fmt.parseFloat(f32, wb);
        return c;
    }
    if (std.mem.eql(u8, k, "r")) {
        c.ref = true;
        c.w = -1;
        c.depth = try std.fmt.parseInt(u32, f.next().?, 10);
        if (f.next()) |pl| c.planes = try std.fmt.parseInt(u8, pl, 10);
        if (f.next()) |n| c.norm = n[0] == '1';
        return c;
    }
    return error.BadConfig;
}

fn configure(m: *Model, c: Cfg) void {
    m.loopcd = c.w;
    m.loopcd_wb = c.wb;
    m.loopcd_depth = c.depth;
    m.loopcd_planes = c.planes;
    m.loopcd_norm = c.norm;
}

/// A scored variant: a configuration's own logits, or l:W:D:ALPHA:ADAPT (Logits from off and r:D;
/// z' = z_R + w (z_R - z_ref) where p_R >= alpha max p_R, adaptive w = w_max (1 - (p1 - p2))).
const Variant = struct {
    name: []const u8,
    own: ?usize = null,
    base: usize = 0,
    ref: usize = 0,
    w: f32 = 0,
    alpha: f32 = 0,
    adaptive: bool = false,
};

fn findCfg(cfgs: []const Cfg, name: []const u8) !usize {
    for (cfgs, 0..) |c, i| if (std.mem.eql(u8, c.name, name)) return i;
    std.debug.print("missing configuration {s}\n", .{name});
    return error.MissingConfig;
}

fn parseVariant(a: std.mem.Allocator, cfgs: []const Cfg, item: []const u8) !Variant {
    var f = std.mem.splitScalar(u8, item, ':');
    if (!std.mem.eql(u8, f.next().?, "l")) return error.BadVariant;
    const w = try std.fmt.parseFloat(f32, f.next().?);
    const d = f.next().?;
    const alpha = try std.fmt.parseFloat(f32, f.next().?);
    const adaptive = f.next().?[0] == '1';
    const ref_name = try std.fmt.allocPrint(a, "r:{s}", .{d});
    return .{ .name = item, .base = try findCfg(cfgs, "off"), .ref = try findCfg(cfgs, ref_name), .w = w, .alpha = alpha, .adaptive = adaptive };
}

fn list(a: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, x);
    return out.items;
}

fn numbers(a: std.mem.Allocator, text: []const u8) ![]const u32 {
    var ids: std.ArrayList(u32) = .empty;
    var words = std.mem.tokenizeScalar(u8, text, ' ');
    while (words.next()) |w| try ids.append(a, try std.fmt.parseInt(u32, w, 10));
    return ids.items;
}

/// "ctx | cont | cont ..." per line; a TF line "first ids..." becomes ctx ids[0..first+1], cont the rest.
const Group = struct { ctx: []const u32, conts: []const []const u32 };

fn groups(a: std.mem.Allocator, name: []const u8) ![]Group {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(1 << 28));
    var out: std.ArrayList(Group) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, '|') == null) {
            const all = try numbers(a, line);
            const first = all[0];
            const ids = all[1..];
            const conts = try a.alloc([]const u32, 1);
            conts[0] = ids[first + 1 ..];
            try out.append(a, .{ .ctx = ids[0 .. first + 1], .conts = conts });
            continue;
        }
        var parts = std.mem.splitScalar(u8, line, '|');
        const ctx = try numbers(a, parts.next().?);
        var conts: std.ArrayList([]const u32) = .empty;
        while (parts.next()) |p| try conts.append(a, try numbers(a, p));
        try out.append(a, .{ .ctx = ctx, .conts = conts.items });
    }
    return out.items;
}

fn secs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// A prefilled state into a cache of the same capacity: every GPU buffer blitted, then the length.
fn copyCache(m: *Model, from: *const st.Cache, to: *st.Cache) void {
    const cb = m.queue.commandBuffer();
    const e = cb.blit();
    e.copy(from.kv, 0, to.kv, 0, st.kvBytes(from.capacity));
    e.copy(from.conv, 0, to.conv, 0, st.conv_bytes);
    e.copy(from.taps, 0, to.taps, 0, st.taps_bytes);
    e.copy(from.backup, 0, to.backup, 0, st.conv_bytes);
    e.end();
    cb.commit();
    cb.wait();
    to.len = from.len;
}

/// A pooled cache back to an empty sequence: convolution history zero, length zero (stale KV is never read).
fn resetCache(m: *Model, c: *st.Cache) void {
    const cb = m.queue.commandBuffer();
    const e = cb.blit();
    e.fill(c.conv, 0, st.conv_bytes, 0);
    e.end();
    cb.commit();
    cb.wait();
    c.len = 0;
    c.pending = null;
}

const temps = [_]f32{ 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0 };
const nt = temps.len;
const RowStat = struct { hit: bool, lp: [nt]f32, h: [nt]f32 };

const V = @Vector(16, f32);

/// 2^x for x <= 0 (clamped at -126): round-to-nearest split, degree-7 Taylor on [-0.5, 0.5].
inline fn exp2v(x: V) V {
    const xc = @max(x, @as(V, @splat(-126.0)));
    const n = @round(xc);
    const f = (xc - n) * @as(V, @splat(0.6931471805599453));
    var p: V = @splat(1.0 / 5040.0);
    p = p * f + @as(V, @splat(1.0 / 720.0));
    p = p * f + @as(V, @splat(1.0 / 120.0));
    p = p * f + @as(V, @splat(1.0 / 24.0));
    p = p * f + @as(V, @splat(1.0 / 6.0));
    p = p * f + @as(V, @splat(0.5));
    p = p * f + @as(V, @splat(1.0));
    p = p * f + @as(V, @splat(1.0));
    const bits: @Vector(16, i32) = (@as(@Vector(16, i32), @intFromFloat(n)) + @as(@Vector(16, i32), @splat(127))) << @splat(23);
    return p * @as(V, @bitCast(bits));
}

fn rowStat(z: []const f32, target: usize) RowStat {
    var most: f32 = z[0];
    var arg: usize = 0;
    for (z, 0..) |x, i| if (x > most) {
        most = x;
        arg = i;
    };
    var out: RowStat = .{ .hit = arg == target, .lp = undefined, .h = undefined };
    const n = z.len / 16 * 16;
    for (temps, 0..) |t, k| {
        const inv: f32 = 1.0 / t;
        const c2: V = @splat(inv * 1.4426950408889634);
        const mv: V = @splat(most);
        var s: V = @splat(0);
        var e: V = @splat(0);
        var i: usize = 0;
        while (i < n) : (i += 16) {
            const d: V = z[i..][0..16].* - mv;
            const x = exp2v(d * c2);
            s += x;
            e += x * d;
        }
        var ss: f64 = @reduce(.Add, s);
        var ee: f64 = @reduce(.Add, e);
        while (i < z.len) : (i += 1) {
            const d = z[i] - most;
            const x = @exp(@as(f64, d * inv));
            ss += x;
            ee += x * d;
        }
        const ls = @log(ss);
        out.lp[k] = @floatCast(@as(f64, (z[target] - most) * inv) - ls);
        out.h[k] = @floatCast(ls - ee * inv / ss);
    }
    return out;
}

fn guide(out: []f32, zr: []const f32, zref: []const f32, v: Variant) void {
    var most: f32 = zr[0];
    var second: f32 = -std.math.inf(f32);
    for (zr[1..]) |x| {
        if (x > most) {
            second = most;
            most = x;
        } else if (x > second) second = x;
    }
    var w = v.w;
    if (v.adaptive) {
        var sum: f64 = 0;
        for (zr) |x| sum += @exp(@as(f64, x - most));
        const p1 = 1.0 / sum;
        const p2 = @exp(@as(f64, second - most)) / sum;
        w = @floatCast(@as(f64, v.w) * (1 - (p1 - p2)));
    }
    const cut: f32 = if (v.alpha > 0) most + @log(v.alpha) else -std.math.inf(f32);
    for (out, zr, zref) |*o, r, f| o.* = if (r >= cut) r + w * (r - f) else r;
}

const Eval = struct {
    store: []const []f32,
    targets: []const u32,
    variants: []const Variant,
    stats: []RowStat, // [variant][row]
    next: std.atomic.Value(usize) = .init(0),
    vocab: usize,

    fn work(self: *Eval) void {
        const rows = self.targets.len;
        const buf = gpa.alloc(f32, self.vocab) catch unreachable;
        defer gpa.free(buf);
        while (true) {
            const r = self.next.fetchAdd(1, .monotonic);
            if (r >= rows) return;
            for (self.variants, 0..) |v, k| {
                const z: []const f32 = if (v.own) |c| self.store[c][r * self.vocab ..][0..self.vocab] else blk: {
                    guide(buf, self.store[v.base][r * self.vocab ..][0..self.vocab], self.store[v.ref][r * self.vocab ..][0..self.vocab], v);
                    break :blk buf;
                };
                self.stats[k * rows + r] = rowStat(z, self.targets[r]);
            }
        }
    }
};

const Job = struct { g: usize, k: ?usize, cache: usize, toks: []const u32, fed: usize = 0, held: []const usize };

test "research: scored items under LoopCD variants" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    const name = sys.getenv("TF_LOOPCD_RESEARCH_ITEMS") orelse return error.SkipZigTest;
    const out_path: [*:0]const u8 = if (sys.getenv("TF_LOOPCD_RESEARCH_OUT")) |o| o.ptr else "research-items.txt";
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cfgs: std.ArrayList(Cfg) = .empty;
    for (try list(a, sys.getenv("TF_LOOPCD_RESEARCH_CFG") orelse "off")) |c| try cfgs.append(a, try parseCfg(c));
    var variants: std.ArrayList(Variant) = .empty;
    for (cfgs.items, 0..) |c, i| if (!c.ref) try variants.append(a, .{ .name = c.name, .own = i });
    if (sys.getenv("TF_LOOPCD_RESEARCH_VAR")) |vs| for (try list(a, vs)) |v| try variants.append(a, try parseVariant(a, cfgs.items, v));
    const all = try groups(a, name);
    const lo = try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_FROM") orelse "0", 10);
    const hi = @min(all.len, try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_TO") orelse "100000000", 10));
    const chunk_rows = try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_CHUNK") orelse "2048", 10);
    const threads = try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_THREADS") orelse "24", 10);
    var cap: usize = 16;
    for (all[lo..hi]) |g| {
        var most: usize = 0;
        for (g.conts) |c| most = @max(most, c.len);
        cap = @max(cap, g.ctx.len + most + 8);
    }
    const m = try Model.load(std.testing.allocator, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const v = m.config.text.unpadded_vocab;
    const pool_n = st.max_segments + 8;
    const caches = try a.alloc(st.Cache, pool_n);
    for (caches) |*c| c.* = try st.Cache.init(m.device, cap, &m.weight_plan);
    defer for (caches) |*c| c.deinit();
    const store = try a.alloc([]f32, cfgs.items.len);
    for (store) |*x| x.* = &.{};
    defer for (store) |x| if (x.len > 0) gpa.free(x);
    const t0 = secs();
    var gpu_secs: f64 = 0;
    var eval_secs: f64 = 0;
    var tokens: usize = 0;
    var at = lo;
    while (at < hi) {
        // one chunk of whole groups; scored row offsets per (group, continuation)
        var end = at;
        var rows: usize = 0;
        while (end < hi) {
            var n: usize = 0;
            for (all[end].conts) |c| n += c.len;
            if (rows > 0 and rows + n > chunk_rows) break;
            rows += n;
            end += 1;
        }
        const gs = all[at..end];
        const offs = try gpa.alloc([]usize, gs.len);
        defer gpa.free(offs);
        const targets = try gpa.alloc(u32, rows);
        defer gpa.free(targets);
        var o: usize = 0;
        for (gs, offs) |g, *off| {
            off.* = try a.alloc(usize, g.conts.len);
            for (g.conts, off.*) |c, *x| {
                x.* = o;
                @memcpy(targets[o..][0..c.len], c);
                o += c.len;
            }
        }
        for (store) |*x| if (x.len < rows * v) {
            if (x.len > 0) gpa.free(x.*);
            x.* = try gpa.alloc(f32, rows * v);
        };
        const g0 = secs();
        for (cfgs.items, store) |c, buf| {
            configure(m, c);
            var free: std.ArrayList(usize) = .empty;
            defer free.deinit(gpa);
            for (0..pool_n) |i| try free.append(gpa, i);
            var jobs: std.ArrayList(Job) = .empty;
            defer jobs.deinit(gpa);
            var next: usize = 0;
            while (next < gs.len or jobs.items.len > 0) {
                // admit groups while their caches fit: one for the context, one more per long continuation
                while (next < gs.len) {
                    var need: usize = 0;
                    for (gs[next].conts) |x| need += @intFromBool(x.len >= 2);
                    need = @max(need, 1);
                    if (free.items.len < need) break;
                    const held = try a.alloc(usize, need);
                    for (held) |*h| h.* = free.pop().?;
                    resetCache(m, &caches[held[0]]);
                    try jobs.append(gpa, .{ .g = next, .k = null, .cache = held[0], .toks = gs[next].ctx, .held = held });
                    next += 1;
                }
                var segs: [st.max_segments]fwd.Segment = undefined;
                var take: [st.max_segments]usize = undefined;
                var ids: [st.max_rows]u32 = undefined;
                var nseg: usize = 0;
                var used: usize = 0;
                for (jobs.items) |*j| {
                    if (nseg == st.max_segments or used == st.max_rows) break;
                    const n = @min(j.toks.len - j.fed, st.max_rows - used);
                    segs[nseg] = .{ .cache = &caches[j.cache], .rows = n };
                    take[nseg] = n;
                    @memcpy(ids[used..][0..n], j.toks[j.fed..][0..n]);
                    used += n;
                    nseg += 1;
                }
                try fwd.runShared(m, &s, segs[0..nseg], ids[0..used], .{ .all = true });
                tokens += used;
                const logits = s.logits_all.slice(f32, used * v);
                var base: usize = 0;
                var done: std.ArrayList(usize) = .empty;
                defer done.deinit(gpa);
                for (jobs.items[0..nseg], take[0..nseg], 0..) |*j, n, ji| {
                    const off = offs[j.g];
                    for (0..n) |r| {
                        const p = j.fed + r;
                        const row = logits[(base + r) * v ..][0..v];
                        if (j.k) |k| {
                            @memcpy(buf[(off[k] + p + 1) * v ..][0..v], row);
                        } else if (p == j.toks.len - 1) {
                            for (off) |x| @memcpy(buf[x * v ..][0..v], row);
                        }
                    }
                    caches[j.cache].len += n;
                    j.fed += n;
                    base += n;
                    if (j.fed == j.toks.len) try done.append(gpa, ji);
                }
                // finished jobs leave in reverse order; a finished context hands its cache to the long continuations
                var d = done.items.len;
                while (d > 0) {
                    d -= 1;
                    const j = jobs.orderedRemove(done.items[d]);
                    if (j.k != null) {
                        try free.append(gpa, j.cache);
                        continue;
                    }
                    const g = gs[j.g];
                    var h: usize = 0;
                    var first_long: ?usize = null;
                    for (g.conts, 0..) |x, k| {
                        if (x.len < 2) continue;
                        if (first_long == null) {
                            first_long = k;
                            h += 1;
                            continue;
                        }
                        copyCache(m, &caches[j.held[0]], &caches[j.held[h]]);
                        try jobs.append(gpa, .{ .g = j.g, .k = k, .cache = j.held[h], .toks = x[0 .. x.len - 1], .held = &.{} });
                        h += 1;
                    }
                    if (first_long) |k| {
                        try jobs.append(gpa, .{ .g = j.g, .k = k, .cache = j.held[0], .toks = g.conts[k][0 .. g.conts[k].len - 1], .held = &.{} });
                    } else try free.append(gpa, j.held[0]);
                }
            }
        }
        configure(m, .{});
        const e0 = secs();
        gpu_secs += e0 - g0;
        // every variant's per-row statistics, threaded over rows
        const stats = try gpa.alloc(RowStat, variants.items.len * rows);
        defer gpa.free(stats);
        var ev: Eval = .{ .store = store, .targets = targets, .variants = variants.items, .stats = stats, .vocab = v };
        const pool = try gpa.alloc(std.Thread, threads);
        defer gpa.free(pool);
        for (pool) |*t| t.* = try std.Thread.spawn(.{}, Eval.work, .{&ev});
        for (pool) |t| t.join();
        eval_secs += secs() - e0;
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        for (variants.items, 0..) |vr, k| for (gs, offs, 0..) |g, off, gi| for (g.conts, off, 0..) |c, x, ci| {
            var hits: usize = 0;
            var lp: [nt]f64 = @splat(0);
            var h: [nt]f64 = @splat(0);
            for (stats[k * rows + x ..][0..c.len]) |rs| {
                hits += @intFromBool(rs.hit);
                for (0..nt) |ti| {
                    lp[ti] += rs.lp[ti];
                    h[ti] += rs.h[ti];
                }
            }
            try text.print(gpa, "S\t{s}\t{d}\t{d}\t{d}\t{d}\t", .{ vr.name, at + gi, ci, c.len, hits });
            for (lp, 0..) |y, ti| try text.print(gpa, "{s}{d:.5}", .{ if (ti == 0) "" else ",", y });
            try text.print(gpa, "\t", .{});
            for (h, 0..) |y, ti| try text.print(gpa, "{s}{d:.4}", .{ if (ti == 0) "" else ",", y });
            try text.print(gpa, "\n", .{});
        };
        const f = std.c.fopen(out_path, "a") orelse return error.OpenFailed;
        _ = std.c.fwrite(text.items.ptr, 1, text.items.len, f);
        _ = std.c.fclose(f);
        std.debug.print("research items {d}/{d}: rows {d}, tokens {d}, gpu {d:.0}s, eval {d:.0}s, wall {d:.0}s\n", .{ end, hi, rows, tokens, gpu_secs, eval_secs, secs() - t0 });
        at = end;
    }
}

test "research: per-position state snapshots" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    const name = sys.getenv("TF_LOOPCD_RESEARCH_SNAP") orelse return error.SkipZigTest;
    const out_path = sys.getenv("TF_LOOPCD_RESEARCH_OUT") orelse "research-snap.bin";
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const all = try groups(a, name);
    const m = try Model.load(std.testing.allocator, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const h = st.hidden;
    const snap = try m.device.buffer(7 * st.max_rows * h * 2, mtl.ResourceOptions.shared);
    defer snap.deinit();
    m.research_snap = snap;
    defer m.research_snap = null;
    configure(m, .{});
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    for (all) |g| {
        var seq: std.ArrayList(u32) = .empty;
        try seq.appendSlice(a, g.ctx);
        for (g.conts) |c| try seq.appendSlice(a, c);
        var cache = try st.Cache.init(m.device, seq.items.len + 8, &m.weight_plan);
        defer cache.deinit();
        var at: usize = 0;
        while (at < seq.items.len) : (at += st.max_rows) {
            const n: usize = @min(st.max_rows, seq.items.len - at);
            try fwd.run(m, &s, &cache, seq.items[at..][0..n], .{ .all = true });
            cache.len += n;
            // [row][7][H] BF16, positions in order; only rows from the context's last on (the scored ones)
            const all_snap = snap.slice(u16, 7 * n * h);
            for (0..n) |r| {
                if (at + r + 1 < g.ctx.len) continue;
                for (0..7) |k| try out.appendSlice(gpa, std.mem.sliceAsBytes(all_snap[(k * n + r) * h ..][0..h]));
            }
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = out.items });
}

// ---------------------------------------------------------------- generation

fn pick(row: []const f32, t: f32, rng: *std.Random.DefaultPrng) u32 {
    var most: f32 = row[0];
    var arg: usize = 0;
    for (row, 0..) |x, i| if (x > most) {
        most = x;
        arg = i;
    };
    if (t == 0) return @intCast(arg);
    var sum: f64 = 0;
    for (row) |x| sum += @exp(@as(f64, (x - most) / t));
    var u = rng.random().float(f64) * sum;
    for (row, 0..) |x, i| {
        u -= @exp(@as(f64, (x - most) / t));
        if (u <= 0) return @intCast(i);
    }
    return @intCast(arg);
}

const Stream = struct {
    prompt: usize,
    temperature: f32,
    seed: u64,
    main: st.Cache,
    ref: ?st.Cache,
    tokens: std.ArrayList(u32) = .empty,
    rng: std.Random.DefaultPrng,
    done: bool = false,
};

/// Last-row logits of the prompt fed into one cache.
fn prefill(m: *Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: []f32) !void {
    const v = m.config.text.unpadded_vocab;
    var at: usize = 0;
    while (at < ids.len) : (at += st.prompt_rows) {
        const n = @min(st.prompt_rows, ids.len - at);
        try fwd.run(m, s, cache, ids[at..][0..n], .{ .all = true });
        if (at + n == ids.len) @memcpy(out, s.logits_all.slice(f32, n * v)[(n - 1) * v ..][0..v]);
        cache.len += n;
    }
}

// Generation setting: a configuration name, or a Logits variant l:W:D:ALPHA:ADAPT (reference r:D).
test "research: generated replies under LoopCD variants" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    const name = sys.getenv("TF_LOOPCD_RESEARCH_GEN") orelse return error.SkipZigTest;
    const out: [*:0]const u8 = if (sys.getenv("TF_LOOPCD_RESEARCH_OUT")) |o| o.ptr else "research-gen.txt";
    const max_new = try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_MAX") orelse "384", 10);
    // thinking replies end only at <|content_model_end_sampling|>; a forced text reply also at <|end_message|>
    const stop_message = sys.getenv("TF_LOOPCD_RESEARCH_THINK") == null;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const settings = try list(a, sys.getenv("TF_LOOPCD_RESEARCH_SET") orelse return error.SkipZigTest);
    const all = try groups(a, name);
    var samples: std.ArrayList(struct { t: f32, seed: u64 }) = .empty;
    var sit = std.mem.tokenizeScalar(u8, sys.getenv("TF_LOOPCD_RESEARCH_SAMPLES") orelse "0:0", ',');
    while (sit.next()) |item| {
        var f = std.mem.splitScalar(u8, item, ':');
        try samples.append(a, .{ .t = try std.fmt.parseFloat(f32, f.next().?), .seed = try std.fmt.parseInt(u64, f.next().?, 10) });
    }
    const m = try Model.load(std.testing.allocator, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const v = m.config.text.unpadded_vocab;
    const zr = try gpa.alloc(f32, v);
    defer gpa.free(zr);
    const zf = try gpa.alloc(f32, v);
    defer gpa.free(zf);
    const g = try gpa.alloc(f32, v);
    defer gpa.free(g);
    const batch = (st.max_segments / samples.items.len) * samples.items.len;
    const main_rows = try gpa.alloc(f32, st.max_segments * v);
    defer gpa.free(main_rows);
    const ref_rows = try gpa.alloc(f32, st.max_segments * v);
    defer gpa.free(ref_rows);
    for (settings) |setting| {
        const logits_variant = setting[0] == 'l';
        var main_config: Cfg = .{};
        var ref_config: Cfg = .{};
        var lv: Variant = .{ .name = setting };
        if (logits_variant) {
            var f = std.mem.splitScalar(u8, setting, ':');
            _ = f.next();
            lv.w = try std.fmt.parseFloat(f32, f.next().?);
            ref_config = .{ .w = -1, .depth = try std.fmt.parseInt(u32, f.next().?, 10), .ref = true };
            lv.alpha = try std.fmt.parseFloat(f32, f.next().?);
            lv.adaptive = f.next().?[0] == '1';
        } else main_config = try parseCfg(setting);
        var active: std.ArrayList(Stream) = .empty;
        defer active.deinit(gpa);
        var next_prompt: usize = 0;
        while (true) {
            while (next_prompt < all.len and active.items.len + samples.items.len <= batch) : (next_prompt += 1) {
                const p = next_prompt;
                var whole: std.ArrayList(u32) = .empty;
                try whole.appendSlice(a, all[p].ctx);
                for (all[p].conts) |c| try whole.appendSlice(a, c);
                const ids = whole.items;
                const cap = ids.len + max_new + 8;
                const first = active.items.len;
                for (samples.items) |sm| try active.append(gpa, .{
                    .prompt = p,
                    .temperature = sm.t,
                    .seed = sm.seed,
                    .main = try st.Cache.init(m.device, cap, &m.weight_plan),
                    .ref = if (logits_variant) try st.Cache.init(m.device, cap, &m.weight_plan) else null,
                    .rng = .init(sm.seed *% 0x9E3779B97F4A7C15 +% p),
                });
                const group = active.items[first..];
                configure(m, main_config);
                try prefill(m, &s, &group[0].main, ids, zr);
                if (group[0].ref) |*r| {
                    configure(m, ref_config);
                    try prefill(m, &s, r, ids, zf);
                }
                for (group[1..]) |*x| {
                    copyCache(m, &group[0].main, &x.main);
                    if (x.ref) |*r| copyCache(m, &group[0].ref.?, r);
                }
                for (group) |*x| {
                    const row: []const f32 = if (x.ref != null) blk: {
                        guide(g, zr, zf, lv);
                        break :blk g;
                    } else zr;
                    const tok = pick(row, x.temperature, &x.rng);
                    try x.tokens.append(gpa, tok);
                    if ((stop_message and tok == 200010) or tok == 200006 or x.tokens.items.len >= max_new) x.done = true;
                }
            }
            var i: usize = 0;
            while (i < active.items.len) {
                const x = &active.items[i];
                if (!x.done) {
                    i += 1;
                    continue;
                }
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(gpa);
                for (x.tokens.items) |t| try buf.print(gpa, " {d}", .{t});
                const f = std.c.fopen(out, "a") orelse return error.OpenFailed;
                const line = try std.fmt.allocPrint(gpa, "GEN\t{s}\t{d}\t{d:.2}\t{d}\t{s}\n", .{ setting, x.prompt, x.temperature, x.seed, buf.items });
                defer gpa.free(line);
                _ = std.c.fwrite(line.ptr, 1, line.len, f);
                _ = std.c.fclose(f);
                x.main.deinit();
                if (x.ref) |*r| r.deinit();
                x.tokens.deinit(gpa);
                _ = active.swapRemove(i);
            }
            if (active.items.len == 0 and next_prompt == all.len) break;
            if (active.items.len == 0) continue;
            const count = active.items.len;
            var segs: [st.max_segments]fwd.Segment = undefined;
            var refs: [st.max_segments]fwd.Segment = undefined;
            var ids: [st.max_segments]u32 = undefined;
            for (active.items, 0..) |*x, j| {
                segs[j] = .{ .cache = &x.main, .rows = 1 };
                if (x.ref) |*r| refs[j] = .{ .cache = r, .rows = 1 };
                ids[j] = x.tokens.items[x.tokens.items.len - 1];
            }
            configure(m, main_config);
            try fwd.runShared(m, &s, segs[0..count], ids[0..count], .{ .all = true });
            @memcpy(main_rows[0 .. count * v], s.logits_all.slice(f32, count * v));
            if (logits_variant) {
                configure(m, ref_config);
                try fwd.runShared(m, &s, refs[0..count], ids[0..count], .{ .all = true });
                @memcpy(ref_rows[0 .. count * v], s.logits_all.slice(f32, count * v));
            }
            for (active.items, 0..) |*x, j| {
                x.main.len += 1;
                if (x.ref) |*r| r.len += 1;
                const z = main_rows[j * v ..][0..v];
                const row: []const f32 = if (logits_variant) blk: {
                    guide(g, z, ref_rows[j * v ..][0..v], lv);
                    break :blk g;
                } else z;
                const tok = pick(row, x.temperature, &x.rng);
                try x.tokens.append(gpa, tok);
                if ((stop_message and tok == 200010) or tok == 200006 or x.tokens.items.len >= max_new) x.done = true;
            }
        }
        configure(m, .{});
    }
}

test "research: forward timing" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    if (sys.getenv("TF_LOOPCD_RESEARCH_TIMING") == null) return error.SkipZigTest;
    const m = try Model.load(std.testing.allocator, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37);
    for ([_]usize{ 1, 8, 32, 64 }) |rows| for ([_]bool{ true, false }) |all_rows| {
        var cache = try st.Cache.init(m.device, 1024, &m.weight_plan);
        defer cache.deinit();
        var best: f64 = 1e9;
        for (0..4) |_| {
            const t = secs();
            try fwd.run(m, &s, &cache, ids[0..rows], .{ .all = all_rows });
            best = @min(best, secs() - t);
            cache.len += rows;
        }
        std.debug.print("timing rows {d} all {}: {d:.3}s\n", .{ rows, all_rows, best });
    };
}
