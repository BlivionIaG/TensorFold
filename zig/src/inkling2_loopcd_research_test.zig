//! Research harness, not for merge: LoopCD variants on the released checkpoint, teacher-forced and generated.
const std = @import("std");
const Model = @import("families/inkling2/metal/model.zig").Model;
const fwd = @import("families/inkling2/metal/forward.zig");
const st = @import("families/inkling2/metal/state.zig");

const gpa = std.heap.page_allocator;
const io = std.testing.io;
const sys = @import("sys");

const Kind = enum { off, hidden, logits };
/// off; h:W:REF:PLANES; l:W:REF:ALPHA:ADAPT (alpha 0: no mask; outside the mask the token keeps z_R).
const Setting = struct {
    kind: Kind = .off,
    w: f32 = 0,
    ref: u32 = 1,
    planes: u8 = 3,
    alpha: f32 = 0,
    adaptive: bool = false,
    name: []const u8 = "off",
};

fn parseSettings(a: std.mem.Allocator, text: []const u8) ![]Setting {
    var out: std.ArrayList(Setting) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |item| {
        var f = std.mem.splitScalar(u8, item, ':');
        const k = f.next().?;
        var s: Setting = .{ .name = item };
        if (std.mem.eql(u8, k, "off")) {
            s.kind = .off;
        } else if (std.mem.eql(u8, k, "h")) {
            s.kind = .hidden;
            s.w = try std.fmt.parseFloat(f32, f.next().?);
            s.ref = try std.fmt.parseInt(u32, f.next().?, 10);
            s.planes = try std.fmt.parseInt(u8, f.next().?, 10);
        } else if (std.mem.eql(u8, k, "l")) {
            s.kind = .logits;
            s.w = try std.fmt.parseFloat(f32, f.next().?);
            s.ref = try std.fmt.parseInt(u32, f.next().?, 10);
            s.alpha = try std.fmt.parseFloat(f32, f.next().?);
            s.adaptive = (try std.fmt.parseInt(u8, f.next().?, 10)) != 0;
        } else return error.BadSetting;
        try out.append(a, s);
    }
    return out.items;
}

fn lines(a: std.mem.Allocator, name: []const u8) ![]const []const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(1 << 26));
    var out: std.ArrayList([]const u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        var ids: std.ArrayList(u32) = .empty;
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        while (words.next()) |w| try ids.append(a, try std.fmt.parseInt(u32, w, 10));
        try out.append(a, ids.items);
    }
    return out.items;
}

fn say(path: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    var buf: [1 << 16]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    const f = std.c.fopen(path, "a") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(line.ptr, 1, line.len, f);
}

/// The GPU configuration one setting's logits come from: null strength is the unguided model.
const Config = struct { w: ?f32, ref: u32 = 1, planes: u8 = 3 };

fn configure(m: *Model, c: Config) void {
    m.loopcd = c.w;
    m.loopcd_ref = c.ref;
    m.loopcd_planes = c.planes;
}

fn top(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return best;
}

fn logOf(row: []const f32, id: usize) f64 {
    var most: f32 = row[0];
    for (row) |x| most = @max(most, x);
    var sum: f64 = 0;
    for (row) |x| sum += @exp(@as(f64, x - most));
    return @as(f64, row[id] - most) - @log(sum);
}

/// z' = z_R + w (z_R - z_ref) inside the plausibility mask (p_R >= alpha max p_R); w adaptive when asked.
fn guide(out: []f32, zr: []const f32, zref: []const f32, s: Setting) void {
    var most: f32 = zr[0];
    var second: f32 = -std.math.inf(f32);
    for (zr[1..]) |x| {
        if (x > most) {
            second = most;
            most = x;
        } else if (x > second) second = x;
    }
    var w = s.w;
    if (s.adaptive) {
        var sum: f64 = 0;
        for (zr) |x| sum += @exp(@as(f64, x - most));
        const p1 = 1.0 / sum;
        const p2 = @exp(@as(f64, second - most)) / sum;
        w = @floatCast(@as(f64, s.w) * (1 - (p1 - p2)));
    }
    const cut: f32 = if (s.alpha > 0) most + @log(s.alpha) else -std.math.inf(f32);
    for (out, zr, zref) |*o, r, f| o.* = if (r >= cut) r + w * (r - f) else r;
}

/// Every row's logits of `ids` from an empty cache, in chunks of the prompt grid.
fn allRows(m: *Model, s: *st.Scratch, ids: []const u32, out: []f32) !void {
    const v = m.config.text.unpadded_vocab;
    var cache = try st.Cache.init(m.device, ids.len + 8, &m.weight_plan);
    defer cache.deinit();
    var at: usize = 0;
    while (at < ids.len) : (at += st.prompt_rows) {
        const n = @min(st.prompt_rows, ids.len - at);
        try fwd.run(m, s, &cache, ids[at..][0..n], .{ .all = true });
        @memcpy(out[at * v ..][0 .. n * v], s.logits_all.slice(f32, n * v));
        cache.len += n;
    }
}

const Tally = struct { n: u64 = 0, logprob: f64 = 0, hits: u64 = 0, agree: u64 = 0, better: u32 = 0, worse: u32 = 0, at: [temps.len]f64 = @splat(0) };
/// Temperatures for the calibration-free comparison: each setting is also scored at its best one.
const temps = [_]f32{ 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6 };

fn logOfTemps(row: []const f32, id: usize, out: *[temps.len]f64) void {
    var most: f32 = row[0];
    for (row) |x| most = @max(most, x);
    for (temps, out) |t, *o| {
        const inv = 1 / t;
        var sum: f32 = 0;
        for (row) |x| sum += @exp((x - most) * inv);
        o.* += @as(f64, (row[id] - most) * inv) - @log(@as(f64, sum));
    }
}

test "research: teacher-forced LoopCD variants" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    const name = sys.getenv("TF_LOOPCD_RESEARCH_TF") orelse return error.SkipZigTest;
    const out: [*:0]const u8 = if (sys.getenv("TF_LOOPCD_RESEARCH_OUT")) |o| o.ptr else "research-tf.txt";
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const settings = try parseSettings(a, sys.getenv("TF_LOOPCD_RESEARCH_SET") orelse return error.SkipZigTest);
    const prompts = try lines(a, name);
    const calibrate = sys.getenv("TF_LOOPCD_RESEARCH_CALIBRATE") != null;
    const m = try Model.load(std.testing.allocator, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const v = m.config.text.unpadded_vocab;
    const tallies = try a.alloc(Tally, settings.len);
    @memset(tallies, .{});
    const guided = try gpa.alloc(f32, v);
    defer gpa.free(guided);
    for (prompts, 0..) |line, pi| {
        const first = line[0];
        const ids = line[1..];
        const size = ids.len * v;
        const base = try gpa.alloc(f32, size);
        defer gpa.free(base);
        configure(m, .{ .w = null });
        try allRows(m, &s, ids, base);
        var refs: [3]?[]f32 = .{ null, null, null };
        defer for (refs) |r| if (r) |x| gpa.free(x);
        const own = try gpa.alloc(f32, size);
        defer gpa.free(own);
        var base_sum: f64 = 0;
        for (settings, tallies, 0..) |set, *t, k| {
            switch (set.kind) {
                .off => {},
                .hidden => {
                    configure(m, .{ .w = set.w, .ref = set.ref, .planes = set.planes });
                    try allRows(m, &s, ids, own);
                },
                .logits => if (refs[set.ref] == null) {
                    const r = try gpa.alloc(f32, size);
                    configure(m, .{ .w = -1, .ref = set.ref, .planes = if (set.ref == 0) 1 else 3 });
                    try allRows(m, &s, ids, r);
                    refs[set.ref] = r;
                },
            }
            var sum: f64 = 0;
            for (first..ids.len - 1) |r| {
                const zr = base[r * v ..][0..v];
                const row: []const f32 = switch (set.kind) {
                    .off => zr,
                    .hidden => own[r * v ..][0..v],
                    .logits => blk: {
                        guide(guided, zr, refs[set.ref].?[r * v ..][0..v], set);
                        break :blk guided;
                    },
                };
                const lp = logOf(row, ids[r + 1]);
                sum += lp;
                t.n += 1;
                t.logprob += lp;
                const hit = top(row) == ids[r + 1];
                t.hits += @intFromBool(hit);
                t.agree += @intFromBool(top(row) == top(zr));
                if (calibrate) logOfTemps(row, ids[r + 1], &t.at);
            }
            if (k == 0) base_sum = sum else if (sum > base_sum) t.better += 1 else if (sum < base_sum) t.worse += 1;
            say(out, "ROW prompt {d} setting {s} rows {d} logprob {d:.6} hits {d} agree {d}\n", .{ pi, set.name, ids.len - 1 - first, sum, t.hits, t.agree });
        }
        configure(m, .{ .w = null });
    }
    for (settings, tallies) |set, t| {
        const n: f64 = @floatFromInt(t.n);
        say(out, "TF {s}: rows {d} mean_logprob {d:.5} top1 {d} agree {d} better {d} worse {d}\n", .{ set.name, t.n, t.logprob / n, t.hits, t.agree, t.better, t.worse });
        if (calibrate) for (temps, t.at) |tt, x| say(out, "TEMP {s} T {d:.2} mean_logprob {d:.5}\n", .{ set.name, tt, x / n });
    }
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

fn pick(row: []const f32, t: f32, rng: *std.Random.DefaultPrng) u32 {
    if (t == 0) return @intCast(top(row));
    var most: f32 = row[0];
    for (row) |x| most = @max(most, x);
    var sum: f64 = 0;
    for (row) |x| sum += @exp(@as(f64, (x - most) / t));
    var u = rng.random().float(f64) * sum;
    for (row, 0..) |x, i| {
        u -= @exp(@as(f64, (x - most) / t));
        if (u <= 0) return @intCast(i);
    }
    return @intCast(top(row));
}

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

/// A prefilled state into a fresh cache of the same capacity: every GPU buffer blitted, then the length.
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

test "research: generated replies under LoopCD variants" {
    const dir = sys.getenv("TF_LOOPCD_RESEARCH_DIR") orelse return error.SkipZigTest;
    const name = sys.getenv("TF_LOOPCD_RESEARCH_GEN") orelse return error.SkipZigTest;
    const out: [*:0]const u8 = if (sys.getenv("TF_LOOPCD_RESEARCH_OUT")) |o| o.ptr else "research-gen.txt";
    const max_new = try std.fmt.parseInt(usize, sys.getenv("TF_LOOPCD_RESEARCH_MAX") orelse "384", 10);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const settings = try parseSettings(a, sys.getenv("TF_LOOPCD_RESEARCH_SET") orelse return error.SkipZigTest);
    const prompts = try lines(a, name);
    // T:SEED pairs; T 0 is greedy.
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
    for (settings) |set| {
        const main_config: Config = if (set.kind == .hidden) .{ .w = set.w, .ref = set.ref, .planes = set.planes } else .{ .w = null };
        const ref_config: Config = .{ .w = -1, .ref = set.ref, .planes = if (set.ref == 0) 1 else 3 };
        var active: std.ArrayList(Stream) = .empty;
        defer active.deinit(gpa);
        var next_prompt: usize = 0;
        var finished: usize = 0;
        while (true) {
            // admit whole prompts while their samples fit: prefill a lead, copy its state, pick first tokens
            while (next_prompt < prompts.len and active.items.len + samples.items.len <= batch) : (next_prompt += 1) {
                const p = next_prompt;
                const ids = prompts[p][1..];
                const cap = prompts[p].len + max_new + 8;
                const first = active.items.len;
                for (samples.items) |sm| try active.append(gpa, .{
                    .prompt = p,
                    .temperature = sm.t,
                    .seed = sm.seed,
                    .main = try st.Cache.init(m.device, cap, &m.weight_plan),
                    .ref = if (set.kind == .logits) try st.Cache.init(m.device, cap, &m.weight_plan) else null,
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
                        guide(g, zr, zf, set);
                        break :blk g;
                    } else zr;
                    const tok = pick(row, x.temperature, &x.rng);
                    try x.tokens.append(gpa, tok);
                    if (tok == 200010 or tok == 200006 or x.tokens.items.len >= max_new) x.done = true;
                }
            }
            // retire finished streams
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
                say(out, "GEN\t{s}\t{d}\t{d:.2}\t{d}\t{s}\n", .{ set.name, x.prompt, x.temperature, x.seed, buf.items });
                x.main.deinit();
                if (x.ref) |*r| r.deinit();
                x.tokens.deinit(gpa);
                _ = active.swapRemove(i);
                finished += 1;
                if (finished % 50 == 0) std.debug.print("research gen {s}: {d} streams done\n", .{ set.name, finished });
            }
            if (active.items.len == 0 and next_prompt == prompts.len) break;
            if (active.items.len == 0) continue;
            // one decode row for every active stream
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
            if (set.kind == .logits) {
                configure(m, ref_config);
                try fwd.runShared(m, &s, refs[0..count], ids[0..count], .{ .all = true });
                @memcpy(ref_rows[0 .. count * v], s.logits_all.slice(f32, count * v));
            }
            for (active.items, 0..) |*x, j| {
                x.main.len += 1;
                if (x.ref) |*r| r.len += 1;
                const z = main_rows[j * v ..][0..v];
                const row: []const f32 = if (set.kind == .logits) blk: {
                    guide(g, z, ref_rows[j * v ..][0..v], set);
                    break :blk g;
                } else z;
                const tok = pick(row, x.temperature, &x.rng);
                try x.tokens.append(gpa, tok);
                if (tok == 200010 or tok == 200006 or x.tokens.items.len >= max_new) x.done = true;
            }
        }
        configure(m, .{ .w = null });
    }
}
