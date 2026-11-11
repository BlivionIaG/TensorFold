//! The report of expert_cache_sim.py: the job results gathered per (configuration, dataset), and the four markdown tables, line for line.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const data = @import("expert_cache_sim_data.zig");
const cfg = @import("expert_cache_sim_config.zig");
const run = @import("expert_cache_sim_run.zig");
const fmt = @import("expert_cache_sim_fmt.zig");

const Writer = std.Io.Writer;
const bucket_names = [_][]const u8{ "0-8", "8-32", "32-128", "128-512", "512+" };

pub fn fixed(w: *Writer, x: f64, prec: u8) Writer.Error!void {
    return fmt.fixed(w, x, prec, false);
}

pub const Agg = struct {
    have: bool = false,
    acc: u64 = 0,
    adm: u64 = 0,
    evi: u64 = 0,
    tok: u64 = 0,
    miss: u64 = 0,
    pos: [run.buckets.len][2]u64 = @splat(.{ 0, 0 }),
    /// misses per decode token of each prompt, in prompt-name order (short, long)
    per: []f64 = &.{},
};

pub const Results = struct {
    configs: []const cfg.Config,
    /// the datasets present, short first
    sets: [2]?data.Set,
    aggs: [][3]Agg,
    /// index of the "LRU global" configuration, if selected
    base: ?usize,

    pub fn init(a: std.mem.Allocator, configs: []const cfg.Config, short: data.Set, long: ?data.Set, jobs: []const run.Job, results: []const run.Result) !Results {
        const aggs = try a.alloc([3]Agg, configs.len);
        for (aggs) |*row| row.* = @splat(.{});
        const sets = [2]?data.Set{ short, long };
        for (jobs, results) |job, r| {
            const g = &aggs[job.config][@backingInt(job.dataset)];
            g.have = true;
            g.acc += r.acc;
            g.adm += r.adm;
            g.evi += r.evi;
            g.tok += r.tok;
            g.miss += r.miss;
            if (job.dataset == .lock4) continue;
            if (g.per.len == 0) g.per = try a.alloc(f64, sets[@backingInt(job.dataset)].?.traces.len);
            g.per[job.item] = @as(f64, @floatFromInt(r.miss)) / @as(f64, @floatFromInt(r.tok));
            for (&g.pos, r.pos) |*dst, src| {
                dst[0] += src[0];
                dst[1] += src[1];
            }
        }
        var base: ?usize = null;
        for (configs, 0..) |c, ci| {
            if (std.mem.startsWith(u8, c.label, "LRU global")) {
                base = ci;
                break;
            }
        }
        return .{ .configs = configs, .sets = sets, .aggs = aggs, .base = base };
    }

    fn get(self: *const Results, ci: usize, ds: run.Dataset) ?*const Agg {
        const g = &self.aggs[ci][@backingInt(ds)];
        return if (g.have) g else null;
    }

    /// `hit(ci, ds)`: 100 * (1 - miss / accesses), percent
    fn hit(self: *const Results, ci: usize, ds: run.Dataset) ?f64 {
        const g = self.get(ci, ds) orelse return null;
        return 100.0 * (1 - ratio(g.miss, g.acc));
    }

    /// `mpt(ci, ds)`: link reads per decode token
    fn mpt(self: *const Results, ci: usize, ds: run.Dataset) ?f64 {
        const g = self.get(ci, ds) orelse return null;
        return ratio(g.miss, g.tok);
    }

    fn present(self: *const Results, ds: run.Dataset) bool {
        return self.sets[@backingInt(ds)] != null;
    }
};

fn ratio(a: u64, b: u64) f64 {
    return @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
}

/// One table row: `| a | b | c |`.
const Line = struct {
    w: *Writer,
    n: usize = 0,

    fn start(w: *Writer) !Line {
        try w.writeAll("| ");
        return .{ .w = w };
    }

    fn sep(self: *Line) !void {
        if (self.n > 0) try self.w.writeAll(" | ");
        self.n += 1;
    }

    fn text(self: *Line, s: []const u8) !void {
        try self.sep();
        try self.w.writeAll(s);
    }

    fn num(self: *Line, x: f64, prec: u8, plus: bool) !void {
        try self.sep();
        try fmt.fixed(self.w, x, prec, plus);
    }

    /// `f(x, nd)`: "-" for None
    fn opt(self: *Line, x: ?f64, prec: u8) !void {
        if (x) |v| try self.num(v, prec, false) else try self.text("-");
    }

    fn end(self: *Line) !void {
        try self.w.writeAll(" |\n");
    }
};

fn separator(w: *Writer, cells: usize) !void {
    try w.writeAll("|");
    for (0..cells) |_| try w.writeAll("---|");
    try w.writeAll("\n");
}

fn datasets(r: *const Results, buf: *[2]run.Dataset) []const run.Dataset {
    var n: usize = 0;
    for ([_]run.Dataset{ .short, .long }) |d| if (r.present(d)) {
        buf[n] = d;
        n += 1;
    };
    return buf[0..n];
}

/// Descending by short+long hit rate, equal rates keeping the configuration order (`sorted(..., reverse=True)` is stable).
fn comb(r: *const Results, ci: usize, dss: []const run.Dataset) f64 {
    var a: u64 = 0;
    var b: u64 = 0;
    for (dss) |d| {
        const g = r.get(ci, d).?;
        a += g.miss;
        b += g.acc;
    }
    return 1 - ratio(a, b);
}

pub const Error = Writer.Error || error{ NoBaseline, OutOfMemory };

pub fn tables(a: std.mem.Allocator, w: *Writer, r: *const Results) Error!void {
    var dbuf: [2]run.Dataset = undefined;
    const dss = datasets(r, &dbuf);
    const configs = r.configs;
    const base = r.base;

    // ---- table 1
    try w.writeAll("## 1. cold-start replay, every policy admit-on-miss, one global pool unless stated\n");
    try w.writeAll("hit% = decode expert accesses served from VRAM; miss/tok = link reads per decode token (320 accesses/token); lock4 miss is per sequence-token;\n");
    try w.writeAll("adm/evi per token on the short set (admissions that evict count in both).\n\n");
    var head: Writer.Allocating = .init(a);
    try head.writer.writeAll("| policy | ");
    for (dss, 0..) |d, i| {
        if (i > 0) try head.writer.writeAll(" | ");
        try head.writer.print("{t} hit% | {t} miss/tok", .{ d, d });
    }
    try head.writer.writeAll(" | lock4 hit% | lock4 miss/seq-tok | adm/tok | evict/tok | short Δ vs LRU (pts) |");
    try w.print("{s}\n", .{head.written()});
    try separator(w, std.mem.count(u8, head.written(), "|") - 1);
    for (configs, 0..) |c, ci| {
        if (c.init != .cold) continue;
        var line = try Line.start(w);
        try line.text(c.label);
        for (dss) |d| {
            try line.opt(r.hit(ci, d), 1);
            try line.opt(r.mpt(ci, d), 2);
        }
        try line.opt(r.hit(ci, .lock4), 1);
        try line.opt(r.mpt(ci, .lock4), 2);
        const s = r.get(ci, .short).?;
        try line.num(ratio(s.adm, s.tok), 2, false);
        try line.num(ratio(s.evi, s.tok), 2, false);
        if (base) |b| try line.num(r.hit(ci, .short).? - r.hit(b, .short).?, 2, true) else try line.text("-");
        try line.end();
    }

    // ---- the two best non-LRU candidates (hit on short+long weighted by tokens)
    var best: [2]usize = undefined;
    var nbest: usize = 0;
    {
        var cand: std.ArrayList(usize) = .empty;
        defer cand.deinit(a);
        for (configs, 0..) |c, ci| {
            if (c.init == .cold and !std.mem.startsWith(u8, c.label, "LRU global")) try cand.append(a, ci);
        }
        var i: usize = 1;
        while (i < cand.items.len) : (i += 1) {
            var j = i;
            while (j > 0 and comb(r, cand.items[j - 1], dss) < comb(r, cand.items[j], dss)) : (j -= 1) std.mem.swap(usize, &cand.items[j - 1], &cand.items[j]);
        }
        nbest = @min(2, cand.items.len);
        @memcpy(best[0..nbest], cand.items[0..nbest]);
    }
    try w.writeAll("\n## 2. evictions / admissions per token: LRU baseline and the two best other candidates (short+long hit rate)\n\n");
    try w.writeAll("| policy | ");
    for (dss, 0..) |d, i| {
        if (i > 0) try w.writeAll(" | ");
        try w.print("{t} adm/tok | {t} evict/tok | {t} miss/tok", .{ d, d, d });
    }
    try w.writeAll(" |\n");
    try separator(w, 1 + 3 * dss.len);
    var rows: [3]usize = undefined;
    var nrows: usize = 0;
    if (base) |b| {
        rows[0] = b;
        nrows = 1;
    }
    @memcpy(rows[nrows..][0..nbest], best[0..nbest]);
    nrows += nbest;
    for (rows[0..nrows]) |ci| {
        var line = try Line.start(w);
        try line.text(configs[ci].label);
        for (dss) |d| {
            const g = r.get(ci, d).?;
            try line.num(ratio(g.adm, g.tok), 2, false);
            try line.num(ratio(g.evi, g.tok), 2, false);
            try line.num(ratio(g.miss, g.tok), 2, false);
        }
        try line.end();
    }
    if (base) |b| {
        try w.writeAll("\nper-prompt paired comparison against LRU (miss/token difference, negative = fewer link reads than LRU):\n\n");
        try w.writeAll("| policy | set | prompts better / equal(<0.05) / worse | mean diff miss/tok | std error |\n");
        try w.writeAll("|---|---|---|---|---|\n");
        for (best[0..nbest]) |ci| {
            for (dss) |d| {
                const cur = r.get(ci, d).?.per;
                const ref = r.get(b, d).?.per;
                const diff = try a.alloc(f64, cur.len);
                defer a.free(diff);
                const scratch = try a.alloc(f64, cur.len);
                defer a.free(scratch);
                var lt: usize = 0;
                var eq: usize = 0;
                var gt: usize = 0;
                for (diff, cur, ref) |*x, c, rf| {
                    x.* = c - rf;
                    if (x.* < -0.05) lt += 1;
                    if (@abs(x.*) <= 0.05) eq += 1;
                    if (x.* > 0.05) gt += 1;
                }
                const se: f64 = if (diff.len > 1) fmt.stdDdof1(diff, scratch) / @sqrt(@as(f64, @floatFromInt(diff.len))) else std.math.nan(f64);
                try w.print("| {s} | {t} | {d} / {d} / {d} | ", .{ configs[ci].label, d, lt, eq, gt });
                try fmt.fixed(w, fmt.mean(diff), 2, true);
                try w.writeAll(" | ");
                try fmt.fixed(w, se, 2, false);
                try w.writeAll(" |\n");
            }
        }
    }

    // ---- table 3: seeding / warm start by decode position
    var seedrows: std.ArrayList(usize) = .empty;
    defer seedrows.deinit(a);
    for (configs, 0..) |c, ci| {
        if (std.mem.startsWith(u8, c.label, "LRU global") or c.init != .cold) try seedrows.append(a, ci);
    }
    try w.writeAll("\n## 3. cold vs seeded vs prefill-warm LRU: misses per decode token by decode position (320 accesses/token)\n\n");
    try w.writeAll("| set | start | ");
    for (bucket_names, 0..) |n, i| try w.print("{s}{s}", .{ if (i > 0) " | " else "", n });
    try w.writeAll(" | all tokens |\n");
    try separator(w, 3 + bucket_names.len);
    for (dss) |d| {
        for (seedrows.items) |ci| {
            const g = r.get(ci, d).?;
            var line = try Line.start(w);
            try line.text(@tagName(d));
            try line.text(if (configs[ci].init != .cold) configs[ci].label else "LRU cold start");
            for (g.pos) |p| {
                try line.sep();
                if (p[1] != 0) {
                    try fmt.fixed(w, ratio(p[0], p[1]), 1, false);
                    try w.print(" (n={d})", .{p[1]});
                } else try w.writeAll("-");
            }
            try line.num(ratio(g.miss, g.tok), 1, false);
            try line.end();
        }
    }

    // steady-state reference: LRU's miss/token in 512+ / 128-512, and tokens before each warm start catches up
    try w.writeAll("\n## 4. what the warm starts save (LRU cold minus warm, link reads per prompt, summed over decode tokens, per bucket)\n\n");
    try w.writeAll("| set | start | ");
    for (bucket_names, 0..) |n, i| try w.print("{s}{s}", .{ if (i > 0) " | " else "", n });
    try w.writeAll(" | total reads saved per prompt | prompts |\n");
    try separator(w, 4 + bucket_names.len);
    for (dss) |d| {
        const nprompt: f64 = @floatFromInt(r.sets[@backingInt(d)].?.traces.len);
        for (seedrows.items) |ci| {
            if (configs[ci].init == .cold) continue;
            const b = base orelse return error.NoBaseline;
            const bg = r.get(b, d).?;
            const g = r.get(ci, d).?;
            var line = try Line.start(w);
            try line.text(@tagName(d));
            try line.text(configs[ci].label);
            var tot: i64 = 0;
            for (0..run.buckets.len) |bi| {
                const dm = @as(i64, @intCast(bg.pos[bi][0])) - @as(i64, @intCast(g.pos[bi][0]));
                tot += dm;
                if (bg.pos[bi][1] != 0) try line.num(@as(f64, @floatFromInt(dm)) / nprompt, 0, true) else try line.text("-");
            }
            try line.num(@as(f64, @floatFromInt(tot)) / nprompt, 0, true);
            try line.text(try std.fmt.allocPrint(a, "{d}", .{r.sets[@backingInt(d)].?.traces.len}));
            try line.end();
        }
    }
}
