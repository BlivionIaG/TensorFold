//! Request-stable DSpark enrollment, not an in-request hold controller.
//! Prices pay the whole prompt/context debt in the prospective round, without amortizing it over an
//! unknown reply length. Cold and recovery admissions are bounded experiments, not speed guarantees:
//! calibration omits tap-staging/host noise and cannot predict an unseen request's acceptance. The
//! unchanged same-head, zero-tolerance full gate remains the performance authority.
const std = @import("std");
const lanes = @import("lanes");
const hp = lanes.hold_price;
const Table = lanes.table.Table;

pub const max_depth = 16;
pub const prior: [max_depth]f64 = @splat(0.84);
const margin = (lanes.depth.Constants{}).plain_margin;

pub const Prices = struct {
    target: ?*const Table,
    holds: []const hp.HoldCost,
    contexts: []const hp.ContextCost,
    streams: usize,
    heads: usize,
    target_rows: u32,
    pass_rows: u32,
    block_rows: u32,
    depth: u32,
    /// Largest complete prompt or outstanding peer debt, not a fictitious zero-debt first hold.
    context_rows: usize,
    context_chunk: u32,
};

pub const Shape = struct {
    heads: usize,
    depth: u32,
    rows: u32,
    passes: usize,
    gain: f64,
    ms: f64,
    tokens: f64,
};

pub const Quote = struct {
    plain_ms: f64,
    streams: usize,
    best: ?Shape = null,

    pub fn pays(q: Quote) bool {
        const b = q.best orelse return false;
        return b.tokens * q.plain_ms > margin * @as(f64, @floatFromInt(q.streams)) * b.ms;
    }
};

fn positive(x: f64) bool {
    return x > 0 and std.math.isFinite(x);
}

/// Uniform retained prefixes of every legal cohort size, including physical full-query splits.
/// A smaller prefix never buys a smaller query block. Missing or invalid quotes stay unknown, rather
/// than flowing through hold_price's intentionally free empty-grid convention.
pub fn quote(p: Prices, rates: []const f64) ?Quote {
    const target = p.target orelse return null;
    if (p.streams == 0 or p.streams > p.target_rows or p.block_rows == 0 or p.pass_rows < p.block_rows or p.context_chunk == 0) return null;
    if (p.holds.len == 0 or p.contexts.len == 0) return null;
    for (p.holds) |h| if (h.streams == 0 or h.depth == 0 or !positive(h.ms)) return null;
    for (p.contexts) |c| if (c.streams == 0 or c.rows == 0 or !positive(c.ms)) return null;
    const plain = target.get(@intCast(p.streams)) orelse return null;
    if (!positive(plain)) return null;
    var result: Quote = .{ .plain_ms = plain, .streams = p.streams };
    const room = p.target_rows - @as(u32, @intCast(p.streams));
    const depth = @min(p.depth, @min(p.block_rows, rates.len));
    var survival: f64 = 1;
    var gain: f64 = 0;
    for (rates[0..depth], 1..) |rate, n| {
        if (!(rate >= 0 and rate <= 1) or !std.math.isFinite(rate)) return null;
        survival *= rate;
        gain += survival;
        const heads = @min(p.heads, room / n);
        for (1..heads + 1) |count| {
            const rows = p.streams + count * n;
            const verify = target.get(@intCast(rows)) orelse return null;
            if (!positive(verify)) return null;
            // Overflow may have flushed individual streams before a hold. Charge those full chunks
            // individually; the remaining debt pays at least the grouped AND the individual quote.
            const debt = @max(p.context_rows, n + 1);
            const overflow = (debt - 1) / p.context_chunk;
            const remainder: u32 = @intCast(debt - overflow * p.context_chunk);
            const one = hp.contextMs(p.contexts, 1, remainder);
            const flushed = if (overflow == 0) 0 else hp.contextMs(p.contexts, 1, p.context_chunk) * @as(f64, @floatFromInt(overflow));
            if (!positive(one) or !(flushed >= 0) or !std.math.isFinite(flushed)) return null;
            const View = struct {
                count: usize,
                depth: i64,
                debt: u32,
                pub fn len(v: @This()) usize {
                    return v.count;
                }
                pub fn at(v: @This(), _: usize) i64 {
                    return v.depth;
                }
                pub fn contextRows(v: @This(), _: usize) u32 {
                    return v.debt;
                }
            };
            const view: View = .{ .count = count, .depth = @intCast(n), .debt = remainder };
            const clean = hp.rowsMs(p.holds, &.{}, p.pass_rows, p.block_rows, view);
            const grouped = hp.rowsMs(p.holds, p.contexts, p.pass_rows, p.block_rows, view);
            if (!positive(clean) or !positive(grouped)) return null;
            const count_f: f64 = @floatFromInt(count);
            const ms = verify + @max(grouped, clean + one * count_f) + flushed * count_f;
            if (!positive(ms)) return null;
            const tokens = @as(f64, @floatFromInt(p.streams)) + gain * count_f;
            if (result.best == null or tokens * result.best.?.ms > result.best.?.tokens * ms) {
                const per_pass = p.pass_rows / p.block_rows;
                result.best = .{ .heads = count, .depth = @intCast(n), .rows = @intCast(rows), .passes = (count + per_pass - 1) / per_pass, .gain = gain * count_f, .ms = ms, .tokens = tokens };
            }
        }
    }
    return result;
}

/// Provider/model/precision-local, and separate for greedy and sampled requests. Four prior trials
/// are finite, and the effective sample count saturates so later genuine observations can recover.
/// Only accepted conditionals and the first rejected conditional were attempted; deeper ones did not fail.
pub const Recent = struct {
    rates: [max_depth]f64 = prior,
    weight: [max_depth]f64 = @splat(4),
    observed: bool = false,
    remaining: u16 = 0,
    interval: u16 = 64,

    pub fn observe(r: *Recent, proposed: u32, accepted: u32) void {
        if (proposed == 0 or accepted > proposed) return;
        const tried = @min(max_depth, @min(@as(u64, proposed), @as(u64, accepted) + 1));
        for (0..tried) |i| {
            r.weight[i] = @min(32, r.weight[i] + 1);
            const hit: f64 = if (i < accepted) 1 else 0;
            r.rates[i] += (hit - r.rates[i]) / r.weight[i];
        }
        r.observed = true;
    }

    /// Unknown prices preserve direct diagnostics. A cold or unprofitable history gets at most one
    /// enrollment per growing 64..1024-request interval, and only if perfect acceptance could pay.
    /// Merely consuming a probe admission changes no acceptance data, even if it runs zero holds.
    pub fn enroll(r: *Recent, p: Prices, conservative_rates: []const f64) bool {
        const priced = quote(p, conservative_rates) orelse return true;
        const pays = priced.pays();
        if (r.observed and pays) {
            r.remaining = 64;
            r.interval = 64;
            return true;
        }
        if (!pays) {
            const perfect: [max_depth]f64 = @splat(1);
            const possible = quote(p, perfect[0..@min(max_depth, p.depth)]) orelse return true;
            if (!possible.pays()) return false;
        }
        if (r.remaining != 0) {
            r.remaining -= 1;
            return false;
        }
        r.remaining = r.interval;
        r.interval = @min(1024, r.interval * 2);
        return true;
    }
};

pub const History = struct {
    modes: [2]Recent = .{ .{}, .{} },

    pub fn recent(h: *History, sampled: bool) *Recent {
        return &h.modes[@intFromBool(sampled)];
    }
};
