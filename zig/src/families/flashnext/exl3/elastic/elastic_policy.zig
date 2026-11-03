//! Elastic memory policy: pure host arithmetic splitting device bytes between the K/V arena and the expert lines.
const std = @import("std");

/// Pools a rank holds at most (the decoder layers' and the MTP layer's).
pub const max_pools = 64;
/// Steps one plan holds at most; a longer move continues at the next tick.
pub const max_steps = 24;

fn roundUp(n: u64, chunk: u64) u64 {
    return (n + chunk - 1) / chunk * chunk;
}

fn sat(a: u64, b: u64) u64 {
    return a -| b;
}

/// The K/V arena: `mapped` pages usable of `count`, `stride` device bytes a page, backed in whole `chunk`s.
pub const Kv = struct {
    mapped: usize,
    count: usize,
    /// The fewest pages the arena keeps (a page and a run for every lane).
    min: usize,
    /// The fewest pages a shrink could keep now (kv_pool.Pool.tailFloor): pages above hold nothing.
    floor: usize,
    /// Pages leased, promised or kept by the prompt store.
    demand: usize,
    stride: u64,
    chunk: u64,

    /// Device bytes the first `pages` pages hold mapped.
    pub fn bytes(k: Kv, pages: usize) u64 {
        return roundUp(@as(u64, pages) * k.stride, k.chunk);
    }

    /// The share of the mapped pages in use, in parts per million (past a million when nothing is mapped).
    pub fn utilization(k: Kv) u64 {
        if (k.mapped == 0) return 2_000_000;
        return @as(u64, k.demand) * 1_000_000 / k.mapped;
    }
};

/// One expert pool: `active` lines of `cap` map `bytes` each, packed in whole `chunk`s.
pub const Line = struct {
    active: u32,
    floor: u32,
    cap: u32,
    bytes: u64,
    chunk: u64,

    /// Device bytes `n` lines hold mapped.
    pub fn mapped(l: Line, n: u32) u64 {
        return roundUp(@as(u64, n) * l.bytes, l.chunk);
    }
};

/// Everything the policy reads of one rank at a quiescent moment.
pub const Memory = struct {
    kv: Kv,
    pools: [max_pools]Line = undefined,
    n_pools: usize = 0,
    /// Device bytes free now (the smaller of the ranks' in a pair).
    free: u64,
    /// Bytes that stay free: the server's reserve and the allocator margin.
    reserve: u64,
    /// The engine's phase: a tick in a prefill leaves the lines (the ring's slots) alone and grows none.
    phase: Phase = .idle,
    /// Pages the queued requests lack beyond those free or reclaimable (the next ones the free lanes take).
    ahead: usize = 0,
    /// Streams running plus requests waiting.
    load: u32 = 0,

    pub fn lines(m: *const Memory) []const Line {
        return m.pools[0..m.n_pools];
    }

    /// Bytes mapped for lines if every pool were cut by `k` lines (to its floor at most) fewer than now.
    pub fn release(m: *const Memory, k: u32) u64 {
        var sum: u64 = 0;
        for (m.lines()) |l| sum += l.mapped(l.active) - l.mapped(@max(l.floor, l.active -| k));
        return sum;
    }

    /// Bytes mapped for lines if every pool gained `k` lines (to its cap at most).
    pub fn cost(m: *const Memory, k: u32) u64 {
        var sum: u64 = 0;
        for (m.lines()) |l| sum += l.mapped(@min(l.cap, l.active +| k)) - l.mapped(l.active);
        return sum;
    }

    /// The most lines one uniform step may take from a pool, and the most it may add.
    pub fn shrinkRoom(m: *const Memory) u32 {
        var most: u32 = 0;
        for (m.lines()) |l| most = @max(most, l.active - @min(l.active, l.floor));
        return most;
    }

    pub fn growRoom(m: *const Memory) u32 {
        var most: u32 = 0;
        for (m.lines()) |l| most = @max(most, l.cap - @min(l.cap, l.active));
        return most;
    }

    /// Lines held across the pools.
    pub fn total(m: *const Memory) u64 {
        var sum: u64 = 0;
        for (m.lines()) |l| sum += l.active;
        return sum;
    }

    fn spare(m: *const Memory) u64 {
        return sat(m.free, m.reserve);
    }
};

/// What the engine is doing between two rounds: no stream, a prompt being filled (some stream), or decoding only.
pub const Phase = enum { idle, prefill, decode };

/// One move of one side: K/V to `kv` pages, or every pool `lines` lines more (negative: fewer, to its floor at most).
pub const Step = union(enum) { kv: usize, lines: i32 };

pub const Reason = enum { none, request, queued, pressure, idle };

/// Steps in order; each is funded when it runs.
pub const Plan = struct {
    steps: [max_steps]Step = undefined,
    n: usize = 0,
    reason: Reason = .none,
    /// Bytes mapped plus bytes unmapped over the whole plan.
    moved: u64 = 0,

    pub fn slice(p: *const Plan) []const Step {
        return p.steps[0..p.n];
    }

    fn push(p: *Plan, s: Step) bool {
        if (p.n == max_steps) return false;
        p.steps[p.n] = s;
        p.n += 1;
        return true;
    }
};

/// Applies `s` to the model and returns the bytes it mapped or unmapped; the device's `free` follows the mapped bytes.
pub fn apply(m: *Memory, s: Step) u64 {
    switch (s) {
        .kv => |pages| {
            const from = m.kv.bytes(m.kv.mapped);
            const to = m.kv.bytes(pages);
            m.kv.mapped = pages;
            m.free = if (to >= from) sat(m.free, to - from) else m.free + (from - to);
            return if (to >= from) to - from else from - to;
        },
        .lines => |k| {
            var moved: u64 = 0;
            for (m.pools[0..m.n_pools]) |*l| {
                const was = l.mapped(l.active);
                const want: u32 = if (k >= 0) @min(l.cap, l.active +| @as(u32, @intCast(k))) else @max(l.floor, l.active -| @as(u32, @intCast(-@as(i64, k))));
                const now = l.mapped(want);
                l.active = want;
                m.free = if (now >= was) sat(m.free, now - was) else m.free + (was - now);
                moved += if (now >= was) now - was else was - now;
            }
            return moved;
        },
    }
}

/// The knobs: watermarks in parts per million of the mapped K/V pages, and tick counts.
pub const Config = struct {
    /// K/V use above this share grows the arena; below `low_ppm` for `idle_ticks` ticks lets lines take its memory.
    high_ppm: u32 = 850_000,
    low_ppm: u32 = 350_000,
    /// The share a resize aims at, so the next change is a long way off.
    fill_ppm: u32 = 600_000,
    idle_ticks: u32 = 8,
    /// Ticks between two plans that are not urgent.
    dwell_ticks: u32 = 16,
    /// Ticks nothing is attempted after a plan failed.
    backoff_ticks: u32 = 40,
    /// Bytes one step moves at most (one chunk step at least).
    step_bytes: u64 = 512 << 20,
    /// Ticks K/V gives no memory back after two or more streams were running (0: it does at once).
    hold_ticks: u32 = 120,

    pub fn valid(c: Config) bool {
        return c.low_ppm > 0 and c.low_ppm < c.fill_ppm and c.fill_ppm < c.high_ppm and c.high_ppm <= 1_000_000 and c.idle_ticks > 0 and c.step_bytes > 0;
    }
};

/// What carries from tick to tick.
pub const State = struct {
    /// Consecutive ticks with K/V use at or below the low watermark.
    calm: u32 = 0,
    /// Ticks since the last plan was applied (or since the start).
    dwell: u32 = 0,
    /// Ticks left in which nothing is attempted.
    blocked: u32 = 0,
    /// Ticks left in which K/V keeps what it holds; two or more streams (running or queued) renew it.
    hold: u32 = 0,

    /// One tick passes: the counters see this tick's K/V use and load.
    pub fn advance(s: *State, c: Config, m: *const Memory) void {
        s.calm = if (m.kv.utilization() <= c.low_ppm) s.calm +| 1 else 0;
        s.dwell +|= 1;
        s.blocked -|= 1;
        s.hold = if (m.load >= 2) c.hold_ticks else s.hold -| 1;
    }

    /// A plan went through: both sides start over.
    pub fn applied(s: *State) void {
        s.dwell = 0;
        s.calm = 0;
    }

    /// A plan failed and was rolled back: nothing is attempted for a while.
    pub fn failed(s: *State, c: Config) void {
        s.blocked = c.backoff_ticks;
        s.calm = 0;
    }
};

fn pagesFor(demand: usize, fill_ppm: u32) usize {
    return @intCast((@as(u64, demand) * 1_000_000 + fill_ppm - 1) / fill_ppm);
}

/// The most K/V pages reachable now: free bytes above the reserve, plus (with `cuts`) every line cut to its floor.
pub fn reachableWith(m: *const Memory, cuts: bool) usize {
    const budget = m.spare() + if (cuts) m.release(m.shrinkRoom()) else 0;
    var lo = m.kv.mapped;
    var hi = m.kv.count;
    const base = m.kv.bytes(m.kv.mapped);
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (m.kv.bytes(mid) - base <= budget) lo = mid else hi = mid - 1;
    }
    return lo;
}

pub fn reachable(m: *const Memory) usize {
    return reachableWith(m, true);
}

/// Whether `extra` more K/V pages than mapped can be had (the admission's question: could it fit after a plan).
pub fn canGrow(m: *const Memory, extra: usize) bool {
    return extra == 0 or reachable(m) >= m.kv.mapped + extra;
}

/// The next K/V page count at most `step` bytes above `kv.mapped` toward `goal` (the current count when nothing fits).
fn kvUp(m: *const Memory, goal: usize, step: u64) usize {
    var lo = m.kv.mapped;
    var hi = goal;
    const base = m.kv.bytes(m.kv.mapped);
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (m.kv.bytes(mid) - base <= step) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// The page count one mapping chunk (at least) below `kv.mapped`, no lower than `goal` and within `step` bytes.
fn kvDown(m: *const Memory, goal: usize, step: u64) usize {
    var lo = goal;
    var hi = m.kv.mapped;
    const base = m.kv.bytes(m.kv.mapped);
    // the lowest count whose release stays within the step
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (base - m.kv.bytes(mid) <= step) hi = mid else lo = mid + 1;
    }
    if (m.kv.bytes(lo) == base) return m.kv.mapped; // inside one chunk: no memory to give back
    return lo;
}

/// The least `k` in [1, room] whose line release reaches `need`; null when even `room` falls short.
fn shrinkFor(m: *const Memory, need: u64) ?u32 {
    const room = m.shrinkRoom();
    if (room == 0 or m.release(room) < need) return null;
    var lo: u32 = 1;
    var hi: u32 = room;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (m.release(mid) >= need) hi = mid else lo = mid + 1;
    }
    return lo;
}

/// The most `k` in [0, room] with `cost(k)` within `budget` (a step's, so a granule above it is a single line).
fn growWithin(m: *const Memory, room: u32, budget: u64) u32 {
    var lo: u32 = 0;
    var hi: u32 = room;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (m.cost(mid) <= budget) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// K/V up to `goal` pages in bounded steps: grow into spare bytes, else cut the lines to pay for the next one.
fn growKv(c: Config, m: *const Memory, goal: usize, cuts: bool, out: *Plan) void {
    var sim = m.*;
    while (sim.kv.mapped < goal) {
        const spare = sim.spare();
        const up = kvUp(&sim, goal, @min(c.step_bytes, spare));
        if (up > sim.kv.mapped) {
            if (!out.push(.{ .kv = up })) return;
            out.moved += apply(&sim, .{ .kv = up });
            continue;
        }
        // the next chunk of the goal does not fit a step's spare bytes: the lines pay for it, or one chunk goes whole
        var next = sim.kv.mapped + 1;
        while (next < goal and sim.kv.bytes(next) == sim.kv.bytes(sim.kv.mapped)) next += 1;
        const one = sim.kv.bytes(next) - sim.kv.bytes(sim.kv.mapped);
        if (one <= spare) { // a chunk is larger than the step: it goes whole
            if (!out.push(.{ .kv = next })) return;
            out.moved += apply(&sim, .{ .kv = next });
            continue;
        }
        const want = @max(one, @min(c.step_bytes, sim.kv.bytes(goal) - sim.kv.bytes(sim.kv.mapped)));
        if (!cuts) return; // only the free bytes may pay
        const k = shrinkFor(&sim, want - spare) orelse return;
        // within the step when it can be: the largest cut that stays under it, one line at least
        var step_k = k;
        while (step_k > 1 and sim.release(step_k) > c.step_bytes) step_k -= 1;
        if (!out.push(.{ .lines = -@as(i32, @intCast(step_k)) })) return;
        out.moved += apply(&sim, .{ .lines = -@as(i32, @intCast(step_k)) });
    }
}

/// Lines `goal` more (uniform) in bounded steps, K/V cut to `kv_goal` pages where the free bytes run out.
fn growLines(c: Config, m: *const Memory, goal: u32, kv_goal: usize, out: *Plan) void {
    var sim = m.*;
    var left = goal;
    while (left > 0) {
        const budget = @min(c.step_bytes, sim.spare());
        var k = growWithin(&sim, left, budget);
        if (k == 0 and sim.kv.mapped <= kv_goal and sim.cost(1) > sim.spare()) return;
        if (k == 0 and sim.cost(1) <= sim.spare()) k = 1; // one line is a whole step
        if (k > 0) {
            if (!out.push(.{ .lines = @intCast(k) })) return;
            out.moved += apply(&sim, .{ .lines = @intCast(k) });
            left -= k;
            continue;
        }
        const down = kvDown(&sim, kv_goal, c.step_bytes);
        if (down >= sim.kv.mapped) return;
        if (!out.push(.{ .kv = down })) return;
        out.moved += apply(&sim, .{ .kv = down });
    }
}

/// The plan for this tick (`st` advanced), or an empty one: urgent K/V growth on a request or high use, else lines.
pub fn plan(config: Config, st: State, m: *const Memory, request: usize) Plan {
    var out: Plan = .{};
    if (st.blocked > 0 or m.n_pools == 0 or m.kv.count < m.kv.mapped) return out;
    var c = config;
    c.step_bytes = @max(c.step_bytes, m.kv.chunk); // a step is at least the chunk it maps
    for (m.lines()) |l| c.step_bytes = @max(c.step_bytes, l.chunk);
    const util = m.kv.utilization();
    // a request or its forecast may cut lines; a prefill tick that only sees high use grows K/V into free bytes
    const ask = @max(request, m.ahead);
    const cuts = ask > 0 or m.phase != .prefill;
    if (ask > 0 or util > c.high_ppm) {
        const need_min = m.kv.mapped + @max(ask, 1);
        const soft = pagesFor(m.kv.demand + ask, c.fill_ppm);
        const goal = @min(m.kv.count, @max(need_min, soft));
        const reach = reachableWith(m, cuts);
        if (need_min > m.kv.count or goal <= m.kv.mapped or reach < need_min) return out;
        growKv(c, m, @min(goal, reach), cuts, &out);
        if (out.n > 0) out.reason = if (request > 0) .request else if (ask > 0) .queued else .pressure;
        return out;
    }
    if (m.phase == .prefill) return out; // lines (and the K/V that pays for them) wait for decode or idle
    if (util > c.low_ppm or st.calm < c.idle_ticks or st.dwell < c.dwell_ticks) return out;
    const room = m.growRoom();
    if (room == 0) return out;
    // after a burst of streams the arena keeps its pages: lines may still grow into free bytes
    const kv_goal = if (st.hold > 0) m.kv.mapped else @min(m.kv.mapped, @max(@max(m.kv.min, m.kv.floor), pagesFor(m.kv.demand, c.fill_ppm)));
    const give = m.kv.bytes(m.kv.mapped) - m.kv.bytes(kv_goal);
    const goal = growWithin(m, room, m.spare() + give);
    if (goal == 0) return out;
    // the arena gives back only what the lines will map beyond the spare bytes
    const owed = sat(m.cost(goal), m.spare());
    var keep = kv_goal;
    var hi = m.kv.mapped;
    while (keep < hi) { // the highest count whose release still covers `owed`
        const mid = keep + (hi - keep + 1) / 2;
        if (m.kv.bytes(m.kv.mapped) - m.kv.bytes(mid) >= owed) keep = mid else hi = mid - 1;
    }
    growLines(c, m, goal, if (owed == 0) m.kv.mapped else keep, &out);
    if (out.n > 0) out.reason = .idle;
    return out;
}

test {
    _ = @import("elastic_policy_test.zig");
}
