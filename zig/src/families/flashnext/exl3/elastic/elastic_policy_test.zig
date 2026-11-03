//! The elastic memory policy: watermarks, hysteresis, floors, conservation, bounded steps, determinism, a random walk.
const std = @import("std");
const policy = @import("elastic_policy.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const mib: u64 = 1 << 20;
const line_bytes: u64 = 1_843_200;

fn memory(pools: usize, free: u64) policy.Memory {
    var m: policy.Memory = .{
        .kv = .{ .mapped = 400, .count = 400, .min = 20, .floor = 10, .demand = 40, .stride = 3 * mib, .chunk = 2 * mib },
        .free = free,
        .reserve = 100 * mib,
        .n_pools = pools,
    };
    for (m.pools[0..pools]) |*l| l.* = .{ .active = 20, .floor = 10, .cap = 80, .bytes = line_bytes, .chunk = 2 * mib };
    return m;
}

/// Bytes held by both sides and free: what a plan must keep whole.
fn ledger(m: *const policy.Memory) u64 {
    var sum = m.free + m.kv.bytes(m.kv.mapped);
    for (m.lines()) |l| sum += l.mapped(l.active);
    return sum;
}

const cfg: policy.Config = .{ .step_bytes = 64 * mib };

/// Applies `p` to a copy of `m0` step by step, asserting every invariant, and returns the result.
fn walk(c: policy.Config, m0: policy.Memory, p: policy.Plan) !policy.Memory {
    var m = m0;
    const total = ledger(&m);
    var granule: u64 = @max(c.step_bytes, m.kv.chunk);
    for (m.lines()) |l| granule = @max(granule, l.chunk);
    for (p.slice()) |s| {
        const before = m;
        const moved = policy.apply(&m, s);
        switch (s) {
            .kv => |pages| {
                try expect(pages >= m.kv.min and pages <= m.kv.count);
                if (pages < before.kv.mapped) try expect(pages >= m.kv.floor); // never below what the pool holds
                try expect(moved <= granule);
                // a step that maps is paid for by spare bytes before it runs
                if (pages > before.kv.mapped) try expect(moved == 0 or before.free >= before.reserve + moved);
            },
            .lines => |k| {
                try expect(k != 0);
                for (m.lines(), before.lines()) |l, was| {
                    try expect(l.active >= l.floor and l.active <= l.cap);
                    try expect(if (k > 0) l.active >= was.active else l.active <= was.active);
                }
                if (k > 0) try expect(moved == 0 or before.free >= before.reserve + moved);
            },
        }
        try equal(total, ledger(&m)); // conservation: what one side lost, the other or free gained
    }
    return m;
}

fn expectSame(a: policy.Plan, b: policy.Plan) !void {
    try equal(a.n, b.n);
    try equal(a.reason, b.reason);
    try equal(a.moved, b.moved);
    for (a.slice(), b.slice()) |x, y| try std.testing.expect(std.meta.eql(x, y));
}

test "exl3: elastic policy: a request grows K/V from spare bytes alone when they reach" {
    var m = memory(4, 100 * mib + 900 * mib);
    m.kv.mapped = 100;
    m.kv.demand = 95;
    const p = policy.plan(cfg, .{}, &m, 30);
    try equal(policy.Reason.request, p.reason);
    for (p.slice()) |s| try expect(s == .kv); // free bytes cover it: no line moves
    const after = try walk(cfg, m, p);
    try expect(after.kv.mapped >= 130);
    try equal(m.total(), after.total());
}

test "exl3: elastic policy: a request the spare bytes do not reach cuts lines first, then grows K/V" {
    var m = memory(4, 100 * mib); // nothing spare
    m.kv.mapped = 100;
    m.kv.demand = 100;
    const p = policy.plan(cfg, .{}, &m, 20);
    try expect(p.n >= 2);
    try expect(p.steps[0] == .lines and p.steps[0].lines < 0);
    var seen_kv = false;
    for (p.slice()) |s| seen_kv = seen_kv or s == .kv;
    try expect(seen_kv);
    const after = try walk(cfg, m, p);
    try expect(after.kv.mapped >= 120);
    try expect(after.total() < m.total());
    for (after.lines()) |l| try expect(l.active >= l.floor);
}

test "exl3: elastic policy: a request beyond what lines at their floor and spare bytes can pay for is not planned" {
    var m = memory(2, 100 * mib);
    m.kv.mapped = 100;
    m.kv.demand = 100;
    const hold = policy.reachable(&m);
    try expect(hold > 100 and hold < 400);
    try expect(policy.canGrow(&m, hold - 100));
    try expect(!policy.canGrow(&m, hold - 99));
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, hold - 99).n);
    const p = policy.plan(cfg, .{}, &m, hold - 100);
    try expect(p.n > 0);
    try expect((try walk(cfg, m, p)).kv.mapped >= hold);
    m.kv.mapped = m.kv.count; // nothing above the arena's own count either
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, 1).n);
}

test "exl3: elastic policy: pressure above the high watermark grows toward the fill level, inside the band nothing moves" {
    var m = memory(3, 100 * mib + 600 * mib);
    m.kv.mapped = 100;
    m.kv.demand = 86; // 86%: above 85%
    const p = policy.plan(cfg, .{}, &m, 0);
    try equal(policy.Reason.pressure, p.reason);
    const after = try walk(cfg, m, p);
    try expect(after.kv.mapped >= 143); // 86 / 0.60
    m.kv.demand = 85; // at the watermark: not above it
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, 0).n);
    m.kv.demand = 50; // between the watermarks: the band does nothing, however long it lasts
    var st: policy.State = .{};
    for (0..100) |_| st.advance(cfg, &m);
    try equal(@as(usize, 0), policy.plan(cfg, st, &m, 0).n);
}

test "exl3: elastic policy: lines grow only after idle_ticks calm ticks and dwell_ticks since the last plan" {
    var m = memory(4, 100 * mib);
    m.kv.demand = 10; // 2.5%: below the low watermark
    var st: policy.State = .{};
    const c: policy.Config = .{ .step_bytes = 64 * mib, .idle_ticks = 5, .dwell_ticks = 9 };
    for (0..8) |_| {
        st.advance(c, &m);
        try equal(@as(usize, 0), policy.plan(c, st, &m, 0).n); // dwell (9) not reached
    }
    st.advance(c, &m);
    const p = policy.plan(c, st, &m, 0);
    try equal(policy.Reason.idle, p.reason);
    // K/V gives memory back and lines take it: a K/V cut comes before the first line step
    try expect(p.steps[0] == .kv);
    const after = try walk(c, m, p);
    try expect(after.total() > m.total());
    try expect(after.kv.mapped >= @max(m.kv.min, m.kv.floor));
    // a K/V use that rises above the low watermark resets the calm count
    st.applied();
    m.kv.demand = 200;
    st.advance(c, &m);
    try equal(@as(u32, 0), st.calm);
    try equal(@as(u32, 1), st.dwell);
}

test "exl3: elastic policy: a prefill's tick grows K/V into free bytes only, a request still cuts lines, lazy lines wait" {
    var m = memory(3, 100 * mib + 20 * mib); // 20 MiB spare
    m.kv.mapped = 100;
    m.kv.demand = 90; // above the high watermark
    m.phase = .prefill;
    const p = policy.plan(cfg, .{}, &m, 0);
    try equal(policy.Reason.pressure, p.reason);
    for (p.slice()) |s| try expect(s == .kv); // the lines are the ring's slots: untouched
    const after = try walk(cfg, m, p);
    try expect(after.kv.mapped > 100 and after.kv.bytes(after.kv.mapped) - m.kv.bytes(100) <= 20 * mib);
    try equal(m.total(), after.total());
    m.free = m.reserve; // no spare at all: the tick plans nothing rather than cut lines
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, 0).n);
    // a stream asking for pages is admitted by cutting lines whatever the phase
    const asked = policy.plan(cfg, .{}, &m, 10);
    try equal(policy.Reason.request, asked.reason);
    try expect(asked.steps[0] == .lines);
    // lazy growth of lines (and the K/V give back that pays for it) wait for decode or idle
    var calm = memory(4, 100 * mib);
    calm.kv.demand = 10;
    const c: policy.Config = .{ .step_bytes = 64 * mib, .idle_ticks = 1, .dwell_ticks = 0 };
    var st: policy.State = .{};
    st.advance(c, &calm);
    calm.phase = .prefill;
    try equal(@as(usize, 0), policy.plan(c, st, &calm, 0).n);
    for ([_]policy.Phase{ .decode, .idle }) |phase| {
        calm.phase = phase;
        try equal(policy.Reason.idle, policy.plan(c, st, &calm, 0).reason);
    }
}

test "exl3: elastic policy: a forecast of the queue's pages plans K/V like a request, in any phase" {
    var m = memory(3, 100 * mib + 20 * mib);
    for (m.pools[0..3]) |*l| l.active = 60; // lines to cut
    m.kv.mapped = 100;
    m.kv.demand = 50; // far below the high watermark: only the forecast asks
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, 0).n);
    m.ahead = 40;
    for ([_]policy.Phase{ .idle, .decode, .prefill }) |phase| { // no ring is live at a tick: lines may pay in a prefill
        m.phase = phase;
        const p = policy.plan(cfg, .{}, &m, 0);
        try equal(policy.Reason.queued, p.reason);
        const after = try walk(cfg, m, p);
        try expect(after.kv.mapped >= 140);
        try expect(after.total() < m.total()); // 120 MiB of pages, 20 MiB spare: lines paid
    }
    try equal(policy.Reason.request, policy.plan(cfg, .{}, &m, 5).reason); // a request that is asked outranks it
    m.ahead = 100_000; // more than the arena holds
    try equal(@as(usize, 0), policy.plan(cfg, .{}, &m, 0).n);
}

test "exl3: elastic policy: the hold keeps K/V's pages and lets lines grow only into free bytes, then K/V gives back" {
    var m = memory(4, 100 * mib + 40 * mib); // 40 MiB spare
    m.kv.demand = 10;
    const c: policy.Config = .{ .step_bytes = 64 * mib, .idle_ticks = 1, .dwell_ticks = 0, .hold_ticks = 5 };
    var st: policy.State = .{};
    m.load = 3;
    st.advance(c, &m);
    try equal(@as(u32, 5), st.hold);
    const held = policy.plan(c, st, &m, 0);
    try equal(policy.Reason.idle, held.reason);
    for (held.slice()) |s| try expect(s == .lines); // K/V gives nothing back
    const after = try walk(c, m, held);
    try expect(after.kv.mapped == m.kv.mapped and after.total() > m.total());
    m.load = 1; // the burst is over: the hold runs down a tick at a time
    for (0..4) |_| st.advance(c, &m);
    try equal(@as(u32, 1), st.hold);
    for (policy.plan(c, st, &m, 0).slice()) |s| try expect(s == .lines);
    st.advance(c, &m);
    try equal(@as(u32, 0), st.hold);
    const free = policy.plan(c, st, &m, 0);
    var gave = false;
    for (free.slice()) |s| gave = gave or (s == .kv and s.kv < m.kv.mapped);
    try expect(gave);
    m.load = 2; // a second stream renews it
    st.advance(c, &m);
    try equal(@as(u32, 5), st.hold);
    var off = c;
    off.hold_ticks = 0; // and a hold of zero ticks is no hold
    st.advance(off, &m);
    try equal(@as(u32, 0), st.hold);
}

test "exl3: elastic policy: lines never pass their cap, K/V never its floor, and a full pool plans nothing" {
    var m = memory(2, 1 << 40); // endless free memory
    m.kv.demand = 0;
    m.kv.mapped = 400;
    var st: policy.State = .{ .calm = 1000, .dwell = 1000 };
    var rounds: usize = 0;
    while (rounds < 20) : (rounds += 1) {
        const p = policy.plan(cfg, st, &m, 0);
        if (p.n == 0) break;
        m = try walk(cfg, m, p);
    }
    for (m.lines()) |l| try equal(l.cap, l.active);
    try equal(@as(usize, 0), policy.plan(cfg, st, &m, 0).n);
    st = .{ .calm = 1000, .dwell = 1000 };
    var small = memory(2, 100 * mib);
    small.kv.demand = 0;
    small.kv.floor = 300;
    const p = policy.plan(cfg, st, &small, 0);
    for (p.slice()) |s| if (s == .kv) try expect(s.kv >= 300);
}

test "exl3: elastic policy: free bytes alone fund lines, K/V is left alone when it need not give" {
    var m = memory(3, 100 * mib + 400 * mib);
    m.kv.demand = 0;
    const st: policy.State = .{ .calm = 100, .dwell = 100 };
    const p = policy.plan(cfg, st, &m, 0);
    try expect(p.n > 0);
    for (p.slice()) |s| try expect(s == .lines and s.lines > 0);
    const after = try walk(cfg, m, p);
    try equal(m.kv.mapped, after.kv.mapped);
    try expect(after.free >= after.reserve);
}

test "exl3: elastic policy: a backed-off tick plans nothing, urgent or not, until the ticks have passed" {
    var m = memory(2, 100 * mib + 900 * mib);
    m.kv.mapped = 100;
    m.kv.demand = 100;
    var st: policy.State = .{ .calm = 9, .dwell = 99 };
    st.failed(cfg);
    try equal(cfg.backoff_ticks, st.blocked);
    try equal(@as(u32, 0), st.calm);
    try equal(@as(usize, 0), policy.plan(cfg, st, &m, 10).n);
    for (0..cfg.backoff_ticks) |_| st.advance(cfg, &m);
    try expect(policy.plan(cfg, st, &m, 10).n > 0);
}

test "exl3: elastic policy: every step is within a step's bytes, or one chunk or line step when that is larger" {
    var m = memory(4, 100 * mib);
    m.kv.mapped = 100;
    m.kv.demand = 100;
    const small: policy.Config = .{ .step_bytes = 16 * mib };
    const p = policy.plan(small, .{}, &m, 20);
    try expect(p.n > 0);
    _ = try walk(small, m, p);
    var sim = m;
    for (p.slice()) |s| {
        const moved = policy.apply(&sim, s);
        try expect(moved <= 16 * mib or (s == .lines and @abs(s.lines) == 1));
    }
    try expect(sim.kv.mapped >= 120);
}

test "exl3: elastic policy: plans are deterministic, and a uniform cut then the same grow gives the lines back" {
    var m = memory(5, 100 * mib);
    m.kv.mapped = 150;
    m.kv.demand = 150;
    const a = policy.plan(cfg, .{}, &m, 25);
    try expectSame(a, policy.plan(cfg, .{}, &m, 25));
    var back = m;
    _ = policy.apply(&back, .{ .lines = -7 });
    _ = policy.apply(&back, .{ .lines = 7 });
    for (back.lines(), m.lines()) |x, y| try equal(y.active, x.active);
    try equal(m.free, back.free);
    // a cut clamps at the floors, so the pools at them stay put
    _ = policy.apply(&back, .{ .lines = -1000 });
    for (back.lines()) |l| try equal(l.floor, l.active);
}

test "exl3: elastic policy: the watermarks and tick counts are valid only in order" {
    try expect((policy.Config{}).valid());
    try expect(!(policy.Config{ .low_ppm = 600_000 }).valid());
    try expect(!(policy.Config{ .high_ppm = 500_000 }).valid());
    try expect(!(policy.Config{ .high_ppm = 1_000_001 }).valid());
    try expect(!(policy.Config{ .idle_ticks = 0 }).valid());
    try expect(!(policy.Config{ .step_bytes = 0 }).valid());
    try expect(!(policy.Config{ .low_ppm = 0 }).valid());
}

test "exl3: elastic policy: pools of different line sizes, floors and caps keep every bound" {
    var m = memory(3, 100 * mib);
    m.pools[0] = .{ .active = 30, .floor = 12, .cap = 40, .bytes = 1_000_000, .chunk = 2 * mib };
    m.pools[1] = .{ .active = 10, .floor = 10, .cap = 90, .bytes = 2_500_000, .chunk = 2 * mib };
    m.pools[2] = .{ .active = 80, .floor = 20, .cap = 80, .bytes = 1_843_200, .chunk = 2 * mib };
    m.kv.mapped = 200;
    m.kv.demand = 200;
    const p = policy.plan(cfg, .{}, &m, 30);
    try expect(p.n > 0);
    const after = try walk(cfg, m, p);
    try expect(after.kv.mapped >= 230);
    try equal(@as(u32, 10), after.pools[1].active); // already at its floor: it never moves
    try expect(after.pools[0].active < 30 and after.pools[2].active < 80);
}

/// A random rank: sizes, floors and caps vary; every walk starts from a legal state.
fn randomMemory(rnd: std.Random) policy.Memory {
    const pools = 1 + rnd.uintLessThan(usize, 6);
    var m: policy.Memory = undefined;
    m.n_pools = pools;
    const chunk: u64 = @as(u64, 1) << @intCast(18 + rnd.uintLessThan(u6, 4)); // 256 KiB .. 2 MiB
    const count = 30 + rnd.uintLessThan(usize, 300);
    const stride = 200_000 + rnd.uintLessThan(u64, 4_000_000);
    const min = 1 + rnd.uintLessThan(usize, 20);
    const mapped = min + rnd.uintAtMost(usize, count - min);
    m.kv = .{ .mapped = mapped, .count = count, .min = min, .floor = 0, .demand = 0, .stride = stride, .chunk = chunk };
    m.phase = .idle;
    m.kv.floor = min + rnd.uintAtMost(usize, mapped - min);
    m.kv.demand = rnd.uintAtMost(usize, mapped);
    for (m.pools[0..pools]) |*l| {
        const floor: u32 = 10 + rnd.uintLessThan(u32, 4);
        const cap: u32 = floor + rnd.uintLessThan(u32, 90);
        l.* = .{ .active = floor + rnd.uintAtMost(u32, cap - floor), .floor = floor, .cap = cap, .bytes = 400_000 + rnd.uintLessThan(u64, 3_000_000), .chunk = chunk };
    }
    m.reserve = rnd.uintLessThan(u64, 300 * mib);
    m.free = m.reserve + rnd.uintLessThan(u64, 400 * mib);
    return m;
}

test "exl3: elastic policy: walks over random demand traces keep every invariant and never thrash inside the dwell" {
    var grew_kv: usize = 0;
    var grew_lines: usize = 0;
    var requests: usize = 0;
    var request_plans: usize = 0;
    var prefill_plans: usize = 0;
    var queued_plans: usize = 0;
    for (0..200) |seed| {
        var prng = std.Random.DefaultPrng.init(0xe1a5_0000 + seed);
        const rnd = prng.random();
        var m = randomMemory(rnd);
        const c: policy.Config = .{ .step_bytes = (8 + rnd.uintLessThan(u64, 120)) * mib, .idle_ticks = 1 + rnd.uintLessThan(u32, 6), .dwell_ticks = rnd.uintLessThan(u32, 20) };
        var st: policy.State = .{};
        var last_idle: ?usize = null;
        for (0..400) |tick| {
            // demand drifts, spikes and drains; the pool's tail floor follows it (a shrink may not pass a held page)
            switch (rnd.uintLessThan(u8, 10)) {
                0 => m.kv.demand = rnd.uintAtMost(usize, m.kv.mapped),
                1 => m.kv.demand = m.kv.demand * 3 / 2 + 1,
                2 => m.kv.demand /= 3,
                else => m.kv.demand = m.kv.demand + rnd.uintAtMost(usize, 3) -| rnd.uintAtMost(usize, 3),
            }
            m.kv.demand = @min(m.kv.demand, m.kv.mapped);
            m.kv.floor = @min(m.kv.mapped, @max(m.kv.min, m.kv.demand + rnd.uintAtMost(usize, 4)));
            m.phase = switch (rnd.uintLessThan(u8, 4)) {
                0 => .prefill,
                1 => .decode,
                else => .idle,
            };
            m.load = rnd.uintLessThan(u32, 5);
            m.ahead = if (rnd.uintLessThan(u8, 10) == 0) 1 + rnd.uintLessThan(usize, 40) else 0;
            const before = m;
            st.advance(c, &m);
            const request: usize = if (rnd.uintLessThan(u8, 12) == 0) 1 + rnd.uintLessThan(usize, 40) else 0;
            requests += @intFromBool(request > 0);
            const p = policy.plan(c, st, &m, request);
            try expectSame(p, policy.plan(c, st, &m, request));
            if (p.n == 0) {
                continue;
            }
            try expect(p.reason != .none);
            if (request == 0 and before.ahead == 0 and before.phase == .prefill) { // free bytes only, no line step, no lazy plan
                try expect(p.reason == .pressure);
                for (p.slice()) |s| try expect(s == .kv and s.kv > before.kv.mapped);
                prefill_plans += 1;
            }
            try expect((p.reason == .request) == (request > 0));
            if (p.reason == .queued) {
                try expect(request == 0 and before.ahead > 0);
                queued_plans += 1;
                try expect(p.n > 0);
            }
            m = try walk(c, m, p);
            try equal(before.kv.count, m.kv.count);
            if (p.reason == .queued and p.n < policy.max_steps) try expect(m.kv.mapped >= before.kv.mapped + before.ahead);
            if (p.reason == .request) {
                request_plans += 1;
                if (p.n < policy.max_steps) try expect(m.kv.mapped >= before.kv.mapped + request);
            }
            if (p.reason == .idle) {
                if (st.hold > 0) for (p.slice()) |s| if (s == .kv) try expect(s.kv >= before.kv.mapped); // the hold: no give back
                // a lazy plan never follows another inside the dwell, and never comes while use is above the low mark
                if (last_idle) |at| try expect(tick - at >= c.dwell_ticks);
                last_idle = tick;
                try expect(before.kv.utilization() <= c.low_ppm);
            }
            for (p.slice()) |s| switch (s) {
                .kv => |pages| grew_kv += @intFromBool(pages > before.kv.mapped),
                .lines => |k| grew_lines += @intFromBool(k > 0),
            };
            st.applied();
        }
        for (m.lines()) |l| try expect(l.active >= l.floor and l.active <= l.cap);
        try expect(m.kv.mapped >= m.kv.min and m.kv.mapped <= m.kv.count);
    }
    try expect(grew_kv > 0 and grew_lines > 0 and requests > 0 and request_plans > 0 and prefill_plans > 0 and queued_plans > 0);
}
