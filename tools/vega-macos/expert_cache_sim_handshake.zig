//! The hit/miss split of the expert offload's layer handshake (docs/LATENCY.md M1), replayed on real routing traces with production's own pool,
//! `core.expert_offload.Cache` (the one hip/expert_store.zig's worker drives): a layer whose eight picks are all resident is passed by the card itself
//! and replayed by the host later (touches only); any other layer asks and is served as before, after the hits that came before it.
//!
//! Two pools run side by side over every decode token: the one that serves every layer (the handshake as it was) and the one that splits them. The
//! pools must be identical after every token, in lines, stamps, residency and counters: that is what the worker's replay has to guarantee. The tally
//! says how many layers the card passed (the layers whose handshake the split removes) and how they spread over layers and tokens.
const std = @import("std");
const offload = @import("expert_offload");
const l = @import("expert_cache_sim_list.zig");
const data = @import("expert_cache_sim_data.zig");
const fmt = @import("expert_cache_sim_fmt.zig");

pub const Config = struct {
    /// Cache lines (production: what the device has left after the streams, 2876 on the 8 GiB card).
    cap: usize,
    /// The SLRU's protected share of the lines (production default 75; 0 is plain LRU).
    protect_pct: u8 = 75,
    /// Misses a round may admit (production default: every one).
    copies: usize = 4096,
    /// Tokens at the start of each prompt that the steady-state figures leave out (the cache fills).
    warm_tokens: usize = 64,
};

pub const Tally = struct {
    prompts: u64 = 0,
    tokens: u64 = 0,
    /// Layer stages, and those the card passed itself; the same two after the first `warm_tokens` of a prompt.
    stages: u64 = 0,
    passed: u64 = 0,
    steady_tokens: u64 = 0,
    steady_stages: u64 = 0,
    steady_passed: u64 = 0,
    /// Per layer: how often the card passed it (steady state), and over how many tokens.
    per_layer: [l.NL]u64 = @splat(0),
    /// Tokens (steady state) by the number of layers the card passed.
    by_passed: [l.NL + 1]u64 = @splat(0),
    /// Router picks, and those that were not resident, over the whole replay.
    accesses: u64 = 0,
    misses: u64 = 0,
    /// Tokens after which the split pool differed from the always-asking one (must be 0).
    mismatches: u64 = 0,
    /// Picks, over all stages, whose residency word the card reads (the split pool's) differed from the line the always-asking host would put in the table row (must be 0):
    /// a stale entry after an eviction, or one a late replay has not reached.
    row_mismatches: u64 = 0,
    /// Picks of stages the card passed, and those whose VRAM table entry would not be (line, 0) if only the miss stages' row copies ever wrote the table (what the card finds
    /// when the check kernel's own row writes are lost: an expert admitted by a miss keeps the entry (window slot, line) until a miss stage names it again).
    hit_picks: u64 = 0,
    stale_if_check_lost: u64 = 0,
};

const Pool = struct {
    cache: offload.Cache,
    resident: []u64,
    line_addr: []u64,

    fn init(a: std.mem.Allocator, cfg: Config) !Pool {
        const resident = try a.alloc(u64, l.NK);
        const line_addr = try a.alloc(u64, cfg.cap);
        for (line_addr, 0..) |*x, i| x.* = 0x2000_0000 + i * 4096;
        const cache = try offload.Cache.init(a, .{ .keys = l.NK, .lines = cfg.cap, .picks_max = l.TOPK, .copy_experts = cfg.copies, .expert_bytes = 1769472, .protect_pct = cfg.protect_pct }, resident, line_addr);
        return .{ .cache = cache, .resident = resident, .line_addr = line_addr };
    }

    fn deinit(p: *Pool, a: std.mem.Allocator) void {
        p.cache.deinit();
        a.free(p.line_addr);
        a.free(p.resident);
    }
};

fn same(a: *const offload.Cache, b: *const offload.Cache) bool {
    return a.now == b.now and a.budget == b.budget and std.mem.eql(offload.Id, a.slots, b.slots) and std.mem.eql(offload.Line, a.line, b.line) and
        std.mem.eql(u32, a.stamp, b.stamp) and std.mem.eql(i32, a.where, b.where) and std.mem.eql(bool, a.flags, b.flags) and std.mem.eql(u64, a.resident, b.resident);
}

fn ids(row: []const u16, out: *[l.TOPK]offload.Id) []const offload.Id {
    for (row, out) |key, *id| id.* = key;
    return out[0..row.len];
}

/// One prompt's decode tokens from a cold cache, `tally` updated.
fn replayTrace(a: std.mem.Allocator, cfg: Config, t: data.Trace, tally: *Tally) !void {
    var base = try Pool.init(a, cfg);
    defer base.deinit(a);
    var split = try Pool.init(a, cfg);
    defer split.deinit(a);
    var late: [l.NL]usize = undefined;
    const ent_line = try a.alloc(u64, l.NK);
    defer a.free(ent_line);
    const ent_adm = try a.alloc(bool, l.NK);
    defer a.free(ent_adm);
    @memset(ent_line, 0);
    @memset(ent_adm, true);
    var buf: [l.TOPK]offload.Id = undefined;
    tally.prompts += 1;
    for (0..t.decodeTokens()) |tok| {
        base.cache.beginRound();
        split.cache.beginRound();
        var nlate: usize = 0;
        var passed: u64 = 0;
        const steady = tok >= cfg.warm_tokens;
        for (0..l.NL) |layer| {
            const keys = ids(data.Trace.picks(t.dec, tok, layer), &buf);
            for (keys) |key| {
                const host_line: u64 = if (base.cache.address(key)) |line| line else 0;
                if (split.resident[key] != host_line) tally.row_mismatches += 1;
            }
            @memcpy(base.cache.ids[0..keys.len], keys);
            const x = try base.cache.decide(keys.len);
            try base.cache.commit(x.started);
            tally.accesses += keys.len;
            tally.misses += keys.len - x.hits;
            if (split.cache.allResident(keys)) {
                for (keys) |key| {
                    tally.hit_picks += 1;
                    if (ent_adm[key] or ent_line[key] != split.resident[key]) tally.stale_if_check_lost += 1;
                }
                late[nlate] = layer;
                nlate += 1;
                passed += 1;
                if (steady) tally.per_layer[layer] += 1;
                continue;
            }
            for (late[0..nlate]) |ll| try split.cache.replay(ids(data.Trace.picks(t.dec, tok, ll), &buf));
            nlate = 0;
            const keys2 = ids(data.Trace.picks(t.dec, tok, layer), &buf);
            @memcpy(split.cache.ids[0..keys2.len], keys2);
            const y = try split.cache.decide(keys2.len);
            try split.cache.commit(y.started);
            for (keys2) |key| {
                const cur = split.resident[key];
                var admitted = false;
                for (split.cache.copies[0..y.started]) |cp| admitted = admitted or cp.id == key;
                ent_line[key] = cur;
                ent_adm[key] = cur == 0 or admitted;
            }
        }
        for (late[0..nlate]) |ll| try split.cache.replay(ids(data.Trace.picks(t.dec, tok, ll), &buf));
        tally.tokens += 1;
        tally.stages += l.NL;
        tally.passed += passed;
        if (steady) {
            tally.steady_tokens += 1;
            tally.steady_stages += l.NL;
            tally.steady_passed += passed;
            tally.by_passed[passed] += 1;
        }
        if (!same(&base.cache, &split.cache)) tally.mismatches += 1;
    }
}

/// Every prompt of `set` (cold start each), in name order.
pub fn replay(a: std.mem.Allocator, cfg: Config, set: data.Set) !Tally {
    var tally: Tally = .{};
    for (set.by_name) |ti| {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        try replayTrace(arena.allocator(), cfg, set.traces[ti], &tally);
    }
    return tally;
}

fn pct(n: u64, of: u64) f64 {
    return if (of == 0) 0 else 100.0 * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(of));
}

/// `x` with `prec` decimals, right-aligned in `width`.
fn pad(out: *std.Io.Writer, x: f64, width: usize, prec: u8) !void {
    var buf: [32]u8 = undefined;
    const text = fmt.fixedBuf(&buf, x, prec, false);
    for (text.len..width) |_| try out.writeAll(" ");
    try out.writeAll(text);
}

/// The report: how much of the handshake the split removes, where, and the check that the worker's replay leaves the pool as the handshake did.
pub fn write(out: *std.Io.Writer, cfg: Config, set: data.Set) !void {
    const t = try replay(std.heap.page_allocator, cfg, set);
    try out.print("# hit/miss split of the layer handshake (docs/LATENCY.md M1): {d} lines, SLRU {d}% protected, up to {d} admissions a round, cold start per prompt\n", .{ cfg.cap, cfg.protect_pct, cfg.copies });
    try out.print("{d} prompts, {d} decode tokens, {d} layer stages; first {d} tokens of a prompt are left out of the steady-state figures ({d} tokens)\n\n", .{ t.prompts, t.tokens, t.stages, cfg.warm_tokens, t.steady_tokens });
    try out.writeAll("layers the card passes itself (all 8 picks resident, no request, no ack from the host):\n");
    try out.writeAll("  all tokens    ");
    try fmt.fixed(out, pct(t.passed, t.stages), 1, false);
    try out.print("% of {d} stages\n  steady state  ", .{t.stages});
    try fmt.fixed(out, pct(t.steady_passed, t.steady_stages), 1, false);
    try out.print("% of {d} stages: ", .{t.steady_stages});
    try fmt.fixed(out, @as(f64, @floatFromInt(t.steady_passed)) / @as(f64, @floatFromInt(@max(t.steady_tokens, 1))), 1, false);
    try out.writeAll(" of 40 layers a token\n");
    try out.writeAll("  picks not resident (miss rate of accesses) ");
    try fmt.fixed(out, pct(t.misses, t.accesses), 1, false);
    try out.writeAll("%\n\nsteady state, share of tokens that the card passes this layer, by layer:\n");
    for (0..l.NL) |layer| {
        if (layer % 10 == 0) try out.print("  L{d:0>2}-{d:0>2} ", .{ layer, layer + 9 });
        try out.writeAll(" ");
        try pad(out, pct(t.per_layer[layer], t.steady_tokens), 5, 1);
        if (layer % 10 == 9) try out.writeAll("\n");
    }
    try out.writeAll("\nsteady state, tokens by the number of layers the card passes:\n");
    const bands = [_][2]usize{ .{ 0, 4 }, .{ 5, 9 }, .{ 10, 14 }, .{ 15, 19 }, .{ 20, 24 }, .{ 25, 29 }, .{ 30, 34 }, .{ 35, 40 } };
    for (bands) |band| {
        var n: u64 = 0;
        for (band[0]..band[1] + 1) |k| n += t.by_passed[k];
        try out.print("  {d:>2}-{d:>2} layers  ", .{ band[0], band[1] });
        try pad(out, pct(n, t.steady_tokens), 5, 1);
        try out.writeAll("% of tokens\n");
    }
    try out.print("\npool after every token, split vs. every layer asking: {d} of {d} tokens differ (must be 0)\n", .{ t.mismatches, t.tokens });
    try out.print("residency word the card reads vs. the line the host's table row would name, per pick: {d} of {d} differ (must be 0)\n", .{ t.row_mismatches, t.accesses });
    try out.print("picks of passed stages whose table entry is not (line, 0) if only miss stages write the table: {d} of {d} ({d:.1}%)\n", .{ t.stale_if_check_lost, t.hit_picks, pct(t.stale_if_check_lost, t.hit_picks) });
}
