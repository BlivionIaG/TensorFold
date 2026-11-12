//! The replay jobs of expert_cache_sim.py (`run_single`, `run_lock`) and the worker threads that run them. A job's result is a handful of
//! integers, so the report does not depend on which worker ran what.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const data = @import("expert_cache_sim_data.zig");
const cfg = @import("expert_cache_sim_config.zig");

pub const buckets = [_][2]usize{ .{ 0, 8 }, .{ 8, 32 }, .{ 32, 128 }, .{ 128, 512 }, .{ 512, 1_000_000_000 } };

pub const Dataset = enum(u8) { short, long, lock4 };

pub const Job = struct {
    config: u32,
    dataset: Dataset,
    /// position in `Set.by_name` (short, long) or the lockstep group index
    item: u32,
    /// decode tokens, to start the heaviest jobs first
    weight: u32,
};

pub const Result = struct {
    /// cache accesses: tokens x 320 (single) or the summed sizes of the per-layer key unions (lock4)
    acc: u64 = 0,
    adm: u64 = 0,
    evi: u64 = 0,
    miss: u64 = 0,
    /// decode tokens (single) or sequence-tokens (lock4)
    tok: u64 = 0,
    /// misses and tokens per decode position bucket
    pos: [buckets.len][2]u64 = @splat(.{ 0, 0 }),
};

pub const Ctx = struct {
    cap: usize,
    configs: []const cfg.Config,
    short: data.Set,
    long: ?data.Set,
    seeds: data.Seeds,
    groups: []const []const usize,
    jobs: []const Job,
    results: []Result,
    next: std.atomic.Value(usize) = .init(0),
    /// set by the first job that cannot run: the message
    failure: std.atomic.Value(?*const []const u8) = .init(null),

    fn set(self: *const Ctx, ds: Dataset) data.Set {
        return if (ds == .long) self.long.? else self.short;
    }
};

const page = std.heap.page_allocator;

/// Worker loop: take jobs from the shared counter until none is left.
pub fn worker(ctx: *Ctx) void {
    while (true) {
        const i = ctx.next.fetchAdd(1, .monotonic);
        if (i >= ctx.jobs.len or ctx.failure.load(.monotonic) != null) return;
        var arena: std.heap.ArenaAllocator = .init(page);
        defer arena.deinit();
        ctx.results[i] = runJob(ctx, arena.allocator(), ctx.jobs[i]) catch |err| {
            const msg = page.create([]const u8) catch return;
            msg.* = @errorName(err);
            ctx.failure.store(msg, .monotonic);
            return;
        };
    }
}

fn runJob(ctx: *const Ctx, a: std.mem.Allocator, job: Job) !Result {
    const config = ctx.configs[job.config];
    var policy: cfg.Policy = try .create(a, config.kind, ctx.cap);
    return if (job.dataset == .lock4) runLock(ctx, &policy, job) else runSingle(ctx, a, &policy, config.init, job);
}

fn reversed(a: std.mem.Allocator, keys: []const u16) ![]u16 {
    const out = try a.dupe(u16, keys);
    std.mem.reverse(u16, out);
    return out;
}

/// The keys to preload (oldest first) for a seeded or prefill-warm start.
fn warmKeys(ctx: *const Ctx, a: std.mem.Allocator, init: cfg.Init, t: data.Trace) ![]u16 {
    const seed = ctx.seeds.get(t.cat) orelse return error.CategoryNotInShortSet;
    if (init == .seed) return reversed(a, seed[0..@min(seed.len, ctx.cap)]);
    const pk = try data.prefillKeys(a, t.pre, ctx.cap, t.name);
    if (init == .pf) return reversed(a, pk);
    const have = try a.alloc(bool, l.NK);
    @memset(have, false);
    for (pk) |k| have[k] = true;
    var fill: std.ArrayList(u16) = .empty;
    const room = ctx.cap -| pk.len;
    for (seed) |k| {
        if (fill.items.len >= room) break;
        if (!have[k]) try fill.append(a, k);
    }
    std.mem.reverse(u16, fill.items);
    const pk_rev = try reversed(a, pk);
    try fill.appendSlice(a, pk_rev);
    return fill.items;
}

fn runSingle(ctx: *const Ctx, a: std.mem.Allocator, policy: *cfg.Policy, init: cfg.Init, job: Job) !Result {
    const set = ctx.set(job.dataset);
    const t = set.traces[set.by_name[job.item]];
    switch (init) {
        .cold => {},
        .seed, .pf, .pf_seed => policy.preload(try warmKeys(ctx, a, init, t)),
        .pfreplay => {
            for (0..t.pre.len / data.row) |tok| for (0..l.NL) |layer| {
                _ = policy.step(layer, data.Trace.picks(t.pre, tok, layer));
            };
        },
    }
    policy.resetCounters();
    const n = t.decodeTokens();
    var r: Result = .{ .acc = n * data.row, .tok = n };
    for (0..n) |tok| {
        var m: u32 = 0;
        for (0..l.NL) |layer| m += policy.step(layer, data.Trace.picks(t.dec, tok, layer));
        r.miss += m;
        for (buckets, 0..) |b, bi| {
            if (tok >= b[0] and tok < b[1]) {
                r.pos[bi][0] += m;
                r.pos[bi][1] += 1;
            }
        }
    }
    r.adm = policy.adm();
    r.evi = policy.evi();
    return r;
}

/// A group of short prompts decoded in lockstep: each layer's call is the sorted union of the live sequences' picks
/// (a lone live sequence's picks as they are).
fn runLock(ctx: *const Ctx, policy: *cfg.Policy, job: Job) !Result {
    const group = ctx.groups[job.item];
    var longest: usize = 0;
    for (group) |g| longest = @max(longest, ctx.short.traces[g].decodeTokens());
    var r: Result = .{};
    for (0..longest) |tok| {
        var live: [l.group_size]*const data.Trace = undefined;
        var nlive: usize = 0;
        for (group) |g| {
            if (tok < ctx.short.traces[g].decodeTokens()) {
                live[nlive] = &ctx.short.traces[g];
                nlive += 1;
            }
        }
        r.tok += nlive;
        for (0..l.NL) |layer| {
            var buf: [l.max_keys]u16 = undefined;
            var keys: []const u16 = undefined;
            if (nlive > 1) {
                var n: usize = 0;
                for (live[0..nlive]) |t| for (data.Trace.picks(t.dec, tok, layer)) |k| {
                    buf[n] = k;
                    n += 1;
                };
                std.mem.sortUnstable(u16, buf[0..n], {}, std.sort.asc(u16));
                var u: usize = 0;
                for (buf[0..n]) |k| {
                    if (u == 0 or buf[u - 1] != k) {
                        buf[u] = k;
                        u += 1;
                    }
                }
                keys = buf[0..u];
            } else keys = data.Trace.picks(live[0].dec, tok, layer);
            r.acc += keys.len;
            r.miss += policy.step(layer, keys);
        }
    }
    r.adm = policy.adm();
    r.evi = policy.evi();
    return r;
}

/// Run every job on `workers` threads (the calling thread among them); returns the first failure's name, if any.
pub fn runAll(ctx: *Ctx, workers: usize) !void {
    const spawned = try page.alloc(std.Thread, workers - 1);
    defer page.free(spawned);
    var started: usize = 0;
    while (started < spawned.len) : (started += 1) {
        spawned[started] = std.Thread.spawn(.{}, worker, .{ctx}) catch break;
    }
    worker(ctx);
    for (spawned[0..started]) |th| th.join();
    if (ctx.failure.load(.monotonic)) |msg| {
        std.debug.print("expert_cache_sim: {s}\n", .{msg.*});
        return error.JobFailed;
    }
}
