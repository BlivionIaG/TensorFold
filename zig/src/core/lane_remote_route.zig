//! Remote prefill's gate and decider: one pull at a time reaches the pair, and a prompt goes there only when it pays.
const std = @import("std");
const breaker = @import("lane_remote_breaker.zig");
const curve = @import("lane_remote_curve.zig");
const queue = @import("lane_remote_queue.zig");

/// TF_PREFILL_ROUTE: `auto` weighs each prompt (see `Route.weigh`); `always` sends every long prompt, behind any pull.
pub const Mode = enum { auto, always };

pub const Outcome = breaker.Outcome;
pub const Door = breaker.Door;
pub const Change = breaker.Change;
pub const pairDown = breaker.pairDown;
pub const Curve = curve.Curve;
pub const Rate = curve.Rate;
pub const Estimate = queue.Estimate;
pub const Gate = queue.Gate;
pub const queue_capacity = queue.capacity;

/// Where a request wants its prompt prefilled: `force` skips the weighing, `never` keeps it off the pair.
pub const Directive = enum {
    auto,
    force,
    never,

    pub fn parse(word: []const u8) ?Directive {
        return std.meta.stringToEnum(Directive, word);
    }
};

/// A queued prompt as the source sees it: its length, the engine's cached tokens, the streams decoding, its directive.
pub const Ask = struct { len: u64, cached: u64, decoding: u32, directive: Directive = .auto };

/// A source's answer for a queued prompt: the pair takes it, it never qualifies (`small`, `off`), or not now (`later`).
pub const Verdict = enum {
    remote,
    small,
    off,
    busy,
    slower,
    down,

    /// A reason that holds for now only: the prompt is weighed again the next time the queue is looked at.
    pub fn later(v: Verdict) bool {
        return switch (v) {
            .busy, .slower, .down => true,
            .remote, .small, .off => false,
        };
    }

    pub fn why(v: Verdict) []const u8 {
        return switch (v) {
            .remote => "the pair takes it",
            .small => "it is too short to pull",
            .off => "remote prefill is off",
            .busy => "prefilling here ends before the pair's queue does",
            .slower => "this machine alone is predicted faster",
            .down => "the pair is not answering",
        };
    }
};

/// A pass the lane host or a pull timed: this machine's own prompt pass, a pull end to end, or the last-token pass.
pub const Pass = enum { local, remote, tail };
/// `end` is the prompt position the pass reached (0: it began at 0, so `rows`).
pub const Measured = struct { pass: Pass, rows: u64, ns: u64, end: u64 = 0 };

/// Prompt rows still to prefill here, from position `from` to `to`.
pub const Span = struct { from: u64, to: u64 };

/// This machine's own prefill work and decode load: `share` is the wall-time part a prompt pass gets beside decoders.
pub const Busy = struct { decoding: u32 = 0, share: f64 = 1, spans: []const Span = &.{} };

/// How a prompt's wait at the gate ended.
pub const Admission = union(enum) {
    /// the pair is its own until `leave(id)`; `waited_ns` is the wait
    entered: struct { id: u64, waited_ns: u64 },
    /// prefilling here finishes sooner than waiting for the pair
    local,
    /// the server is closing or the request is gone
    stop,
};

/// How long a pull waiting for the pair sleeps between looks (ms).
const poll_ms = 5;
/// A pass of fewer rows says little about a rate: a fixed cost of 0.5 s or more a forward dominates it.
const min_rows = 1024;
/// Spans of the host's own work that count (the rest are ignored): filling prompts and those kept off the pair.
pub const max_spans = 32;

/// One pair, one pull at a time. Several threads may ask; the lock only guards the bookkeeping, never a pull.
pub const Route = struct {
    mode: Mode = .auto,
    lock: std.atomic.Mutex = .unlocked,
    /// the pulls at the gate and the prompts sent here, in arrival order
    queue: queue.Queue = .{},
    /// this machine's prompt prefill by length (seed 2050 a second to 32k rows; 1860 at 131k, 1695 at 491k: decay 0.07)
    local: Curve = .{ .base = .{ .v = 2050 }, .decay = 0.07 },
    /// a pull's prefix tokens a second, end to end (seed 2400 to 32k rows; 1430 over 968k: `decay` 0.15)
    remote: Curve = .{ .base = .{ .v = 2400 }, .decay = 0.15 },
    /// the Mac's pass after the pull, in seconds: restore, the last token, its first reply (seed: 0.6)
    tail_s: Rate = .{ .v = 0.6 },
    /// closed, open or half-open: failures of the pair or its link keep prompts here for a cool-down
    breaker: breaker.Breaker = .{},
    /// TF_PREFILL_TIMEOUT_S (0: `deadlineS` derives one from the predicted pull)
    timeout_s: u64 = 0,

    /// The monotonic clock every `weigh`, `join` and `poll` caller must use.
    pub fn nowNs(io: std.Io) u64 {
        return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
    }

    /// The knobs, as their values: `mode` is the word's index in TF_PREFILL_ROUTE's list.
    pub const Knobs = struct {
        mode: u64 = 0,
        local_tps: f64 = 2050,
        remote_tps: f64 = 2400,
        local_decay: f64 = 0.07,
        remote_decay: f64 = 0.15,
        stall: f64 = 1,
        trip: u64 = 3,
        cooldown_s: u64 = 30,
        timeout_s: u64 = 0,
        /// curves fitted to measured points (TF_PREFILL_LOCAL_CURVE, _REMOTE_CURVE) replace the rate and decay
        local_fit: ?Curve = null,
        remote_fit: ?Curve = null,
    };

    /// Knobs from TF_PREFILL_ROUTE, _LOCAL_TPS, _REMOTE_TPS, _LOCAL_DECAY, _REMOTE_DECAY, _STALL and the breaker ones.
    pub fn fromEnv() !Route {
        const sys = @import("sys");
        const env = sys.environment;
        const d: Knobs = .{};
        const local_tps = try env.read(f64, "TF_PREFILL_LOCAL_TPS", d.local_tps);
        const remote_tps = try env.read(f64, "TF_PREFILL_REMOTE_TPS", d.remote_tps);
        const local_decay = try env.read(f64, "TF_PREFILL_LOCAL_DECAY", d.local_decay);
        const remote_decay = try env.read(f64, "TF_PREFILL_REMOTE_DECAY", d.remote_decay);
        const local_fit = try calibrate("TF_PREFILL_LOCAL_CURVE", env.setting("TF_PREFILL_LOCAL_CURVE"), .{ .base = .{ .v = local_tps }, .decay = local_decay });
        const remote_fit = try calibrate("TF_PREFILL_REMOTE_CURVE", env.setting("TF_PREFILL_REMOTE_CURVE"), .{ .base = .{ .v = remote_tps }, .decay = remote_decay });
        if (local_fit) |c| std.log.info("remote prefill: this machine's curve {d:.0} tokens a second to 32768 rows, decay {d:.2}", .{ c.base.v, c.decay });
        if (remote_fit) |c| std.log.info("remote prefill: the pair's curve {d:.0} tokens a second to 32768 rows, decay {d:.2}", .{ c.base.v, c.decay });
        return build(.{
            .mode = try env.read(u64, "TF_PREFILL_ROUTE", d.mode),
            .local_tps = local_tps,
            .remote_tps = remote_tps,
            .local_decay = local_decay,
            .remote_decay = remote_decay,
            .stall = try env.read(f64, "TF_PREFILL_STALL", d.stall),
            .trip = try env.read(u64, "TF_PREFILL_BREAKER", d.trip),
            .cooldown_s = try env.read(u64, "TF_PREFILL_COOLDOWN_S", d.cooldown_s),
            .timeout_s = try env.read(u64, "TF_PREFILL_TIMEOUT_S", d.timeout_s),
            .local_fit = local_fit,
            .remote_fit = remote_fit,
        });
    }

    /// The curve fitted to a setting's `rows:seconds` points (seeded with `seed`'s decay); null when unset or blank.
    pub fn calibrate(name: [:0]const u8, raw: ?[]const u8, seed: Curve) !?Curve {
        const text = (try @import("sys").environment.path(name, raw)) orelse return null;
        var buf: [curve.max_points]curve.Point = undefined;
        const points = curve.parsePoints(text, &buf) orelse {
            if (!@import("builtin").is_test) std.log.err("{s}: expected 1 to {d} rows:seconds pairs, each over a positive row count", .{ name, curve.max_points });
            return error.InvalidEnvironment;
        };
        return curve.fit(points, seed);
    }

    pub fn build(k: Knobs) Route {
        const always = k.mode != 0;
        return .{
            .mode = if (always) .always else .auto,
            .queue = .{ .stall = k.stall },
            .local = k.local_fit orelse .{ .base = .{ .v = k.local_tps }, .decay = k.local_decay },
            .remote = k.remote_fit orelse .{ .base = .{ .v = k.remote_tps }, .decay = k.remote_decay },
            .breaker = .{ .trip = k.trip, .cooldown_ns = k.cooldown_s * std.time.ns_per_s },
            .timeout_s = k.timeout_s,
        };
    }

    fn acquire(r: *Route) void {
        while (!r.lock.tryLock()) std.atomic.spinLoopHint();
    }

    fn cost(r: *const Route, prefix: u64, cached: u64, len: u64) Estimate {
        return .{
            .remote_ns = r.remote.spanNs(0, prefix) +| @as(u64, @intFromFloat(r.tail_s.v * std.time.ns_per_s)),
            .local_ns = r.local.spanNs(cached, len),
        };
    }

    /// A prefix of `prefix` tokens pulled, against the prompt's rows from `cached` to `len` prefilled here.
    pub fn estimate(r: *Route, prefix: u64, cached: u64, len: u64) Estimate {
        r.acquire();
        defer r.lock.unlock();
        return r.cost(prefix, cached, len);
    }

    /// `estimate` for a request with `directive`: one that forces the pair waits for it.
    pub fn estimateFor(r: *Route, prefix: u64, cached: u64, len: u64, directive: Directive) Estimate {
        var e = r.estimate(prefix, cached, len);
        e.pinned = directive == .force;
        return e;
    }

    /// A pass this machine or a pull timed: measured rates replace the seeds, a third to three times at a time.
    pub fn observe(r: *Route, m: Measured) void {
        if (m.ns == 0) return;
        const to = @max(m.end, m.rows);
        const from = to - m.rows;
        r.acquire();
        defer r.lock.unlock();
        switch (m.pass) {
            .local => if (m.rows >= min_rows) r.local.observe(from, to, m.ns),
            .remote => if (m.rows >= min_rows) r.remote.observe(from, to, m.ns),
            .tail => r.tail_s.blend(@as(f64, @floatFromInt(m.ns)) / 1e9),
        }
    }

    /// A pull of `prefix` tokens took `ns`; `producer_cached` of them came from the producer's cache, not its compute.
    pub fn observePull(r: *Route, prefix: u64, producer_cached: u64, ns: u64) void {
        r.observe(.{ .pass = .remote, .rows = prefix -| producer_cached, .ns = ns, .end = prefix });
    }

    /// The host's own load at `now`: its decoding streams and the rows of prefill it has queued or under way.
    pub fn busy(r: *Route, now: u64, b: Busy) void {
        r.acquire();
        defer r.lock.unlock();
        var ns: u64 = 0;
        for (b.spans) |s| ns +|= r.local.spanNs(s.from, s.to);
        r.queue.publish(now, .{ .decoding = b.decoding, .share = b.share, .local_ns = ns });
    }

    /// Where a queued prompt goes: the pair unless prefilling here ends sooner, queue and decode load counted.
    pub fn weigh(r: *Route, now: u64, ask: Ask, prefix: u64) Verdict {
        r.acquire();
        defer r.lock.unlock();
        if (r.breaker.shut(now)) return .down;
        if (r.mode == .always or ask.directive == .force) return .remote;
        var e = r.cost(prefix, ask.cached, ask.len);
        e.pinned = false;
        if (!r.queue.local(now, e, ask.decoding)) return .remote;
        return if (r.queue.len > 0) .busy else .slower;
    }

    /// The prompt joins the pair's queue; null: it is full, and the prompt is prefilled here.
    pub fn join(r: *Route, now: u64, e: Estimate) ?u64 {
        r.acquire();
        defer r.lock.unlock();
        return r.queue.join(now, e);
    }

    /// The pair for the prompt (`entered`, until `leave`), a short wait for the pulls ahead, or the Mac's prefill.
    pub fn poll(r: *Route, now: u64, id: u64) Gate {
        r.acquire();
        defer r.lock.unlock();
        return r.queue.poll(now, id, r.mode == .auto);
    }

    /// A pull the gate let in asks the breaker: go, go after a liveness check (the trial), or leave it to the Mac.
    pub fn door(r: *Route, now: u64) Door {
        r.acquire();
        defer r.lock.unlock();
        return r.breaker.door(now);
    }

    /// The pull `door` let through ended as `outcome`; the answer says whether the breaker opened or closed.
    pub fn finish(r: *Route, now: u64, outcome: Outcome) Change {
        r.acquire();
        defer r.lock.unlock();
        return r.breaker.finish(now, outcome);
    }

    /// A pull's time limit: TF_PREFILL_TIMEOUT_S, else 15 s and three times its predicted time, 30 s to 30 min.
    pub fn deadlineS(r: *Route, tokens: u64) u64 {
        if (r.timeout_s != 0) return r.timeout_s;
        r.acquire();
        defer r.lock.unlock();
        const predicted = r.remote.span(0, tokens) + r.tail_s.v;
        return @intFromFloat(std.math.clamp(@ceil(15 + 3 * predicted), 30, 1800));
    }

    /// The pull that `poll` let in is over, or the prompt stopped waiting.
    pub fn leave(r: *Route, id: u64) void {
        r.acquire();
        defer r.lock.unlock();
        r.queue.leave(id);
    }

    /// Joins the queue and waits for the pair only while that pays; `local`: prefill here, `stop`: closing or `gone`.
    pub fn admit(r: *Route, io: std.Io, e: Estimate, closing: *const std.atomic.Value(bool), gone: *const std.atomic.Value(bool)) Admission {
        const arrived = Route.nowNs(io);
        const id = r.join(arrived, e) orelse return .local;
        while (!closing.load(.acquire) and !gone.load(.acquire)) {
            const now = Route.nowNs(io);
            switch (r.poll(now, id)) {
                .entered => return .{ .entered = .{ .id = id, .waited_ns = now - arrived } },
                .local => return .local,
                .wait => std.Io.sleep(io, .fromMilliseconds(poll_ms), .awake) catch break,
            }
        }
        r.leave(id);
        return .stop;
    }
};
