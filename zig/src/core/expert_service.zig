//! The host worker that answers an offloaded forward's layer stages: the device requests, the worker acks.
const std = @import("std");
const platform = @import("sys");
const tier_mod = @import("expert_tier.zig");
const stats_mod = @import("expert_stats.zig");

pub const Job = tier_mod.Job;
pub const Ports = tier_mod.Ports;
pub const Tier = tier_mod.Tier;
pub const Shape = tier_mod.Shape;
pub const Window = tier_mod.Window;
pub const Error = tier_mod.Error;

/// Layer stages that may be posted and not yet served (a forward posts one a layer; a prompt queues several forwards).
pub const ring: u32 = 4096;

pub const Service = struct {
    tier: Tier,
    ports: Ports = undefined,
    since: stats_mod.Stats = .{},
    span: stats_mod.Span = .{},
    jobs: []Job,
    /// Jobs posted (the posting thread only), published to the worker, and served.
    posted: u32 = 0,
    published: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,

    pub fn init(tier: Tier) Error!Service {
        const jobs = try tier.gpa.alloc(Job, ring);
        return .{ .tier = tier, .jobs = jobs };
    }

    pub fn deinit(s: *Service) void {
        s.report("total");
        s.stopWorker();
        s.tier.gpa.free(s.jobs);
        s.tier.deinit();
        s.* = undefined;
    }

    /// Starts the worker; `ports` is the device's side of every stage, and the service must not move afterwards.
    pub fn start(s: *Service, ports: Ports) Error!void {
        if (s.worker != null) return;
        s.ports = ports;
        s.worker = std.Thread.spawn(.{}, workerMain, .{s}) catch return error.OutOfMemory;
    }

    fn stopWorker(s: *Service) void {
        const w = s.worker orelse return;
        s.stop.store(true, .release);
        w.join();
        s.worker = null;
    }

    /// Waits until the worker has served every posted stage: no policy or window state moves under the caller after.
    fn quiesce(s: *Service) void {
        while (s.done.load(.acquire) != s.posted) std.atomic.spinLoopHint();
    }

    /// Posts a layer's stage, which the worker serves once the device has traced the picks and written the serial.
    pub fn post(s: *Service, job: Job) error{BadShape}!u32 {
        if (job.rows == 0 or job.rows > s.tier.shape.trace_rows or s.worker == null) return error.BadShape;
        while (s.posted -% s.done.load(.acquire) >= ring - 1) std.atomic.spinLoopHint();
        s.posted +%= 1;
        s.jobs[s.posted % ring] = job;
        s.published.store(s.posted, .release);
        return s.posted;
    }

    /// The word the device writes a stage's serial to, and the one it waits on for the answer.
    pub fn reqWord(s: *const Service) *u32 {
        return s.tier.win.word(Window.req_at);
    }

    pub fn ackWord(s: *const Service) *u32 {
        return s.tier.win.word(Window.ack_at);
    }

    fn workerMain(s: *Service) void {
        const req: *volatile u32 = s.reqWord();
        var waited: u64 = nowNs();
        while (!s.stop.load(.acquire)) {
            const n = s.done.load(.monotonic) +% 1;
            if (s.published.load(.acquire) -% n >= 0x8000_0000) {
                napNs(50_000);
                waited = nowNs();
                continue;
            }
            if (@atomicLoad(u32, @as(*const u32, @volatileCast(req)), .acquire) -% n >= 0x8000_0000) {
                std.atomic.spinLoopHint();
                continue;
            }
            const began = nowNs();
            s.tier.stats.idle_ns += began - waited;
            const job = s.jobs[n % ring];
            if (job.prefill and job.layer == 0) s.span = .{ .began = began, .bytes = s.tier.stats.scratch_bytes };
            s.serve(job);
            const ended = nowNs();
            s.tier.stats.stage_ns += ended - began;
            s.tier.stats.stages += 1;
            if (job.prefill) {
                s.span.stage_ns += ended - began;
                if (job.layer + 1 == s.tier.shape.layers) stats_mod.reportSpan(s.span, job.rows, ended, s.tier.stats.scratch_bytes);
            }
            s.done.store(n, .release);
            waited = nowNs();
        }
    }

    /// Stages `job` and answers: a failed stage is answered too, so the device never waits on a dead worker.
    fn serve(s: *Service, job: Job) void {
        s.tier.stage(job, s.ports) catch s.failed.store(true, .release);
        @atomicStore(u32, s.ackWord(), s.done.load(.monotonic) +% 1, .release);
    }

    /// A round's first stage may be served: the worker has answered every stage posted so far.
    pub fn beginRound(s: *Service) void {
        s.quiesce();
    }

    /// The round over its first `rows` rows is finished: its time joins the counters and the admission budget renews.
    pub fn endRound(s: *Service, rows: usize, ns: u64) error{StageFailed}!void {
        s.quiesce();
        if (s.failed.load(.acquire)) return error.StageFailed;
        s.tier.endRound(rows, ns);
        if (s.tier.stats.rounds % stats_mod.report_every == 0) s.report("interval");
    }

    /// One log line for the rounds since the last (or since the start).
    pub fn report(s: *Service, label: []const u8) void {
        const d = s.tier.stats.since(s.since);
        s.since = s.tier.stats;
        stats_mod.report(label, d, s.tier.shape.blobBytes(), s.tier.lineCount());
    }
};

fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn napNs(ns: u64) void {
    var ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(ns) };
    _ = std.c.nanosleep(&ts, null);
}
