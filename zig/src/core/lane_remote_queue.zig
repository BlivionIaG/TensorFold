//! The pair's queue as this machine sees it: each prompt goes to whichever side, pair or here, finishes it first.
const std = @import("std");

/// A prompt's cost on each side in ns (a pull end to end, its own prefill); `pinned` (forced) waits out any pull.
pub const Estimate = struct { remote_ns: u64, local_ns: u64, pinned: bool = false };

/// This machine's own load: streams decoding, the wall-time share of a prompt pass beside them, queued prefill work.
pub const Load = struct { decoding: u32 = 0, share: f64 = 1, local_ns: u64 = 0 };

/// What `Queue.poll` says to a waiting prompt: the pair is its own now, wait a little for it, or prefill here.
pub const Gate = enum { entered, wait, local };

/// A prompt waiting, pulling (the pair is its own) or sent here (its prefill is reserved until the host counts it).
const State = enum { waiting, pulling, local };

const Entry = struct { id: u64, state: State, est: Estimate, at: u64, ends: u64 = 0 };

/// A prompt sent here stays reserved this long (ns), until the host's own count of its work includes it.
pub const grace_ns: u64 = 500 * std.time.ns_per_ms;

/// Pulls waiting at once, at most; one more is prefilled here (TF_PREFILL_AHEAD, 4 by default, keeps it far below).
pub const capacity = 64;

pub const Queue = struct {
    entries: [capacity]Entry = undefined,
    len: usize = 0,
    next_id: u64 = 1,
    load: Load = .{},
    /// each decoding stream's lost decode time counts this much against prefilling here (seconds per second)
    stall: f64 = 1,

    /// The prompt joins the end of the queue; null: it is full, the prompt is prefilled here.
    pub fn join(q: *Queue, now: u64, est: Estimate) ?u64 {
        if (q.len == capacity) return null;
        const id = q.next_id;
        q.next_id += 1;
        q.entries[q.len] = .{ .id = id, .state = .waiting, .est = est, .at = now };
        q.len += 1;
        return id;
    }

    /// The prompt leaves: its pull ended or it stopped waiting; one sent here stays reserved until `publish` frees it.
    pub fn leave(q: *Queue, id: u64) void {
        const at = q.find(id) orelse return;
        if (q.entries[at].state == .local) return;
        q.remove(at);
    }

    /// The host's count of its own work, taken at `now`: it replaces the reservations made at least a grace ago.
    pub fn publish(q: *Queue, now: u64, load: Load) void {
        q.load = load;
        var i: usize = 0;
        while (i < q.len) {
            const e = q.entries[i];
            if (e.state == .local and e.at +| grace_ns <= now) q.remove(i) else i += 1;
        }
    }

    /// Whether a pull is under way.
    pub fn held(q: *const Queue) bool {
        for (q.entries[0..q.len]) |e| if (e.state == .pulling) return true;
        return false;
    }

    /// The prompt's turn: the pair if no pull is under way and no earlier prompt waits; else wait, or go here.
    pub fn poll(q: *Queue, now: u64, id: u64, sendable: bool) Gate {
        if (sendable) q.assign(now);
        const at = q.find(id) orelse return .local;
        const e = &q.entries[at];
        switch (e.state) {
            .local => return .local,
            .pulling => return .entered,
            .waiting => {},
        }
        if (q.held()) return .wait;
        for (q.entries[0..at]) |x| if (x.state == .waiting) return .wait;
        e.state = .pulling;
        e.ends = now +| e.est.remote_ns;
        return .entered;
    }

    /// Whether a new prompt, queued behind everything now, is better off here, `decoding` streams being live.
    pub fn local(q: *Queue, now: u64, est: Estimate, decoding: u32) bool {
        if (est.pinned) return false;
        q.assign(now);
        return q.better(q.sums(now), est, decoding);
    }

    /// Time ahead on each side, in ns: what a prompt sent to either would meet.
    const Sums = struct { remote: u64 = 0, local: u64 = 0 };

    fn sums(q: *const Queue, now: u64) Sums {
        var s: Sums = .{ .local = q.load.local_ns };
        for (q.entries[0..q.len]) |e| switch (e.state) {
            .pulling => s.remote +|= left(e, now),
            .waiting => s.remote +|= e.est.remote_ns,
            .local => s.local +|= e.est.local_ns,
        };
        return s;
    }

    /// Whether prefilling here ends sooner than waiting on the pair, given the work ahead on each side.
    fn better(q: *const Queue, s: Sums, est: Estimate, decoding: u32) bool {
        const remote_done: f64 = @floatFromInt(s.remote +| est.remote_ns);
        const p = if (decoding > 0) std.math.clamp(q.load.share, 0.01, 1) else 1;
        const own: f64 = @floatFromInt(est.local_ns);
        const ahead: f64 = @floatFromInt(s.local);
        const lost = q.stall * @as(f64, @floatFromInt(decoding)) * own;
        return (ahead + own) / p + lost < remote_done;
    }

    /// Every waiting prompt, earliest first, to the side that finishes it sooner given the prompts decided before it.
    fn assign(q: *Queue, now: u64) void {
        var s: Sums = .{ .local = q.load.local_ns };
        for (q.entries[0..q.len]) |*e| switch (e.state) {
            .pulling => s.remote +|= left(e.*, now),
            .local => s.local +|= e.est.local_ns,
            .waiting => {
                if (!e.est.pinned and q.better(s, e.est, q.load.decoding)) {
                    e.state = .local;
                    e.at = now;
                    s.local +|= e.est.local_ns;
                } else s.remote +|= e.est.remote_ns;
            },
        };
    }

    /// Time a pull under way still needs: its estimate's rest, and once overrun, as much again as the overrun.
    fn left(e: Entry, now: u64) u64 {
        return if (now < e.ends) e.ends - now else now - e.ends;
    }

    fn find(q: *const Queue, id: u64) ?usize {
        for (q.entries[0..q.len], 0..) |e, i| if (e.id == id) return i;
        return null;
    }

    fn remove(q: *Queue, at: usize) void {
        std.mem.copyForwards(Entry, q.entries[at .. q.len - 1], q.entries[at + 1 .. q.len]);
        q.len -= 1;
    }
};
