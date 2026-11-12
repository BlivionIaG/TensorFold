//! Counters of the expert offload and their log lines.
const std = @import("std");

pub const Stats = struct {
    rounds: u64 = 0,
    /// Router picks of a row decided by the policy, and how many were resident.
    accesses: u64 = 0,
    hits: u64 = 0,
    /// Misses given a cache line (their GEMVs write what they read there).
    admissions: u64 = 0,
    /// Bytes of prompt-span experts the copy engine moved window to scratch.
    scratch_bytes: u64 = 0,
    /// Rows decoded, and the wall nanoseconds of the rounds.
    tokens: u64 = 0,
    round_ns: u64 = 0,
    /// Layer stages, the bytes the host copied into the window, and the worker's nanoseconds from request to ack.
    stages: u64 = 0,
    window_bytes: u64 = 0,
    stage_ns: u64 = 0,
    /// Nanoseconds the worker waited for the device's request (the device computing, not stage work).
    idle_ns: u64 = 0,

    /// The counters since `base`.
    pub fn since(s: Stats, b: Stats) Stats {
        return .{
            .rounds = s.rounds - b.rounds,
            .accesses = s.accesses - b.accesses,
            .hits = s.hits - b.hits,
            .admissions = s.admissions - b.admissions,
            .scratch_bytes = s.scratch_bytes - b.scratch_bytes,
            .tokens = s.tokens - b.tokens,
            .round_ns = s.round_ns - b.round_ns,
            .stages = s.stages - b.stages,
            .window_bytes = s.window_bytes - b.window_bytes,
            .stage_ns = s.stage_ns - b.stage_ns,
            .idle_ns = s.idle_ns - b.idle_ns,
        };
    }
};

/// A prompt span's counters, from its first stage to its last answer.
pub const Span = struct { began: u64 = 0, stage_ns: u64 = 0, bytes: u64 = 0 };

/// An `interval` log line every this many rounds.
pub const report_every: u64 = 256;

fn fl(x: u64) f64 {
    return @floatFromInt(x);
}

/// One log line for the rounds in `d`: hit rate, misses and window traffic a token, the stage time.
pub fn report(label: []const u8, d: Stats, blob_bytes: usize, lines: usize) void {
    if (d.rounds == 0) return;
    const a = fl(@max(d.accesses, 1));
    const t = fl(@max(d.tokens, 1));
    const miss = fl(d.accesses - d.hits);
    const mib: f64 = 1024.0 * 1024.0;
    std.log.info(
        "experts {s}: {d} rounds {d} tokens, hit {d:.1}% miss {d:.1}/token ({d:.1} MiB/token over the link), " ++
            "window fill {d:.1} MiB/token, admissions {d:.1}/round, stage {d:.2} ms/token (device compute+wait {d:.2}), " ++
            "round {d:.2} ms ({d:.2} ms/token), {d} lines",
        .{
            label,
            d.rounds,
            d.tokens,
            100.0 * fl(d.hits) / a,
            miss / t,
            miss / t * fl(blob_bytes) / mib,
            fl(d.window_bytes) / t / mib,
            fl(d.admissions) / fl(d.rounds),
            fl(d.stage_ns) / t / 1e6,
            fl(d.idle_ns) / t / 1e6,
            fl(d.round_ns) / fl(d.rounds) / 1e6,
            fl(d.round_ns) / t / 1e6,
            lines,
        },
    );
}

/// One log line for a prompt span: its wall time, the time in stages, the experts copied to the scratch.
pub fn reportSpan(span: Span, rows: u32, ended: u64, scratch_bytes: u64) void {
    const wall = fl(ended - span.began);
    const staged_ns = fl(span.stage_ns);
    const mib = fl(scratch_bytes - span.bytes) / (1 << 20);
    std.log.info(
        "experts span: {d} rows, {d:.2} s first stage to last answer, {d:.2} s in stages, " ++
            "{d:.0} MiB copied to the scratch ({d:.2} GB/s of the stages' time)",
        .{ rows, wall / 1e9, staged_ns / 1e9, mib, mib * 1.048576e6 / staged_ns },
    );
}

test "the counters since a base are their differences" {
    const base: Stats = .{ .rounds = 2, .hits = 5, .stage_ns = 100 };
    const now: Stats = .{ .rounds = 5, .hits = 9, .stage_ns = 400, .tokens = 7 };
    const d = now.since(base);
    try std.testing.expectEqual(@as(u64, 3), d.rounds);
    try std.testing.expectEqual(@as(u64, 4), d.hits);
    try std.testing.expectEqual(@as(u64, 300), d.stage_ns);
    try std.testing.expectEqual(@as(u64, 7), d.tokens);
}
