const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const session = @import("session.zig");
const rounds = @import("decode_round.zig");
const allocation = @import("draft_allocation.zig");
const sampling = @import("sampling.zig");
const shared = @import("qwen_shared.zig");
const G = session.Generation(model.Model);

pub const Result = struct { done: bool = false, failure: ?anyerror = null };
pub const Candidate = struct { slot: usize, served: u64, activated: u64 };

pub fn select(candidates: []Candidate, max_rows: usize) []Candidate {
    std.mem.sort(Candidate, candidates, {}, struct {
        fn less(_: void, a: Candidate, b: Candidate) bool {
            return if (a.served == b.served) a.activated < b.activated else a.served < b.served;
        }
    }.less);
    return candidates[0..@min(candidates.len, max_rows)];
}

pub const Coordinator = struct {
    max_rows: usize = 128,
    costs: [128]allocation.Cost = undefined,
    cost_count: usize = 0,
    overhead_ms: f64 = 0,
    rows: usize = 0,
    streams: usize = 0,
    peak_bytes: u64 = 0,

    pub fn calibrate(c: *Coordinator, s: *session.Session, max_streams: usize) !void {
        if (s.backend != .qwen) return;
        if (max_streams == 0 or max_streams > 8 or c.max_rows == 0 or c.max_rows > 128) return error.InvalidSharedLimits;
        const m = &s.backend.qwen;
        var base = try G.init(m, &s.tokenizer, mx.allocator, &.{ 1000, 1001, 1002, 1003 }, .{ .max_tokens = 1 }, .{}, null);
        defer base.deinit();
        while (base.phase == .prefill) _ = try base.step(m);
        const limit = @min(c.max_rows, max_streams * 16);
        var measured: [9]allocation.Cost = undefined;
        var measured_count: usize = 0;
        var width: usize = 1;
        while (true) {
            var best = std.math.inf(f64);
            for (0..2) |_| {
                var states: [8]shared.State = undefined;
                var initialized: usize = 0;
                defer for (states[0..initialized]) |*state| state.deinit();
                var streams: [8]shared.Stream = undefined;
                var tokens: [8][16]i32 = undefined;
                var parents: [8][16]i32 = undefined;
                const n = @min(max_streams, width);
                var remaining = width;
                for (0..n) |i| {
                    states[i] = try base.state.clone();
                    initialized += 1;
                    const count = (remaining + n - i - 1) / (n - i);
                    remaining -= count;
                    for (0..count) |j| {
                        tokens[i][j] = @intCast(1000 + j);
                        parents[i][j] = @as(i32, @intCast(j)) - 1;
                    }
                    streams[i] = .{ .state = &states[i], .tokens = tokens[i][0..count], .parents = parents[i][0..count] };
                }
                try mx.check(mx.c.mlx_clear_cache());
                const resident = try @import("memory_runtime.zig").activeBytes();
                try mx.check(mx.c.mlx_reset_peak_memory());
                const started = @import("server_live.zig").now(s.io);
                var pass = try m.forwardStreams(streams[0..n]);
                defer pass.deinit();
                var paths: [8][]const i32 = @splat(&.{0});
                try pass.commit(paths[0..n]);
                try mx.check(mx.c.mlx_synchronize(mx.stream));
                best = @min(best, (@import("server_live.zig").now(s.io) - started) * 1000);
                var peak: usize = 0;
                try mx.check(mx.c.mlx_get_peak_memory(&peak));
                c.peak_bytes = @max(c.peak_bytes, peak -| resident);
            }
            measured[measured_count] = .{ .rows = width, .ms = @max(best, 0.001) };
            measured_count += 1;
            if (width == limit) break;
            width = @min(width * 2, limit);
        }
        for (1..limit + 1) |rows| {
            var upper: usize = 0;
            while (measured[upper].rows < rows) : (upper += 1) {}
            const hi = measured[upper];
            const lo = measured[upper -| 1];
            const fraction = if (hi.rows == lo.rows) 0 else @as(f64, @floatFromInt(rows - lo.rows)) / @as(f64, @floatFromInt(hi.rows - lo.rows));
            c.costs[rows - 1] = .{ .rows = rows, .ms = lo.ms + fraction * (hi.ms - lo.ms) };
        }
        c.cost_count = limit;
        std.debug.print("Shared decode measured {d} widths, interpolated to {d}; peak work={d} bytes\n", .{ measured_count, limit, c.peak_bytes });
    }

    pub fn step(c: *Coordinator, m: *model.Model, requests: []const *G, results: []Result) !void {
        if (requests.len == 0 or requests.len > 8 or requests.len > c.max_rows or results.len != requests.len) return error.InvalidSharedLimits;
        if (m.round_owner.stage != .idle) return error.ModelRoundActive;
        for (requests, 0..) |g, i| {
            if (g.model != m) return error.WrongGenerationModel;
            if (g.in_round or g.state.borrowed) return error.GenerationRoundActive;
            for (requests[0..i]) |prior| if (g == prior) return error.DuplicateStream;
        }
        @memset(results, .{});
        c.rows = 0;
        c.streams = 0;
        const started = @import("server_live.zig").now(std.Options.debug_io);
        var windows: [8]rounds.Window = undefined;
        var indexes: [8]usize = undefined;
        var count: usize = 0;
        defer for (requests) |g| {
            g.in_round = false;
        };
        errdefer for (indexes[0..count]) |i| {
            requests[i].phase = .failed;
        };
        for (requests, 0..) |g, i| {
            const window = g.prepareShared() catch |err| {
                results[i].failure = err;
                g.phase = .failed;
                continue;
            };
            if (window) |w| {
                g.in_round = true;
                windows[count] = w;
                indexes[count] = i;
                count += 1;
            } else results[i].done = true;
        }
        if (count == 0) return;
        var chances: [8][15]f64 = undefined;
        var probabilities: [8][]const f64 = undefined;
        var fixed: [8]usize = @splat(1);
        for (windows[0..count], 0..) |w, i| {
            probabilities[i] = chances[i][0..w.draft.len];
            if (w.from_neural) {
                @memcpy(chances[i][0..w.draft.len], w.draft.probabilities[0..w.draft.len]);
            } else {
                const structural = if (requests[indexes[i]].proposer) |p| p.last_structural else false;
                try allocation.chainProbabilities(&.{if (structural) 1 else 0.94}, chances[i][0..w.draft.len]);
            }
        }
        const granted = try allocation.allocate(mx.allocator, fixed[0..count], probabilities[0..count], c.costs[0..c.cost_count], c.overhead_ms, c.max_rows);
        defer mx.allocator.free(granted);
        var streams: [8]shared.Stream = undefined;
        for (windows[0..count], granted, 0..) |*w, extra, i| {
            const g = requests[indexes[i]];
            if (g.proposer) |*p| if (p.last_structural) {
                p.structural_tokens -= w.draft.len - extra;
            };
            w.draft.len = extra;
            w.count = extra + 1;
            streams[i] = .{ .state = &g.state, .tokens = w.tokens[0..w.count], .parents = w.parents[0..w.count] };
            c.rows += w.count;
        }
        c.streams = count;
        var retained: [8]model.Pass = @splat(.{});
        defer for (retained[0..count]) |*pass| pass.deinit();
        var selections: [8]G.Selection = undefined;
        var paths: [8][]const i32 = undefined;
        const forward_started = @import("server_live.zig").now(std.Options.debug_io);
        var pass = try m.forwardStreams(streams[0..count]);
        var forwarded_ms = (@import("server_live.zig").now(std.Options.debug_io) - forward_started) * 1000;
        var active = true;
        defer if (active) pass.deinit();
        for (windows[0..count], 0..) |*w, j| {
            const i = indexes[j];
            const g = requests[i];
            const view = try pass.view(j);
            selections[j] = blk: {
                const ids = sampling.rows(&m.kernels, &retained[j].scope, view.logits, w.positions[0..w.count], g.settings) catch |err| {
                    results[i].failure = err;
                    break :blk .{ .count = 0 };
                };
                defer mx.allocator.free(ids);
                break :blk g.selectDecode(m, w, ids) catch |err| {
                    results[i].failure = err;
                    break :blk .{ .count = 0 };
                };
            };
            paths[j] = selections[j].rows[0..selections[j].count];
            retained[j].start = view.start;
            retained[j].count = view.count;
            if (g.sink.draft_budget > 0 and g.sink.drafter != null and results[i].failure == null) for (view.taps, &retained[j].taps) |tap, *out| {
                out.* = try retained[j].scope.own(try mx.retain(tap));
            };
        }
        const commit_started = @import("server_live.zig").now(std.Options.debug_io);
        try pass.commit(paths[0..count]);
        forwarded_ms += (@import("server_live.zig").now(std.Options.debug_io) - commit_started) * 1000;
        pass.deinit();
        active = false;
        for (windows[0..count], 0..) |*w, j| {
            const i = indexes[j];
            const g = requests[i];
            if (results[i].failure == null) {
                if (g.sink.drafter) |d| g.state.swapDFlash(d);
                g.finishDecode(m, w, &retained[j], selections[j]) catch |err| {
                    results[i].failure = err;
                };
                if (g.sink.drafter) |d| g.state.swapDFlash(d);
            }
            if (results[i].failure != null) g.phase = .failed else results[i].done = g.phase == .finished;
        }
        const elapsed_ms = (@import("server_live.zig").now(std.Options.debug_io) - started) * 1000;
        c.overhead_ms = 0.8 * c.overhead_ms + 0.2 * @max(0, elapsed_ms - forwarded_ms);
    }
};

test "row-limited turns give every eligible request its mandatory pending row" {
    var candidates = [_]Candidate{ .{ .slot = 0, .served = 9, .activated = 1 }, .{ .slot = 1, .served = 0, .activated = 3 }, .{ .slot = 2, .served = 0, .activated = 2 } };
    const first = select(&candidates, 2);
    try std.testing.expectEqual(@as(usize, 2), first.len);
    try std.testing.expectEqual(@as(usize, 2), first[0].slot);
    try std.testing.expectEqual(@as(usize, 1), first[1].slot);
    for (first) |*candidate| candidate.served = 10;
    try std.testing.expectEqual(@as(usize, 0), select(&candidates, 1)[0].slot);
}

test "overlapping shared rounds fail before accessing another request" {
    var m: model.Model = undefined;
    m.round_owner = .{};
    const ticket = try m.round_owner.begin();
    defer ticket.release();
    var request: G = undefined;
    var results: [1]Result = undefined;
    var coordinator = Coordinator{};
    try std.testing.expectError(error.ModelRoundActive, coordinator.step(&m, &.{&request}, &results));
    try ticket.expect(.bound);
}
