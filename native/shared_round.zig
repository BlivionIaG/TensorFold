const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const session = @import("session.zig");
const rounds = @import("decode_round.zig");
const allocation = @import("draft_allocation.zig");
const sampling = @import("sampling.zig");
const G = session.Generation(model.Model);
pub const max_streams = @import("nemotron.zig").Model.max_shared_streams;

pub fn modelStreamLimit(comptime M: type) usize {
    return if (@hasDecl(M, "max_shared_streams")) M.max_shared_streams else 8;
}

pub fn streamLimit(backend: session.Backend) usize {
    return switch (backend) {
        .qwen => if (mx.tensor_units) model.Model.max_shared_streams else 32,
        inline else => |m| modelStreamLimit(@TypeOf(m)),
    };
}

fn Shared(comptime M: type) type {
    if (M == model.Model) return @import("qwen_shared.zig");
    if (M == @import("gemma.zig").Model) return @import("gemma_shared.zig");
    if (M == @import("nemotron.zig").Model) return @import("nemotron_shared.zig");
    if (M == @import("flash.zig").Model) return @import("flash_shared.zig");
    @compileError("Unsupported shared model");
}

pub fn rowLimit(backend: session.Backend) usize {
    return switch (backend) {
        .qwen => if (mx.tensor_units) model.Model.max_shared_rows else 32,
        inline .gemma, .nemotron, .flash => |m| @TypeOf(m).max_shared_rows,
        else => 0,
    };
}

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
    overhead_ms: [max_streams]?f64 = @splat(null),
    mtp_costs: ?@import("draft_depth.zig").Adaptive = null,
    rows: usize = 0,
    streams: usize = 0,
    peak_bytes: u64 = 0,
    timing: @import("server_live.zig").RoundTiming = .{},

    fn overhead(c: *const Coordinator, streams: usize) f64 {
        var best: f64 = 8;
        var distance: usize = std.math.maxInt(usize);
        for (c.overhead_ms, 1..) |measured, count| if (measured) |ms| {
            const delta = @max(streams, count) - @min(streams, count);
            if (delta < distance) {
                distance = delta;
                best = ms;
            }
        };
        return best;
    }

    fn observeOverhead(c: *Coordinator, streams: usize, ms: f64) void {
        const sample = @max(0, ms);
        const slot = &c.overhead_ms[streams - 1];
        slot.* = if (slot.*) |previous| previous + 0.2 * (sample - previous) else sample;
    }

    fn observeRound(c: *Coordinator, streams: usize, rows: usize, elapsed_ms: f64, forwarded_ms: f64) void {
        var target_ms = forwarded_ms;
        for (c.costs[0..c.cost_count]) |cost| if (cost.rows == rows) {
            target_ms = cost.ms;
            break;
        };
        c.observeOverhead(streams, elapsed_ms - target_ms);
    }

    pub fn calibrate(c: *Coordinator, s: *session.Session, stream_count: usize) !void {
        switch (s.backend) {
            inline .qwen, .gemma, .nemotron, .flash => |*m| {
                try c.calibrateModel(s, m, stream_count);
                const M = @TypeOf(m.*);
                if (comptime adaptiveMtp(M) or M == @import("gemma.zig").Model) if (s.draft_options.enabled and s.draft_options.max_draft > 0 and @import("neural_draft.zig").enabled(m, null)) {
                    const prior = if (@hasDecl(M, "draft_prior")) M.draft_prior else &@import("draft_depth.zig").flash_prior;
                    var policy = try @import("draft_depth.zig").Adaptive.init(@min(s.draft_options.max_draft, c.max_rows - 1), prior);
                    if (comptime adaptiveMtp(M)) try @import("mtp_calibration.zig").measureStep(M, m, s.io, &policy);
                    policy.forward_ms = @splat(0);
                    for (c.costs[0..c.cost_count]) |cost| if (cost.rows < policy.forward_ms.len) {
                        policy.forward_ms[cost.rows] = cost.ms;
                    };
                    c.mtp_costs = policy;
                };
            },
            else => {},
        }
    }

    fn calibrateModel(c: *Coordinator, s: *session.Session, m: anytype, stream_count: usize) !void {
        const M = @TypeOf(m.*);
        const S = Shared(M);
        const capacity = comptime modelStreamLimit(M);
        const qwen_costs = M == model.Model;
        const nemotron_costs = M == @import("nemotron.zig").Model;
        const gemma_costs = M == @import("gemma.zig").Model;
        if (stream_count == 0 or stream_count > streamLimit(s.backend) or c.max_rows == 0 or c.max_rows > 128) return error.InvalidSharedLimits;
        c.max_rows = @min(c.max_rows, rowLimit(s.backend));
        var prompt: [64]i32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(1000 + i);
        var calibration_tokens: [64]i32 = undefined;
        if (nemotron_costs or gemma_costs) {
            try @import("mtp_calibration.zig").checkTokens(&s.tokenizer, &calibration_tokens, @intCast(M.vocab));
            @memcpy(prompt[0..48], calibration_tokens[0..48]);
        }
        var base = try session.Generation(M).init(m, &s.tokenizer, mx.allocator, prompt[0..if (qwen_costs) @as(usize, 64) else if (nemotron_costs or gemma_costs) 48 else 4], .{ .max_tokens = 1 }, .{}, null);
        defer base.deinit();
        while (base.phase == .prefill) _ = try base.step(m);
        const limit = @min(c.max_rows, stream_count * 16);
        var measured: [32]allocation.Cost = undefined;
        var measured_count: usize = 0;
        var widths: [32]usize = undefined;
        const count_widths = if (qwen_costs) qwenCalibrationWidths(limit, mx.tensor_units, &widths).len else if (nemotron_costs or gemma_costs) nemotronCalibrationWidths(limit, &widths).len else blk: {
            var width: usize = 1;
            var n: usize = 0;
            while (true) {
                widths[n] = width;
                n += 1;
                if (width == limit) break;
                width = @min(width * 2, limit);
            }
            break :blk n;
        };
        const hc_tiles = m.kernels.flash_rows.hc_tiles_on;
        if (M == @import("flash.zig").Model) m.kernels.flash_rows.hc_tiles_on = false;
        defer m.kernels.flash_rows.hc_tiles_on = hc_tiles;
        var commit_bytes: u64 = 0;
        for (widths[0..count_widths]) |width| {
            var best = std.math.inf(f64);
            // Extra streams probe workspace without charging their state cost to row growth.
            const single_cost = qwen_costs or ((nemotron_costs or gemma_costs) and width <= 16);
            const settlement_probe = qwen_costs and width == @min(stream_count, 8);
            const geometries: usize = if (single_cost and stream_count > 1 and width > 1 and (!qwen_costs or width == limit or settlement_probe)) 2 else 1;
            for (0..geometries) |geometry| {
                var states: [capacity]S.State = undefined;
                var initialized: usize = 0;
                defer for (states[0..initialized]) |*state| state.deinit();
                var streams: [capacity]S.Stream = undefined;
                var tokens: [capacity][if (M == model.Model) 128 else 16]i32 = undefined;
                var parents: [capacity][if (M == model.Model) 128 else 16]i32 = undefined;
                const n = if (single_cost and geometry == 0) 1 else @min(stream_count, if (settlement_probe) width else if (qwen_costs) @max(1, width / 2) else if (!single_cost and (nemotron_costs or gemma_costs)) (width + 1) / 2 else width);
                var remaining = width;
                for (0..n) |i| {
                    states[i] = try base.state.clone();
                    initialized += 1;
                    const count = (remaining + n - i - 1) / (n - i);
                    remaining -= count;
                    for (0..count) |j| {
                        tokens[i][j] = if ((nemotron_costs or gemma_costs) and j < 16) calibration_tokens[48 + j] else @intCast((if (qwen_costs) @as(usize, 2000) else 1000) + j);
                        parents[i][j] = @as(i32, @intCast(j)) - 1;
                    }
                    streams[i] = .{ .state = &states[i], .tokens = tokens[i][0..count], .parents = parents[i][0..count] };
                }
                for (0..4) |repetition| {
                    if (repetition > 0) for (states[0..n]) |*state| {
                        const fresh = try base.state.clone();
                        state.deinit();
                        state.* = fresh;
                    };
                    if (geometry != 0 or repetition == 0) try mx.check(mx.c.mlx_clear_cache());
                    try mx.check(mx.c.mlx_synchronize(mx.stream));
                    const resident = try @import("memory_runtime.zig").activeBytes();
                    try mx.check(mx.c.mlx_reset_peak_memory());
                    const started = @import("server_live.zig").now(s.io);
                    var pass = try m.forwardStreams(streams[0..n]);
                    defer pass.deinit();
                    try mx.eval(pass.logits);
                    if (geometry == 0 and repetition != 0) best = @min(best, (@import("server_live.zig").now(s.io) - started) * 1000);
                    var forward_bytes: u64 = 0;
                    var before_commit: u64 = 0;
                    var logit_workspace: u64 = 0;
                    if (qwen_costs) {
                        var peak: usize = 0;
                        try mx.check(mx.c.mlx_get_peak_memory(&peak));
                        forward_bytes = peak -| resident;
                        // Host sampling and a DFlash lattice can retain FP32 logits alongside target work.
                        const draft_rows = if (s.drafter != null) @as(usize, @min(stream_count, 8)) * 16 else 0;
                        logit_workspace = try std.math.mul(u64, width + draft_rows, @as(u64, @intCast(mx.dim(pass.logits, -1))) * @sizeOf(f32));
                        if (geometry != 0) {
                            // One-row forwards stage cache settlement in the GDN kernel itself.
                            // Measure a kernel group to bound that path without materializing every idle stream.
                            if (settlement_probe) commit_bytes = @max(commit_bytes, try std.math.divCeil(u64, forward_bytes, n));
                            // Python's shared workspace probe forwards two rows per stream without settlement.
                            // Reserve each stream's measured settlement peak without allocating idle caches.
                            const committed = try std.math.add(u64, forward_bytes, try std.math.mul(u64, commit_bytes, n));
                            c.peak_bytes = @max(c.peak_bytes, try std.math.add(u64, committed, logit_workspace));
                            continue;
                        }
                        before_commit = try @import("memory_runtime.zig").activeBytes();
                        try mx.check(mx.c.mlx_reset_peak_memory());
                    }
                    var paths: [capacity][]const i32 = @splat(&.{0});
                    try pass.commit(paths[0..n]);
                    try mx.check(mx.c.mlx_synchronize(mx.stream));
                    var peak: usize = 0;
                    try mx.check(mx.c.mlx_get_peak_memory(&peak));
                    if (qwen_costs) {
                        commit_bytes = @max(commit_bytes, peak -| before_commit);
                        c.peak_bytes = @max(c.peak_bytes, try std.math.add(u64, try std.math.add(u64, forward_bytes, commit_bytes), logit_workspace));
                    } else c.peak_bytes = @max(c.peak_bytes, peak -| resident);
                }
            }
            measured[measured_count] = .{ .rows = width, .ms = @max(best, 0.001) };
            measured_count += 1;
        }
        if (qwen_costs or nemotron_costs) std.debug.print("{s} calibrated target forward costs: {any}\n", .{ if (qwen_costs) "Qwen" else "Nemotron", measured[0..measured_count] });
        for (1..limit + 1) |rows| {
            var upper: usize = 0;
            while (measured[upper].rows < rows) : (upper += 1) {}
            const hi = measured[upper];
            const lo = measured[upper -| 1];
            const fraction = if (hi.rows == lo.rows) 0 else @as(f64, @floatFromInt(rows - lo.rows)) / @as(f64, @floatFromInt(hi.rows - lo.rows));
            c.costs[rows - 1] = .{ .rows = rows, .ms = lo.ms + fraction * (hi.ms - lo.ms) };
        }
        c.cost_count = limit;
        std.debug.print("Shared decode measured {d} widths, interpolated to {d}; workspace reserve={d} bytes\n", .{ measured_count, limit, c.peak_bytes });
    }

    pub fn step(c: *Coordinator, m: anytype, requests: []const *session.Generation(@TypeOf(m.*)), results: []Result) !void {
        c.timing = .{};
        const M = @TypeOf(m.*);
        const capacity = comptime modelStreamLimit(M);
        if (@hasDecl(M, "max_shared_rows")) c.max_rows = @min(c.max_rows, M.max_shared_rows);
        const stream_limit = if (M == model.Model and !mx.tensor_units) 32 else capacity;
        if (M == model.Model and !mx.tensor_units) c.max_rows = @min(c.max_rows, 32);
        if (requests.len == 0 or requests.len > stream_limit or requests.len > c.max_rows or results.len != requests.len) return error.InvalidSharedLimits;
        if (m.round_owner.stage != .idle) return error.ModelRoundActive;
        var shared_drafter: ?*@import("drafter.zig").Drafter = null;
        for (requests, 0..) |g, i| {
            if (g.model != m) return error.WrongGenerationModel;
            if (g.in_round or g.state.borrowed) return error.GenerationRoundActive;
            for (requests[0..i]) |prior| if (g == prior) return error.DuplicateStream;
            if (M == model.Model and g.sink.drafter != null) {
                if (shared_drafter != null and shared_drafter != g.sink.drafter) return error.IncompatibleSharedDraft;
                shared_drafter = g.sink.drafter;
            }
        }
        @memset(results, .{});
        c.rows = 0;
        c.streams = 0;
        if (requests.len == 1 and !requests[0].options.draft) {
            const g = requests[0];
            results[0].done = g.step(m) catch |err| blk: {
                results[0].failure = err;
                g.phase = .failed;
                break :blk false;
            };
            c.rows = g.round_rows;
            c.streams = @intFromBool(c.rows > 0);
            c.timing = g.round_timing;
            return;
        }
        return c.stepShared(m, requests, results, shared_drafter);
    }

    // Single-request dispatch must not allocate the full shared round's stack frame.
    noinline fn stepShared(c: *Coordinator, m: anytype, requests: []const *session.Generation(@TypeOf(m.*)), results: []Result, shared_drafter: ?*@import("drafter.zig").Drafter) !void {
        const M = @TypeOf(m.*);
        const S = Shared(M);
        const array_targets = M == @import("nemotron.zig").Model or M == @import("gemma.zig").Model;
        const capacity = comptime modelStreamLimit(M);
        const Generation = session.Generation(M);
        const Pass = @typeInfo(@typeInfo(@TypeOf(M.forward)).@"fn".return_type.?).error_union.payload;
        if (@hasDecl(M, "forwardAfter")) {
            for (requests) |g| g.discardPreview();
        }
        const started = @import("server_live.zig").now(std.Options.debug_io);
        const singleton_mtp = (adaptiveMtp(M) or M == @import("gemma.zig").Model) and requests.len == 1 and c.mtp_costs != null;
        var adaptive_window = false;
        var windows: [capacity]rounds.Window = undefined;
        var indexes: [capacity]usize = undefined;
        var count: usize = 0;
        defer for (requests) |g| {
            g.in_round = false;
        };
        errdefer for (indexes[0..count]) |i| {
            requests[i].phase = .failed;
        };
        if (M == model.Model) {
            for (requests) |g| g.round_draft_budget = @min(g.sink.draft_budget, g.shared_draft_grant + 2);
        } else {
            var prior: [capacity][15]f64 = undefined;
            var probabilities: [capacity][]const f64 = undefined;
            var mandatory: [capacity]usize = @splat(1);
            for (requests, 0..) |g, i| {
                const budget = if (g.options.draft and @import("neural_draft.zig").enabled(m, g.sink.drafter)) @min(15, g.sink.draft_budget, @max(1, g.draftRoom() -| 1)) else 0;
                probabilities[i] = prior[i][0..budget];
                try g.draft_depth.chances(prior[i][0..budget]);
            }
            const budgets = try allocation.allocate(mx.allocator, mandatory[0..requests.len], probabilities[0..requests.len], c.costs[0..c.cost_count], c.overhead(requests.len), c.max_rows);
            defer mx.allocator.free(budgets);
            for (requests, budgets) |g, budget| g.round_draft_budget = @max(1, budget);
        }
        defer for (requests) |g| {
            g.round_draft_budget = 15;
            g.defer_neural = false;
        };
        for (requests, 0..) |g, i| {
            g.defer_neural = M == model.Model or M == @import("gemma.zig").Model or @hasDecl(M, "draftStepStreams");
            const window = g.prepareShared() catch |err| {
                results[i].failure = err;
                g.phase = .failed;
                continue;
            };
            if (window) |w| {
                const forcing = g.budget.forced.len > 0 or (if (g.sink.gate) |gate| gate.forced.len > 0 else false);
                if (singleton_mtp and w.from_neural and !forcing) {
                    g.round_draft_budget = g.draft_depth.next_depth orelse g.draft_depth.choose(&c.mtp_costs.?, g.sink.draft_budget, g.draftRoom());
                    adaptive_window = true;
                }
                g.draft_depth.next_depth = null;
                g.in_round = true;
                windows[count] = w;
                indexes[count] = i;
                count += 1;
            } else results[i].done = true;
        }
        if (count == 0) return;
        var pending_proposals = @import("neural_draft.zig").PendingProposals{};
        defer pending_proposals.deinit();
        var head_verification = @import("neural_draft.zig").HeadVerification{};
        defer head_verification.deinit();
        var head_slots: [capacity]?usize = @splat(null);
        var head_updates: [max_streams]@import("neural_draft.zig").HeadUpdate = @splat(.{});
        defer for (&head_updates) |*update| update.deinit();
        var pending_windows: [capacity]usize = undefined;
        var pending_slots: [capacity]?usize = @splat(null);
        var input_scope = mx.Scope{};
        defer input_scope.deinit();
        if (M == model.Model or M == @import("gemma.zig").Model or @hasDecl(M, "draftStepStreams")) {
            const neural = @import("neural_draft.zig");
            const DraftStream = if (M == model.Model) @import("drafter.zig").Stream else neural.Stream(M);
            var draft_streams: [capacity]DraftStream = undefined;
            var proposals: [capacity]@import("drafter.zig").Proposal = undefined;
            var draft_indexes: [capacity]usize = undefined;
            var drafting: usize = 0;
            var drafter: ?*neural.Drafter = null;
            for (windows[0..count], 0..) |window, j| if (window.from_neural) {
                const g = requests[indexes[j]];
                draft_indexes[drafting] = j;
                const budget = @min(g.round_draft_budget, g.sink.draft_budget, @max(1, g.draftRoom() -| 1));
                draft_streams[drafting] = if (M == model.Model) .{ .state = &g.state, .anchor = window.tokens[0], .budget = budget, .settings = g.settings } else .{ .state = &g.state, .first = window.tokens[0], .budget = budget, .settings = g.settings };
                if (M == model.Model) {
                    if (drafter != null and drafter != g.sink.drafter) return error.IncompatibleSharedDraft;
                    drafter = g.sink.drafter;
                }
                drafting += 1;
            };
            if (drafting > 0) {
                var gpu_targets = array_targets;
                for (indexes[0..count]) |index| {
                    const cfg = requests[index].settings;
                    gpu_targets = gpu_targets and (cfg.metal or cfg.temperature == 0);
                }
                if (M == model.Model) {
                    try drafter.?.proposeStreams(m, draft_streams[0..drafting], proposals[0..drafting]);
                } else if (gpu_targets) {
                    pending_proposals = try neural.proposeStreamsLazy(m, draft_streams[0..drafting]);
                    try pending_proposals.metadata(proposals[0..drafting]);
                    for (draft_indexes[0..drafting], 0..) |j, i| {
                        pending_windows[i] = j;
                        pending_slots[j] = i;
                    }
                } else if (M == @import("gemma.zig").Model) {
                    for (draft_streams[0..drafting], proposals[0..drafting]) |stream, *proposal| {
                        stream.state.swap(m);
                        defer stream.state.swap(m);
                        proposal.* = try neural.propose(m, stream.state, null, stream.first, stream.budget, stream.settings);
                    }
                } else try neural.proposeStreams(m, draft_streams[0..drafting], proposals[0..drafting]);
                for (draft_indexes[0..drafting], proposals[0..drafting]) |j, proposal| {
                    const g = requests[indexes[j]];
                    var proposed = proposal;
                    if (M != model.Model) try g.draft_depth.chances(proposed.probabilities[0..proposed.len]);
                    windows[j] = try rounds.Window.init(g.next, g.state.position, proposed, true);
                }
            }
        }
        var chances: [capacity][15]f64 = undefined;
        var probabilities: [capacity][]const f64 = undefined;
        var fixed: [capacity]usize = @splat(1);
        for (windows[0..count], 0..) |w, i| {
            probabilities[i] = chances[i][0..w.draft.len];
            if (w.from_neural) {
                @memcpy(chances[i][0..w.draft.len], w.draft.probabilities[0..w.draft.len]);
            } else {
                const structural = if (requests[indexes[i]].proposer) |p| p.last_structural else false;
                try allocation.chainProbabilities(&.{if (structural) 1 else 0.94}, chances[i][0..w.draft.len]);
            }
        }
        const singleton_grant = [_]usize{windows[0].draft.len};
        const allocated = if (adaptive_window) null else try allocation.allocate(mx.allocator, fixed[0..count], probabilities[0..count], c.costs[0..c.cost_count], c.overhead(count), c.max_rows);
        defer if (allocated) |granted| mx.allocator.free(granted);
        const granted: []const usize = allocated orelse &singleton_grant;
        var streams: [capacity]S.Stream = undefined;
        var array_streams: [capacity]if (array_targets) S.ArrayStream else void = undefined;
        var token_parts: [capacity]mx.Array = undefined;
        for (windows[0..count], granted, 0..) |*w, extra, i| {
            const g = requests[indexes[i]];
            if (w.from_neural) g.shared_draft_grant = extra;
            if (g.proposer) |*p| if (p.last_structural) {
                p.structural_tokens -= w.draft.len - extra;
            };
            w.draft.len = extra;
            w.count = extra + 1;
            streams[i] = .{ .state = &g.state, .tokens = w.tokens[0..w.count], .parents = w.parents[0..w.count] };
            if (array_targets and pending_proposals.count > 0) {
                array_streams[i] = .{ .state = &g.state, .count = w.count, .parents = w.parents[0..w.count] };
                const host = try input_scope.cast(try input_scope.ints(if (pending_slots[i] != null) w.tokens[0..1] else w.tokens[0..w.count]), mx.c.MLX_UINT32);
                token_parts[i] = if (pending_slots[i]) |slot| (if (extra > 0) try input_scope.cat(&.{ host, try input_scope.slice(pending_proposals.tokens[slot], 0, 0, @intCast(extra)) }, 0) else host) else host;
            }
            c.rows += w.count;
        }
        c.streams = count;
        var retained: [capacity]Pass = @splat(if (M == @import("gemma.zig").Model) .{ .position = 0, .generation = 0, .rows = 0 } else .{});
        defer for (retained[0..count]) |*pass| pass.deinit();
        var selections: [capacity]Generation.Selection = undefined;
        var paths: [capacity][]const i32 = undefined;
        const forward_started = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.prepare_seconds = forward_started - started;
        var pass = if (array_targets and pending_proposals.count > 0)
            try S.forwardArray(m, array_streams[0..count], try input_scope.cat(token_parts[0..count], 0))
        else if (M == model.Model)
            try @import("qwen_shared.zig").forwardQueued(m, streams[0..count])
        else
            try m.forwardStreams(streams[0..count]);
        const forward_ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.forward_seconds = forward_ended - forward_started;
        var active = true;
        defer if (active) pass.deinit();
        var sample_scope = mx.Scope{};
        defer sample_scope.deinit();
        var positions: [128]i32 = undefined;
        var settings: [128]sampling.Sampling = undefined;
        var sampled_rows: usize = 0;
        for (windows[0..count], 0..) |w, j| {
            @memcpy(positions[sampled_rows..][0..w.count], w.positions[0..w.count]);
            @memset(settings[sampled_rows..][0..w.count], requests[indexes[j]].settings);
            sampled_rows += w.count;
        }
        const logits = try sample_scope.reshape(pass.logits, &.{ @intCast(sampled_rows), mx.dim(pass.logits, -1) });
        var head_gpu_targets = array_targets;
        for (indexes[0..count]) |index| {
            const cfg = requests[index].settings;
            head_gpu_targets = head_gpu_targets and (cfg.metal or cfg.temperature == 0);
        }
        const ids = if (array_targets and head_gpu_targets) blk: {
            const selected = try @import("gpu_sampling.zig").sampleRows(&m.kernels, &sample_scope, logits, positions[0..sampled_rows], settings[0..sampled_rows], null);
            var proposal_arrays: [max_streams]mx.Array = undefined;
            const ready = pending_proposals.arrays(&proposal_arrays);
            var arrays: [max_streams + 1]mx.Array = undefined;
            arrays[0] = selected;
            @memcpy(arrays[1..][0..ready.len], ready);
            try mx.evalMany(arrays[0 .. ready.len + 1], false);
            var proposals: [capacity]@import("drafter.zig").Proposal = undefined;
            try pending_proposals.read(proposals[0..pending_proposals.count]);
            for (pending_windows[0..pending_proposals.count], proposals[0..pending_proposals.count]) |j, proposal| {
                const w = &windows[j];
                var draft = w.draft;
                @memcpy(draft.tokens[0..draft.len], proposal.tokens[0..draft.len]);
                w.* = try rounds.Window.init(w.tokens[0], requests[indexes[j]].state.position, draft, true);
            }
            const result = try mx.allocator.alloc(i32, sampled_rows);
            for (result, mx.c.mlx_array_data_uint32(selected)[0..sampled_rows]) |*token, id| token.* = @intCast(id);
            break :blk result;
        } else try sampling.streamRowsMapped(&m.kernels, &sample_scope, logits, positions[0..sampled_rows], settings[0..sampled_rows], null);
        defer mx.allocator.free(ids);
        const sample_ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.sample_seconds = sample_ended - forward_ended;
        var sampled: usize = 0;
        var heads: [capacity]@import("neural_draft.zig").HeadStream = undefined;
        var head_hidden: [capacity]mx.Array = undefined;
        var head_tokens: [capacity]mx.Array = undefined;
        var head_keeps: [capacity]@import("neural_draft.zig").HeadVerification.Keep = undefined;
        var keeping_heads: usize = 0;
        for (windows[0..count], 0..) |*w, j| {
            const i = indexes[j];
            const g = requests[i];
            const view = if (M == model.Model) try pass.contextView(j) else try pass.view(j);
            selections[j] = blk: {
                break :blk g.selectDecode(m, w, ids[sampled..][0..w.count]) catch |err| {
                    results[i].failure = err;
                    break :blk .{ .count = 0 };
                };
            };
            sampled += w.count;
            paths[j] = selections[j].rows[0..selections[j].count];
            if (M == @import("nemotron.zig").Model and head_gpu_targets) {
                if (g.options.draft and g.sink.draft_budget > 0 and g.state.draft_hidden.ctx != null and results[i].failure == null and selections[j].count > 0) {
                    const slot = keeping_heads;
                    const keep = selections[j].count;
                    const row: usize = @intCast(selections[j].rows[selections[j].count - 1]);
                    const token = ids[sampled - w.count + row];
                    heads[slot] = .{ .state = &g.state, .anchor = w.tokens[0], .count = keep, .settings = g.settings };
                    head_hidden[slot] = try sample_scope.slice(view.hidden, 0, 0, @intCast(keep));
                    head_tokens[slot] = try sample_scope.cast(try sample_scope.ints(ids[sampled - w.count ..][0..keep]), mx.c.MLX_UINT32);
                    head_slots[j] = slot;
                    head_keeps[slot] = .{ .slot = slot, .count = keep, .token = token, .predict = g.phase == .decode and g.draftRoom() > 1 and g.next == token };
                    keeping_heads += 1;
                }
            }
            if (M == @import("gemma.zig").Model) {
                retained[j].position = view.position;
                retained[j].generation = view.generation;
                retained[j].rows = view.rows;
            } else {
                retained[j].start = view.start;
                retained[j].count = view.count;
            }
            const needs_draft = g.options.draft and g.sink.draft_budget > 0;
            if (needs_draft and results[i].failure == null) {
                if (M == model.Model) {
                    if (g.sink.drafter != null) for (view.taps, &retained[j].taps) |tap, *out| {
                        out.* = try retained[j].scope.own(try mx.retain(tap));
                    };
                } else {
                    retained[j].hidden = try retained[j].scope.own(try mx.retain(view.hidden));
                    if (M == @import("gemma.zig").Model and view.taps.ctx != null)
                        retained[j].taps = try retained[j].scope.own(try mx.retain(view.taps));
                }
            }
        }
        if (M == @import("nemotron.zig").Model and keeping_heads > 0) head_verification = try @import("neural_draft.zig").HeadVerification.init(m, heads[0..keeping_heads], try sample_scope.cat(head_hidden[0..keeping_heads], 0), try sample_scope.cat(head_tokens[0..keeping_heads], 0));
        const select_ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.select_seconds = select_ended - sample_ended;
        if (M == @import("nemotron.zig").Model) {
            if (keeping_heads > 0) head_updates = try head_verification.prepareBatch(head_keeps[0..keeping_heads]);
            const neural = @import("neural_draft.zig");
            var draft_states: [capacity]@import("request_state.zig").State(M) = undefined;
            var draft_streams: [capacity]neural.Stream(M) = undefined;
            var draft_slots: [capacity]usize = undefined;
            var draft_rates: [capacity][15]f64 = undefined;
            var draft_chances: [capacity][]const f64 = undefined;
            const mandatory: [capacity]usize = @splat(1);
            var drafting: usize = 0;
            for (indexes[0..count], 0..) |index, j| if (head_slots[j]) |slot| {
                const g = requests[index];
                const update = &head_updates[slot];
                if (update.prediction.first.ctx == null) continue;
                var depth = g.draft_depth;
                if (windows[j].from_neural) depth.observe(windows[j].draft.len, selections[j].accepted);
                const budget = @min(g.sink.draft_budget, @max(1, g.draftRoom() -| 2));
                draft_chances[drafting] = draft_rates[drafting][0..budget];
                try depth.chances(draft_rates[drafting][0..budget]);
                // Borrow only the head fields; target caches are still awaiting atomic commit.
                draft_states[drafting] = g.state;
                draft_states[drafting].borrowed = false;
                draft_states[drafting].position = update.prediction.position;
                draft_states[drafting].head_cache = update.cache;
                draft_states[drafting].draft_hidden = update.hidden;
                draft_states[drafting].head_prediction = update.prediction;
                draft_streams[drafting] = .{ .state = &draft_states[drafting], .first = update.prediction.token, .settings = g.settings, .budget = if (adaptive_window) depth.choose(&c.mtp_costs.?, g.sink.draft_budget, g.draftRoom() -| 1) else budget };
                draft_slots[drafting] = slot;
                drafting += 1;
            };
            if (drafting > 0) {
                if (!adaptive_window) {
                    const budgets = try allocation.allocate(mx.allocator, mandatory[0..drafting], draft_chances[0..drafting], c.costs[0..c.cost_count], c.overhead(drafting), c.max_rows);
                    defer mx.allocator.free(budgets);
                    for (draft_streams[0..drafting], budgets) |*stream, budget| stream.budget = @max(1, budget);
                }
                var queued = try neural.proposeStreamsLazy(m, draft_streams[0..drafting]);
                defer queued.deinit();
                for (draft_slots[0..drafting], 0..) |slot, i| {
                    head_updates[slot].prediction.drafts = queued.tokens[i];
                    queued.tokens[i] = mx.empty;
                }
            }
            var head_arrays: [max_streams * 7]mx.Array = undefined;
            var ready: usize = 0;
            for (head_updates[0..head_verification.count]) |update| if (update.hidden.ctx != null) {
                head_arrays[ready] = update.cache.a;
                head_arrays[ready + 1] = update.cache.b;
                head_arrays[ready + 2] = update.hidden;
                ready += 3;
                if (update.prediction.first.ctx != null) {
                    head_arrays[ready] = update.prediction.cache.a;
                    head_arrays[ready + 1] = update.prediction.cache.b;
                    head_arrays[ready + 2] = update.prediction.hidden;
                    head_arrays[ready + 3] = update.prediction.first;
                    ready += 4;
                }
            };
            if (ready > 0 and drafting != keeping_heads) try mx.evalMany(head_arrays[0..ready], true);
        }
        try pass.commit(paths[0..count]);
        const commit_ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.commit_seconds = commit_ended - select_ended;
        const forwarded_ms = (commit_ended - forward_started) * 1000;
        pass.deinit();
        active = false;
        if (M == @import("nemotron.zig").Model) for (indexes[0..count], 0..) |index, j| if (head_slots[j]) |slot| {
            if (head_updates[slot].hidden.ctx != null) {
                head_updates[slot].publish(&requests[index].state);
                head_updates[slot].deinit();
            }
        };
        head_verification.deinit();
        const release_ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.release_seconds = release_ended - commit_ended;
        if (M == model.Model) if (shared_drafter) |d| {
            var absorption: [capacity]@import("drafter.zig").AbsorbStream = undefined;
            var absorbing: usize = 0;
            for (windows[0..count], 0..) |*w, j| {
                const i = indexes[j];
                const g = requests[i];
                if (results[i].failure != null or !g.options.draft or g.sink.drafter == null or g.sink.draft_budget == 0) continue;
                absorption[absorbing] = .{ .state = &g.state, .pass = &retained[j], .rows = selections[j].rows[0..selections[j].count], .tokens = w.tokens[0..w.count] };
                absorbing += 1;
            }
            try d.absorbStreams(m, absorption[0..absorbing]);
        };
        if (@hasDecl(M, "absorbDraftStreams")) {
            const neural = @import("neural_draft.zig");
            var absorption: [capacity]neural.AbsorbStream(M) = undefined;
            var absorbing: usize = 0;
            for (windows[0..count], 0..) |*w, j| {
                const i = indexes[j];
                const g = requests[i];
                if (results[i].failure != null or !g.options.draft or g.sink.draft_budget == 0) continue;
                if (M == @import("nemotron.zig").Model and head_slots[j] != null) continue;
                absorption[absorbing] = .{ .state = &g.state, .hidden = if (@hasDecl(M, "draftHidden")) M.draftHidden(&retained[j]) else retained[j].hidden, .rows = selections[j].rows[0..selections[j].count], .tokens = w.tokens[0..w.count] };
                absorbing += 1;
            }
            try neural.absorbStreams(m, absorption[0..absorbing]);
        }
        for (windows[0..count], 0..) |*w, j| {
            const i = indexes[j];
            const g = requests[i];
            if (results[i].failure == null) {
                g.state.swap(m);
                defer g.state.swap(m);
                if (g.sink.drafter) |d| g.state.swapDFlash(d);
                defer if (g.sink.drafter) |d| g.state.swapDFlash(d);
                finishShared(g, m, w, &retained[j], selections[j]) catch |err| {
                    results[i].failure = err;
                };
            }
            if (results[i].failure != null) g.phase = .failed else results[i].done = g.phase == .finished;
        }
        const ended = @import("server_live.zig").now(std.Options.debug_io);
        c.timing.finish_seconds = ended - release_ended;
        const elapsed_ms = (ended - started) * 1000;
        if (adaptive_window and results[indexes[0]].failure == null) {
            const g = requests[indexes[0]];
            // Python queues the next head depth before recording this round's cost.
            if (g.phase == .decode and g.draftRoom() > 1) g.draft_depth.next_depth = g.draft_depth.choose(&c.mtp_costs.?, g.sink.draft_budget, g.draftRoom() - 1);
            try c.mtp_costs.?.observeElapsed(windows[0].draft.len, elapsed_ms, g.draft_depth.rounds == 0);
        }
        for (indexes[0..count]) |i| if (results[i].failure == null) {
            requests[i].draft_depth.rounds += 1;
        };
        c.observeRound(count, c.rows, elapsed_ms, forwarded_ms);
    }
};

fn builtinMtp(comptime M: type) bool {
    return M == @import("nemotron.zig").Model or M == @import("flash.zig").Model;
}

fn adaptiveMtp(comptime M: type) bool {
    return builtinMtp(M) and (if (@hasDecl(M, "adaptive_mtp_depth")) M.adaptive_mtp_depth else false);
}

test "singleton MTP choices use per-request rates and shared measured costs" {
    const Adaptive = @import("draft_depth.zig").Adaptive;
    const Depth = @import("neural_draft.zig").Depth;
    var costs = try Adaptive.init(3, &@import("draft_depth.zig").nemotron_prior);
    costs.forward_ms[2] = 5;
    costs.forward_ms[3] = 8;
    costs.forward_ms[4] = 12;
    costs.mtp_ms = 1;
    var first = Depth.init(@import("nemotron.zig").Model);
    var second = Depth.init(@import("nemotron.zig").Model);
    const rates = first.rates;
    for (0..7) |_| try std.testing.expectEqual(@as(usize, 1), first.choose(&costs, 3, 32));
    try std.testing.expectEqual(@as(usize, 2), first.choose(&costs, 3, 32));
    try std.testing.expectEqual(@as(usize, 0), second.choices);
    try std.testing.expectEqualSlices(f64, &rates, &first.rates);
    try costs.observeElapsed(2, 4, false);
    try std.testing.expectEqual(@as(usize, 2), second.choose(&costs, 3, 32));
    try std.testing.expectEqual(@as(usize, 1), second.choose(&costs, 3, 2));
    try std.testing.expectEqual(@as(usize, 0), second.choose(&costs, 0, 32));
    try std.testing.expectEqual(@as(usize, 2), second.choices);
    try std.testing.expectEqualSlices(f64, &rates, &second.rates);
    try std.testing.expect(builtinMtp(@import("nemotron.zig").Model));
    try std.testing.expect(builtinMtp(@import("flash.zig").Model));
    try std.testing.expect(!builtinMtp(model.Model));
    try std.testing.expect(!builtinMtp(@import("gemma.zig").Model));
    try std.testing.expect(adaptiveMtp(@import("nemotron.zig").Model));
    try std.testing.expect(!adaptiveMtp(@import("flash.zig").Model));
    try std.testing.expect(!adaptiveMtp(model.Model));
    try std.testing.expect(!adaptiveMtp(@import("gemma.zig").Model));
}

fn qwenCalibrationWidths(limit: usize, tensor: bool, out: *[32]usize) []const usize {
    std.debug.assert(limit > 0 and limit <= 128);
    var count: usize = 0;
    for ([_]usize{ 1, 2, 4, 8, 9, 12, 16, 17, 24, 25, 32, 33, 48, 64, 65, 96, 128 }) |width| {
        if (width > limit) break;
        if (tensor and (width == 9 or width == 25)) continue;
        out[count] = width;
        count += 1;
    }
    if (out[count - 1] != limit) {
        out[count] = limit;
        count += 1;
    }
    return out[0..count];
}

fn nemotronCalibrationWidths(limit: usize, out: *[32]usize) []const usize {
    std.debug.assert(limit > 0 and limit <= 128);
    var count: usize = @min(limit, 16);
    for (out[0..count], 1..) |*width, row| width.* = row;
    for ([_]usize{ 17, 32, 48, 64, 96, 128 }) |width| {
        if (width > limit) break;
        out[count] = width;
        count += 1;
    }
    if (out[count - 1] != limit) {
        out[count] = limit;
        count += 1;
    }
    return out[0..count];
}

test "Qwen forward cost probes separate tensor tile boundaries" {
    var widths: [32]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 8, 12, 16 }, qwenCalibrationWidths(16, true, &widths));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 8, 12, 16, 17 }, qwenCalibrationWidths(17, true, &widths));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128 }, qwenCalibrationWidths(128, true, &widths));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, qwenCalibrationWidths(3, true, &widths));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 8, 9, 12, 16, 17, 24, 25, 32 }, qwenCalibrationWidths(32, false, &widths));
}

test "Nemotron forward cost probes every serial width and wide shared boundaries" {
    var widths: [32]usize = undefined;
    const serial = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    try std.testing.expectEqualSlices(usize, serial[0..3], nemotronCalibrationWidths(3, &widths));
    try std.testing.expectEqualSlices(usize, &serial, nemotronCalibrationWidths(16, &widths));
    try std.testing.expectEqualSlices(usize, &(serial ++ [_]usize{17}), nemotronCalibrationWidths(17, &widths));
    try std.testing.expectEqualSlices(usize, &(serial ++ [_]usize{ 17, 24 }), nemotronCalibrationWidths(24, &widths));
    try std.testing.expectEqualSlices(usize, &(serial ++ [_]usize{ 17, 32, 48, 64, 96, 128 }), nemotronCalibrationWidths(128, &widths));
}

fn finishShared(g: anytype, m: anytype, window: *const rounds.Window, pass: anytype, selected: anytype) !void {
    const M = @TypeOf(m.*);
    if (M == @import("gemma.zig").Model and g.options.draft and g.sink.draft_budget > 0) {
        if (m.draft) |*draft| {
            if (draft.position != pass.position or pass.taps.ctx == null) return error.InvalidDraftContext;
            try draft.absorb(try pass.scope.slice(pass.taps, 0, 0, @intCast(selected.count)));
        }
    }
    try g.finishDecode(m, window, pass, selected, M != model.Model and !@hasDecl(M, "absorbDraftStreams"));
}

test "shared overhead follows stream count with nearest measured fallback" {
    var coordinator = Coordinator{};
    try std.testing.expectEqual(@as(f64, 8), coordinator.overhead(3));
    coordinator.observeOverhead(1, 2);
    coordinator.observeOverhead(8, 16);
    try std.testing.expectEqual(@as(f64, 2), coordinator.overhead(1));
    try std.testing.expectEqual(@as(f64, 16), coordinator.overhead(8));
    try std.testing.expectEqual(@as(f64, 2), coordinator.overhead(3));
    try std.testing.expectEqual(@as(f64, 16), coordinator.overhead(6));
    coordinator.observeOverhead(1, 7);
    try std.testing.expectEqual(@as(f64, 3), coordinator.overhead(1));
    try std.testing.expectEqual(@as(f64, 16), coordinator.overhead(8));
}

test "draft allocation includes sampling and head work in round overhead" {
    var calibrated = Coordinator{ .cost_count = 2 };
    calibrated.costs[0] = .{ .rows = 2, .ms = 10 };
    calibrated.costs[1] = .{ .rows = 3, .ms = 13.5 };
    calibrated.observeRound(2, 3, 18.5, 16.5);
    try std.testing.expectEqual(@as(f64, 5), calibrated.overhead(2));

    var uncalibrated = Coordinator{};
    uncalibrated.observeRound(2, 3, 18.5, 16.5);
    try std.testing.expectEqual(@as(f64, 2), uncalibrated.overhead(2));
    const a = std.testing.allocator;
    const costs = calibrated.costs[0..calibrated.cost_count];
    const before = try allocation.allocate(a, &.{ 1, 1 }, &.{ &.{0.5}, &.{} }, costs, uncalibrated.overhead(2), 3);
    defer a.free(before);
    const after = try allocation.allocate(a, &.{ 1, 1 }, &.{ &.{0.5}, &.{} }, costs, calibrated.overhead(2), 3);
    defer a.free(after);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, before);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, after);

    calibrated.observeRound(4, 4, 18.5, 16.5);
    try std.testing.expectEqual(@as(f64, 2), calibrated.overhead(4));
    calibrated.observeRound(8, 3, 12, 10);
    try std.testing.expectEqual(@as(f64, 0), calibrated.overhead(8));
}

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
