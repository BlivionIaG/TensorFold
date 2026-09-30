const std = @import("std");
const mx = @import("mlx.zig");
const session = @import("session.zig");

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    chunks: std.ArrayList(usize) = .empty,
    cancelled: bool = false,
    cancel_after: ?usize = null,
    fn deinit(c: *Capture) void {
        c.bytes.deinit(mx.allocator);
        c.chunks.deinit(mx.allocator);
    }
    fn emit(raw: ?*anyopaque, value: []const u8) !void {
        const c: *Capture = @ptrCast(@alignCast(raw.?));
        try c.bytes.appendSlice(mx.allocator, value);
        try c.chunks.append(mx.allocator, value.len);
        if (c.cancel_after) |count| if (c.chunks.items.len >= count) {
            c.cancelled = true;
        };
    }
    fn cancellation(raw: ?*anyopaque) !void {
        const c: *Capture = @ptrCast(@alignCast(raw.?));
        if (c.cancelled) return error.RequestCancelled;
    }
    fn sink(c: *Capture) session.Sink {
        return .{ .context = c, .emit = emit, .cancellation = .{ .context = c, .callback = cancellation } };
    }
};

pub fn checkShared(io: std.Io, directory: []const u8, drafter: ?[]const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var s = try session.Session.initWithDraft(io, directory, .{ .enabled = drafter != null, .directory = drafter, .max_draft = 15 });
    defer s.deinit();
    if (s.backend != .qwen) return error.UnsupportedSharedModel;
    const G = session.Generation(@import("model.zig").Model);
    const m = &s.backend.qwen;
    const shared = @import("shared_round.zig");
    var tokens: [8][73]i32 = undefined;
    var prompts: [8][]const i32 = undefined;
    var options: [8]session.Options = undefined;
    var expected: [8]session.Reply = undefined;
    var baseline: [8]Capture = @splat(.{});
    defer for (&baseline) |*capture| capture.deinit();
    var completed: usize = 0;
    defer for (expected[0..completed]) |*reply| reply.deinit(mx.allocator);
    for (0..8) |i| {
        const count = 17 + i * 8;
        for (tokens[i][0..count], 0..) |*token, j| token.* = @intCast(1000 + i * 73 + j);
        prompts[i] = tokens[i][0..count];
        options[i] = .{ .max_tokens = 24 + i, .ignore_eos = true, .draft = false, .seed = 123 + i, .sampling = .{ .temperature = if (i % 2 == 0) 0 else 0.7, .top_k = 12, .top_p = 0.8, .metal = true } };
        var g = try G.init(m, &s.tokenizer, mx.allocator, prompts[i], options[i], s.draftSink(baseline[i].sink()), null);
        defer g.deinit();
        while (!try g.step(m)) {}
        expected[i] = try g.takeReply();
        completed += 1;
    }
    for ([_]usize{ 1, 2, 8, 128 }) |rows| {
        var requests: [8]G = undefined;
        var captures: [8]Capture = @splat(.{});
        defer for (&captures) |*capture| capture.deinit();
        var initialized: usize = 0;
        defer for (requests[0..initialized]) |*g| g.deinit();
        for (0..8) |i| {
            var opts = options[i];
            opts.draft = true;
            requests[i] = try G.init(m, &s.tokenizer, mx.allocator, prompts[i], opts, s.draftSink(captures[i].sink()), null);
            initialized += 1;
            if (drafter == null) {
                requests[i].context.clearRetainingCapacity();
                try requests[i].context.appendSlice(mx.allocator, prompts[i]);
                for (expected[i].tokens.items) |token| try requests[i].context.append(mx.allocator, @intCast(token));
                try requests[i].context.appendSlice(mx.allocator, prompts[i]);
                requests[i].proposer.?.prompt_len = requests[i].context.items.len;
            }
            while (requests[i].phase == .prefill) _ = try requests[i].step(m);
        }
        captures[0].cancel_after = 3;
        var coordinator = shared.Coordinator{ .max_rows = rows };
        var served: [8]u64 = @splat(0);
        var done: [8]bool = @splat(false);
        for (1..512) |turn| {
            if (std.mem.allEqual(bool, &done, true)) break;
            var candidates: [8]shared.Candidate = undefined;
            var count: usize = 0;
            for (done, 0..) |finished, i| if (!finished) {
                candidates[count] = .{ .slot = i, .served = served[i], .activated = i };
                count += 1;
            };
            const selected = shared.select(candidates[0..count], rows);
            var gs: [8]*G = undefined;
            var results: [8]shared.Result = undefined;
            for (selected, 0..) |candidate, i| {
                gs[i] = &requests[candidate.slot];
                served[candidate.slot] = turn;
            }
            try coordinator.step(m, gs[0..selected.len], results[0..selected.len]);
            try std.testing.expect(coordinator.rows <= rows);
            for (selected, results[0..selected.len]) |candidate, result| {
                if (result.failure) |err| {
                    try std.testing.expectEqual(@as(usize, 0), candidate.slot);
                    try std.testing.expectEqual(error.RequestCancelled, err);
                    done[0] = true;
                } else done[candidate.slot] = result.done;
            }
            try std.testing.expectEqual(@as(i32, 0), m.position);
            try std.testing.expectEqual(@import("decode_round.zig").Stage.idle, m.round_owner.stage);
            for (&requests) |*g| try std.testing.expect(!g.in_round and !g.state.borrowed);
        }
        try std.testing.expect(std.mem.allEqual(bool, &done, true));
        for (1..8) |i| {
            var reply = try requests[i].takeReply();
            defer reply.deinit(mx.allocator);
            try same(expected[i], reply, baseline[i], captures[i]);
        }
        try std.testing.expectEqual(.failed, requests[0].phase);
        std.debug.print("PASS: 8 shared requests, row cap {d}, exact greedy/sampled {s} output and streaming, cancellation isolation and fair turns\n", .{ rows, if (drafter != null) "neural" else "copy" });
    }
}

fn same(expected: session.Reply, actual: session.Reply, before: Capture, after: Capture) !void {
    try std.testing.expectEqualSlices(u32, expected.tokens.items, actual.tokens.items);
    try std.testing.expectEqualSlices(u8, expected.content, actual.content);
    try std.testing.expectEqual(expected.finish_reason, actual.finish_reason);
    try std.testing.expectEqual(expected.prompt_tokens, actual.prompt_tokens);
    try std.testing.expectEqualSlices(u8, before.bytes.items, after.bytes.items);
    try std.testing.expectEqualSlices(usize, before.chunks.items, after.chunks.items);
}

fn firstTokenPublication(s: *session.Session, prompt: []const i32, options: session.Options, expected: session.Reply) !void {
    for ([_]usize{ 1, 2 }) |limit| {
        var opts = options;
        opts.max_tokens = limit;
        opts.draft = false;
        var captures: [2]Capture = @splat(.{});
        defer for (&captures) |*capture| capture.deinit();
        var requests: [2]session.RequestGeneration = undefined;
        var initialized: usize = 0;
        defer for (requests[0..initialized]) |*request| request.deinit();
        for (&requests, &captures) |*request, *capture| {
            request.* = try session.RequestGeneration.init(s, mx.allocator, prompt, opts, capture.sink(), null);
            initialized += 1;
        }
        var finished: [2]bool = @splat(false);
        for (&requests, &captures, &finished, 0..) |*request, *capture, *done, i| {
            while (request.progress().prefilled < prompt.len) done.* = try request.step(s);
            try std.testing.expectEqual(limit == 1, done.*);
            try std.testing.expectEqualSlices(u32, expected.tokens.items[0..1], request.tokens());
            try std.testing.expectEqual(prompt.len, request.memoryLengths().now);
            try std.testing.expectEqual(null, try request.snapshot());
            const first = try s.tokenizer.decode(mx.allocator, expected.tokens.items[0..1], false);
            defer mx.allocator.free(first);
            if (std.unicode.utf8ValidateSlice(first) and !std.mem.endsWith(u8, first, "�")) {
                try std.testing.expectEqualSlices(u8, first, capture.bytes.items);
                try std.testing.expect(capture.chunks.items.len > 0);
            }
            if (i == 0) {
                try std.testing.expectEqual(@as(usize, 0), requests[1].progress().prefilled);
                try std.testing.expectEqual(@as(usize, 0), requests[1].tokens().len);
                try std.testing.expectEqual(@as(usize, 0), captures[1].chunks.items.len);
            }
        }
        while (!std.mem.allEqual(bool, &finished, true)) for (&requests, &finished) |*request, *done| {
            if (!done.*) done.* = try request.step(s);
        };
        for (&requests, captures) |*request, capture| {
            try std.testing.expectEqual(prompt.len + limit - 1, request.memoryLengths().now);
            var actual = try request.takeReply();
            defer actual.deinit(mx.allocator);
            try std.testing.expectEqualSlices(u32, expected.tokens.items[0..limit], actual.tokens.items);
            try std.testing.expectEqualSlices(u8, actual.content, capture.bytes.items);
            try std.testing.expectEqual(.length, actual.finish_reason);
        }
    }
}

fn firstTokenEdges(m: anytype, tok: *@import("vendor/tokenizer.zig").Tokenizer, prompt: []const i32) !void {
    const M = @TypeOf(m.*);
    const G = session.Generation(M);
    const vocab: usize = if (M == @import("model.zig").Model) 248320 else if (@hasField(M, "vocab")) @intCast(m.vocab) else @intCast(M.vocab);
    const end: u32 = blk: {
        for (0..vocab) |id| {
            const token: i32 = @intCast(id);
            const eos = if (M == @import("model.zig").Model) token == 248044 or token == 248046 else if (@hasDecl(M, "isEos")) m.isEos(token) else M.eos(token);
            if (eos) break :blk @intCast(id);
        }
        return error.MissingEosToken;
    };
    {
        var capture = Capture{};
        defer capture.deinit();
        var g = try G.init(m, tok, mx.allocator, prompt, .{ .max_tokens = 2, .draft = false }, capture.sink(), null);
        defer g.deinit();
        // Force a known EOS independently of the model's sampled continuation.
        g.budget.forced = &.{end};
        var done = false;
        while (g.phase == .prefill) done = try g.step(m);
        try std.testing.expect(done);
        try std.testing.expect(try g.step(m));
        try std.testing.expectEqual(prompt.len, @as(usize, @intCast(g.state.position)));
        try std.testing.expectEqual(@as(usize, 0), capture.chunks.items.len);
        var reply = try g.takeReply();
        defer reply.deinit(mx.allocator);
        try std.testing.expectEqual(@as(usize, 0), reply.tokens.items.len);
        try std.testing.expectEqualStrings("", reply.content);
        try std.testing.expectEqual(.stop, reply.finish_reason);
    }
    {
        var capture = Capture{};
        defer capture.deinit();
        var g = try G.init(m, tok, mx.allocator, prompt, .{ .max_tokens = 2, .draft = false, .ignore_eos = true, .thinking_budget = 1 }, capture.sink(), null);
        defer g.deinit();
        if (g.budget.limit == 0) return;
        const close = g.budget.close;
        try std.testing.expect(close.len >= 2);
        while (g.phase == .prefill) try std.testing.expect(!try g.step(m));
        try std.testing.expectEqualSlices(u32, close[0..1], g.reply.tokens.items);
        try std.testing.expectEqualSlices(u32, close[1..], g.budget.forced);
        try std.testing.expect(!g.budget.open);
        try std.testing.expectEqual(prompt.len, @as(usize, @intCast(g.state.position)));
        while (!try g.step(m)) {}
        try std.testing.expectEqualSlices(u32, close[2..], g.budget.forced);
        try std.testing.expectEqual(prompt.len + 1, @as(usize, @intCast(g.state.position)));
        var reply = try g.takeReply();
        defer reply.deinit(mx.allocator);
        try std.testing.expectEqualSlices(u32, close[0..2], reply.tokens.items);
        try std.testing.expectEqualSlices(u8, reply.content, capture.bytes.items);
        try std.testing.expectEqual(.length, reply.finish_reason);
    }
}

fn verifiedCopies(m: anytype, tok: *@import("vendor/tokenizer.zig").Tokenizer, prompt: []const i32, options: session.Options, expected: session.Reply, baseline: Capture) !void {
    const a = mx.allocator;
    for ([_]bool{ false, true }) |reject| {
        var capture = Capture{};
        defer capture.deinit();
        var g = try session.Generation(@TypeOf(m.*)).init(m, tok, a, prompt, options, capture.sink(), null);
        defer g.deinit();
        errdefer std.debug.print("Copy verification {s}: reject={any}, proposed={d}, accepted={d}, expected={any}, actual={any}\n", .{ @typeName(@TypeOf(m.*)), reject, g.proposed, g.accepted, expected.tokens.items, g.reply.tokens.items });
        // Proposals may be arbitrary: seed the lookup with a known continuation to
        // exercise full acceptance and a rejected suffix regardless of model prose.
        g.context.clearRetainingCapacity();
        try g.context.appendSlice(a, prompt);
        for (expected.tokens.items, 0..) |token, i| try g.context.append(a, @intCast(if (reject and i == 7) token ^ 1 else token));
        try g.context.appendSlice(a, prompt);
        g.proposer.?.prompt_len = g.context.items.len;
        while (!try g.step(m)) {}
        try std.testing.expect(g.proposed > 0);
        try std.testing.expect(g.accepted > 0);
        if (reject) try std.testing.expect(g.proposed > g.accepted);
        try std.testing.expectEqual(prompt.len + expected.tokens.items.len - 1, @as(usize, @intCast(g.state.position)));
        try std.testing.expectEqual(@as(i32, 0), m.position);
        var reply = try g.takeReply();
        defer reply.deinit(a);
        try same(expected, reply, baseline, capture);
    }
}

fn prefixReuse(s: *session.Session, prompt: []const i32, options: session.Options, expected: session.Reply, baseline: Capture) !void {
    const a = mx.allocator;
    const Store = @import("prompt_cache.zig").Store(session.Snapshot);
    var store = try Store.init(a, 2, null);
    defer store.deinit();
    const chunks = try (try s.prefillPlan()).chunks(a, prompt);
    defer chunks.deinit(a);
    const count = chunks.next(0);
    const disk_path = "build/native-checks/session-prefix.safetensors";
    const identity = try @import("snapshot_store.zig").identity(s);
    defer a.free(identity);
    const boundary = @import("prompt_cache.zig").Boundary{ .starts = chunks.starts };
    {
        var donor = try session.RequestGeneration.init(s, a, prompt, options, .{}, null);
        defer donor.deinit();
        try std.testing.expectEqual(null, try donor.snapshot());
        try std.testing.expect(!try donor.step(s));
        try std.testing.expectEqual(@as(usize, 0), donor.tokens().len);
        const saved = (try donor.snapshot()) orelse return error.MissingPrefixSnapshot;
        saved.save(s.io, disk_path, identity, prompt[0..count]) catch |err| {
            var owned = saved;
            owned.deinit();
            return err;
        };
        try std.testing.expectEqual(count, saved.position());
        try std.testing.expect(saved.nbytes() > 0);
        try store.insertOwned(prompt[0..count], saved, prompt, false);
    }
    if (s.backend == .flash) {
        const decision = s.backend.flash.kernels.flash_prefill.decision.?;
        defer s.backend.flash.kernels.flash_prefill.decision = decision;
        s.backend.flash.kernels.flash_prefill.decision = !decision;
        const changed = try @import("snapshot_store.zig").identity(s);
        defer a.free(changed);
        try std.testing.expectError(error.IncompatibleSnapshot, session.Snapshot.load(s.io, disk_path, changed, prompt[0..count], .flash));
    }
    try std.testing.expectEqual(@as(usize, 0), store.longest(prompt[0..count], boundary));
    var captures: [4]Capture = @splat(.{});
    defer for (&captures) |*capture| capture.deinit();
    var requests: [4]session.RequestGeneration = undefined;
    var initialized: usize = 0;
    defer for (requests[0..initialized]) |*request| request.deinit();
    for (&requests, &captures, 0..) |*request, *capture, index| {
        request.* = try session.RequestGeneration.init(s, a, prompt, options, capture.sink(), null);
        initialized += 1;
        if (index == 3) {
            var loaded = try session.Snapshot.load(s.io, disk_path, identity, prompt[0..count], @as(std.meta.Tag(session.Backend), s.backend));
            defer loaded.deinit();
            try request.restoreOwnedPrefix(&loaded);
            try std.testing.expectEqual(count, request.memoryLengths().now);
            continue;
        }
        var hit = (try store.match(prompt, boundary, index == 2)) orelse return error.MissingPrefixHit;
        defer hit.deinit(a);
        var invalid = try hit.cache.clone();
        defer invalid.deinit();
        switch (invalid) {
            inline else => |*state| state.position += 1,
        }
        try std.testing.expectError(error.IncompatibleSnapshotBoundary, request.restorePrefix(&invalid));
        try std.testing.expectError(error.IncompatibleSnapshotBoundary, request.restoreOwnedPrefix(&invalid));
        try std.testing.expectEqual(count + 1, invalid.position());
        try std.testing.expectEqual(@as(u64, 0), request.memoryLengths().now);
        if (index == 0) {
            try request.restorePrefix(&hit.cache);
        } else {
            const pointer = switch (hit.cache) {
                inline else => |state| @intFromPtr(state.cache.ptr),
            };
            try request.restoreOwnedPrefix(&hit.cache);
            try std.testing.expectEqual(@as(usize, 0), hit.cache.position());
            switch (request.*) {
                inline else => |state| try std.testing.expectEqual(pointer, @intFromPtr(state.state.cache.ptr)),
            }
        }
        try std.testing.expectEqual(count, request.memoryLengths().now);
        try std.testing.expectError(error.InvalidSnapshotState, request.restorePrefix(&hit.cache));
    }
    // Copied requests remain independent after the final request takes the stored owner.
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), store.nbytes());
    var finished = [_]bool{ false, false, false, false };
    while (!std.mem.allEqual(bool, &finished, true)) {
        for (&requests, &finished) |*request, *done| if (!done.*) {
            done.* = try request.step(s);
        };
    }
    for (&requests, captures) |*request, capture| {
        var actual = try request.takeReply();
        defer actual.deinit(a);
        try same(expected, actual, baseline, capture);
        try std.testing.expectEqual(null, try request.snapshot());
    }
    std.debug.print("PASS: persisted {s} prefix resumes identical tokens and streaming at boundary {d}\n", .{ @tagName(s.backend), count });
}

fn interleaved(s: *session.Session, m: anytype, tok: *@import("vendor/tokenizer.zig").Tokenizer) !void {
    const M = @TypeOf(m.*);
    const G = session.Generation(M);
    const a = mx.allocator;
    var storage: [3][2051]i32 = undefined;
    const prompts = [_][]const i32{ storage[0][0..37], storage[1][0..if (@hasDecl(M, "prefill")) @as(usize, 2051) else 71], storage[2][0..21] };
    for (prompts, 0..) |prompt, j| for (@constCast(prompt), 0..) |*id, i| {
        id.* = @intCast(10 + (i * 7 + j * 37) % 93);
    };
    var options = [_]session.Options{
        .{ .max_tokens = 12, .ignore_eos = true, .seed = 17 },
        .{ .max_tokens = 16, .ignore_eos = true, .seed = 123, .sampling = .{ .temperature = 0.7, .top_k = 12, .top_p = 0.8, .metal = true } },
        .{ .max_tokens = 18, .ignore_eos = true, .seed = 991, .thinking_budget = 3, .sampling = .{ .temperature = 1, .top_k = 5, .top_p = 0.9, .metal = true } },
    };
    var expected: [3]session.Reply = undefined;
    var baseline: [3]Capture = @splat(.{});
    defer for (&baseline) |*capture| capture.deinit();
    var completed: usize = 0;
    defer for (expected[0..completed]) |*reply| reply.deinit(a);
    for (prompts, options, &baseline, &expected) |prompt, opt, *capture, *reply| {
        var serial = opt;
        serial.draft = false;
        var g = try G.init(m, tok, a, prompt, serial, s.draftSink(capture.sink()), null);
        defer g.deinit();
        while (!try g.step(m)) {}
        reply.* = try g.takeReply();
        completed += 1;
    }
    for ([_]usize{ 0, 2 }) |i| try firstTokenPublication(s, prompts[i], options[i], expected[i]);
    try firstTokenEdges(m, tok, prompts[0]);
    try prefixReuse(s, prompts[1], options[1], expected[1], baseline[1]);
    for (prompts[0..2], options[0..2], expected[0..2], baseline[0..2]) |prompt, opt, reply, capture| try verifiedCopies(m, tok, prompt, opt, reply, capture);
    if (s.prefillStep() > 256) {
        const plan = try s.prefillPlan();
        if (plan.assistant.len == 0) return error.MissingAssistantPrefillMarker;
        const adaptive_prompt = try a.dupe(i32, prompts[1]);
        defer a.free(adaptive_prompt);
        for ([_]usize{ 320, 640 }) |at| @memcpy(adaptive_prompt[at..][0..plan.assistant.len], plan.assistant);
        var adaptive_capture = Capture{};
        defer adaptive_capture.deinit();
        var cold = try G.init(m, tok, a, adaptive_prompt, options[1], s.draftSink(adaptive_capture.sink()), null);
        defer cold.deinit();
        try cold.setPlan(plan);
        try std.testing.expectEqual(@as(usize, 320), cold.chunks.next(0));
        while (!try cold.step(m)) {}
        var reference = try cold.takeReply();
        defer reference.deinit(a);
        try prefixReuse(s, adaptive_prompt, options[1], reference, adaptive_capture);
    }
    var captured: [3]Capture = @splat(.{});
    defer for (&captured) |*capture| capture.deinit();
    var active: [3]G = undefined;
    var initialized: usize = 0;
    defer for (active[0..initialized]) |*g| g.deinit();
    for (prompts, options, &captured, &active) |prompt, opt, *capture, *g| {
        g.* = try G.init(m, tok, a, prompt, opt, s.draftSink(capture.sink()), null);
        if (s.draft_options.enabled) g.proposer.?.fallback_enabled = false;
        initialized += 1;
    }
    var cancelled_capture = Capture{};
    defer cancelled_capture.deinit();
    var cancelled = try G.init(m, tok, a, prompts[0], options[0], s.draftSink(cancelled_capture.sink()), null);
    defer cancelled.deinit();
    while (cancelled.phase == .prefill) _ = try cancelled.step(m);
    try std.testing.expectEqualSlices(u32, expected[0].tokens.items[0..1], cancelled.reply.tokens.items);
    try std.testing.expectEqual(prompts[0].len, @as(usize, @intCast(cancelled.state.position)));
    cancelled_capture.cancelled = true;
    try std.testing.expectError(error.RequestCancelled, cancelled.step(m));
    try std.testing.expectError(error.FailedGeneration, cancelled.step(m));
    var abort_capture = Capture{};
    defer abort_capture.deinit();
    var aborted = try G.init(m, tok, a, prompts[0], options[0], s.draftSink(abort_capture.sink()), null);
    defer aborted.deinit();
    while (aborted.phase == .prefill) _ = try aborted.step(m);
    {
        var decode = try aborted.beginRound(m);
        defer decode.deinit();
        try std.testing.expect(try decode.prepare());
        try decode.forward();
        abort_capture.cancelled = true;
        try std.testing.expectError(error.RequestCancelled, decode.settle());
        try std.testing.expectEqual(.failed, m.round_owner.stage);
        try std.testing.expectError(error.InvalidRoundStage, decode.settle());
    }
    try std.testing.expectError(error.FailedGeneration, aborted.step(m));
    try std.testing.expectEqual(@as(i32, 0), m.position);
    try std.testing.expectEqual(.idle, m.round_owner.stage);
    var finished = [_]bool{ false, false, false };
    var round: usize = 0;
    while (!std.mem.allEqual(bool, &finished, true)) : (round += 1) {
        if (round > 256) return error.GenerationDidNotFinish;
        for (0..active.len) |j| {
            const index = (j + round) % active.len;
            if (!finished[index]) {
                const g = &active[index];
                if (g.phase == .decode) {
                    var decode = try g.beginRound(m);
                    defer decode.deinit();
                    try std.testing.expectError(error.GenerationRoundActive, g.step(m));
                    try std.testing.expectError(error.GenerationRoundActive, g.takeReply());
                    const other = &active[(index + 1) % active.len];
                    if (other.phase == .decode or other.phase == .prefill) try std.testing.expectError(error.ModelRoundActive, other.beginRound(m));
                    try std.testing.expectError(error.InvalidRoundStage, decode.forward());
                    if (try decode.prepare()) {
                        try std.testing.expectError(error.InvalidRoundStage, decode.prepare());
                        try std.testing.expectError(error.InvalidRoundStage, decode.settle());
                        try decode.forward();
                        try std.testing.expectError(error.InvalidRoundStage, decode.forward());
                        try decode.settle();
                        try std.testing.expectError(error.InvalidRoundStage, decode.settle());
                    }
                    finished[index] = g.phase == .finished;
                    decode.deinit();
                    try std.testing.expectError(error.StaleRound, decode.prepare());
                } else finished[index] = try g.step(m);
            }
            try std.testing.expectEqual(@as(i32, 0), m.position);
        }
    }
    for (&active, expected, baseline, captured) |*g, before, before_sink, after_sink| {
        if (s.draft_options.enabled) try std.testing.expect(g.neural_proposed > 0);
        var actual = try g.takeReply();
        defer actual.deinit(a);
        try same(before, actual, before_sink, after_sink);
        try std.testing.expect(try g.step(m));
    }
    // A stop string ending inside a token must preserve the serial streaming boundary.
    if (expected[0].content.len >= 3) {
        options[0].stops = &.{expected[0].content[0..3]};
        var stopped_capture = Capture{};
        defer stopped_capture.deinit();
        var stopped = try G.init(m, tok, a, prompts[0], options[0], s.draftSink(stopped_capture.sink()), null);
        defer stopped.deinit();
        while (!try stopped.step(m)) {}
        var reply = try stopped.takeReply();
        defer reply.deinit(a);
        try std.testing.expectEqual(.stop, reply.finish_reason);
        try std.testing.expectEqual(@as(usize, 0), reply.content.len);
        try std.testing.expectEqual(@as(usize, 0), stopped_capture.bytes.items.len);
    }
    var zero = try G.init(m, tok, a, prompts[0], .{ .max_tokens = 0 }, .{}, null);
    defer zero.deinit();
    try std.testing.expect(try zero.step(m));
    var empty = try zero.takeReply();
    defer empty.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), empty.tokens.items.len);
    std.debug.print("PASS: {s} isolated/interleaved prompts, immediate first token, one/two-token limits, reusable prefixes, sampling, streaming, thinking budget, cancellation after forward, round ownership, stale handles, stop strings and zero-token requests\n", .{@typeName(M)});
}

pub fn check(io: std.Io, dir: []const u8) !void {
    return checkWithDraft(io, dir, .{});
}

pub fn checkNeural(io: std.Io, dir: []const u8, drafter: []const u8) !void {
    return checkWithDraft(io, dir, .{ .enabled = true, .directory = if (std.mem.eql(u8, drafter, "-")) null else drafter, .max_draft = 15 });
}

fn checkWithDraft(io: std.Io, dir: []const u8, options: @import("neural_draft.zig").Options) !void {
    try mx.init();
    defer mx.shutdown();
    {
        var s = try session.Session.initWithDraft(io, dir, options);
        defer s.deinit();
        switch (s.backend) {
            inline else => |*m| try interleaved(&s, m, &s.tokenizer),
        }
    }
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
}

pub fn checkImages(io: std.Io, dir: []const u8, path: []const u8) !void {
    return checkImagesWithDraft(io, dir, path, .{});
}

pub fn checkNeuralImages(io: std.Io, dir: []const u8, path: []const u8, drafter: []const u8) !void {
    return checkImagesWithDraft(io, dir, path, .{ .enabled = true, .directory = drafter, .max_draft = 15 });
}

fn checkImagesWithDraft(io: std.Io, dir: []const u8, path: []const u8, draft_options: @import("neural_draft.zig").Options) !void {
    try mx.init();
    defer mx.shutdown();
    {
        const a = mx.allocator;
        var s = try session.Session.initWithDraft(io, dir, draft_options);
        defer s.deinit();
        if (s.backend != .qwen) return error.ExpectedQwen;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(20 * 1024 * 1024));
        defer a.free(bytes);
        var tokens: std.ArrayList(i32) = .empty;
        defer tokens.deinit(a);
        try tokens.appendSlice(a, &.{ 10, 20, 30, 40 });
        const memory_before = try @import("memory_runtime.zig").activeBytes();
        var cpu = try @import("vision.zig").Prepared.init(io, dir, &.{.{ .bytes = bytes }}, tokens.items);
        defer cpu.deinit();
        try std.testing.expectEqual(memory_before, try @import("memory_runtime.zig").activeBytes());
        tokens.clearRetainingCapacity();
        try tokens.appendSlice(a, cpu.tokens.items);
        try mx.check(mx.c.mlx_reset_peak_memory());
        var prepared = try cpu.encode(io, dir, &s.backend.qwen.weights);
        defer prepared.deinit();
        var peak: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        try std.testing.expect(peak -| memory_before <= cpu.workspaceBytes());
        try std.testing.expect(prepared.positions.delta != 0);
        const G = session.Generation(@import("model.zig").Model);
        const prompts = [_][]const i32{ tokens.items, &.{ 50, 60, 70, 80 } };
        const images = [_]?*@import("vision.zig").Prompt{ &prepared, null };
        const options = session.Options{ .max_tokens = 12, .ignore_eos = true, .seed = 317, .sampling = .{ .temperature = 0.7, .top_k = 12, .top_p = 0.8, .metal = true } };
        var reference: [2]Capture = @splat(.{});
        var capture: [2]Capture = @splat(.{});
        defer for (&reference) |*v| v.deinit();
        defer for (&capture) |*v| v.deinit();
        var expected: [2]session.Reply = undefined;
        var finished: usize = 0;
        defer for (expected[0..finished]) |*reply| reply.deinit(a);
        for (prompts, images, &reference, &expected) |prompt, image, *output, *reply| {
            var serial = options;
            serial.draft = false;
            var g = try G.init(&s.backend.qwen, &s.tokenizer, a, prompt, serial, s.draftSink(output.sink()), image);
            defer g.deinit();
            while (!try g.step(&s.backend.qwen)) {}
            reply.* = try g.takeReply();
            finished += 1;
        }
        var active: [2]G = undefined;
        var initialized: usize = 0;
        defer for (active[0..initialized]) |*g| g.deinit();
        for (prompts, images, &capture, &active) |prompt, image, *output, *g| {
            g.* = try G.init(&s.backend.qwen, &s.tokenizer, a, prompt, options, s.draftSink(output.sink()), image);
            if (draft_options.enabled) g.proposer.?.fallback_enabled = false;
            initialized += 1;
        }
        var done = [_]bool{ false, false };
        while (!std.mem.allEqual(bool, &done, true)) {
            for (&active, &done) |*g, *ended| if (!ended.*) {
                ended.* = try g.step(&s.backend.qwen);
                try std.testing.expectEqual(@as(i32, 0), s.backend.qwen.rope_delta);
            };
        }
        for (&active, expected, reference, capture) |*g, before, before_sink, after_sink| {
            if (draft_options.enabled) try std.testing.expect(g.neural_proposed > 0);
            var actual = try g.takeReply();
            defer actual.deinit(a);
            try same(before, actual, before_sink, after_sink);
        }
    }
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: interleaved image/text requests preserve tokens, streaming and independent multimodal positions\n", .{});
}

pub fn checkSyntheticNeural(m: anytype, io: std.Io) !void {
    var prompt: [33]i32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(i + 1);
    return checkSyntheticNeuralPrompt(m, io, &prompt);
}

pub fn checkSyntheticNeuralPrompt(m: anytype, io: std.Io, prompt: []const i32) !void {
    const M = @TypeOf(m.*);
    const G = session.Generation(M);
    if (prompt.len < 2) return error.InvalidSnapshotState;
    const plan = @import("prefill_plan.zig").Plan{ .step = @min(G.chunk_size, prompt.len - 1) };
    const a = mx.allocator;
    m.reset();
    defer m.reset();
    var tok = @import("vendor/tokenizer.zig").Tokenizer.initEmptyForTests(a, .wordpiece);
    defer tok.deinit();
    for (0..@intCast(m.vocab)) |id| {
        const word = try std.fmt.allocPrint(a, "t{d}", .{id});
        try tok.vocab.put(word, @intCast(id));
        try tok.id_to_token.put(@intCast(id), word);
    }
    for ([_]f64{ 0, 0.7 }) |temperature| {
        const options = session.Options{ .max_tokens = 18, .ignore_eos = true, .seed = 819, .sampling = .{ .temperature = temperature, .top_k = 12, .top_p = 0.8, .metal = true } };
        var reference = Capture{};
        defer reference.deinit();
        var serial_options = options;
        serial_options.draft = false;
        var baseline = try G.init(m, &tok, a, prompt, serial_options, reference.sink(), null);
        defer baseline.deinit();
        try baseline.setPlan(plan);
        while (!try baseline.step(m)) {}
        for ([_]usize{ 1, 3, 15 }) |depth| {
            var captures: [3]Capture = @splat(.{});
            defer for (&captures) |*capture| capture.deinit();
            var generations: [3]G = undefined;
            var initialized: usize = 0;
            defer for (generations[0..initialized]) |*g| g.deinit();
            for (&generations, &captures) |*g, *capture| {
                var sink = capture.sink();
                sink.draft_budget = depth;
                g.* = try G.init(m, &tok, a, prompt, options, sink, null);
                initialized += 1;
                try g.setPlan(plan);
                g.proposer.?.fallback_enabled = false;
            }
            try std.testing.expect(!try generations[0].step(m));
            var owned = try generations[0].state.clone();
            defer owned.deinit();
            const pointer = owned.cache.ptr;
            try generations[1].restoreOwnedPrefix(&owned);
            try std.testing.expectEqual(pointer, generations[1].state.cache.ptr);
            try std.testing.expectEqual(@as(i32, 0), owned.position);
            const disk = @import("snapshot_file.zig");
            const disk_path = "build/native-checks/synthetic-prefix.safetensors";
            try disk.save(io, disk_path, @typeName(M), prompt[0..generations[0].offset], generations[0].state);
            var reader = try disk.Reader.open(io, disk_path, @typeName(M));
            defer reader.deinit();
            var restored = try reader.load(@TypeOf(generations[0].state));
            defer restored.deinit();
            try generations[2].restoreOwnedPrefix(&restored);
            var done = [_]bool{ false, false, false };
            while (!std.mem.allEqual(bool, &done, true)) {
                for (&generations, &done) |*g, *ended| if (!ended.*) {
                    ended.* = try g.step(m);
                };
                try std.testing.expectEqual(@as(i32, 0), m.position);
            }
            for (&generations, captures) |*g, capture| {
                try std.testing.expect(g.neural_proposed > 0);
                try same(baseline.reply, g.reply, reference, capture);
                try std.testing.expectEqual(baseline.state.position, g.state.position);
                var scope = mx.Scope{};
                defer scope.deinit();
                for (baseline.state.cache, g.state.cache) |expected, actual| inline for (comptime std.meta.fieldNames(@TypeOf(actual))) |field| {
                    if (@FieldType(@TypeOf(actual), field) == mx.Array) {
                        const x = @field(expected, field);
                        const y = @field(actual, field);
                        try std.testing.expectEqual(x.ctx == null, y.ctx == null);
                        if (x.ctx != null) try @import("sampling_checks.zig").equal(&scope, x, y);
                    }
                };
            }
        }
    }
    std.debug.print("PASS: synthetic {s} request-local neural depths 1/3/15, serial cache/output parity, sampling and in-memory/disk prefix restoration\n", .{@typeName(M)});
}
