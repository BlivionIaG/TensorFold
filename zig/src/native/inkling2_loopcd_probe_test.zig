//! Opt-in LoopCD probes on the released checkpoint: teacher-forced likelihood and a greedy reply smoke.
const std = @import("std");
const api = @import("engine_api");
const native = @import("inkling2.zig");
const serving = @import("inkling2_serving_test.zig");
const Model = @import("../families/inkling2/metal/model.zig").Model;
const fwd = @import("../families/inkling2/metal/forward.zig");
const st = @import("../families/inkling2/metal/state.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const probe_strengths = [_]?f32{ null, 0.1, 0.3, 0.5, 1.0 };

/// Lines of "first id id ...": the row whose next token is the first one scored, then the sequence.
fn lines(a: std.mem.Allocator, name: []const u8) ![]const []const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(1 << 22));
    var out: std.ArrayList([]const u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        var ids: std.ArrayList(u32) = .empty;
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        while (words.next()) |w| try ids.append(a, try std.fmt.parseInt(u32, w, 10));
        try out.append(a, ids.items);
    }
    return out.items;
}

fn top(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return best;
}

fn logOf(row: []const f32, id: usize) f64 {
    var most: f32 = row[0];
    for (row) |x| most = @max(most, x);
    var sum: f64 = 0;
    for (row) |x| sum += @exp(@as(f64, x - most));
    return @as(f64, row[id] - most) - @log(sum);
}

const Tally = struct { n: u64 = 0, logprob: f64 = 0, hits: u64 = 0, agree: u64 = 0, better: u32 = 0, worse: u32 = 0 };

test "LoopCD probe: teacher-forced likelihood of the scored rows at each strength" {
    const dir = try serving.checkpoint();
    const name = @import("sys").getenv("TF_LOOPCD_PROBE") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompts = try lines(a, name);
    const m = try Model.load(gpa, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const v = m.config.text.unpadded_vocab;
    var tallies: [probe_strengths.len]Tally = @splat(.{});
    for (prompts) |line| {
        const first = line[0];
        const ids = line[1..];
        const all = try a.alloc(f32, probe_strengths.len * ids.len * v);
        for (probe_strengths, 0..) |w, k| {
            m.loopcd = w;
            var cache = try st.Cache.init(m.device, 1024, &m.weight_plan);
            defer cache.deinit();
            var at: usize = 0;
            while (at < ids.len) : (at += st.prompt_rows) {
                const n = @min(st.prompt_rows, ids.len - at);
                try fwd.run(m, &s, &cache, ids[at..][0..n], .{ .all = true });
                @memcpy(all[(k * ids.len + at) * v ..][0 .. n * v], s.logits_all.slice(f32, n * v));
                cache.len += n;
            }
        }
        m.loopcd = null;
        var base_sum: f64 = 0;
        for (probe_strengths, &tallies, 0..) |_, *t, k| {
            var sum: f64 = 0;
            for (first..ids.len - 1) |r| {
                const row = all[(k * ids.len + r) * v ..][0..v];
                const base = all[r * v ..][0..v];
                const lp = logOf(row, ids[r + 1]);
                sum += lp;
                t.n += 1;
                t.logprob += lp;
                t.hits += @intFromBool(top(row) == ids[r + 1]);
                t.agree += @intFromBool(top(row) == top(base));
            }
            if (k == 0) base_sum = sum else if (sum > base_sum) t.better += 1 else if (sum < base_sum) t.worse += 1;
        }
    }
    for (probe_strengths, tallies) |w, t| {
        const n: f64 = @floatFromInt(t.n);
        std.debug.print("LOOPCD probe w={d:.1} (0: unguided): {d} rows, mean logprob {d:.5}, top-1 {d}/{d}, agree with unguided {d}/{d}, prompts better {d} worse {d}\n", .{ w orelse 0, t.n, t.logprob / n, t.hits, t.n, t.agree, t.n, t.better, t.worse });
    }
}

test "LoopCD probe: greedy replies with the flag off and at a strength" {
    const dir = try serving.checkpoint();
    const name = @import("sys").getenv("TF_LOOPCD_SMOKE") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompts = try lines(a, name);
    for ([_]?f32{ null, 0.3 }) |w| {
        var why: []const u8 = "";
        const engine = (try native.open(gpa, gpa, io, .{ .dir = dir, .model_type = "inkling2_loop", .context = 512, .lanes = 1, .prompt_cache_gib = 0, .decode_share = 0.5, .loopcd = w != null, .loopcd_w = w }, &why)) orelse {
            std.debug.print("refused: {s}\n", .{why});
            return error.Refused;
        };
        defer engine.close(engine.ctx);
        for (prompts, 0..) |line, i| {
            var reply: serving.Reply = .{};
            defer reply.deinit();
            try engine.engine.submit(i + 1, &.{ .prompt = line[1..], .max_tokens = 96, .drafts = false }, .{ .ctx = &reply, .event = serving.Reply.event });
            _ = try reply.wait();
            std.debug.print("LOOPCD smoke w={d:.1} prompt {d}: {any}\n", .{ w orelse 0, i, reply.tokens.items });
        }
    }
}
