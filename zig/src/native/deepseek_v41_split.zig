//! DeepSeek V4.1 layer split on the Mac: one export pulled while the pair still computes. Its boundary rows feed the
//! engine's passes through a `Feed`; its layers-below-the-cut state lands in a blank Snapshot. Opt-in, mixed arithmetic.
const std = @import("std");
const tf = @import("tensorfold");
const api = @import("engine_api");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const fabric = tf.fabric;
const fd = tf.deepseek_v41_split_feed;
const split = ho.split;
const remote = api.LaneHost.Remote;
const env = @import("sys").environment;

/// How a pull reaches the pair: the trigger (blocks until the pair is done), the pull of the export, its release.
pub const Transport = struct {
    ptr: *anyopaque,
    trigger: *const fn (ptr: *anyopaque, id: [16]u8, tokens: []const u32, cut: u32, gone: *const std.atomic.Value(bool), limit_s: u64) anyerror!void,
    /// the export into `sink`, returning the payload bytes; waits for the pair to open it until `expect.open_timeout_ns`
    pull: *const fn (ptr: *anyopaque, arena: std.mem.Allocator, id: [16]u8, expect: fabric.decoder.Expect, sink: fabric.decoder.Sink, reason: *fabric.decoder.Reason) anyerror!u64,
    release: *const fn (ptr: *anyopaque, id: [16]u8) void,
};

/// What `run` needs of the server: the cut (TF_LS_CUT), the checkpoint's name, the route to learn into.
pub const Spec = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *ds.Model,
    route: *remote.Route,
    checkpoint: []const u8,
    transport: Transport,
    cut: u32,
    closing: *const std.atomic.Value(bool),
};

/// TF_LS_CUT (2, 8 or 14; default 14): the layer the Mac enters at.
pub fn cutFromEnv() !u32 {
    return env.read(u32, "TF_LS_CUT", split.default_cut);
}

/// One split's state, shared by the pulling thread and the engine thread through its `Feed`.
pub const Session = struct {
    gpa: std.mem.Allocator,
    model: *ds.Model,
    stage: fabric.row_stage.Stage,
    snap: *ds.Model.Snapshot,
    landing: ho.landing.Landing,
    cut: u32,
    upto: u64,
    t0: u64,
    announce: remote.Announce,
    feedv: fd.Feed,
    announced: bool = false,
    abort: std.atomic.Value(bool) = .init(false),
    state: std.atomic.Value(u8) = .init(@backingInt(fd.State.pulling)),
    first_row_ns: std.atomic.Value(u64) = .init(0),
    rows_ns: std.atomic.Value(u64) = .init(0),
    state_ns: std.atomic.Value(u64) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
    /// the engine's pass is over: written once by `finish` (stats first, then `done` with release)
    done: std.atomic.Value(bool) = .init(false),
    ok: bool = false,
    stats: fd.Stats = .{},

    fn init(spec: Spec, upto: u64, announce: remote.Announce) !*Session {
        const s = try spec.gpa.create(Session);
        errdefer spec.gpa.destroy(s);
        const at: u32 = std.math.cast(u32, upto) orelse return error.PrefixTooLong;
        s.* = .{ .gpa = spec.gpa, .model = spec.model, .stage = try fabric.row_stage.Stage.init(spec.gpa, upto, split.row_bytes), .snap = undefined, .landing = undefined, .cut = spec.cut, .upto = upto, .t0 = fabric.words.nowNs(), .announce = announce, .feedv = undefined };
        errdefer s.stage.deinit();
        s.snap = try spec.model.snapshotBlank(at);
        errdefer spec.model.snapshotDrop(s.snap);
        try s.landing.initCut(spec.gpa, at, spec.cut, .{ .@"packed" = @import("deepseek_v41_remote.zig").packedOf(spec.model, s.snap) });
        s.feedv = .{ .ptr = s, .cut = spec.cut, .upto = upto, .vtable = &vtable };
        return s;
    }

    fn deinit(s: *Session) void {
        s.landing.deinit();
        s.model.snapshotDrop(s.snap);
        s.stage.deinit();
        s.gpa.destroy(s);
    }

    fn since(s: *const Session) u64 {
        return fabric.words.nowNs() - s.t0;
    }

    fn fail(s: *Session) void {
        s.state.store(@backingInt(fd.State.failed), .release);
        s.stage.fail();
        s.abort.store(true, .release);
    }

    /// The pull's sink: boundary rows to the stage (the first one opens the engine's way in), the rest to the landing.
    fn put(ptr: *anyopaque, layer: *const fabric.manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const s: *Session = @ptrCast(@alignCast(ptr));
        if (split.cutOfLayer(layer.index) == null) return s.landing.decoderSink().put_fn(s.landing.decoderSink().ptr, layer, row_start, rows, bytes);
        if (split.cutOfLayer(layer.index).? != s.cut or layer.rows() != s.upto) return error.Mismatch;
        try s.stage.put(row_start, bytes);
        const now = s.since();
        if (!s.announced) {
            s.announced = true;
            s.first_row_ns.store(now, .release);
            s.announce.feed(&s.feedv);
        }
        s.rows_ns.store(now, .release);
    }

    fn self(p: *anyopaque) *Session {
        return @ptrCast(@alignCast(p));
    }

    fn ready(p: *anyopaque) u64 {
        return self(p).stage.available();
    }

    fn rowsOf(p: *anyopaque, row0: u64, n: u64) []const u8 {
        return self(p).stage.slice(row0, n);
    }

    fn wait(p: *anyopaque, end: u64, timeout_ns: u64) fd.Wait {
        const s = self(p);
        return switch (s.stage.wait(end, timeout_ns, null)) {
            .ready => .ready,
            .timeout, .aborted => .timeout,
            .failed => .failed,
        };
    }

    fn stateOf(p: *anyopaque) fd.State {
        return @fromBackingInt(@intCast(self(p).state.load(.acquire)));
    }

    fn pullOf(p: *anyopaque) fd.Pull {
        const s = self(p);
        return .{ .first_row_ns = s.first_row_ns.load(.acquire), .rows_ns = s.rows_ns.load(.acquire), .state_ns = s.state_ns.load(.acquire), .bytes = s.bytes.load(.acquire) };
    }

    fn snapshot(p: *anyopaque) ?*anyopaque {
        const s = self(p);
        return if (stateOf(s) == .landed) s.snap else null;
    }

    /// The engine is done with the feed, once: it never touches it again, and the pulling thread may free it.
    fn finish(p: *anyopaque, ok: bool, stats: fd.Stats) void {
        const s = self(p);
        if (s.done.load(.acquire)) return;
        s.ok = ok;
        s.stats = stats;
        if (!ok) s.abort.store(true, .release); // the stream is no use now: stop pulling it
        s.done.store(true, .release);
    }

    const vtable: fd.Feed.VTable = .{ .ready = ready, .rows = rowsOf, .wait = wait, .state = stateOf, .pull = pullOf, .snapshot = snapshot, .finish = finish };
};

/// The trigger on its own thread: it returns when the pair has finished its prefill, the pull runs meanwhile.
const Trigger = struct {
    spec: Spec,
    id: [16]u8,
    tokens: []const u32,
    gone: *const std.atomic.Value(bool),
    limit_s: u64,
    session: *Session,
    result: anyerror!void = {},

    fn run(t: *Trigger) void {
        t.result = t.spec.transport.trigger(t.spec.transport.ptr, t.id, t.tokens, t.spec.cut, t.gone, t.limit_s);
        if (t.result) |_| {} else |_| t.session.fail(); // the pair never produced: the pull stops, the engine falls back
    }
};

/// The whole split, the gate held: trigger and pull at once, the engine fed through `announce`'s feed; returns when the
/// engine finished the feed (or the request is gone or the time is up). Errors are the pair's, mapped by `outcome`.
pub fn run(spec: Spec, prompt: []const u32, upto: u64, gone: *const std.atomic.Value(bool), limit_s: u64, announce: remote.Announce, outcome: *remote.Outcome) !void {
    if (upto >= prompt.len) return error.PrefixTooLong; // a token past the prefix goes too
    var id: [16]u8 = undefined;
    spec.io.randomSecure(id[0..8]) catch return error.NoEntropy;
    std.mem.writeInt(u64, id[8..16], fabric.words.nowNs(), .little);
    const s = try Session.init(spec, upto, announce);
    var leak = false;
    defer if (!leak) s.deinit();
    var trig: Trigger = .{ .spec = spec, .id = id, .tokens = prompt[0 .. upto + 1], .gone = gone, .limit_s = limit_s, .session = s };
    const thread = try std.Thread.spawn(.{}, Trigger.run, .{&trig});
    var arena: std.heap.ArenaAllocator = .init(spec.gpa);
    defer arena.deinit();
    const expect: fabric.decoder.Expect = .{ .model = spec.checkpoint, .tokens = prompt[0..upto], .protocol = fabric.wire.recurrent_version, .abort = &s.abort, .open_timeout_ns = limit_s * std.time.ns_per_s };
    var reason: fabric.decoder.Reason = .{};
    const pulled = spec.transport.pull(spec.transport.ptr, arena.allocator(), id, expect, .{ .ptr = s, .put_fn = Session.put }, &reason);
    if (pulled) |n| {
        s.bytes.store(n, .release);
        landed(s, prompt) catch s.fail();
        if (s.state.load(.acquire) != @backingInt(fd.State.failed)) s.state_ns.store(s.since(), .release);
    } else |err| {
        if (err == error.Refused and reason.len > 0) std.log.warn("layer split: the pair refused ({s})", .{reason.text()});
        s.fail();
    }
    spec.transport.release(spec.transport.ptr, id);
    // the trigger returns once the pair is done (or the pull's failure cut it short through `gone`)
    thread.join();
    outcome.* = if (trig.result) |_| .ok else |err| if (remote.pairDown(err)) .down else .other;
    var failure: ?anyerror = null; // the trigger's error tells more than the pull's abort it caused
    trig.result catch |err| {
        failure = err;
    };
    if (failure == null) _ = pulled catch |err| {
        failure = err;
    };
    // the engine ends the feed; a pass that never began (the request went) is ended by `gone`
    const deadline = fabric.words.nowNs() + (limit_s + 60) * std.time.ns_per_s;
    while (s.announced and !s.done.load(.acquire)) { // a feed never announced never reached the engine
        if (gone.load(.acquire) or spec.closing.load(.acquire)) s.abort.store(true, .release);
        if (fabric.words.nowNs() >= deadline) {
            std.log.warn("layer split: the engine never ended its feed; the session is kept rather than freed under it", .{});
            leak = true;
            return error.EngineStalled;
        }
        std.Io.sleep(spec.io, .fromMilliseconds(1), .awake) catch {};
    }
    if (s.ok) {
        const total = s.state_ns.load(.acquire);
        spec.route.observe(.{ .pass = .split, .rows = upto, .ns = total });
        spec.route.observe(.{ .pass = .split_m5, .rows = s.stats.rows, .ns = s.stats.compute_ns });
        std.log.info("layer split: {d} rows at cut {d}: first row {d:.0} ms, last {d:.0} ms, state {d:.0} ms; Mac waited {d:.0} ms, computed {d:.0} ms", .{ upto, s.cut, ms(s.first_row_ns.load(.acquire)), ms(s.rows_ns.load(.acquire)), ms(total), ms(s.stats.wait_ns), ms(s.stats.compute_ns) });
        return;
    }
    if (failure) |err| return err;
    return error.SplitFailed;
}

/// The pull is whole: the state lands in the snapshot (sealed for `prompt`) and the engine may restore it.
fn landed(s: *Session, prompt: []const u32) !void {
    _ = try s.landing.complete();
    try s.model.snapshotSeal(s.snap, prompt[0..@intCast(s.upto)], @intCast(s.upto));
    s.state.store(@backingInt(fd.State.landed), .release);
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

test "the cut comes from TF_LS_CUT, 14 unless set" {
    if (std.c.getenv("TF_LS_CUT") != null) return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 14), try cutFromEnv());
}
