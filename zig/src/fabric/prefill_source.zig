//! Where a stream's prompt state comes from: its own prefill (native), or another machine's over the fabric (remote).
const std = @import("std");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const decoder = @import("decoder.zig");
const words = @import("words.zig");

/// macOS and glibc both provide it; Zig's std.c declares it on some systems only.
extern "c" fn getentropy(buf: [*]u8, len: usize) c_int;

/// A remote prefix never counts as the engine's own exact state.
pub const Provenance = enum { native, remote };

/// The engine may keep a prompt prefix in its exact cache only when it computed it itself.
pub fn cacheable(p: Provenance) bool {
    return p == .native;
}

/// What a source fills: prompt tokens [0, upto); the backend prefills [upto, prompt_len) and draws the first token.
pub const Offer = struct { upto: u64, provenance: Provenance };

/// Opt-in and experimental: off unless the server was started with the fabric, and outside the exactness contract.
pub const Policy = struct {
    enabled: bool = false,
    min_prefix: u64 = 1024,
    chunk: u64 = 512,
    /// Prompt tokens left to the backend (not 0): the prefix is `prompt_len - tail` and `chunk` is not used.
    tail: u64 = 0,

    /// The remote prefix: all but `tail` tokens, else whole chunks and one token left local; nothing below `cached`.
    pub fn upto(p: Policy, prompt_len: u64, cached: u64) ?u64 {
        if (!p.enabled or prompt_len < p.min_prefix) return null;
        if (p.tail != 0) return if (prompt_len > p.tail and prompt_len - p.tail > cached) prompt_len - p.tail else null;
        if (p.chunk == 0) return null;
        const n = (prompt_len - 1) / p.chunk * p.chunk;
        return if (n > cached) n else null;
    }
};

/// Where filled state lands: the stream's caches in the backend; nothing is visible to rounds before `commit`.
pub const StateSink = struct {
    ptr: *anyopaque,
    put_fn: *const fn (ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void,
    commit_fn: *const fn (ptr: *anyopaque, upto: u64, provenance: Provenance) void,
    abort_fn: *const fn (ptr: *anyopaque) void,
    /// Shard `k`'s sink when the state arrives from several producers at once (each `put_fn` of a shard may run on its
    shard_fn: ?*const fn (ptr: *anyopaque, shard: u32) decoder.Sink = null,
};

/// A provider the lane engine asks before Backend.prefill; the engine owns the stream and the caches throughout.
pub const PrefillSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// How much of this prompt the source fills now; null leaves the whole prompt to the backend.
        offer: *const fn (ptr: *anyopaque, prompt: []const u32, cached: u64) ?Offer,
        /// Fill prompt[0..offer.upto] through `sink`, then commit; on error the sink is aborted and nothing is kept.
        fill: *const fn (ptr: *anyopaque, prompt: []const u32, offer: Offer, sink: StateSink) anyerror!void,
    };

    pub fn offer(s: PrefillSource, prompt: []const u32, cached: u64) ?Offer {
        return s.vtable.offer(s.ptr, prompt, cached);
    }

    pub fn fill(s: PrefillSource, prompt: []const u32, o: Offer, sink: StateSink) !void {
        s.vtable.fill(s.ptr, prompt, o, sink) catch |err| {
            sink.abort_fn(sink.ptr);
            return err;
        };
    }
};

/// Asks a producer to prefill `tokens` under a handoff id: its prefix and one more token (all but it is exported).
pub const Trigger = struct {
    ptr: *anyopaque,
    start_fn: *const fn (ptr: *anyopaque, handoff: [16]u8, tokens: []const u32) anyerror!void,

    /// One trigger exchange, on its own thread or inline: `live` while it runs, `broken` once it failed.
    pub const Run = struct {
        r: Trigger,
        id: [16]u8,
        tokens: []const u32,
        live: std.atomic.Value(bool) = .init(true),
        broken: std.atomic.Value(bool) = .init(false),
        result: anyerror!void = error.NotRun,
        ns: u64 = 0,

        pub fn run(t: *Run) void {
            const began = words.nowNs();
            t.result = t.r.start_fn(t.r.ptr, t.id, t.tokens);
            t.ns = words.nowNs() - began;
            if (t.result) |_| {} else |_| t.broken.store(true, .release);
            t.live.store(false, .release);
        }
    };
};

/// Makes every link usable again once the producer's prefill is over: a daemon may have re-linked while it ran.
pub const Renew = struct {
    ptr: *anyopaque,
    /// error.LinkDown: a link is not up yet (the pull waits for it); any other error ends the fill
    renew_fn: *const fn (ptr: *anyopaque) anyerror!void,
};

/// How long a pull waits for a link that dropped during the producer's prefill to come back (a re-link takes seconds).
pub const link_wait_ns = 30 * std.time.ns_per_s;
const link_poll_ns = 100 * std.time.ns_per_ms;

/// The fabric's source: one handoff at a time per link, pulled with the decoder into the stream's caches.
pub const Remote = struct {
    gpa: std.mem.Allocator,
    policy: Policy,
    client: *mailbox.Client,
    /// The other producers of a KV_SPLIT export.
    peers: []const *mailbox.Client = &.{},
    trigger: Trigger,
    model: []const u8,
    protocol: u16 = 1, // 2 to take a producer's recurrent state (MKVH protocol 2); 1 refuses such an export
    busy: bool = false,
    next_id: u64 = 1,
    reason: decoder.Reason = .{},
    /// the pull's request is gone (set before each fill, which one gate serialises); the decoder stops between calls
    abort: ?*const std.atomic.Value(bool) = null,
    /// TF_PREFILL_STREAM: the pull runs while the trigger's prefill does (the producer exports as it prefills)
    stream: bool = false,
    /// the last fill's trigger exchange (the producer's prefill) in ns, and the payload bytes it pulled
    trigger_ns: u64 = 0,
    bytes: u64 = 0,
    /// called between the trigger and the pull, when set
    renew: ?Renew = null,
    /// the longest wait for a link to come back after the trigger
    link_wait_ns: u64 = link_wait_ns,

    pub fn source(r: *Remote) PrefillSource {
        return .{ .ptr = r, .vtable = &.{ .offer = offerFn, .fill = fillFn } };
    }

    /// `renew` until every link answers; only a link that is not up yet is waited for, for `link_wait_ns`.
    fn renewLinks(r: *Remote) !void {
        const rn = r.renew orelse return;
        const deadline = words.nowNs() +| r.link_wait_ns;
        while (true) {
            rn.renew_fn(rn.ptr) catch |err| {
                if (err != error.LinkDown or words.nowNs() >= deadline) return err;
                if (r.abort) |gone| if (gone.load(.acquire)) return err;
                words.pause(link_poll_ns);
                continue;
            };
            return;
        }
    }

    fn offerFn(ptr: *anyopaque, prompt: []const u32, cached: u64) ?Offer {
        const r: *Remote = @ptrCast(@alignCast(ptr));
        if (r.busy or r.client.poisoned) return null;
        for (r.peers) |c| if (c.poisoned) return null;
        const n = r.policy.upto(prompt.len, cached) orelse return null;
        return .{ .upto = n, .provenance = .remote };
    }

    fn fillFn(ptr: *anyopaque, prompt: []const u32, o: Offer, sink: StateSink) anyerror!void {
        const r: *Remote = @ptrCast(@alignCast(ptr));
        r.busy = true;
        defer r.busy = false;
        var id: [16]u8 = undefined;
        if (getentropy(&id, 8) != 0) return error.NoEntropy;
        std.mem.writeInt(u64, id[8..16], r.next_id, .little);
        r.next_id += 1;
        if (o.upto >= prompt.len) return error.PrefixTooLong; // a token past the prefix goes too
        const tokens = prompt[0..@intCast(o.upto)];
        var trig: Trigger.Run = .{ .r = r.trigger, .id = id, .tokens = prompt[0..@intCast(o.upto + 1)] };
        // TF_PREFILL_STREAM: the pull starts with the trigger and takes parts as the producer finishes them
        const thread: ?std.Thread = if (r.stream) std.Thread.spawn(.{}, Trigger.Run.run, .{&trig}) catch null else null;
        if (thread == null) {
            trig.run();
            r.trigger_ns = trig.ns;
            try trig.result;
        }
        r.renewLinks() catch |err| { // streaming: the links as they stand now; one that drops during the prefill fails the pull
            if (thread) |t| t.join();
            return err;
        };
        const pulled = r.pullAll(id, tokens, sink, &trig);
        if (thread) |t| {
            t.join();
            r.trigger_ns = trig.ns;
            try trig.result; // the producer's failure is the cause, not the pull's Aborted
        }
        try pulled;
        sink.commit_fn(sink.ptr, o.upto, .remote);
    }

    /// Every shard's export (or the one's) pulled into `sink`; `trig` says whether the producer is still prefilling.
    fn pullAll(r: *Remote, id: [16]u8, tokens: []const u32, sink: StateSink, trig: *const Trigger.Run) !void {
        if (r.peers.len > 0) return r.pullShards(id, tokens, sink, trig);
        var arena: std.heap.ArenaAllocator = .init(r.gpa);
        defer arena.deinit();
        const pass: decoder.Sink = .{ .ptr = sink.ptr, .put_fn = sink.put_fn };
        const got = try decoder.pull(arena.allocator(), r.client, id, r.expect(tokens, 0, 1, trig), pass, &r.reason);
        if (got.manifest.first_token != 0) return error.PartialPrefix;
        r.bytes = got.bytes;
    }

    fn expect(r: *const Remote, tokens: []const u32, rank: u64, size: u64, trig: *const Trigger.Run) decoder.Expect {
        return .{ .model = r.model, .tokens = tokens, .protocol = r.protocol, .tp_rank = rank, .tp_size = size, .abort = r.abort, .streaming = if (r.stream) &trig.live else null, .broken = if (r.stream) &trig.broken else null };
    }

    /// One shard's pull on its own link; the thread's result and the producer's reason stay in the job.
    const Job = struct {
        client: *mailbox.Client,
        id: [16]u8,
        expect: decoder.Expect,
        sink: decoder.Sink,
        arena: std.heap.ArenaAllocator,
        reason: decoder.Reason = .{},
        result: anyerror!decoder.Pulled = error.NotRun,

        fn run(j: *Job) void {
            j.result = decoder.pull(j.arena.allocator(), j.client, j.id, j.expect, j.sink, &j.reason);
        }
    };

    /// Every shard's export pulled at once, each over its own link into its own sink; the first failure is returned.
    fn pullShards(r: *Remote, id: [16]u8, tokens: []const u32, sink: StateSink, trig: *const Trigger.Run) !void {
        const shard_fn = sink.shard_fn orelse return error.SinkNotSharded;
        const n: u32 = @intCast(1 + r.peers.len);
        const jobs = try r.gpa.alloc(Job, n);
        defer r.gpa.free(jobs);
        var made: usize = 0;
        defer for (jobs[0..made]) |*j| j.arena.deinit();
        for (jobs, 0..) |*j, k| {
            j.* = .{ .client = if (k == 0) r.client else r.peers[k - 1], .id = id, .expect = r.expect(tokens, k, n, trig), .sink = shard_fn(sink.ptr, @intCast(k)), .arena = .init(r.gpa) };
            made += 1;
        }
        const threads = try r.gpa.alloc(?std.Thread, n);
        defer r.gpa.free(threads);
        @memset(threads, null);
        for (jobs[1..], threads[1..]) |*j, *t| t.* = std.Thread.spawn(.{}, Job.run, .{j}) catch null;
        jobs[0].run();
        for (jobs[1..], threads[1..]) |*j, t| if (t) |th| th.join() else j.run();
        r.bytes = 0;
        for (jobs) |*j| {
            const got = j.result catch |err| {
                r.reason = j.reason;
                return err;
            };
            if (got.manifest.first_token != 0) return error.PartialPrefix;
            r.bytes += got.bytes;
        }
    }
};

test "only native prefixes are cacheable, and a remote prefix leaves a local tail on a chunk boundary" {
    try std.testing.expect(cacheable(.native) and !cacheable(.remote));
    const p: Policy = .{ .enabled = true, .min_prefix = 1024, .chunk = 512 };
    try std.testing.expectEqual(@as(?u64, 1536), p.upto(2000, 0));
    try std.testing.expectEqual(@as(?u64, 1536), p.upto(2048, 0));
    try std.testing.expectEqual(@as(?u64, 2048), p.upto(2049, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(1000, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(2000, 1536));
    try std.testing.expectEqual(@as(?u64, null), (Policy{}).upto(4096, 0));
}

test "a tail policy leaves exactly its tokens local, whatever the chunk, and nothing below what is cached" {
    const p: Policy = .{ .enabled = true, .min_prefix = 1024, .chunk = 512, .tail = 1 };
    try std.testing.expectEqual(@as(?u64, 1999), p.upto(2000, 0));
    try std.testing.expectEqual(@as(?u64, 2047), p.upto(2048, 0));
    try std.testing.expectEqual(@as(?u64, 8191), p.upto(8192, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(1000, 0)); // under min_prefix
    try std.testing.expectEqual(@as(?u64, null), p.upto(2000, 1999)); // the cache already holds all but the tail
    try std.testing.expectEqual(@as(?u64, 1999), p.upto(2000, 1998));
    var t64 = p;
    t64.tail = 64;
    try std.testing.expectEqual(@as(?u64, 1936), t64.upto(2000, 0));
    t64.min_prefix = 0;
    try std.testing.expectEqual(@as(?u64, null), t64.upto(64, 0)); // nothing would be pulled
    try std.testing.expectEqual(@as(?u64, 1), t64.upto(65, 0));
    var off = p;
    off.enabled = false;
    try std.testing.expectEqual(@as(?u64, null), off.upto(4096, 0));
}
