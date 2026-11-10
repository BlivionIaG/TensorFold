//! #1594 step 2b for DeepSeek V4.1 on the Mac: a prompt prefix the GB10 pair prefilled, pulled over an MKVH protocol 2 link
//! and landed in a blank Snapshot of the served lane (the state `Backend.prefill` restores a kept prompt state from).
//!
//! The export is bf16 and f32 parts, not Metal bytes (docs/deepseek-v41-handoff.md): `Landing` packs them into the Snapshot's
//! buffers with the engine's own quantisers, the drafter's rings are dropped, and `Model.snapshotSeal` sets the cursor and
//! Engram record. An imported prefix is outside the exactness contract and is kept only as a remote state.
const std = @import("std");
const tf = @import("tensorfold");
const api = @import("engine_api");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const fabric = tf.fabric;
const ps = fabric.prefill_source;
const q27 = @import("qwen27_remote.zig");
const trigger = @import("remote_trigger.zig");
const Filled = api.LaneHost.Remote.Filled;
const Flags = api.LaneHost.Remote.Flags;

pub const policy = q27.policy;
pub const keeps = q27.keeps;
pub const checkpointId = api.LaneHost.Remote.checkpointId;

/// A slot's Packed layout over a Snapshot's own buffers (host-visible: nothing is staged).
pub fn packedOf(m: *const ds.Model, snap: *ds.Model.Snapshot) ho.metal.Packed {
    const plan = m.caches[0].plan;
    return .{ .live = snap.live, .storage = snap.storage, .layers = plan.layers, .capacity = plan.capacity };
}

/// The lane host's remote source: one pull at a time over its link, each into a blank Snapshot of this model's shape.
pub const Pull = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *ds.Model,
    remote: ps.Remote,
    mutex: std.Io.Mutex = .init,
    pulls: u64 = 0,
    ns: u64 = 0,
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    off: std.atomic.Value(bool) = .init(false),
    linked: bool = true,

    pub fn source(p: *Pull) api.LaneHost.Remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running };
    }

    /// A prefix of `prompt` beyond `cached` on a whole chunk, filled remotely and landed whole in a blank Snapshot.
    pub fn pull(p: *Pull, prompt: []const u32, cached: u64) !?Filled {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return null;
        p.remote.client.renew() catch |err| {
            if (p.linked) std.log.warn("remote prefill: the link is not usable ({s}); prompts are prefilled here", .{@errorName(err)});
            p.linked = false;
            return null;
        };
        if (!p.linked) std.log.info("remote prefill: the link is back", .{});
        p.linked = true;
        p.remote.reason.len = 0;
        const src = p.remote.source();
        const offer = src.offer(prompt, cached) orelse return null;
        const at: u32 = std.math.cast(u32, offer.upto) orelse return error.PrefixTooLong;
        const t0 = fabric.words.nowNs();
        const snap = try p.model.snapshotBlank();
        errdefer p.model.snapshotDrop(snap);
        var landing: ho.landing.Landing = undefined;
        try landing.init(p.gpa, at, .{ .@"packed" = packedOf(p.model, snap) });
        defer landing.deinit();
        src.fill(prompt, offer, landing.state()) catch |err| {
            if (err == error.Refused and p.remote.reason.len > 0) std.log.warn("remote prefill: the GB10 refused ({s})", .{p.remote.reason.text()});
            p.settle(err);
            return err;
        };
        _ = landing.complete() catch return error.PartialState;
        try p.model.snapshotSeal(snap, prompt, at);
        p.pulls += 1;
        p.ns += fabric.words.nowNs() - t0;
        return .{ .saved = snap, .at = at };
    }

    /// A mismatch no retry mends (the checkpoint, the prompt's tokens, the protocol) turns remote prefill off for good.
    fn settle(p: *Pull, err: anyerror) void {
        if (err != error.Mismatch and err != error.Protocol) return;
        p.off.store(true, .release);
        std.log.warn("remote prefill off for this server: the GB10's export does not match ({s})", .{@errorName(err)});
    }

    fn wants(ptr: *anyopaque, len: u64, cached: u64) bool {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return false;
        const n = p.remote.policy.upto(len, cached) orelse return false;
        return n - cached >= p.remote.policy.min_prefix;
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64) anyerror!?Filled {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        return p.pull(prompt, cached);
    }

    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        p.model.snapshotDrop(@ptrCast(@alignCast(saved)));
    }
};

/// What serve refuses before loading: one flag without the other, or a separate drafter model.
pub fn refusal(f: Flags, drafter: bool) ?[]const u8 {
    if (!f.on()) return null;
    if (f.url == null or f.link == null) return "--prefill-from and --mcdma-link go together: the GB10 server's URL and the mcdma-rpcd link it answers on";
    if (drafter) return "--prefill-from serves DeepSeek V4.1 with its own MTP drafter only: serve without --drafter";
    return null;
}

/// A served link: the connect daemon's mailbox, its client, the HTTP trigger and the pull, as long as the server runs.
pub const Link = struct {
    box: fabric.mailbox.Mailbox,
    client: fabric.mailbox.Client,
    http: trigger.Http,
    pull: Pull,

    /// `id` names the checkpoint both ends load (the manifest's model must equal it).
    pub fn start(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, f: Flags, id: []const u8) !*Link {
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        const rules = try policy();
        l.box = try fabric.mailbox.Mailbox.openConnect(f.link.?);
        errdefer l.box.close();
        l.client = try fabric.mailbox.Client.init(&l.box, fabric.words.native, 30 * std.time.ns_per_s);
        const base = std.mem.trimEnd(u8, f.url.?, "/");
        const key = try q27.apiKey(gpa, io);
        errdefer if (key) |k| gpa.free(k);
        l.http = .{ .gpa = gpa, .io = io, .url = try std.fmt.allocPrint(gpa, "{s}/v1/completions", .{base}), .model = "", .closing = &l.pull.closing, .key = key };
        l.pull = .{ .gpa = gpa, .io = io, .model = m, .remote = .{ .gpa = gpa, .policy = rules, .client = &l.client, .trigger = l.http.trigger(), .model = id, .protocol = fabric.wire.recurrent_version } };
        return l;
    }

    /// After the lane host stopped: fills still under way end first.
    pub fn stop(l: *Link, gpa: std.mem.Allocator) void {
        l.pull.closing.store(true, .release);
        while (l.pull.running.load(.acquire) > 0) std.Io.sleep(l.pull.io, .fromMilliseconds(1), .awake) catch {};
        if (l.pull.pulls > 0) std.log.info("remote prefill: {d} prefixes pulled, {d:.1} ms each on average", .{ l.pull.pulls, @as(f64, @floatFromInt(l.pull.ns)) / @as(f64, @floatFromInt(l.pull.pulls)) / 1e6 });
        gpa.free(l.http.url);
        if (l.http.key) |k| {
            std.crypto.secureZero(u8, @constCast(k));
            gpa.free(k);
        }
        l.box.close();
        gpa.destroy(l);
    }
};

/// --prefill-from on a served host `h`: starts its link and gives the lane host the pull and the keep rule.
pub fn attach(h: anytype, io: std.Io, a: std.mem.Allocator, f: Flags, id: []const u8, why: *[]const u8) !void {
    if (!f.on()) return;
    h.link = Link.start(h.gpa, io, h.model, f, id) catch |err| {
        why.* = if (err == error.InvalidEnvironment) "a TF_PREFILL_* setting is malformed: the line above names it, its value and its accepted forms" else try std.fmt.allocPrint(a, "--mcdma-link {s}: the link could not open ({s}); is mcdma-rpcd's connect daemon serving it?", .{ f.link.?, @errorName(err) });
        return err;
    };
    h.host.remote = h.link.?.pull.source();
    h.host.keep_remote = keeps() catch |err| {
        why.* = "TF_PREFILL_KEEP takes 1 (keep states built on a pulled prefix, marked remote) or 0";
        h.link.?.stop(h.gpa);
        h.link = null;
        return err;
    };
    if (!h.host.keep_remote) std.log.info("remote prefill: states built on a pulled prefix are not kept (TF_PREFILL_KEEP=0)", .{});
}

test "remote prefill for DeepSeek V4.1 refuses half its flags and a separate drafter, by name" {
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{}, true));
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000" }, false) != null);
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, true) != null);
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, false));
}

test "the pull wants a prefix only 4096 tokens or more past what the Mac resumes, and stops for good on a mismatch" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null) return error.SkipZigTest;
    var p: Pull = undefined;
    p.closing = .init(false);
    p.off = .init(false);
    p.remote.policy = try policy();
    try std.testing.expect(Pull.wants(&p, 5600, 1000));
    try std.testing.expect(!Pull.wants(&p, 5600, 2000));
    p.settle(error.Refused);
    try std.testing.expect(Pull.wants(&p, 8192, 0));
    p.settle(error.Mismatch);
    try std.testing.expect(!Pull.wants(&p, 8192, 0));
}
