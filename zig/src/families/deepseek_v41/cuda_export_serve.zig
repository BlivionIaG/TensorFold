//! Hand-offs exported over MKVH (`--mcdma-produce LINK`) and/or written to files (`TF_DSV41_HANDOFF_DUMP=DIR`).
const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const cx = @import("cuda_export.zig");
const stream_mod = @import("cuda_export_stream.zig");
const plan = @import("handoff/plan.zig");
const split = @import("handoff/split.zig");
const file = @import("handoff_file.zig");
const producer = fabric.producer;
const log = std.log.scoped(.dsv41);
const spill_mod = @import("cuda_export_spill.zig");
const early_run = @import("cuda_export_early_run.zig");

pub const Options = struct {
    /// mcdma-rpcd's link name (null: no link; the exports go to files only)
    link: ?[]const u8 = null,
    /// a directory the exports are written under (null: none written)
    dump: ?[]const u8 = null,
    /// the checkpoint id both ends give a hand-off
    model: []const u8,
    /// the most export bytes a link holds at once (TF_PRODUCE_HELD_GIB)
    held_bytes: u64 = 32 << 30,
    ttl_ns: u64 = 120 * std.time.ns_per_s,
    /// KV_SPLIT: this producer is shard `shard` of `shards` (each rank exports its own pool pages on its own link)
    shard: u32 = 0,
    shards: u32 = 1,
    /// serve held exports over the server's own HTTP port (`--mcdma-produce http`), same cap and expiry as a link
    http: bool = false,
    /// a hand-off's export is built while its prompt prefills, and pulled as its parts are ready (needs a link)
    stream: bool = false,
    /// TF_PRODUCE_SPILL_DIR: buffers of `spill_min_bytes` and more live in unlinked files there (page cache, not RAM)
    spill_dir: ?[]const u8 = null,
    spill_min_bytes: usize = @as(usize, spill_mod.default_min_mib) << 20,
};

/// What `save` hands `exported`: the engine's state, read back (a streamed export's: without its pool pages).
pub const Saved = struct { cap: cx.Captured, read_ns: u64 = 0, stream: ?*stream_mod.Stream = null, owner: ?*Produce = null };

/// One export the producer holds; a reading stream pins it, so expiry or release only marks it `gone` until it closes.
pub const Held = struct { built: *cx.Built, name: [32]u8, pins: u32 = 0, gone: bool = false };

pub const Produce = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    source: cx.Source,
    opts: Options,
    box: fabric.mailbox.Mailbox = undefined,
    reg: ?fabric.registration.Registration = null, // MODE poll: the listen daemon sends staged replies only to a service
    table: producer.Table,
    serving: producer.Serving = undefined,
    thread: ?std.Thread = null,
    held: std.AutoHashMapUnmanaged([16]u8, *Held) = .empty,
    lock: std.atomic.Mutex = .unlocked, // `held`: the engine thread adds, the serving thread releases
    streams: std.atomic.Value(u32) = .init(0), // reads under way (HTTP): the producer stops after them
    building: std.atomic.Value(u32) = .init(0), // exports converting on their own threads (`exportedBg`)
    pending: std.AutoHashMapUnmanaged([16]u8, void) = .empty, // their ids, under `lock`: a reader waits for them
    active: std.AutoHashMapUnmanaged([16]u8, *stream_mod.Stream) = .empty, // streamed exports not yet cut, under `lock`
    spill: ?*spill_mod.Spill = null, // `Options.spill_dir`: the allocator of every buffer a held export is made of

    /// The most exports converting at once; one more converts on the caller's thread.
    pub const max_building: u32 = 2;

    /// A producer over `source`: attaches the link when `opts.link` is set, makes the dump dir when `opts.dump` is.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, source: cx.Source, opts: Options, problem: *[]const u8) !*Produce {
        const p = try gpa.create(Produce);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .io = io, .source = source, .opts = opts, .table = .{ .gpa = gpa, .ttl_ns = opts.ttl_ns, .clock = fabric.words.nowNs, .cap_bytes = opts.held_bytes } };
        errdefer p.table.deinit();
        if (opts.spill_dir) |dir| {
            std.Io.Dir.cwd().createDirPath(io, dir) catch |err| {
                problem.* = "TF_PRODUCE_SPILL_DIR: the directory cannot be made";
                return err;
            };
            p.spill = spill_mod.Spill.open(gpa, dir, opts.spill_min_bytes) catch |err| {
                problem.* = "TF_PRODUCE_SPILL_DIR: no file can be mapped there (a full or read-only file system?)";
                return err;
            };
            log.info("handoff: exports of {d} MiB and more are held on file-backed pages under {s}", .{ opts.spill_min_bytes >> 20, dir });
        }
        errdefer if (p.spill) |s| s.close();
        if (opts.dump) |dir| std.Io.Dir.cwd().createDirPath(io, dir) catch |err| {
            problem.* = "TF_DSV41_HANDOFF_DUMP: the directory cannot be made";
            return err;
        };
        if (opts.http) log.info("handoff: exports are served over HTTP, GET /v1/handoff/<id> on this server's port", .{});
        const link = opts.link orelse return p;
        if (builtin.os.tag != .linux) {
            problem.* = "--mcdma-produce runs on Linux, beside mcdma-rpcd's listen daemon";
            return error.UnsupportedPlatform;
        }
        p.box = fabric.mailbox.Mailbox.openListen(link) catch |err| {
            problem.* = "--mcdma-produce: no mcdma-rpcd listen daemon serves that link (start `mcdma-rpcd listen` first)";
            return err;
        };
        errdefer p.box.close();
        p.reg = fabric.registration.Registration.attach(link) catch |err| {
            problem.* = "--mcdma-produce: the listen daemon refused the service (another holds the link, or it is not running)";
            return err;
        };
        errdefer if (p.reg) |*r| r.close();
        log.info("mcdma handoff: attached to the listen daemon on link {s}", .{link});
        p.serving.init(gpa, &p.box, fabric.words.native, &p.table, opts.model);
        p.serving.responder.tp_rank = opts.shard;
        p.serving.responder.tp_size = opts.shards;
        errdefer p.serving.deinit();
        p.thread = try std.Thread.spawn(.{}, run, .{p});
        log.info("mcdma handoff: exporting DeepSeek-V4.1 states as MKVH protocol 2 on link {s}", .{link});
        return p;
    }

    /// Stops serving and frees every export still held; the server's socket timeouts bound a read under way.
    pub fn stop(p: *Produce) void {
        if (p.thread) |t| {
            p.serving.stop.store(true, .release);
            t.join();
        }
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        var open_streams = p.active.valueIterator();
        while (open_streams.next()) |st| st.*.abort();
        p.active.deinit(p.gpa);
        p.lock.unlock();
        while (p.streams.load(.acquire) > 0 or p.building.load(.acquire) > 0) std.Io.sleep(p.io, .fromMilliseconds(1), .awake) catch {};
        p.pending.deinit(p.gpa);
        var it = p.held.valueIterator();
        while (it.next()) |h| release(p, h.*);
        p.held.deinit(p.gpa);
        if (p.thread != null) {
            if (p.reg) |*r| r.close();
            p.serving.deinit();
            p.box.close();
        }
        p.table.deinit();
        if (p.spill) |s| s.close(); // after every export is released
        p.gpa.destroy(p);
    }

    fn run(p: *Produce) void {
        var done: std.ArrayList([]const u8) = .empty;
        defer done.deinit(p.gpa);
        var since: u64 = 0;
        while (!p.serving.stop.load(.acquire)) {
            p.table.expire() catch {}; // exports age out and are freed while the link is down too
            p.table.takeFinished(&done) catch {};
            for (done.items) |name| reap(p, name);
            done.clearRetainingCapacity();
            if (!p.attached(&since)) continue;
            // a new connection's HELLO zeroes the request word and its client counts from 1 again
            if (fabric.words.load(p.box.wordAt(fabric.layout.request_word)) == 0) p.serving.service.last = 0;
            if (p.serving.service.next(std.time.ns_per_ms) catch return) |req|
                p.serving.responder.handle(req.seq, req.payload) catch |err| p.serving.responder.failed(req.seq, err) catch return;
        }
    }

    /// Registered with the listen daemon; a dropped registration (the link went down) is retried each second.
    fn attached(p: *Produce, since: *u64) bool {
        if (p.reg) |*r| {
            if (r.alive()) return true;
            r.close();
            p.reg = null;
            log.warn("mcdma handoff: the listen daemon dropped the service on link {s}; attaching again", .{p.opts.link.?});
        }
        const now = fabric.words.nowNs();
        if (now - since.* < std.time.ns_per_s) {
            fabric.words.pause(std.time.ns_per_ms);
            return false;
        }
        since.* = now;
        p.reg = fabric.registration.Registration.attach(p.opts.link.?) catch return false;
        log.info("mcdma handoff: attached to the listen daemon on link {s}", .{p.opts.link.?});
        return true;
    }

    fn reap(p: *Produce, name: []const u8) void {
        var id: [16]u8 = undefined;
        _ = std.fmt.hexToBytes(&id, name) catch return; // a failed handoff's entry holds no export
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const got = p.held.fetchRemove(id);
        const reading = if (got) |kv| kv.value.pins > 0 else false;
        if (reading) got.?.value.gone = true; // the last read to close frees it
        p.lock.unlock();
        if (p.thread != null) _ = p.serving.responder.checksums.remove(id); // CLOSE removes it; an expired export's would stay
        if (got) |kv| if (!reading) release(p, kv.value);
    }

    fn release(p: *Produce, h: *Held) void {
        h.built.deinit(p.gpa); // the frames the responder planned at OPEN go with it
        p.gpa.destroy(h);
    }

    /// Frees exports past their time and those the decoder released (a link's serving thread runs this each ms).
    fn sweep(p: *Produce) void {
        var done: std.ArrayList([]const u8) = .empty;
        defer done.deinit(p.gpa);
        p.table.expire() catch {};
        p.table.takeFinished(&done) catch {};
        for (done.items) |name| reap(p, name);
    }

    /// The longest a reader waits for an export that is still converting.
    const read_wait_ms: u32 = 120_000;

    fn isPending(p: *Produce, id: [16]u8) bool {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        defer p.lock.unlock();
        return p.pending.contains(id);
    }

    /// One decoder's read of a held export (`GET /v1/handoff/<id>`): replies to OPEN and PULL; pinned until `close`.
    pub const Reading = struct {
        p: *Produce,
        held: *Held,
        replies: *file.Replies,

        /// the stream's length in bytes, known before the first
        pub fn size(r: *const Reading) u64 {
            return r.replies.size;
        }

        pub fn next(r: *Reading) !?[]const u8 {
            return r.replies.next();
        }

        pub fn close(r: *Reading) void {
            const p = r.p;
            const held = r.held;
            r.replies.deinit();
            p.gpa.destroy(r);
            p.unlease(held);
        }
    };

    /// The held export under `id`. `NotHeld`: never held, released or expired. `Refused`: it failed (`why` says why).
    pub fn read(p: *Produce, id: [16]u8, why: *[]const u8) error{ NotHeld, Refused, OutOfMemory, Unreadable }!*Reading {
        p.sweep();
        // an export still converting on its own thread is awaited, not reported missing
        var waited_ms: u32 = 0;
        while (p.isPending(id) and waited_ms < read_wait_ms) : (waited_ms += 1) std.Io.sleep(p.io, .fromMilliseconds(1), .awake) catch {};
        const failed_why: ?[]const u8 = if (p.table.get(id)) |e| (if (e == .failed) e.failed else null) else null;
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const held = p.held.get(id) orelse {
            p.lock.unlock();
            why.* = failed_why orelse return error.NotHeld;
            return error.Refused;
        };
        held.pins += 1;
        p.lock.unlock();
        _ = p.streams.fetchAdd(1, .acq_rel);
        errdefer p.unlease(held);
        const r = try p.gpa.create(Reading);
        errdefer p.gpa.destroy(r);
        const replies = file.Replies.open(p.gpa, &held.built.exp, p.opts.model, p.opts.shard, p.opts.shards) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.Unreadable;
        r.* = .{ .p = p, .held = held, .replies = replies };
        return r;
    }

    pub fn unlease(p: *Produce, h: *Held) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        h.pins -= 1;
        const last = h.pins == 0 and h.gone;
        p.lock.unlock();
        if (last) release(p, h);
        _ = p.streams.fetchSub(1, .acq_rel);
    }

    /// The decoder is done with `id` (`DELETE /v1/handoff/<id>`): its export is released now. False when none was held.
    pub fn discard(p: *Produce, id: [16]u8) bool {
        if (p.table.get(id) == null) return false;
        p.table.finish(id) catch return false;
        p.sweep();
        return true;
    }

    /// The allocator of the read-back pages and converted parts: file-backed past the threshold when a spill is open.
    fn big(p: *Produce) std.mem.Allocator {
        return if (p.spill) |s| s.allocator() else p.gpa;
    }

    /// The engine's state at `at` tokens, read back (the lane host's `snapshots.save` at the cut).
    pub fn save(p: *Produce, at: u64) !*Saved {
        const t0 = fabric.words.nowNs();
        const cap = try p.source.take(p.source.ptr, p.big(), at);
        const s = p.gpa.create(Saved) catch |err| {
            p.source.release(cap.ctx);
            return err;
        };
        s.* = .{ .cap = cap, .read_ns = fabric.words.nowNs() - t0 };
        return s;
    }

    pub fn drop(p: *Produce, saved: *Saved) void {
        p.source.release(saved.cap.ctx);
        p.gpa.destroy(saved);
    }

    fn dropSaved(ctx: *anyopaque) void {
        const saved: *Saved = @ptrCast(@alignCast(ctx));
        saved.owner.?.drop(saved);
    }

    /// Engine thread, a hand-off admitted for `tokens` (up to its cut): its export builds as the prompt prefills.
    pub fn begin(p: *Produce, id: [16]u8, tokens: []const u32) void {
        if (!p.opts.stream or p.opts.link == null or tokens.len == 0) return;
        if (p.spill != null) return log.info("handoff {x}: built at the cut (TF_PRODUCE_SPILL_DIR holds the buffers)", .{&id});
        const open_feed = p.source.open_feed orelse return;
        var pl: cx.Plan = undefined;
        const feed = open_feed(p.source.ptr, p.gpa, tokens.len, &pl) catch |err| return log.info("handoff {x}: built at the cut ({s})", .{ &id, @errorName(err) });
        const shape: plan.Shape = .{ .upto = tokens.len, .ds_rows = pl.ds_rows, .shard = pl.shard, .shards = pl.shards, .split = pl.shards > 1 };
        const st = stream_mod.Stream.open(p.gpa, id, tokens, shape, feed, pl.index_format, .{ .ptr = p, .touch = touchStream, .ended = streamEnded }) catch |err| {
            feed.close(feed.ptr);
            return log.info("handoff {x}: built at the cut ({s})", .{ &id, @errorName(err) });
        };
        const held = p.gpa.create(Held) catch return st.discard();
        held.* = .{ .built = st.built, .name = undefined, .pins = 1 }; // the worker's pin
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        st.user = held;
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        var placed = false;
        if (p.active.ensureUnusedCapacity(p.gpa, 1)) |_| {
            if (p.held.getOrPut(p.gpa, id)) |sl| {
                if (!sl.found_existing) {
                    sl.value_ptr.* = held;
                    p.active.putAssumeCapacity(id, st);
                    placed = true;
                }
            } else |_| {}
        } else |_| {}
        p.lock.unlock();
        if (!placed) {
            p.gpa.destroy(held);
            return st.discard(); // a reused id keeps the export it holds
        }
        p.table.addSized(id, &held.name, .{ .ready = &st.built.exp }, st.built.bytes) catch |err| {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            _ = p.held.remove(id);
            _ = p.active.remove(id);
            p.lock.unlock();
            p.gpa.destroy(held);
            log.info("handoff {x}: built at the cut ({s})", .{ &id, @errorName(err) });
            return st.discard();
        };
        _ = p.building.fetchAdd(1, .acq_rel);
        st.start();
    }

    /// Engine thread: the slot of `id`'s prompt stands at `at` tokens, its chunk done; the pages that are whole go on.
    pub fn progress(p: *Produce, id: [16]u8, at: u64) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const st = p.active.get(id);
        p.lock.unlock();
        if (st) |x| x.progress(at);
    }

    /// The engine thread ends `id`'s stream: nothing more comes from the engine (the cut took it, or the pass ended).
    fn endStream(p: *Produce, id: [16]u8) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const got = p.active.fetchRemove(id);
        p.lock.unlock();
        const st = (got orelse return).value;
        const broken = st.failedAlready();
        st.abort();
        if (!broken) p.table.finish(id) catch {}; // a pass that ended: the decoder hears "no export" below, not a stale gate
    }

    fn touchStream(ptr: *anyopaque, id: [16]u8) void {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        p.table.touch(id);
    }

    /// The worker's last word (on its thread): the export is complete, or failed and its gate says why.
    fn streamEnded(ptr: *anyopaque, st: *stream_mod.Stream, end: stream_mod.End) void {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        const id = st.built.exp.handoff;
        if (end.ok) {
            p.table.touch(id);
            log.info("handoff {x}: {d} tokens, {d} parts, {d} MiB of bf16 rows; {d} parts built while prefilling, {d} ms after the cut", .{ &id, st.built.exp.tokens.len, end.parts, end.bytes >> 20, end.streamed, end.cut_ns / std.time.ns_per_ms });
            if (p.opts.dump != null) p.dump(st.built, st.built.exp.tokens) catch |err| log.warn("handoff {x}: not written ({s})", .{ &id, @errorName(err) });
        } else log.warn("handoff {x}: the streamed export failed ({s})", .{ &id, st.reason() });
        const held: *Held = @ptrCast(@alignCast(st.user.?));
        p.unpin(held);
        _ = p.building.fetchSub(1, .acq_rel);
    }

    fn unpin(p: *Produce, h: *Held) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        h.pins -= 1;
        const last = h.pins == 0 and h.gone;
        p.lock.unlock();
        if (last) release(p, h);
    }

    /// `save` at the cut: a streamed export's state has no pool pages (the stream has them); else the plain `save`.
    pub fn saveFor(p: *Produce, id: [16]u8, at: u64) !*Saved {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const got = p.active.fetchRemove(id);
        p.lock.unlock();
        const st = (got orelse return p.save(at)).value;
        if (st.failedAlready()) {
            log.warn("mcdma handoff {x}: the streamed export failed ({s})", .{ &id, st.reason() });
            st.abort();
            return error.StreamFailed;
        }
        const t0 = fabric.words.nowNs();
        const cap = p.source.take_state.?(p.source.ptr, p.gpa, at, st.feed) catch |err| {
            st.abort();
            return err;
        };
        const saved = p.gpa.create(Saved) catch |err| {
            p.source.release(cap.ctx);
            st.abort();
            return err;
        };
        saved.* = .{ .cap = cap, .read_ns = fabric.words.nowNs() - t0, .stream = st, .owner = p };
        return saved;
    }

    /// `saved` (now the producer's) is the state after `tokens`, the prompt but its last token: exported under `id`.
    pub fn exported(p: *Produce, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        if (saved.stream) |st| return st.cut(saved.cap.state, saved, dropSaved); // the worker finishes the export
        defer p.drop(saved);
        p.record(id, tokens, saved.cap.state, saved.read_ns) catch |err| p.failed(id, @errorName(err));
    }

    /// `exported` with the conversion on its own thread (a file-only producer or a full set of converters: inline).
    pub fn exportedBg(p: *Produce, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        if (saved.stream != null) return p.exported(id, tokens, saved);
        if ((p.opts.link == null and !p.opts.http) or p.building.load(.acquire) >= max_building) return p.exported(id, tokens, saved);
        const own = p.gpa.dupe(u32, tokens) catch return p.exported(id, tokens, saved);
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const put = p.pending.put(p.gpa, id, {});
        p.lock.unlock();
        put catch {
            p.gpa.free(own);
            return p.exported(id, tokens, saved);
        };
        _ = p.building.fetchAdd(1, .acq_rel);
        const t = std.Thread.spawn(.{}, convert, .{ p, id, own, saved }) catch {
            p.finishBuild(id);
            p.gpa.free(own);
            return p.exported(id, tokens, saved);
        };
        t.detach();
    }

    fn convert(p: *Produce, id: [16]u8, tokens: []u32, saved: *Saved) void {
        p.exported(id, tokens, saved);
        p.gpa.free(tokens);
        p.finishBuild(id);
    }

    fn finishBuild(p: *Produce, id: [16]u8) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        _ = p.pending.remove(id);
        p.lock.unlock();
        _ = p.building.fetchSub(1, .acq_rel);
    }

    fn record(p: *Produce, id: [16]u8, tokens: []const u32, st: cx.State, read_ns: u64) !void {
        if (p.opts.http) p.sweep(); // what expired or was released makes room under the cap
        const t0 = fabric.words.nowNs();
        const built = try cx.build(p.big(), id, tokens, st);
        const build_ns = fabric.words.nowNs() - t0;
        var kept = false;
        defer if (!kept) built.deinit(p.gpa);
        var frames: u64 = 0;
        for (built.exp.layers) |l| frames += l.parts.len;
        log.info("handoff {x}: {d} tokens, {d} parts, {d} MiB of bf16 rows; read back {d} ms, built {d} ms", .{ &id, tokens.len, frames, built.bytes >> 20, read_ns / std.time.ns_per_ms, build_ns / std.time.ns_per_ms });
        if (p.opts.dump != null) try p.dump(built, tokens);
        if (p.opts.link == null and !p.opts.http) return;
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .built = built, .name = undefined };
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            defer p.lock.unlock();
            const slot = try p.held.getOrPut(p.gpa, id);
            if (slot.found_existing) return error.HandoffHeld; // a reused id: the export already held stays
            slot.value_ptr.* = held;
        }
        p.table.addSized(id, &held.name, .{ .ready = &built.exp }, built.bytes) catch |err| {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            _ = p.held.remove(id);
            p.lock.unlock();
            return err;
        };
        kept = true;
    }

    /// How long a reader waits on a layer-split export's producer for a frame's rows or the state before it stalls.
    pub const early_wait_ns: u64 = 120 * std.time.ns_per_s;

    /// A layer-split export's entry added now, before anything is computed, so the decoder's OPEN finds it
    pub fn openEarly(p: *Produce, id: [16]u8, tokens: []const u32, cut: u32, window: u64) !Early {
        if (p.opts.link == null and !p.opts.http) return error.SplitNeedsLink;
        if (p.opts.shards != 1) return error.SplitNeedsReplicatedPool;
        p.sweep(); // what expired or was released makes room under the cap
        const built = try cx.early(p.gpa, id, tokens, cut, window, early_wait_ns);
        var kept = false;
        defer if (!kept) built.deinit(p.gpa);
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .built = built, .name = undefined, .pins = 1 }; // the run's lease
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            defer p.lock.unlock();
            const slot = try p.held.getOrPut(p.gpa, id);
            if (slot.found_existing) return error.HandoffHeld; // a reused id: the export already held stays
            slot.value_ptr.* = held;
        }
        p.table.addSized(id, &held.name, .{ .ready = &built.exp }, built.bytes) catch |err| {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            _ = p.held.remove(id);
            p.lock.unlock();
            return err;
        };
        _ = p.streams.fetchAdd(1, .acq_rel);
        kept = true;
        log.info("handoff {x}: layer split at {d}, {d} tokens, {d} MiB held (boundary rows {d} MiB of the window)", .{ &id, cut, tokens.len, built.bytes >> 20, (@min(tokens.len, window) * split.row_bytes) >> 20 });
        return .{ .p = p, .held = held, .early = built.early.?, .id = id, .leased = true };
    }

    /// The engine's state of layers [0, cut) at `at` tokens, read back (the slot stands there
    pub fn saveCut(p: *Produce, at: u64, cut: u32) !cx.Captured {
        const take = p.source.take_cut orelse return error.SplitNeedsDevice;
        return take(p.source.ptr, p.gpa, at, cut);
    }

    /// A layer-split export under way on the engine thread.
    pub const Early = early_run.Early;

    /// `<dir>/<id>.mkvh2`, written whole under a `.part` name and renamed after `<id>.tokens.json`: present = complete.
    fn dump(p: *Produce, built: *cx.Built, tokens: []const u32) !void {
        const dir = p.opts.dump.?;
        const cwd = std.Io.Dir.cwd();
        var name_buf: [32 + file.extension.len]u8 = undefined;
        const name = file.fileName(&name_buf, built.exp.handoff);
        const final = try std.fs.path.join(p.gpa, &.{ dir, name });
        defer p.gpa.free(final);
        const part = try std.fmt.allocPrint(p.gpa, "{s}.part", .{final});
        defer p.gpa.free(part);
        const ids = try std.fmt.allocPrint(p.gpa, "{s}/{x}.tokens.json", .{ dir, &built.exp.handoff });
        defer p.gpa.free(ids);
        {
            var json: std.Io.Writer.Allocating = .init(p.gpa);
            defer json.deinit();
            try std.json.Stringify.value(tokens, .{}, &json.writer);
            try cwd.writeFile(p.io, .{ .sub_path = ids, .data = json.written() });
        }
        {
            var f = try cwd.createFile(p.io, part, .{});
            defer f.close(p.io);
            var buf: [64 << 10]u8 = undefined;
            var w = f.writerStreaming(p.io, &buf);
            try file.write(p.gpa, &w.interface, &built.exp, p.opts.model, p.opts.shard, p.opts.shards);
            try w.interface.flush();
        }
        try cwd.rename(part, cwd, final, p.io);
        log.info("handoff {x}: written to {s} (model {s})", .{ &built.exp.handoff, final, p.opts.model });
    }

    /// No state will come for `id`: the decoder is told why (a link), and the reason lands in `<id>.error` (a dump).
    pub fn failed(p: *Produce, id: [16]u8, reason: []const u8) void {
        p.endStream(id);
        const why = if (std.mem.eql(u8, reason, "HeldExportsFull")) "the producer already holds TF_PRODUCE_HELD_GIB of exports; it refuses more until the decoder closes some" else reason;
        log.warn("mcdma handoff {x}: no export ({s})", .{ &id, why });
        if (p.opts.dump) |dir| {
            if (std.fmt.allocPrint(p.gpa, "{s}/{x}.error", .{ dir, &id })) |path| {
                defer p.gpa.free(path);
                std.Io.Dir.cwd().writeFile(p.io, .{ .sub_path = path, .data = why }) catch {};
            } else |_| {}
        }
        if (p.opts.link != null or p.opts.http) p.table.add(id, "failed", .{ .failed = why }) catch {};
    }
};
