//! The backend's heavy vtable entries: verify, keep, draft, tree and release.
const std = @import("std");
const mtl = @import("metal");
const ds = @import("model_metal.zig");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const greedy = @import("greedy.zig");
const Backend = @import("backend.zig").Backend;
const self = Backend.self;
const slotOf = Backend.slotOf;
const put = Backend.put;
const free = Backend.free;
const draw = Backend.draw;
const noBatchInstrument = Backend.noBatchInstrument;
const batchOn = Backend.batchOn;
const batchMax = Backend.batchMax;
const penBump = Backend.penBump;
const sampleRow = Backend.sampleRow;
const policy = @import("backend_policy.zig");
const fastArgmax = policy.fastArgmax;
const lookupTokens = policy.lookupTokens;
const noRefeedOn = policy.noRefeedOn;
const mtpDepthMax = policy.mtpDepthMax;
const lookupDepth = policy.lookupDepth;
const mtpDepthFor = policy.mtpDepthFor;
const Offer = policy.Offer;

pub fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const b = self(ptr);
    // A solo round's ms is what the depth policy prices; a shared round's rows are another machine's question.
    const t0: ?f64 = if (windows.len == 1) mtl.clock.seconds() else null;
    for (windows) |w| {
        if (w.parents != null or w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
    }
    // A round wider than one batched forward (TF_DSV41_BATCH_MAX, four by default) runs as several.
    const wide = batchMax();
    if (windows.len > wide and batchOn() and noBatchInstrument()) {
        var all_one = true;
        for (windows) |w| if (w.rows() != 1) {
            all_one = false;
            break;
        };
        if (all_one) {
            var done: usize = 0;
            while (done < windows.len) {
                const n = @min(windows.len - done, wide);
                try verify(ptr, windows[done..][0..n], out[done..][0..n]);
                done += n;
            }
            return;
        }
    }
    // A shared round's plain windows are one row each.
    const batch = batchOn() and windows.len > 1 and windows.len <= wide and noBatchInstrument() and blk: {
        for (windows) |w| if (w.rows() != 1) break :blk false;
        break :blk true;
    };
    var batch_logits: ?[]const f32 = null;
    var batch_feeds: [8][]u32 = undefined;
    var batch_off: [8]u32 = undefined;
    if (batch) {
        var segs_buf: [8]ds.Seg = undefined;
        var off: u32 = 0;
        var made: usize = 0;
        var begun: usize = 0;
        errdefer for (batch_feeds[0..made]) |f| b.gpa.free(f);
        errdefer for (segs_buf[0..begun]) |sg| b.model.commitSlot(sg.slot) catch {};
        for (windows, 0..) |w, wi| {
            const slot = try b.slotOf(w.stream);
            const st = &b.model.caches[slot];
            if (w.positions.len != 1 or w.positions[0] != st.position + 1) return error.PositionMismatch;
            b.model.commitSlot(slot) catch {};
            try b.model.beginSlot(slot);
            begun += 1;
            batch_feeds[wi] = try b.gpa.alloc(u32, 1);
            made += 1;
            batch_feeds[wi][0] = w.pending;
            batch_off[wi] = off;
            segs_buf[wi] = .{ .slot = slot, .ids = batch_feeds[wi], .p0 = st.position };
            off += 1;
        }
        batch_logits = try b.model.forwardSegs(segs_buf[0..windows.len], .all);
        if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("batch fwd: windows={d} gpu={d:.3} ms", .{ windows.len, b.model.last_gpu_seconds * 1e3 });
    }
    for (windows, out, 0..) |w, o, wi| {
        const i = try b.slotOf(w.stream);
        const state = &b.model.caches[i];
        const rows = w.rows();
        var feeds: []u32 = undefined;
        var opened = false;
        if (batch) {
            feeds = batch_feeds[wi];
        } else {
            if (w.positions.len != rows) return error.WindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != state.position + r + 1) return error.PositionMismatch;
            b.model.commitSlot(i) catch {}; // a window the engine accepted whole leaves its transaction to close here
            try b.model.beginSlot(i);
            opened = true;
            feeds = try b.gpa.alloc(u32, rows);
            feeds[0] = w.pending;
            const held0 = b.held[i];
            if (w.held > held0.len) return error.WindowOutOfStep;
            @memcpy(feeds[1 .. 1 + w.held], held0[0..w.held]);
            @memcpy(feeds[1 + w.held ..], w.tokens);
            // A row that is not a token (a stale host-held draft) is a wrong guess, never an error.
            for (feeds[1..]) |*t| {
                if (t.* >= 129280) t.* = feeds[0];
            }
        }
        defer b.gpa.free(feeds);
        errdefer if (opened) b.model.commitSlot(i) catch {};
        const held = b.held[i];
        if (w.held > held.len) return error.WindowOutOfStep;
        var row: []const f32 = undefined;
        if (batch_logits) |lg| {
            row = lg[@as(usize, batch_off[wi]) * 129280 ..];
        } else {
            row = try b.model.forwardOn(i, feeds, .all);
            if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("single fwd: slot={d} rows={d} gpu={d:.3} ms", .{ i, rows, b.model.last_gpu_seconds * 1e3 });
        }
        // TF_DSV41_ROUTE_DUMP=N: the last layer's routes of the first N windows.
        if (std.c.getenv("TF_DSV41_ROUTE_DUMP")) |raw| {
            const S = struct {
                var n: u32 = 0;
            };
            const max = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 0;
            const slots = rows * 6;
            if (S.n < max and slots > 0 and slots <= 96) {
                S.n += 1;
                const v = b.model.workspace.view(.route_ids);
                const ids: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                var sizes: [96]u32 = @splat(0);
                var distinct: u32 = 0;
                for (0..slots) |q| {
                    var first = true;
                    for (0..q) |j| if (ids[j] == ids[q]) {
                        first = false;
                        break;
                    };
                    if (!first) continue;
                    var c: u32 = 0;
                    for (0..slots) |j| if (ids[j] == ids[q]) {
                        c += 1;
                    };
                    sizes[distinct] = c;
                    distinct += 1;
                }
                std.log.info("routes: rows={d} distinct={d} sizes={any}", .{ rows, distinct, sizes[0..distinct] });
            }
        }
        // A stream with a positive temperature samples every row; the greedy path below is untouched.
        const sampled_stream: ?lanes.sampling.Sampling = if (w.stream.sampling) |sp|
            (if (sp.temperature > 0) sp else null)
        else
            null;
        // Presence/frequency penalties, upstream's semantics (lanes.Penalties, the CUDA lane's tf_penalize_rows form).
        const pen_on = w.stream.penalties.active();
        if (pen_on) {
            b.pen_counts.clearRetainingCapacity();
            const pen_start = @min(w.stream.prompt_len, w.stream.context.items.len);
            for (w.stream.context.items[pen_start..]) |t| try penBump(b.gpa, &b.pen_counts, t);
            if (std.c.getenv("TF_DSV41_PEN_DEBUG") != null) std.log.info(
                "pen: context={d} prompt={d} pending={d} last={d} rows={d} distinct={d}",
                .{ w.stream.context.items.len, w.stream.prompt_len, w.pending, if (w.stream.context.items.len > 0) w.stream.context.items[w.stream.context.items.len - 1] else 0, rows, b.pen_counts.count() },
            );
        }
        for (0..rows) |r| {
            const logits_row = row[r * 129280 ..][0..129280];
            // With penalties on, row r draws from a penalized copy of its logits.
            const target: []const f32 = if (pen_on) blk: {
                if (r > 0) try penBump(b.gpa, &b.pen_counts, feeds[r]);
                if (b.pen_row.len < 129280) {
                    if (b.pen_row.len > 0) b.gpa.free(b.pen_row);
                    b.pen_row = try b.gpa.alloc(f32, 129280);
                }
                @memcpy(b.pen_row, logits_row);
                var pen_it = b.pen_counts.iterator();
                while (pen_it.next()) |e| b.pen_row[e.key_ptr.*] -= w.stream.penalties.of(e.value_ptr.*);
                break :blk b.pen_row;
            } else logits_row;
            // The row's own logits and its top-2, for the long-prompt equivalence gate.
            if (std.c.getenv("TF_DSV41_LOGIT_DUMP")) |dv| {
                const D = struct {
                    var n: u64 = 0;
                };
                const dir = std.mem.span(dv);
                var nb: [300]u8 = undefined;
                const name = std.fmt.bufPrint(&nb, "{s}/logits-{d}-{d}.bin", .{ dir, @as(u64, @intCast(w.positions[r])), D.n }) catch "logits.bin";
                D.n += 1;
                std.Io.Dir.cwd().writeFile(b.model.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(logits_row) }) catch {};
                var m1: f32 = -std.math.inf(f32);
                var m2: f32 = -std.math.inf(f32);
                var best: u32 = 0;
                for (target, 0..) |v, id| {
                    if (!std.math.isFinite(v)) continue;
                    if (v > m1) {
                        m2 = m1;
                        m1 = v;
                        best = @intCast(id);
                    } else if (v > m2) m2 = v;
                }
                std.log.info("logitdump {s} top1={d} m1={d:.7} m2={d:.7} margin={d:.7}", .{ name, best, m1, m2, m1 - m2 });
            }
            if (sampled_stream) |sp| {
                o.sampled[r] = try sampleRow(target, sp, w.positions[r]);
            } else if (fastArgmax()) {
                const best = greedy.argmax(target);
                if (!std.math.isFinite(target[best])) return error.NonfiniteLogits;
                o.sampled[r] = @intCast(best);
            } else {
                var best: u32 = 0;
                var maximum = -std.math.inf(f32);
                for (target, 0..) |value, id| {
                    if (!std.math.isFinite(value)) return error.NonfiniteLogits;
                    if (value > maximum) {
                        maximum = value;
                        best = @intCast(id);
                    }
                }
                o.sampled[r] = best;
            }
        }
        // The target's pick per row, joined to the proposal's top-2 by the round counter.
        if (std.c.getenv("TF_DSV41_TREE_PRICE") != null) {
            std.log.info("treeprice verify round {d} rows {d} held {d} pending {d} sampled {any}", .{ ds.price_round, rows, w.held, w.pending, o.sampled[0..rows] });
            ds.price_round += 1;
        }
        // The round's own ms by its rows: the depth policy's cost side.
        if (t0) |t| {
            const ms = (mtl.clock.seconds() - t) * 1e3;
            if (std.c.getenv("TF_DSV41_MTP_ADAPT_DEBUG") != null) std.log.info(
                "mtp round: slot {d} rows {d} ms {d:.2} src {s} depth {d} shared {any}",
                .{ i, rows, ms, @tagName(b.src_now[i]), b.depth_now[i], w.stream.shared_round },
            );
            b.mtp.noteRound(@intCast(rows), ms);
        }
        // The window's fed tokens after the pending one landed: the verified drafts, GLM's contract.
        if (o.drafts.len > 0) @memcpy(o.drafts, feeds[1 .. 1 + o.drafts.len]);
        @memcpy(b.last_sampled[0..rows], o.sampled[0..rows]);
    }
}

pub fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const b = self(ptr);
    for (windows, paths) |w, path| {
        const i = try b.slotOf(w.stream);
        const rows = w.rows();
        const kept = path.len;
        // The drafts the window landed: a chain's kept rows past its pending one teach the policy its positions.
        if (w.parents == null and w.masks.len == 0 and b.depth_now[i] != policy.unset and rows > 1)
            b.mtp.noteLanded(b.src_now[i], kept -| 1, rows - 1);
        b.depth_now[i] = policy.unset;
        if (kept >= rows) continue; // the whole window landed
        try b.model.rollbackSlot(i);
        // The ring rows the window's rejected rows published are aliases the next forward can still read.
        try b.model.restoreRing(i, @intCast(kept), @intCast(rows));
        if (kept == 0) continue; // the rollback closed the transaction
        var feeds = try b.gpa.alloc(u32, kept);
        defer b.gpa.free(feeds);
        feeds[0] = w.pending;
        const held = b.held[i];
        const held_kept = @min(w.held, kept - 1);
        if (held_kept > held.len) return error.WindowOutOfStep;
        @memcpy(feeds[1 .. 1 + held_kept], held[0..held_kept]);
        @memcpy(feeds[1 + held_kept ..], w.tokens[0 .. kept - 1 - held_kept]);
        for (feeds[1..]) |*t| {
            if (t.* >= 129280) t.* = feeds[0];
        }
        if (noRefeedOn()) {
            // The no-re-feed port.
            try b.model.portKeep(i, @intCast(kept));
            b.model.commitSlot(i) catch {};
            continue;
        }
        // Diagnostic (TF_DSV41_CLEAR_STALE=1): the rejected rows the window wrote stay in the ring; zero them.
        if (std.c.getenv("TF_DSV41_CLEAR_PARTIALS") != null) b.model.caches[i].clearPartials();
        if (std.c.getenv("TF_DSV41_CLEAR_STALE") != null) {
            const start = b.model.caches[i].position;
            b.model.caches[i].clearRows(start + kept, start + rows);
        }
        _ = b.model.forwardOn(i, feeds, .last) catch |e| {
            std.log.err("keep: refeed failed: {s}", .{@errorName(e)});
            return e;
        };
        // The refeed's own forward closed the transaction; commit only if one is still open.
        b.model.commitSlot(i) catch {};
    }
}

/// The next window's drafts: the stream's own suffix match when it has one, else the trained head's chain.
pub fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const b = self(ptr);
    for (requests) |r| {
        // What this call holds, until drafts land: the engine plans its window from this count.
        r.stream.proposal_depth = 0;
        const i = b.by.get(r.stream) orelse continue;
        if (b.held[i].len > 0) b.gpa.free(b.held[i]);
        b.held[i] = &.{};
        if (std.c.getenv("TF_DSV41_MTP_ADAPT_DEBUG") != null) std.log.info(
            "mtp ask: slot {d} engine {d} depth_now {d} rows {d}",
            .{ i, r.depth, b.depth_now[i], if (r.rows) |x| x.len else 0 },
        );
        if (r.depth == 0) continue;
        if (std.c.getenv("TF_DSV41_DRAFT_COPY") != null) {
            // Diagnostic: every draft repeats the pending token, so a two-row window is [pending, pending].
            const out0 = try b.gpa.alloc(u32, r.depth);
            const pend = r.stream.pending orelse 0;
            for (out0) |*t| t.* = pend;
            b.held[i] = out0;
            r.stream.proposal_depth = r.depth;
            continue;
        }
        if (!policy.wantsDrafts(b, @intCast(i))) continue;
        const ctx = r.stream.context.items;
        // A matching CPU suffix arm comes before a trained forward: it costs no pass.
        var copied: [policy.max_lookup_drafts]u32 = undefined;
        const got = lookupTokens(ctx, copied[0..lookupDepth()]);
        const last_tok: ?u32 = r.stream.pending orelse (if (ctx.len > 0) ctx[ctx.len - 1] else null);
        const trained = b.model.mtp_enabled and b.model.weights.dspark != null and last_tok != null;
        if (got == 0 and !trained) continue;
        const offer: Offer = if (got > 0)
            .{ .src = .lookup, .have = @intCast(got) }
        else
            .{ .src = .trained, .have = mtpDepthMax() };
        const depth = mtpDepthFor(b, @intCast(i), offer);
        if (depth == 0) continue;
        // The pack's own depth decides, whatever the engine asked for; the count it holds rides the stream.
        const n: usize = @min(depth, if (got > 0) got else mtpDepthMax());
        const out = try b.gpa.alloc(u32, n);
        if (got > 0) {
            @memcpy(out, copied[0..n]);
            b.held[i] = out;
            r.stream.proposal_depth = @intCast(n);
            continue;
        }
        const t0 = mtl.clock.seconds();
        const drafts = b.model.proposeTokens(@intCast(i), last_tok.?, @intCast(n)) catch |err| {
            std.log.warn("mtp: drafts unavailable ({s})", .{@errorName(err)});
            b.gpa.free(out);
            continue;
        };
        b.mtp.noteDraft((mtl.clock.seconds() - t0) * 1e3);
        @memcpy(out, drafts[0..n]);
        b.held[i] = out;
        r.stream.proposal_depth = @intCast(n);
        // The round's other hook (tree) asks for the same drafts; keep them here.
        if (i < b.mtp_cache.len) b.mtp_cache[i] = .{ .pending = last_tok.?, .drafts = drafts, .count = @intCast(n) };
    }
}

/// The next window's drafts as the engine's held tokens: the lookup's match, else the trained head's chain.
pub fn tree(ptr: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
    const b = self(ptr);
    // The proposer's forward needs the lane's slot (the trunk's key ring lives there).
    const slot_i = b.by.get(s) orelse return null;
    if (!policy.wantsDrafts(b, slot_i)) return null;
    var copied: [policy.max_lookup_drafts]u32 = undefined;
    const got = lookupTokens(s.context.items, copied[0..lookupDepth()]);
    const trained = b.model.mtp_enabled and b.model.weights.dspark != null and s.pending != null;
    if (got == 0 and !trained) return null;
    const offer: Offer = if (got > 0)
        .{ .src = .lookup, .have = @intCast(got) }
    else
        .{ .src = .trained, .have = mtpDepthMax() };
    const depth = mtpDepthFor(b, slot_i, offer);
    if (depth == 0) return null;
    if (got > 0) {
        const n: usize = @min(depth, got);
        return .{ .count = @intCast(n), .tokens = try gpa.dupe(u32, copied[0..n]) };
    }
    // The round's other hook (draft) already ran the proposer's forward for this pending token.
    const n: usize = @min(depth, mtpDepthMax());
    var drafts: [4]u32 = undefined;
    var cached = false;
    if (slot_i < b.mtp_cache.len) if (b.mtp_cache[slot_i]) |c| if (c.pending == s.pending.? and c.count >= n) {
        drafts = c.drafts;
        cached = true;
    };
    if (!cached) {
        drafts = b.model.proposeTokens(@intCast(slot_i), s.pending.?, @intCast(n)) catch |err| {
            std.log.warn("mtp: proposer unavailable ({s})", .{@errorName(err)});
            return null;
        };
    }
    const out = try gpa.alloc(u32, n);
    @memcpy(out, drafts[0..n]);
    return .{ .count = @intCast(n), .tokens = out };
}

pub fn release(ptr: *anyopaque, s: *Stream) void {
    const b = self(ptr);
    s.proposal_depth = null;
    if (b.by.get(s)) |i| {
        if (b.held[i].len > 0) b.gpa.free(b.held[i]);
        b.held[i] = &.{};
    }
    const entry = b.by.fetchRemove(s) orelse return;
    if (entry.value < b.depth_now.len) b.depth_now[entry.value] = policy.unset;
    // A request's workload is over: its chances start over, the machine's row costs stay.
    b.mtp.reset();
    b.model.resetSlot(entry.value) catch {};
}
