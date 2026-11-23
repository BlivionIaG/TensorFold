//! The drafter's host policy: the window's depth from the round's own measurements, and the lookup hand-off.
const std = @import("std");
const Backend = @import("backend.zig").Backend;
const mtp_cost = @import("draft/mtp_cost.zig");

var fast_argmax_cache: ?bool = null;
pub fn fastArgmax() bool {
    if (fast_argmax_cache) |v| return v;
    const s = std.c.getenv("TF_DSV41_FAST_ARGMAX");
    fast_argmax_cache = if (s) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return fast_argmax_cache.?;
}

/// The tokens that followed the last earlier occurrence of the stream's own tail, at most `out.len` of them.
pub fn lookupTokens(context: []const u32, out: []u32) usize {
    const total = context.len;
    if (total < 4 or out.len == 0) return 0;
    var n: usize = @min(total - 1, 12);
    while (n >= 2) : (n -= 1) {
        const suffix = context[total - n ..];
        var i: usize = 0;
        while (i + n < total) : (i += 1) {
            if (std.mem.eql(u32, context[i..][0..n], suffix)) {
                const start = i + n;
                const avail = @min(out.len, total - start);
                @memcpy(out[0..avail], context[start..][0..avail]);
                return avail;
            }
        }
    }
    return 0;
}

/// The hand-off's cap (the drafts a round); the default is the shipped three.
pub fn mtpDepthMax() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_DEPTH_MAX") orelse return 3;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3;
    return @min(@max(n, 1), 4);
}

pub fn maxDraftDepth(b: *const Backend) u32 {
    const trained: u32 = if (b.model.mtp_enabled and b.model.weights.dspark != null) mtpDepthMax() else 0;
    return @max(lookupDepth(), trained);
}

pub const max_lookup_drafts: u32 = 7;
var lookup_depth: ?u32 = null;
/// TF_DSV41_LOOKUP_DRAFTS=<depth>: draft that many tokens a round from the stream's own suffix matches.
pub fn lookupDepth() u32 {
    if (lookup_depth) |d| return d;
    var out: u32 = 0;
    if (std.c.getenv("TF_DSV41_LOOKUP_DRAFTS")) |v| {
        std.log.info("deepseek-v41: lookupDepth reads TF_DSV41_LOOKUP_DRAFTS={s}", .{std.mem.span(v)});
        out = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
        out = @min(out, max_lookup_drafts);
    }
    lookup_depth = out;
    return out;
}

var no_refeed: ?bool = null;
/// TF_DSV41_NO_REFEED=0 restores the re-feed; the no-re-feed port is the default.
pub fn noRefeedOn() bool {
    if (no_refeed) |v| return v;
    const raw = std.c.getenv("TF_DSV41_NO_REFEED") orelse {
        no_refeed = true;
        return true;
    };
    no_refeed = !std.mem.eql(u8, std.mem.span(raw), "0");
    return no_refeed.?;
}

var mtp_adapt: ?bool = null;
/// TF_DSV41_MTP_ADAPT=0 fixes the depth at the deepest window; unset prices the depths from the rounds.
pub fn mtpAdaptOn() bool {
    if (mtp_adapt) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT") orelse {
        mtp_adapt = true;
        return true;
    };
    mtp_adapt = !std.mem.eql(u8, std.mem.span(raw), "0");
    return mtp_adapt.?;
}

pub const unset: u32 = std.math.maxInt(u32);

/// The drafts this round's source can offer: the lookup's suffix match, else the trained head's cap.
pub const Offer = struct { src: mtp_cost.Source, have: u32 };

/// One decision a round: the `draft` and `tree` hooks of a window ask it once, so their rows agree.
pub fn mtpDepthFor(b: *Backend, slot: u32, offer: Offer) u32 {
    const cap = @min(maxDraftDepth(b), offer.have);
    if (cap == 0 or slot >= b.depth_now.len) return 0;
    if (!mtpAdaptOn()) return cap;
    if (b.depth_now[slot] != unset) return @min(b.depth_now[slot], cap);
    b.mtp.cap = cap;
    const k = @min(b.mtp.pick(offer.src), cap);
    b.depth_now[slot] = @intCast(k);
    b.src_now[slot] = offer.src;
    b.mtp.noteDepth(offer.src, k);
    if (std.c.getenv("TF_DSV41_MTP_ADAPT_DEBUG") != null) std.log.info(
        "mtp depth: slot {d} {s} have {d} depth {d} ms a token {d:.2} {d:.2} {d:.2} {d:.2} pass {d:.2}",
        .{ slot, @tagName(offer.src), offer.have, k, msPer(b, offer.src, 0), msPer(b, offer.src, 1), msPer(b, offer.src, 2), msPer(b, offer.src, 3), b.mtp.rows.draft.get() orelse 0 },
    );
    return @intCast(k);
}

/// The policy's own ms a token at depth `k` (0: nothing priced yet).
fn msPer(b: *const Backend, src: mtp_cost.Source, k: usize) f64 {
    const r = b.mtp.rate(src, k);
    return if (r > 0) 1.0 / r else 0;
}

/// The proposer's last proposal, so a round runs the draft forward once.
pub const MtpCache = struct { pending: u32, drafts: [4]u32, count: u32 = 3 };
