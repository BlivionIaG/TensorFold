//! The placement planner: a Zig port of `plan()` in docs/proposals/metal-tb4-split-moe-probe/placement_model.py.
//! Pure arithmetic over the model geometry, the Mac, the external device and the link; no tensors, no devices.
//! `plan` lists every placement the model admits with a predicted decode time and memory use; `choose` implements `auto`.
//! Decode time for one stream is `T = sum over devices (bytes read / (peak * eff) + launches * t_launch) + hops`.
const std = @import("std");
const geometry = @import("geometry.zig");
const placement = @import("placement.zig");
const receipt = @import("receipt.zig");

const Allocator = std.mem.Allocator;
const Model = geometry.Model;
const Owner = placement.Owner;

const GiB: f64 = @floatFromInt(geometry.GiB);

/// Recurrent state is read and written once per token, so it counts twice in the bytes a token moves (by construction of the
/// recurrence; the 120 MiB a token for the A3B is not separately measured).
pub const state_rw: f64 = 2.0;
/// Kernel launches the head adds per token on a launch-bound device (assumed, not yet verified on a card).
pub const head_dispatches: f64 = 3;
/// The Mac decodes at 0.62 of its peak bandwidth end to end (measured on an M4 Pro: 91.6 tok/s on the A3B and 66 tok/s on
/// the 4B, 164 and 180 GB/s of weights and state against the 273 GB/s peak). Use it when only a peak bandwidth is known.
pub const mac_efficiency: f64 = 0.62;
/// Hop external device to Metal, in nanoseconds: flag visible to the host 5 us, event wake of a pre-committed command
/// buffer 65 us (an idle Mac), flush 20 us (assumed). The probe on an RX Vega 56 measured 159 us at p50 (flag 10 us, wake
/// 149 us, p99 479 us) while the card was in use: replace it with the probe receipt's number; this is only the default.
pub const assumed_hop_to_host_ns: f64 = 90_000;
/// Hop Metal to external device in nanoseconds: host store and poll on the device side. The probe measured 10.5 us at p50
/// (28.5 us at p99) on an RX Vega 56; this default is kept for a run with no receipt.
pub const assumed_hop_to_device_ns: f64 = 15_000;
/// A resident decode still returns a token id to the Mac and takes a sampling bookkeeping step (assumed).
const resident_overhead_s: f64 = 30e-6;
/// Reverse split: the Mac side of one crossing, command-buffer end to the external device's wake (assumed, not yet measured).
const reverse_to_device_s: f64 = 45e-6;
/// Launches the reverse split adds on the external device per offloaded layer (gather, three projections; assumed).
const reverse_launches: f64 = 4;
/// Timeouts written into a chosen placement (assumed; the first card run replaces them).
pub const default_timeouts: placement.Timeouts = .{ .hop = 500, .step = 30_000 };

pub const Request = struct {
    context: u32,
    streams: u32 = 1,
    /// Replaces the Mac's working-set budget (emulates a smaller Mac).
    mac_budget_override: ?u64 = null,
};

pub const Mac = struct {
    /// `recommendedMaxWorkingSetSize` in bytes.
    max_working_set: u64,
    peak_bandwidth: f64,
    efficiency: f64,

    /// A Mac whose end-to-end decode bandwidth was measured: the bandwidth is taken as 1.0 of itself.
    pub fn fromReceipt(r: receipt.Receipt) Mac {
        return .{ .max_working_set = r.mac.max_working_set_bytes.value, .peak_bandwidth = r.mac.decode_bandwidth_bps.value, .efficiency = 1.0 };
    }
};

pub const External = struct {
    name: []const u8,
    vram_usable: u64,
    peak_bandwidth: f64,
    efficiency: f64,
    launch_ns: f64,
    act: receipt.Act,

    pub fn fromReceipt(r: receipt.Receipt) External {
        const d = r.device;
        return .{ .name = d.name, .vram_usable = d.vram_bytes.value, .peak_bandwidth = d.peak_bandwidth_bps.value, .efficiency = d.decode_efficiency.value, .launch_ns = d.launch_ns.value, .act = d.act };
    }
};

pub const Link = struct {
    hop_to_host_ns: f64,
    hop_to_device_ns: f64,

    /// The assumed hop costs (see `assumed_hop_to_host_ns`); not for use once a receipt exists.
    pub const assumed: Link = .{ .hop_to_host_ns = assumed_hop_to_host_ns, .hop_to_device_ns = assumed_hop_to_device_ns };

    /// The receipt's median hop times.
    pub fn fromReceipt(r: receipt.Receipt) Link {
        return .{ .hop_to_host_ns = r.link.hop_to_host_ns.p50, .hop_to_device_ns = r.link.hop_to_device_ns.p50 };
    }
};

/// Where everything lives, independent of the device's settings.
pub const Layout = struct {
    layers: []const Owner,
    expert_layers: []const u32,
    embedding: Owner,
    head: Owner,
};

pub const Plan = struct {
    mode: placement.Mode,
    /// Seconds a token takes: the serial sum, or for the pipelined split the reciprocal of its throughput bound.
    t: f64,
    tok_s: f64,
    fits: bool,
    /// PR 1 executes all_metal, resident and the dense layer split; the rest is information for `tensorfold info`.
    executable: bool,
    /// Streams in flight the predicted `tok_s` needs (2 for the pipelined split).
    min_streams: u32,
    layout: Layout,
    vram_bytes: u64,
    mac_bytes: u64,
    prefill: placement.Prefill,
    /// Whether the Mac holds the whole model, so `fallback: metal` is valid for this plan.
    fallback_ok: bool,
    note: []const u8,

    /// The file `auto` writes: this plan's layout, the device settings, timeouts and the prediction.
    pub fn toPlacement(p: Plan, device: placement.Device, req: Request, fallback: placement.Fallback) error{FallbackImpossible}!placement.Placement {
        if (fallback == .metal and !p.fallback_ok) return error.FallbackImpossible;
        return .{
            .device = device,
            .layers = p.layout.layers,
            .expert_layers = p.layout.expert_layers,
            .embedding = p.layout.embedding,
            .head = p.layout.head,
            .prefill = p.prefill,
            .fallback = fallback,
            .timeouts_ms = default_timeouts,
            .prediction = .{ .mode = p.mode, .decode_tok_s = p.tok_s, .streams = req.streams, .context = req.context },
        };
    }
};

/// Bytes needed against bytes available for one quantity.
pub const Need = struct { needed: u64, available: u64 };

/// What was missing when nothing fits: the Mac alone, the external device alone, or both together (the model, its caches
/// and the workspace of a second device against both budgets).
pub const Shortfall = struct {
    mac: Need,
    vram: Need,
    combined: Need,
    /// Some placement fits but is not executable yet; `allow_unexecutable` would pick it.
    unexecutable_fits: bool,
};

pub const Candidates = struct {
    items: []const Plan,
    mac_budget: u64,
    vram_usable: u64,
    streams: u32,
    scratch: u64,

    pub fn get(c: Candidates, mode: placement.Mode) ?Plan {
        for (c.items) |p| if (p.mode == mode) return p;
        return null;
    }

    pub fn shortfall(c: Candidates) Shortfall {
        const all = c.get(.all_metal).?;
        const res = c.get(.resident).?;
        var unexec = false;
        for (c.items) |p| unexec = unexec or (p.fits and !p.executable);
        return .{
            .mac = .{ .needed = all.mac_bytes, .available = c.mac_budget },
            .vram = .{ .needed = res.vram_bytes, .available = c.vram_usable },
            .combined = .{ .needed = all.mac_bytes + c.scratch, .available = c.mac_budget + c.vram_usable },
            .unexecutable_fits = unexec,
        };
    }
};

const Dev = struct { mem: u64, bandwidth: f64, launch_s: f64 };

fn f(x: u64) f64 {
    return @floatFromInt(x);
}

/// Seconds to decode one token through layers `[lo, hi)` (and the head) on a device that reads `bandwidth` bytes a second.
fn stageSeconds(m: Model, lo: usize, hi: usize, d: Dev, ctx: u64, head: bool) f64 {
    var bytes: f64 = 0;
    var launches: f64 = 0;
    for (m.layers[lo..hi]) |l| {
        bytes += f(l.dense + l.experts_read + l.kv_per_tok * ctx) + state_rw * f(l.state);
        launches += f(l.dispatches);
    }
    if (head) {
        bytes += f(m.headRead());
        launches += head_dispatches;
    }
    return bytes / d.bandwidth + launches * d.launch_s;
}

/// Seconds a whole token takes on one device that reads `bandwidth` bytes a second with `launch_ns` per launch; for
/// calibrating the model against a single-device measurement.
pub fn decodeSeconds(m: Model, bandwidth: f64, launch_ns: f64, ctx: u32) f64 {
    return stageSeconds(m, 0, m.layers.len, .{ .mem = 0, .bandwidth = bandwidth, .launch_s = launch_ns * 1e-9 }, ctx, true);
}

/// Resident bytes of layers `[lo, hi)` with their KV and state for `streams` streams, plus workspace, embedding and head.
fn footprint(m: Model, lo: usize, hi: usize, ctx: u64, streams: u64, head: bool, embed: bool) u64 {
    var w: u64 = m.scratch;
    for (m.layers[lo..hi]) |l| w += l.resident + l.experts + l.kv_per_tok * ctx * streams + l.state * streams;
    if (embed) w += m.embed;
    if (head) w += m.head;
    return w;
}

fn expertBytes(m: Model, lo: usize, hi: usize) u64 {
    var s: u64 = 0;
    for (m.layers[lo..hi]) |l| s += l.experts;
    return s;
}

fn expertRead(m: Model, lo: usize, hi: usize) f64 {
    var s: f64 = 0;
    for (m.layers[lo..hi]) |l| s += f(l.experts_read);
    return s;
}

fn owners(a: Allocator, n: usize, first: Owner, k: usize, second: Owner) Allocator.Error![]const Owner {
    const out = try a.alloc(Owner, n);
    for (out, 0..) |*o, i| o.* = if (i < k) first else second;
    return out;
}

fn upTo(a: Allocator, n: usize) Allocator.Error![]const u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*o, i| o.* = @intCast(i);
    return out;
}

const Split = struct { k: usize, vram: u64, mac: u64, tg: f64, tm: f64, t: f64 };

/// Every placement the model admits, in the order all-Metal, resident, layer split (best serial), layer split (best for 2+
/// streams), expert split, reverse expert split. Slices and notes are allocated from `arena`.
pub fn plan(arena: Allocator, model: Model, req: Request, mac: Mac, ext: External, link: Link) Allocator.Error!Candidates {
    const n = model.layers.len;
    const ctx: u64 = req.context;
    const streams: u64 = req.streams;
    const budget = req.mac_budget_override orelse mac.max_working_set;
    const md: Dev = .{ .mem = budget, .bandwidth = mac.peak_bandwidth * mac.efficiency, .launch_s = 0 };
    const gd: Dev = .{ .mem = ext.vram_usable, .bandwidth = ext.peak_bandwidth * ext.efficiency, .launch_s = ext.launch_ns * 1e-9 };
    const to_mac = link.hop_to_host_ns * 1e-9;
    const to_gpu = link.hop_to_device_ns * 1e-9;
    const routed = model.hasExperts();
    var out: std.ArrayList(Plan) = .empty;

    // all-Metal
    const t_mac = stageSeconds(model, 0, n, md, ctx, true);
    const w_mac = footprint(model, 0, n, ctx, streams, true, true);
    const mac_fits = w_mac <= budget;
    try out.append(arena, .{
        .mode = .all_metal,
        .t = t_mac,
        .tok_s = 1 / t_mac,
        .fits = mac_fits,
        .executable = true,
        .min_streams = 1,
        .layout = .{ .layers = try owners(arena, n, .metal, n, .metal), .expert_layers = &.{}, .embedding = .metal, .head = .metal },
        .vram_bytes = 0,
        .mac_bytes = w_mac,
        .prefill = .metal,
        .fallback_ok = mac_fits,
        .note = try std.fmt.allocPrint(arena, "Mac {d:.1} GiB", .{f(w_mac) / GiB}),
    });

    // resident: the external device runs everything; the Mac only hosts tokenizer and sampling bookkeeping
    const w_res = footprint(model, 0, n, ctx, streams, true, true);
    const t_res = stageSeconds(model, 0, n, gd, ctx, true) + resident_overhead_s;
    try out.append(arena, .{
        .mode = .resident,
        .t = t_res,
        .tok_s = 1 / t_res,
        .fits = w_res <= gd.mem,
        .executable = true,
        .min_streams = 1,
        .layout = .{ .layers = try owners(arena, n, .external, n, .external), .expert_layers = if (routed) try upTo(arena, n) else &.{}, .embedding = .external, .head = .external },
        .vram_bytes = w_res,
        .mac_bytes = 0,
        .prefill = if (mac_fits) .metal else .external,
        .fallback_ok = mac_fits,
        .note = try std.fmt.allocPrint(arena, "VRAM {d:.1} GiB", .{f(w_res) / GiB}),
    });

    // layer split: the external device takes the first k layers and the embedding, the Mac the rest and the head; one hop each way
    var best: ?Split = null;
    var best_pipe: ?Split = null;
    var best_rate: f64 = 0;
    for (1..n) |k| {
        const vram = footprint(model, 0, k, ctx, streams, false, true);
        const mac_w = footprint(model, k, n, ctx, streams, true, false);
        if (vram > gd.mem or mac_w > budget) continue;
        const tg = stageSeconds(model, 0, k, gd, ctx, false);
        const tm = stageSeconds(model, k, n, md, ctx, true);
        const s: Split = .{ .k = k, .vram = vram, .mac = mac_w, .tg = tg, .tm = tm, .t = tg + tm + to_mac + to_gpu };
        if (best == null or s.t < best.?.t) best = s;
        const rate = 1 / @max(tg, tm);
        if (best_pipe == null or rate > best_rate) {
            best_pipe = s;
            best_rate = rate;
        }
    }
    if (best) |s| try out.append(arena, try splitPlan(arena, model, .layer_split, s, s.t, routed, mac_fits));
    if (best_pipe) |s| try out.append(arena, try splitPlan(arena, model, .layer_split_pipelined, s, 1 / best_rate, routed, mac_fits));

    if (routed) {
        // expert split (design A): the external device runs everything except the routed experts of the layers it cannot hold
        const dense_w = footprint(model, 0, n, ctx, streams, true, true) - expertBytes(model, 0, n);
        var best_a: ?Plan = null;
        for (0..n + 1) |r| {
            const need = dense_w + expertBytes(model, 0, r);
            const mac_w = expertBytes(model, r, n);
            if (need > gd.mem or mac_w > budget) continue;
            var t = stageSeconds(model, 0, n, gd, ctx, true);
            // layer i sends x and the ids; the Mac streams the top-k experts; the result comes back
            t += f(@intCast(n - r)) * (to_mac + to_gpu);
            t += expertRead(model, r, n) / md.bandwidth - expertRead(model, r, n) / gd.bandwidth;
            if (best_a == null or t < best_a.?.t) best_a = .{
                .mode = .expert_split,
                .t = t,
                .tok_s = 1 / t,
                .fits = true,
                .executable = false,
                .min_streams = 1,
                .layout = .{ .layers = try owners(arena, n, .external, n, .external), .expert_layers = try upTo(arena, r), .embedding = .external, .head = .external },
                .vram_bytes = need,
                .mac_bytes = mac_w,
                .prefill = .split,
                .fallback_ok = mac_fits,
                .note = try std.fmt.allocPrint(arena, "VRAM {d:.1} GiB, {d} expert layers resident, {d} crossings", .{ f(need) / GiB, r, n - r }),
            };
        }
        if (best_a) |p| try out.append(arena, p);

        // reverse: the Mac keeps attention, dense weights and KV; the external device holds the routed experts of as many layers as fit
        // as in the python model, the Mac's own workspace is not counted on this side
        const mac_dense = footprint(model, 0, n, ctx, streams, true, true) - expertBytes(model, 0, n) - model.scratch;
        var best_r: ?Plan = null;
        for (1..n + 1) |r| {
            const need = expertBytes(model, 0, r) + model.scratch;
            const mac_w = mac_dense + expertBytes(model, r, n);
            if (need > gd.mem or mac_w > budget) continue;
            var t = stageSeconds(model, 0, n, md, ctx, true) - expertRead(model, 0, r) / md.bandwidth;
            // per layer: Mac to device (45 us, of which the assumed 15 us doorbell is replaced by the link's), device to Mac, the read on the device
            t += f(@intCast(r)) * (reverse_to_device_s + to_gpu - assumed_hop_to_device_ns * 1e-9 + to_mac + reverse_launches * gd.launch_s);
            t += expertRead(model, 0, r) / gd.bandwidth;
            if (best_r == null or t < best_r.?.t) best_r = .{
                .mode = .reverse_expert_split,
                .t = t,
                .tok_s = 1 / t,
                .fits = true,
                .executable = false,
                .min_streams = 1,
                .layout = .{ .layers = try owners(arena, n, .metal, n, .metal), .expert_layers = try upTo(arena, r), .embedding = .metal, .head = .metal },
                .vram_bytes = need,
                .mac_bytes = mac_w,
                .prefill = .split,
                .fallback_ok = mac_fits,
                .note = try std.fmt.allocPrint(arena, "VRAM {d:.1} GiB, {d} expert layers on the external device, {d} crossings", .{ f(need) / GiB, r, 2 * r }),
            };
        }
        if (best_r) |p| try out.append(arena, p);
    }
    return .{ .items = out.items, .mac_budget = budget, .vram_usable = ext.vram_usable, .streams = req.streams, .scratch = model.scratch };
}

fn splitPlan(arena: Allocator, model: Model, mode: placement.Mode, s: Split, t: f64, routed: bool, mac_fits: bool) Allocator.Error!Plan {
    const n = model.layers.len;
    return .{
        .mode = mode,
        .t = t,
        .tok_s = 1 / t,
        .fits = true,
        .executable = !routed,
        .min_streams = if (mode == .layer_split_pipelined) 2 else 1,
        .layout = .{ .layers = try owners(arena, n, .external, s.k, .metal), .expert_layers = if (routed) try upTo(arena, s.k) else &.{}, .embedding = .external, .head = .metal },
        .vram_bytes = s.vram,
        .mac_bytes = s.mac,
        .prefill = if (mac_fits) .metal else .split,
        .fallback_ok = mac_fits,
        .note = try std.fmt.allocPrint(arena, "{d} layers on the external device, VRAM {d:.1} GiB, Mac {d:.1} GiB; stages {d:.1}+{d:.1} ms", .{ s.k, f(s.vram) / GiB, f(s.mac) / GiB, s.tg * 1e3, s.tm * 1e3 }),
    };
}

/// Routed-expert bytes read per layer per token at which the reverse split starts to pay (the experts' read-time saving on
/// the external device equals its two crossings); null when the external device's decode bandwidth is below the Mac's.
pub fn reverseBreakEvenBytes(mac: Mac, ext: External, link: Link) ?f64 {
    const saving = 1 / (mac.peak_bandwidth * mac.efficiency) - 1 / (ext.peak_bandwidth * ext.efficiency);
    if (saving <= 0) return null;
    const crossings = reverse_to_device_s + link.hop_to_device_ns * 1e-9 + link.hop_to_host_ns * 1e-9 + reverse_launches * ext.launch_ns * 1e-9;
    return crossings / saving;
}

pub const ChooseOptions = struct {
    /// Allow placements PR 1 cannot execute (expert splits, a MoE layer split); for reporting only.
    allow_unexecutable: bool = false,
};

/// `auto`: the fastest fitting placement the request can use (the pipelined split only with two or more streams). All-Metal
/// is a candidate, so a placement predicted slower than all-Metal is never chosen while all-Metal fits; ties keep the
/// earlier (fewer devices, fewer crossings). Returns `DoesNotFit` when none fits; `Candidates.shortfall` says what is missing.
pub fn choose(c: Candidates, options: ChooseOptions) error{DoesNotFit}!Plan {
    var best: ?Plan = null;
    for (c.items) |p| {
        if (!p.fits or p.min_streams > c.streams) continue;
        if (!p.executable and !options.allow_unexecutable) continue;
        if (best == null or p.tok_s > best.?.tok_s) best = p;
    }
    return best orelse error.DoesNotFit;
}

test {
    _ = @import("planner_test.zig");
}
