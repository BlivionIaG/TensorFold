//! Planner tests: the numbers of placement_results.txt (tok/s to 1% or the table's own rounding, bytes to its 0.1 GiB), the
//! calibration lines, the reverse break-even, and the `auto` rules.
const std = @import("std");
const geometry = @import("geometry.zig");
const fx = @import("fixtures.zig");
const placement = @import("placement.zig");
const planner = @import("planner.zig");

const Mode = placement.Mode;
const GiB: f64 = @floatFromInt(geometry.GiB);

const mac: planner.Mac = .{ .max_working_set = @intFromFloat(37.4 * GiB), .peak_bandwidth = 273e9, .efficiency = planner.mac_efficiency };

fn profile(name: []const u8, vram_gib: f64, peak: f64, eff: f64, launch_us: f64) planner.External {
    return .{ .name = name, .vram_usable = @intFromFloat(vram_gib * GiB), .peak_bandwidth = peak, .efficiency = eff, .launch_ns = launch_us * 1e3, .act = .f16 };
}

const profiles = [_]planner.External{
    profile("RX Vega 56 8 GiB", 7.9, 410e9, 0.37, 4.4),
    profile("RX 7900 XTX 24 GiB", 23.0, 960e9, 0.50, 2.5),
    profile("RTX 5090 32 GiB", 31.0, 1792e9, 0.55, 1.5),
};

/// One row of placement_results.txt: `n` is the layers on the external device (layer split) or the expert layers it holds;
/// 0 means not checked. `vram` and `mac` are GiB, 0 when the row does not print them.
const Row = struct { mode: Mode, fits: bool = true, tok: f64 = 0, n: u32 = 0, vram: f64 = 0, mac: f64 = 0 };

const Case = struct { model: u8, profile: u8, ctx: u32, budget_gib: u8 = 0, rows: []const Row };

fn near(expected: f64, actual: f64, rel: f64, abs: f64) !void {
    try std.testing.expect(@abs(expected - actual) <= @max(expected * rel, abs));
}

fn checkCase(c: Case) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const m = try geometry.fromConfigJson(arena.allocator(), if (c.model == 0) fx.a3b_json else fx.d4_json);
    const budget: ?u64 = if (c.budget_gib == 0) null else @as(u64, c.budget_gib) * geometry.GiB;
    const cands = try planner.plan(arena.allocator(), m, .{ .context = c.ctx, .mac_budget_override = budget }, mac, profiles[c.profile], .assumed);
    for (c.rows) |row| {
        errdefer std.debug.print("case model {d} profile {d} ctx {d} mode {s}\n", .{ c.model, c.profile, c.ctx, @tagName(row.mode) });
        const p = cands.get(row.mode) orelse return error.MissingCandidate;
        try std.testing.expectEqual(row.fits, p.fits);
        if (row.tok > 0) try near(row.tok, p.tok_s, 0.01, 0.51);
        if (row.n > 0) {
            const on_ext = if (row.mode == .layer_split or row.mode == .layer_split_pipelined) p.layout.layers.len - countMetal(p) else p.layout.expert_layers.len;
            try std.testing.expectEqual(row.n, @as(u32, @intCast(on_ext)));
        }
        if (row.vram > 0) try near(row.vram, @as(f64, @floatFromInt(p.vram_bytes)) / GiB, 0.01, 0.051);
        if (row.mac > 0) try near(row.mac, @as(f64, @floatFromInt(p.mac_bytes)) / GiB, 0.01, 0.051);
    }
}

fn countMetal(p: planner.Plan) usize {
    var n: usize = 0;
    for (p.layout.layers) |o| n += @intFromBool(o == .metal);
    return n;
}

test "A3B on the Vega card, 8k and 128k" {
    try checkCase(.{ .model = 0, .profile = 0, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 86, .mac = 18.8 },
        .{ .mode = .resident, .fits = false, .vram = 18.8 },
        .{ .mode = .layer_split, .tok = 84, .n = 1, .vram = 1.1, .mac = 18.1 },
        .{ .mode = .layer_split_pipelined, .tok = 131, .n = 16, .vram = 7.8, .mac = 11.4 },
        .{ .mode = .expert_split, .tok = 52, .n = 14, .vram = 7.8 },
        .{ .mode = .reverse_expert_split, .tok = 85, .n = 1, .vram = 0.8 },
    } });
    try checkCase(.{ .model = 0, .profile = 0, .ctx = 131072, .rows = &.{
        .{ .mode = .all_metal, .tok = 38, .mac = 21.1 },
        .{ .mode = .resident, .fits = false, .vram = 21.1 },
        .{ .mode = .layer_split, .tok = 37, .n = 1, .vram = 1.1, .mac = 20.4 },
        .{ .mode = .layer_split_pipelined, .tok = 54, .n = 14, .vram = 7.6, .mac = 13.9 },
        .{ .mode = .expert_split, .tok = 27, .n = 8, .vram = 7.6 },
        .{ .mode = .reverse_expert_split, .tok = 38, .n = 1, .vram = 0.8 },
    } });
}

test "A3B on a 7900 XTX class card, 8k and 128k" {
    try checkCase(.{ .model = 0, .profile = 1, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 86, .mac = 18.8 },
        .{ .mode = .resident, .tok = 158, .vram = 18.8 },
        .{ .mode = .layer_split, .tok = 131, .n = 39, .vram = 18.1, .mac = 1.1 },
        .{ .mode = .layer_split_pipelined, .tok = 234, .n = 30, .vram = 14.0, .mac = 5.1 },
        .{ .mode = .expert_split, .tok = 159, .n = 40, .vram = 18.8 },
        .{ .mode = .reverse_expert_split, .tok = 85, .n = 1, .vram = 0.8 },
    } });
    try checkCase(.{ .model = 0, .profile = 1, .ctx = 131072, .rows = &.{
        .{ .mode = .all_metal, .tok = 38, .mac = 21.1 },
        .{ .mode = .resident, .tok = 86, .vram = 21.1 },
        .{ .mode = .layer_split, .tok = 72, .n = 39, .vram = 20.2, .mac = 1.3 },
        .{ .mode = .layer_split_pipelined, .tok = 119, .n = 31, .vram = 16.1, .mac = 5.4 },
        .{ .mode = .expert_split, .tok = 87, .n = 40, .vram = 21.1 },
        .{ .mode = .reverse_expert_split, .tok = 38, .n = 1, .vram = 0.8 },
    } });
}

test "A3B on a 5090 class card, 8k and 128k" {
    try checkCase(.{ .model = 0, .profile = 2, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 86, .mac = 18.8 },
        .{ .mode = .resident, .tok = 299, .vram = 18.8 },
        .{ .mode = .layer_split, .tok = 199, .n = 39, .vram = 18.1, .mac = 1.1 },
        .{ .mode = .layer_split_pipelined, .tok = 367, .n = 36, .vram = 16.7, .mac = 2.4 },
        .{ .mode = .expert_split, .tok = 301, .n = 40, .vram = 18.8 },
        .{ .mode = .reverse_expert_split, .tok = 86, .n = 1, .vram = 0.8 },
    } });
    try checkCase(.{ .model = 0, .profile = 2, .ctx = 131072, .rows = &.{
        .{ .mode = .all_metal, .tok = 38, .mac = 21.1 },
        .{ .mode = .resident, .tok = 169, .vram = 21.1 },
        .{ .mode = .layer_split, .tok = 114, .n = 39, .vram = 20.2, .mac = 1.3 },
        .{ .mode = .layer_split_pipelined, .tok = 199, .n = 36, .vram = 18.8, .mac = 2.7 },
        .{ .mode = .expert_split, .tok = 170, .n = 40, .vram = 21.1 },
        .{ .mode = .reverse_expert_split, .tok = 38, .n = 1, .vram = 0.8 },
    } });
}

test "A3B on a 24 GiB Mac (16 GiB budget, 32k context)" {
    try checkCase(.{ .model = 0, .profile = 0, .ctx = 32768, .budget_gib = 16, .rows = &.{
        .{ .mode = .all_metal, .fits = false, .mac = 19.2 },
        .{ .mode = .resident, .fits = false, .vram = 19.2 },
        .{ .mode = .layer_split, .tok = 64, .n = 7, .vram = 3.8, .mac = 15.8 },
        .{ .mode = .layer_split_pipelined, .tok = 100, .n = 15, .vram = 7.5, .mac = 12.2 },
        .{ .mode = .expert_split, .tok = 44, .n = 13, .vram = 7.9 },
        .{ .mode = .reverse_expert_split, .tok = 64, .n = 7, .vram = 3.3 },
    } });
    try checkCase(.{ .model = 0, .profile = 1, .ctx = 32768, .budget_gib = 16, .rows = &.{
        .{ .mode = .all_metal, .fits = false, .mac = 19.2 },
        .{ .mode = .resident, .tok = 135, .vram = 19.2 },
        .{ .mode = .layer_split, .tok = 113, .n = 39, .vram = 18.5, .mac = 1.2 },
        .{ .mode = .layer_split_pipelined, .tok = 196, .n = 30, .vram = 14.4, .mac = 5.3 },
        .{ .mode = .expert_split, .tok = 136, .n = 40, .vram = 19.2 },
        .{ .mode = .reverse_expert_split, .tok = 66, .n = 7, .vram = 3.3 },
    } });
    try checkCase(.{ .model = 0, .profile = 2, .ctx = 32768, .budget_gib = 16, .rows = &.{
        .{ .mode = .all_metal, .fits = false, .mac = 19.2 },
        .{ .mode = .resident, .tok = 259, .vram = 19.2 },
        .{ .mode = .layer_split, .tok = 173, .n = 39, .vram = 18.5, .mac = 1.2 },
        .{ .mode = .layer_split_pipelined, .tok = 314, .n = 36, .vram = 17.1, .mac = 2.5 },
        .{ .mode = .expert_split, .tok = 261, .n = 40, .vram = 19.2 },
        .{ .mode = .reverse_expert_split, .tok = 66, .n = 7, .vram = 3.3 },
    } });
}

test "dense 4B at 8k on the three profiles" {
    try checkCase(.{ .model = 1, .profile = 0, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 62, .mac = 2.8 },
        .{ .mode = .resident, .tok = 51, .vram = 2.8 },
        .{ .mode = .layer_split, .tok = 61, .n = 1, .vram = 0.7, .mac = 2.4 },
        .{ .mode = .layer_split_pipelined, .tok = 109, .n = 16, .vram = 1.7, .mac = 1.4 },
    } });
    try checkCase(.{ .model = 1, .profile = 1, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 62, .mac = 2.8 },
        .{ .mode = .resident, .tok = 149, .vram = 2.8 },
        .{ .mode = .layer_split, .tok = 118, .n = 31, .vram = 2.7, .mac = 0.4 },
        .{ .mode = .layer_split_pipelined, .tok = 207, .n = 26, .vram = 2.4, .mac = 0.7 },
    } });
    try checkCase(.{ .model = 1, .profile = 2, .ctx = 8192, .rows = &.{
        .{ .mode = .all_metal, .tok = 62, .mac = 2.8 },
        .{ .mode = .resident, .tok = 295, .vram = 2.8 },
        .{ .mode = .layer_split, .tok = 177, .n = 31, .vram = 2.7, .mac = 0.4 },
        .{ .mode = .layer_split_pipelined, .tok = 347, .n = 31, .vram = 2.7, .mac = 0.4 },
    } });
}

test "a dense model gets no expert splits and its splits are executable" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const m = try geometry.fromConfigJson(arena.allocator(), fx.d4_json);
    const c = try planner.plan(arena.allocator(), m, .{ .context = 8192 }, mac, profiles[1], .assumed);
    try std.testing.expectEqual(@as(usize, 4), c.items.len);
    try std.testing.expect(c.get(.expert_split) == null and c.get(.reverse_expert_split) == null);
    for (c.items) |p| try std.testing.expect(p.executable);
    try std.testing.expectEqual(@as(usize, 0), c.get(.resident).?.layout.expert_layers.len);
}

test "calibration against the measured single-device decode rates" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a3b = try geometry.fromConfigJson(arena.allocator(), fx.a3b_json);
    const d4 = try geometry.fromConfigJson(arena.allocator(), fx.d4_json);
    const bw = mac.peak_bandwidth * mac.efficiency;
    try near(94, 1 / planner.decodeSeconds(a3b, bw, 0, 512), 0.01, 0.51); // measured 91.6
    try near(68, 1 / planner.decodeSeconds(d4, bw, 0, 512), 0.01, 0.51); // measured 66.0
    const vega = profiles[0];
    try near(55, 1 / planner.decodeSeconds(d4, vega.peak_bandwidth * vega.efficiency, vega.launch_ns, 512), 0.01, 0.51); // measured 52-57
}

test "reverse split break-even bytes" {
    const mib: f64 = 1 << 20;
    try std.testing.expect(planner.reverseBreakEvenBytes(mac, profiles[0], .assumed) == null);
    try near(40, planner.reverseBreakEvenBytes(mac, profiles[1], .assumed).? / mib, 0.01, 0.51);
    try near(30, planner.reverseBreakEvenBytes(mac, profiles[2], .assumed).? / mib, 0.01, 0.51);
}

test "the link comes from the argument: slower hops make the splits slower" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const m = try geometry.fromConfigJson(arena.allocator(), fx.d4_json);
    const slow: planner.Link = .{ .hop_to_host_ns = 1_000_000, .hop_to_device_ns = 1_000_000 };
    const base = try planner.plan(arena.allocator(), m, .{ .context = 8192 }, mac, profiles[1], .assumed);
    const worse = try planner.plan(arena.allocator(), m, .{ .context = 8192 }, mac, profiles[1], slow);
    try std.testing.expect(worse.get(.layer_split).?.t > base.get(.layer_split).?.t + 1.8e-3);
    try std.testing.expectEqual(base.get(.resident).?.t, worse.get(.resident).?.t);
}

test "auto never picks a plan slower than all-Metal when all-Metal fits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const a3b = try geometry.fromConfigJson(a, fx.a3b_json);
    const d4 = try geometry.fromConfigJson(a, fx.d4_json);
    // Vega is slower than the Mac: one stream stays on Metal, 2+ streams take the pipelined split
    const vega_dense = try planner.plan(a, d4, .{ .context = 8192 }, mac, profiles[0], .assumed);
    try std.testing.expectEqual(Mode.all_metal, (try planner.choose(vega_dense, .{})).mode);
    const vega_two = try planner.plan(a, d4, .{ .context = 8192, .streams = 2 }, mac, profiles[0], .assumed);
    try std.testing.expectEqual(Mode.layer_split_pipelined, (try planner.choose(vega_two, .{})).mode);
    // a fast card: resident for one stream, the pipelined split for two
    const xtx_one = try planner.plan(a, d4, .{ .context = 8192 }, mac, profiles[1], .assumed);
    try std.testing.expectEqual(Mode.resident, (try planner.choose(xtx_one, .{})).mode);
    const xtx_two = try planner.plan(a, d4, .{ .context = 8192, .streams = 2 }, mac, profiles[1], .assumed);
    try std.testing.expectEqual(Mode.layer_split_pipelined, (try planner.choose(xtx_two, .{})).mode);
    // the MoE: only all-Metal and resident are executable
    const moe_vega = try planner.plan(a, a3b, .{ .context = 8192, .streams = 2 }, mac, profiles[0], .assumed);
    try std.testing.expectEqual(Mode.all_metal, (try planner.choose(moe_vega, .{})).mode);
    const moe_xtx = try planner.plan(a, a3b, .{ .context = 8192 }, mac, profiles[1], .assumed);
    const pick = try planner.choose(moe_xtx, .{});
    try std.testing.expectEqual(Mode.resident, pick.mode);
    try std.testing.expect(pick.tok_s >= moe_xtx.get(.all_metal).?.tok_s);
    // information-only placements are reachable on request
    const info = try planner.choose(moe_vega, .{ .allow_unexecutable = true });
    try std.testing.expectEqual(Mode.layer_split_pipelined, info.mode);
    try std.testing.expect(!info.executable);
    // every chosen plan beats or ties all-Metal whenever all-Metal fits
    for ([_]planner.Candidates{ vega_dense, vega_two, xtx_one, xtx_two, moe_vega, moe_xtx }) |c| {
        const chosen = try planner.choose(c, .{ .allow_unexecutable = true });
        try std.testing.expect(chosen.tok_s >= c.get(.all_metal).?.tok_s);
    }
}

test "all-Metal that does not fit is replaced, and a hopeless request reports the shortfall" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const a3b = try geometry.fromConfigJson(a, fx.a3b_json);
    // 24 GiB Mac with the XTX: resident fits, all-Metal does not
    const small = try planner.plan(a, a3b, .{ .context = 32768, .mac_budget_override = 16 << 30 }, mac, profiles[1], .assumed);
    try std.testing.expectEqual(Mode.resident, (try planner.choose(small, .{})).mode);
    // 24 GiB Mac with the Vega card: only information-only splits fit, so the default refuses and says what is missing
    const vega = try planner.plan(a, a3b, .{ .context = 32768, .mac_budget_override = 16 << 30 }, mac, profiles[0], .assumed);
    try std.testing.expectError(error.DoesNotFit, planner.choose(vega, .{}));
    const s = vega.shortfall();
    try std.testing.expect(s.unexecutable_fits);
    try std.testing.expect(s.mac.needed > s.mac.available);
    try std.testing.expect(s.vram.needed > s.vram.available);
    try std.testing.expect(s.combined.needed <= s.combined.available);
    // one stream cannot use the pipelined split
    try std.testing.expectEqual(Mode.layer_split, (try planner.choose(vega, .{ .allow_unexecutable = true })).mode);
    // a Mac of 4 GiB and a Vega card cannot hold 19 GiB together
    const hopeless = try planner.plan(a, a3b, .{ .context = 32768, .mac_budget_override = 4 << 30 }, mac, profiles[0], .assumed);
    try std.testing.expectError(error.DoesNotFit, planner.choose(hopeless, .{ .allow_unexecutable = true }));
    const h = hopeless.shortfall();
    try std.testing.expect(!h.unexecutable_fits);
    try std.testing.expect(h.combined.needed > h.combined.available);
    try std.testing.expectEqual(@as(u64, 4 << 30), h.mac.available);
}

test "a plan becomes a valid placement.json value" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d4 = try geometry.fromConfigJson(a, fx.d4_json);
    const req: planner.Request = .{ .context = 8192, .streams = 2 };
    const c = try planner.plan(a, d4, req, mac, profiles[0], .assumed);
    const pick = try planner.choose(c, .{});
    const device: placement.Device = .{ .backend = "hip", .index = 0, .window_mib = 64 };
    const p = try pick.toPlacement(device, req, .metal);
    try placement.validate(p, d4.layers.len);
    try std.testing.expectEqual(Mode.layer_split_pipelined, p.prediction.?.mode);
    try std.testing.expectEqual(placement.Prefill.metal, p.prefill);
    const text = try placement.write(std.testing.allocator, p);
    defer std.testing.allocator.free(text);
    const back = try placement.parse(std.testing.allocator, text);
    defer back.deinit();
    try std.testing.expectEqualSlices(placement.Owner, p.layers, back.value.layers);
    // fallback to Metal is refused when the Mac cannot hold the model
    const a3b = try geometry.fromConfigJson(a, fx.a3b_json);
    const cmoe = try planner.plan(a, a3b, .{ .context = 32768, .mac_budget_override = 16 << 30 }, mac, profiles[1], .assumed);
    const moe_pick = try planner.choose(cmoe, .{});
    try std.testing.expectError(error.FallbackImpossible, moe_pick.toPlacement(device, req, .metal));
    try placement.validate(try moe_pick.toPlacement(device, req, .none), a3b.layers.len);
    try std.testing.expectEqual(placement.Prefill.external, moe_pick.prefill);
}
