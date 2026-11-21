//! LFM2 contract registration and lifetime, including the prompt selector and the existing shared-round controls.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const lfm2_argmax = @import("metal_lfm2_argmax.zig");
const lfm2_streams = @import("metal_lfm2_streams.zig");
const lfm2_tap = @import("metal_lfm2_tap.zig");
const lfm2_controls = @import("metal_lfm2_controls.zig");
const lfm2_dspark = @import("metal_lfm2_dspark.zig");
const lfm2_bf16 = @import("metal_lfm2_bf16.zig");
const lfm2_attn = @import("metal_lfm2_attn.zig");
const lfm2_conf = @import("metal_lfm2_confidence.zig");
const Prompt = @import("metal_lfm2_prompt.zig").Suite;
fn wanted(only: ?[]const u8, name: []const u8) bool {
    return if (only) |o| std.mem.indexOf(u8, name, o) != null else true;
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    prompt: ?Prompt = null,
    argmax: std.ArrayList(*lfm2_argmax.Probe) = .empty,
    streams: std.ArrayList(*lfm2_streams.Probe) = .empty,
    taps: std.ArrayList(*lfm2_tap.Probe) = .empty,
    drafter: std.ArrayList(*lfm2_dspark.Probe) = .empty,
    blocks: std.ArrayList(*lfm2_attn.Probe) = .empty,
    bf16: std.ArrayList(*lfm2_bf16.Probe) = .empty,
    conf: std.ArrayList(*lfm2_conf.Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const a = s.a;
        const device = s.device;
        const queue = s.queue;
        s.prompt = .{ .a = a, .device = device, .queue = queue, .kernels = s.kernels };
        try s.prompt.?.collect(only);
        for (std.enums.values(lfm2_controls.Mutation)) |m| {
            const argmax_side = m == .none or m == .argmax_last_tie or m == .argmax_no_flag;
            if (argmax_side) {
                if (!wanted(only, if (m == .none) lfm2_argmax.name else m.name())) continue;
                const p = try a.create(lfm2_argmax.Probe);
                errdefer a.destroy(p);
                p.* = try lfm2_argmax.Probe.init(a, device, queue, m);
                try s.argmax.append(a, p);
                try s.kernels.append(a, p.kernel());
                if (m != .none) continue;
            }
            for (std.enums.values(lfm2_streams.Which)) |w| {
                const control_for = (m == .conv_first_state and w == .conv) or (m == .attention_first_cache and w == .attention);
                if (m != .none and !control_for) continue;
                if (!wanted(only, if (m == .none) lfm2_streams.spec(w).name else m.name())) continue;
                const p = try a.create(lfm2_streams.Probe);
                errdefer a.destroy(p);
                p.* = try lfm2_streams.Probe.init(a, device, queue, w, m);
                try s.streams.append(a, p);
                try s.kernels.append(a, p.kernel());
            }
        }
        for ([_]lfm2_controls.Mutation{ .none, .tap_packed_rows, .tap_clobbers_column }) |m| {
            if (!wanted(only, if (m == .none) lfm2_tap.name else m.name())) continue;
            const p = try a.create(lfm2_tap.Probe);
            errdefer a.destroy(p);
            p.* = try lfm2_tap.Probe.init(a, device, queue, m);
            try s.taps.append(a, p);
            try s.kernels.append(a, p.kernel());
        }
        for (std.enums.values(lfm2_controls.Mutation)) |m| {
            const drafter_which: ?lfm2_dspark.Which = switch (m) {
                .markov_nan_dropped, .markov_first_prev => .markov_step,
                .markov_bf16_nan_dropped, .markov_bf16_first_prev => .markov_step_bf16,
                .draft_ids_unzeroed => .draft_ids,
                else => null,
            };
            if (m == .none or drafter_which != null) {
                for (std.enums.values(lfm2_dspark.Which)) |w| {
                    if (m != .none and w != drafter_which.?) continue;
                    if (!wanted(only, if (m == .none) lfm2_dspark.spec(w).name else m.name())) continue;
                    const p = try a.create(lfm2_dspark.Probe);
                    errdefer a.destroy(p);
                    p.* = try lfm2_dspark.Probe.init(a, device, queue, w, m);
                    try s.drafter.append(a, p);
                    try s.kernels.append(a, p.kernel());
                }
            }
            const bf16_which: ?lfm2_bf16.Which = switch (m) {
                .bf16_rows_short_k, .bf16_rows_last_row => .rows,
                .bf16_tile_untransposed, .bf16_tile_last_column => .tile,
                else => null,
            };
            if (m == .none or bf16_which != null) {
                for (std.enums.values(lfm2_bf16.Which)) |w| {
                    if (m != .none and w != bf16_which.?) continue;
                    if (!wanted(only, if (m == .none) lfm2_bf16.spec(w).name else m.name())) continue;
                    const p = try a.create(lfm2_bf16.Probe);
                    errdefer a.destroy(p);
                    p.* = try lfm2_bf16.Probe.init(a, device, queue, w, m);
                    try s.bf16.append(a, p);
                    try s.kernels.append(a, p.kernel());
                }
            }
            if (m == .none or m == .block_attention_causal) {
                if (!wanted(only, if (m == .none) lfm2_attn.name else m.name())) continue;
                const p = try a.create(lfm2_attn.Probe);
                errdefer a.destroy(p);
                p.* = try lfm2_attn.Probe.init(a, device, queue, m);
                try s.blocks.append(a, p);
                try s.kernels.append(a, p.kernel());
            }
        }
        for ([_]lfm2_controls.Mutation{ .none, .confidence_no_w1 }) |m| {
            if (!wanted(only, if (m == .none) lfm2_conf.name else m.name())) continue;
            const p = try a.create(lfm2_conf.Probe);
            errdefer a.destroy(p);
            p.* = try lfm2_conf.Probe.init(a, device, queue, m);
            try s.conf.append(a, p);
            try s.kernels.append(a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        if (s.prompt) |*p| p.deinit();
        inline for (.{ "argmax", "streams", "taps", "drafter", "blocks", "bf16", "conf" }) |field| {
            const list = &@field(s, field);
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
            list.deinit(s.a);
        }
    }
};
