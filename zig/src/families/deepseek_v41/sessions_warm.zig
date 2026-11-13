//! Warm resume under the exact replay (TF_DSV41_REPLAY_WARM): a prompt entry holds the decoder layers' exact SWA rings.

const std = @import("std");
const fwd = @import("forward.zig");
const ced = @import("ced.zig");
const pf = @import("forward_prefill.zig");
const pf_cone = @import("forward_prefill_cone.zig");

/// Whether entries are warm: the exact replay with the knob on, and every decoder layer's SWA ring in the snapshot.
pub fn enabled(f: *const fwd.Forward, roles: []const []const u8, backbone: []const u32) bool {
    const st = pf.state(f);
    if (st.mode != .replay or !st.cone_warm) return false;
    const d = ced.decoderStart(f.cfg) catch return false;
    var found: u32 = 0;
    for (backbone) |L| if (L >= d) {
        var buf: [32]u8 = undefined;
        const want = [_][]const u8{ "swa.v", "swa.s" };
        for (want) |part| {
            const name = std.fmt.bufPrint(&buf, "s.L{d}.{s}", .{ L, part }) catch return false;
            for (roles) |r| {
                if (std.mem.eql(u8, r, name)) found += 1;
            }
        }
    };
    var layers: u32 = 0;
    for (backbone) |L| layers += @intFromBool(L >= d);
    return layers > 0 and found == 2 * layers;
}

/// Before an entry is saved: the exact replay up to the slot's position. False: no entry (too short, or no replay).
pub fn beforeSave(f: *fwd.Forward) !bool {
    const pos = f.slot.pos;
    if (pos <= ced.keep) return false;
    if (f.leads()) {
        if (!pf_cone.exactAt(f, pos)) return false;
        try f.finishPrompt();
    }
    return f.slot.exact and f.slot.exact_at == pos;
}
