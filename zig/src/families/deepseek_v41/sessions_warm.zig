//! Warm resume under the exact replay (TF_DSV41_REPLAY_WARM): a prompt entry holds the decoder layers' exact SWA rings.

const std = @import("std");
const fwd = @import("forward.zig");
const ced = @import("ced.zig");
const pf = @import("forward_prefill.zig");
const pf_cone = @import("forward_prefill_cone.zig");
const pf_finish = @import("forward_prefill_finish.zig");
const ft = @import("forward_types.zig");
const cuda = @import("cuda");
const buffers = @import("buffers.zig");

/// The knob's level when entries are warm (the exact replay on, every decoder SWA ring in the snapshot); else 0.
pub fn level(f: *const fwd.Forward, roles: []const []const u8, backbone: []const u32) u8 {
    const st = pf.state(f);
    if (st.mode != .replay or st.cone_warm == 0) return 0;
    const d = ced.decoderStart(f.cfg) catch return 0;
    var found: u32 = 0;
    var layers: u32 = 0;
    for (backbone) |L| if (L >= d) {
        layers += 1;
        var buf: [32]u8 = undefined;
        for ([_][]const u8{ "swa.v", "swa.s" }) |part| {
            const name = std.fmt.bufPrint(&buf, "s.L{d}.{s}", .{ L, part }) catch return 0;
            for (roles) |r| {
                if (std.mem.eql(u8, r, name)) found += 1;
            }
        }
    };
    return if (layers > 0 and found == 2 * layers) st.cone_warm else 0;
}

/// The state that names the cone ring before an entry's replay: level 1 puts it back after.
pub const Saved = struct {
    cone: ft.ConeStash,
    cone_id: u64,
    slot_cone_id: u64,
    exact: bool,
    exact_at: u64,
};

/// Whether an entry can be saved here: it needs the exact replay up to the position (the leader decides).
pub fn canSave(f: *fwd.Forward) bool {
    return f.slot.pos > ced.keep and pf_cone.exactAt(f, f.slot.pos);
}

pub fn capture(f: *const fwd.Forward) Saved {
    const st = pf.state(f);
    return .{ .cone = st.cone, .cone_id = st.cone_id, .slot_cone_id = f.slot.cone_id, .exact = f.slot.exact, .exact_at = f.slot.exact_at };
}

/// The exact replay up to the slot's position, on this rank (every rank runs it inside the entry's save).
pub fn replayNow(f: *fwd.Forward) !bool {
    try pf_finish.finish(f);
    return f.slot.exact and f.slot.exact_at == f.slot.pos;
}

/// Level 1: the prompt's own replay goes on as if the entry's had not run: the ring's ticket and the slot's exactness.
pub fn reinstate(f: *fwd.Forward, s: Saved) void {
    const st = pf.state(f);
    st.cone = s.cone;
    st.cone_id = s.cone_id;
    f.slot.cone_id = s.slot_cone_id;
    f.slot.exact = s.exact;
    f.slot.exact_at = s.exact_at;
    st.replay_exact = false;
    st.taps_rows = 0;
}

/// One device role of the cone ring and its bytes.
pub const Role = struct { role: []const u8, len: usize };

pub fn ringBytes(roles: []const Role) usize {
    var n: usize = 0;
    for (roles) |r| n += r.len;
    return n;
}

/// The cone ring's roles into `out` (the stream idle after the sync).
pub fn download(f: *fwd.Forward, roles: []const Role, out: []u8) !void {
    const r = f.runner;
    try r.stream.synchronize();
    var at: usize = 0;
    for (roles) |x| {
        const addr = r.addressOf(x.role) orelse return error.Unbound;
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = addr, .len = x.len }, 0, out[at..][0..x.len]);
        at += x.len;
    }
}

pub fn upload(f: *fwd.Forward, roles: []const Role, bytes: []const u8) !void {
    var at: usize = 0;
    for (roles) |x| {
        const addr = f.runner.addressOf(x.role) orelse return error.Unbound;
        try f.upload(addr, bytes[at..][0..x.len]);
        at += x.len;
    }
}

/// The cone ring's roles with their sizes, or null when the plan lacks one.
pub fn coneRoles(a: std.mem.Allocator, plan: *const buffers.Plan) !?[]const Role {
    var cr: std.ArrayList(Role) = .empty;
    for (ced.cone_roles) |role| try cr.append(a, .{ .role = role, .len = plan.sizes.get(role) orelse return null });
    return cr.items;
}

/// An entry's save, every rank: the exact replay up to the position; level 1 first copies the cone ring to the host.
pub fn before(f: *fwd.Forward, gpa: std.mem.Allocator, level_: u8, roles: []const Role, bak: *[]u8) !?Saved {
    if (level_ == 0) return null;
    const saved = capture(f);
    if (level_ == 1) {
        if (bak.len == 0) bak.* = try gpa.alloc(u8, ringBytes(roles));
        try download(f, roles, bak.*);
    }
    if (!try replayNow(f)) return error.BadWindow;
    return saved;
}

/// After the snapshot: level 1 puts the cone ring and its ticket back; level 2 leaves the replay as it ran.
pub fn after(f: *fwd.Forward, level_: u8, roles: []const Role, bak: []const u8, saved: ?Saved) !void {
    const sv = saved orelse return;
    if (level_ != 1) return;
    try upload(f, roles, bak);
    reinstate(f, sv);
}
