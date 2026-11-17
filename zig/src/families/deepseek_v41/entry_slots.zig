//! The two entry buffers of the layer-split passes: which one the next pass's rows are staged in, and when that waits.
const std = @import("std");

pub const Slots = struct {
    /// a pass that reads the slot is committed and not yet known complete
    busy: [2]bool = .{ false, false },
    next: u1 = 0,

    pub const Take = struct { slot: u1, wait: bool };

    /// The slot the next staging fills. `wait`: a running pass reads it, so that pass is waited for first.
    pub fn take(s: *Slots) Take {
        const slot = s.next;
        s.next +%= 1;
        return .{ .slot = slot, .wait = s.busy[slot] };
    }

    /// A pass reading `slot` was committed.
    pub fn committed(s: *Slots, slot: u1) void {
        s.busy[slot] = true;
    }

    /// The pass that read `slot` is complete.
    pub fn completed(s: *Slots, slot: u1) void {
        s.busy[slot] = false;
    }
};

test "staging alternates the two buffers and waits only for the pass that last read the one it reuses" {
    var s: Slots = .{};
    var parked: ?u1 = null;
    var waits: u32 = 0;
    // the model's order: stage, commit the pass (the pass before it is then waited for), stage the next while it runs
    for (0..9) |i| {
        const t = s.take();
        try std.testing.expectEqual(@as(u1, @intCast(i % 2)), t.slot);
        if (t.wait) waits += 1;
        s.committed(t.slot);
        if (parked) |p| s.completed(p);
        parked = t.slot;
        // the buffer the running pass reads is never the one the next staging fills
        try std.testing.expect(s.busy[t.slot] and !s.busy[t.slot +% 1]);
    }
    try std.testing.expectEqual(@as(u32, 0), waits);
    // a pass left running while two more are staged: the second reuse of its buffer waits
    var r: Slots = .{};
    const a = r.take();
    r.committed(a.slot);
    const b = r.take();
    try std.testing.expect(!b.wait);
    r.committed(b.slot);
    const c = r.take();
    try std.testing.expect(c.wait and c.slot == a.slot);
    r.completed(a.slot);
    r.committed(c.slot);
    try std.testing.expect(r.busy[0] and r.busy[1]);
}
