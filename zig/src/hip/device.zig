//! The GPU a rank serves on and what the run may use there, chosen once by whoever opens an engine.

const Driver = @import("runtime/driver.zig").Driver;
const Context = @import("runtime/context.zig").Context;
const Caps = @import("caps.zig").Caps;
const Policy = @import("policy.zig").Policy;
const Group = @import("comm/group.zig").Group;

pub const Device = struct {
    /// The ordinal among the visible cards.
    index: c_int,
    caps: Caps,
    /// Rank 0's policy; the other ranks adopt it when they join the group.
    policy: Policy,
    /// The tensor-parallel group this rank belongs to (null: one rank).
    group: ?*Group = null,

    /// The card of `rank`: with every card visible rank r takes card r, with one card a process that card.
    pub fn ordinal(rank: u32) c_int {
        var d = Driver.open() catch return 0;
        defer d.close();
        const count = d.deviceCount() catch return 0;
        return if (count > 0) @intCast(rank % @as(u32, @intCast(count))) else 0;
    }

    /// What the card at `index` can do, or null when it is no usable GPU.
    pub fn capsOf(index: c_int) ?Caps {
        var d = Driver.open() catch return null;
        defer d.close();
        var ctx = Context.init(&d, index) catch return null;
        defer ctx.deinit();
        return ctx.caps() catch null;
    }
};
