//! A stream's prompt pass a chunk at a time, between the other streams' rounds, for backends with `prefill_step`.
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const trail = @import("trail.zig");
const f = @import("events.zig").f;
const str = trail.str;

/// Whether the backend can run a prompt pass a chunk at a time between rounds (`fillStream`).
pub fn fills(e: *const Engine) bool {
    return e.backend.vtable.prefill_step != null;
}

/// One chunk of a stream's prompt pass; true once the stream is opened and joins the next round.
pub fn fillStream(e: *Engine, s: *Stream, first: bool) !bool {
    _ = e.arena.reset(.retain_capacity);
    if (first) try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const chunk = e.backend.vtable.prefill_step orelse return error.NoPrefillSteps;
    const done = chunk(e.backend.ptr, s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s);
        return err;
    };
    if (!done) return false;
    try e.opened(s);
    return true;
}
