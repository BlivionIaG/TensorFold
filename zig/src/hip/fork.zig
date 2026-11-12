//! Branches of a captured round: kernels with no data in common recorded on a side stream between a fork and a join, so the graph has no edge between them.
//!
//! HIP's capture rule (and hipmac's): an event recorded on a capturing stream remembers the stream's newest nodes, and a stream that waits on it
//! joins the capture with those nodes as its first dependencies. A branch is `fork.record(main)`, `side.wait(fork)`, the branch's launches on
//! `side`, `join.record(side)`, `main.wait(join)`. The runtime then runs the nodes of the two branches in any order, also at once, and may drop the
//! barrier between them (`HIPMAC_IB_ELIDE`), so the caller owns the proof that the branches touch no memory in common unless both only read it
//! (`hazard.zig` checks that for a recorded round). Outside a capture nothing forks: `Branch.ops` is the caller's ops and `join` does nothing,
//! the launches keep their order on one stream.

const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Stream = @import("stream.zig").Stream;
const Event = @import("stream.zig").Event;
const counting = @import("launches/util.zig").counting;

/// The side stream and the two events of a round's branches, made once (never during a capture) and reused by every branch of every capture.
pub const Fork = struct {
    side: Stream,
    fork: Event,
    join: Event,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Fork {
        var side = try Stream.init(r);
        errdefer side.deinit();
        var fork = try Event.init(r, false);
        errdefer fork.deinit();
        return .{ .side = side, .fork = fork, .join = try Event.init(r, false) };
    }

    pub fn deinit(f: *Fork) void {
        f.join.deinit();
        f.fork.deinit();
        f.side.deinit();
        f.* = undefined;
    }
};

/// A fork in progress: `side` is where the branch's kernels go (`main` itself when nothing forked), `join` ends it.
pub const Branch = struct {
    main: Stream,
    side: abi.Stream,
    forked: ?*Fork,

    /// The branch's kernels go on `side` after this: everything launched on `main` so far is done before the first of them.
    /// No fork on the counting stream, nor without a `Fork` (an eager round, a tensor-parallel rank).
    pub fn begin(f: ?*Fork, main: Stream) runtime.Error!Branch {
        const fk = f orelse return .{ .main = main, .side = main.handle, .forked = null };
        if (main.handle == counting) return .{ .main = main, .side = main.handle, .forked = null };
        try fk.fork.record(main);
        try fk.side.wait(fk.fork);
        return .{ .main = main, .side = fk.side.handle, .forked = fk };
    }

    /// Later launches on `main` start after the branch's last one.
    pub fn end(b: Branch) runtime.Error!void {
        const fk = b.forked orelse return;
        try fk.join.record(.{ .r = b.main.r, .handle = b.side });
        try b.main.wait(fk.join);
    }
};
