//! What an engine's tick reads of the host's queue: the first asks in order and the counts.

/// One queued request, as much as an admission forecast needs.
pub const Ask = struct { prompt: u32, max_tokens: u32 };

/// What the host holds now: requests waiting, streams running, the lanes it will fill, and the asks copied out.
pub const Snapshot = struct { queued: u32 = 0, running: u32 = 0, lanes: u32 = 0, copied: u32 = 0 };

/// The host's queue, read on demand from the engine thread.
pub const Queue = struct {
    host: *anyopaque,
    read: *const fn (host: *anyopaque, out: []Ask) Snapshot,

    /// Copies the first `out.len` queued asks, in order, and counts the rest.
    pub fn snapshot(q: Queue, out: []Ask) Snapshot {
        return q.read(q.host, out);
    }
};
