//! The chunks a prompt pass runs in, and the gap between the snapshots it keeps.

const pages = @import("../forward/pages.zig");

/// A snapshot this close to the prompt's end saves too little to keep next to the one before it.
pub const min_gap = 256;

/// Cuts sit on page edges, which are multiples of the chunked recurrence's chunk: a resumed span then chunks a prompt as a fresh one does.
pub const step = pages.tokens;

/// Rows a prompt pass advances between two rounds; its steps end on multiples of this, so a prompt is cut the same way.
pub const chunk = 1024;

/// Where the step from `at` ends: the next multiple of `chunk`, or `limit` when that comes first.
pub fn chunkEnd(at: usize, limit: usize) usize {
    return @min(limit, (at / chunk + 1) * chunk);
}
