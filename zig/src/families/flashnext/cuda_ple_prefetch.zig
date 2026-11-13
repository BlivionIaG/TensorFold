//! Advisory page-cache reads of the n-gram rows the next verify window gathers (TF_FLASHNEXT_PLE_PREFETCH).

const ple = @import("cuda_ple.zig");

/// True when the EXL3 n-gram table was opened with prefetch on.
pub fn on(g: *const ple.Ngram) bool {
    const t = g.x3 orelse return false;
    return t.opts.prefetch;
}

/// Asks the kernel for the pages of the rows `token` gathers after `history`; no value is read into a result.
pub fn rows(g: *const ple.Ngram, history: []const i64, token: u32) void {
    const x3 = g.x3 orelse return;
    var ids: [ple.max_heads]i64 = undefined;
    ple.lastIds(g.hash, history, @intCast(token), ids[0..g.hash.heads]) catch return;
    for (ids[0..g.hash.heads]) |id| {
        const loc = g.hash.locate(id) catch return;
        x3.advise(loc.shard, loc.local);
    }
}
