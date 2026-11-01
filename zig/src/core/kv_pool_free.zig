//! The pool's free page ids: a stack, or a min-heap when the pool hands out the lowest id first.
const std = @import("std");
const Pool = @import("kv_pool.zig").Pool;

/// `id` is free again.
pub fn push(p: *Pool, id: u32) void {
    var i = p.nfree;
    p.nfree += 1;
    while (p.lowest and i > 0 and p.free[(i - 1) / 2] > id) : (i = (i - 1) / 2) p.free[i] = p.free[(i - 1) / 2];
    p.free[i] = id;
}

/// The id the pool hands out next: the top of the stack, or the lowest free id.
pub fn pop(p: *Pool) u32 {
    p.nfree -= 1;
    if (!p.lowest) return p.free[p.nfree];
    const lowest = p.free[0];
    p.free[0] = p.free[p.nfree];
    siftDown(p, 0);
    return lowest;
}

fn siftDown(p: *Pool, from: usize) void {
    var i = from;
    while (true) {
        var low = i;
        for ([2]usize{ 2 * i + 1, 2 * i + 2 }) |c| if (c < p.nfree and p.free[c] < p.free[low]) {
            low = c;
        };
        if (low == i) return;
        std.mem.swap(u32, &p.free[i], &p.free[low]);
        i = low;
    }
}

/// The free ids as a heap again after some left in place (no-op for the stack).
pub fn heapify(p: *Pool) void {
    if (!p.lowest) return;
    var i = p.nfree / 2;
    while (i > 0) : (i -= 1) siftDown(p, i - 1);
}
