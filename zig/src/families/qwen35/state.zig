//! A stream's caches on the device: a fixed key/value buffer per full-attention layer, the conv window and DeltaNet
//! state per linear layer, both zeroed fresh (the Python engine's zeros for a cache that does not exist yet).

const std = @import("std");
const hip = @import("hip");
const view = @import("view.zig");
const Allocator = std.mem.Allocator;

pub const LayerCache = union(enum) {
    full: struct { k: hip.DeviceBuffer, v: hip.DeviceBuffer, len: usize },
    linear: struct { conv: hip.DeviceBuffer, state: hip.DeviceBuffer },
};

pub const Caches = struct {
    layers: []LayerCache,
    total: usize,

    /// Buffers for `total` positions; every byte zeroed.
    pub fn init(gpa: Allocator, d: *const hip.Driver, m: *const view.Model, total: usize) !Caches {
        return make(gpa, d, m, total, true);
    }

    /// Buffers for `total` positions, contents left as the allocator gave them (a copy fills them).
    pub fn blank(gpa: Allocator, d: *const hip.Driver, m: *const view.Model, total: usize) !Caches {
        return make(gpa, d, m, total, false);
    }

    fn make(gpa: Allocator, d: *const hip.Driver, m: *const view.Model, total: usize, zero: bool) !Caches {
        const s = m.spec;
        const layers = try gpa.alloc(LayerCache, s.n_layers);
        var made: usize = 0;
        errdefer {
            for (layers[0..made]) |*l| free(l);
            gpa.free(layers);
        }
        for (layers, 0..) |*l, i| {
            if (s.full(i)) {
                const bytes = s.kv_heads * total * s.head_dim * m.act.size();
                var k = try hip.DeviceBuffer.alloc(d, bytes);
                errdefer k.free();
                var v = try hip.DeviceBuffer.alloc(d, bytes);
                errdefer v.free();
                if (zero) {
                    try k.fill8(0, null);
                    try v.fill8(0, null);
                }
                l.* = .{ .full = .{ .k = k, .v = v, .len = 0 } };
            } else {
                var conv = try hip.DeviceBuffer.alloc(d, (s.conv - 1) * view.convChannels(s) * 4);
                errdefer conv.free();
                var state = try hip.DeviceBuffer.alloc(d, s.value_heads * s.value_dim * s.key_dim * 4);
                errdefer state.free();
                if (zero) {
                    try conv.fill8(0, null);
                    try state.fill8(0, null);
                }
                l.* = .{ .linear = .{ .conv = conv, .state = state } };
            }
            made += 1;
        }
        return .{ .layers = layers, .total = total };
    }

    fn free(l: *LayerCache) void {
        switch (l.*) {
            .full => |*f| {
                f.k.free();
                f.v.free();
            },
            .linear => |*x| {
                x.conv.free();
                x.state.free();
            },
        }
    }

    pub fn deinit(c: *Caches, gpa: Allocator) void {
        for (c.layers) |*l| free(l);
        gpa.free(c.layers);
        c.* = undefined;
    }

    /// Bytes the buffers hold.
    pub fn held(c: *const Caches) usize {
        var n: usize = 0;
        for (c.layers) |l| switch (l) {
            .full => |f| n += f.k.len + f.v.len,
            .linear => |x| n += x.conv.len + x.state.len,
        };
        return n;
    }

    /// Copy the first `len` positions of every attention layer and the whole linear state of `src` into `dst` (either may
    /// be the longer buffer); `dst` then holds `len` positions.
    pub fn copyPrefix(dst: *Caches, src: *const Caches, m: *const view.Model, len: usize, stream: hip.abi.Stream) !void {
        const s = m.spec;
        const row = s.head_dim * m.act.size();
        for (dst.layers, src.layers) |*to, from| switch (to.*) {
            .full => |*f| {
                const g = from.full;
                for (0..s.kv_heads) |h| {
                    try f.k.copyFrom(h * dst.total * row, g.k.ptr + h * src.total * row, len * row, stream);
                    try f.v.copyFrom(h * dst.total * row, g.v.ptr + h * src.total * row, len * row, stream);
                }
                f.len = len;
            },
            .linear => |*x| {
                const g = from.linear;
                try x.conv.copyFrom(0, g.conv.ptr, g.conv.len, stream);
                try x.state.copyFrom(0, g.state.ptr, g.state.len, stream);
            },
        };
    }

    /// The attention kernels' view of a full layer's cache.
    pub fn attention(c: *const Caches, m: *const view.Model, index: usize) hip.ops.Ops.Cache {
        const f = c.layers[index].full;
        return .{ .k = f.k.ptr, .v = f.v.ptr, .kind = m.act, .kv_heads = m.spec.kv_heads, .total = c.total, .d = m.spec.head_dim };
    }
};
