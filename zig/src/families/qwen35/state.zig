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
    /// Which stream's buffers these are, for graphs bound to them (set by the engine).
    serial: u64 = 0,

    /// Buffers for `total` positions; every byte zeroed.
    pub fn init(gpa: Allocator, d: *const hip.Driver, m: *const view.Model, total: usize) !Caches {
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
                try k.fill8(0, null);
                try v.fill8(0, null);
                l.* = .{ .full = .{ .k = k, .v = v, .len = 0 } };
            } else {
                var conv = try hip.DeviceBuffer.alloc(d, (s.conv - 1) * view.convChannels(s) * 4);
                errdefer conv.free();
                var state = try hip.DeviceBuffer.alloc(d, s.value_heads * s.value_dim * s.key_dim * 4);
                errdefer state.free();
                try conv.fill8(0, null);
                try state.fill8(0, null);
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

    /// The attention kernels' view of a full layer's cache.
    pub fn attention(c: *const Caches, m: *const view.Model, index: usize) hip.ops.Ops.Cache {
        const f = c.layers[index].full;
        return .{ .k = f.k.ptr, .v = f.v.ptr, .kind = m.act, .kv_heads = m.spec.kv_heads, .total = c.total, .d = m.spec.head_dim };
    }
};
