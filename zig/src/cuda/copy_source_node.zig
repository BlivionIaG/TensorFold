//! The custody node of one copy source: its stream, completion proof and the bytes it keeps alive.
const std = @import("std");
const abi = @import("abi.zig");
const memory = @import("memory.zig");
const budget = @import("copy_source_budget.zig");
const durable = std.heap.page_allocator;
pub const Node = struct {
    charge: usize = @sizeOf(Node),
    next: ?*Node = null,
    work_next: ?*Node = null,
    busy: bool = false,
    ticket: u64 = 0,
    driver: usize = 0,
    context: abi.Context = null,
    stream: abi.Stream = null,
    failed: bool = false,
    reusable: bool = false, // a temporary upload's stage: completion returns it to the spares
    data: union(enum) {
        none,
        cpu: []align(16) u8,
        pinned: struct {
            bytes: []align(16) u8,
            free: @FieldType(abi.Api, "cuMemFreeHost"),
            event: ?abi.Event,
            destroy: ?@FieldType(abi.Api, "cuEventDestroy_v2"),
            query: ?@FieldType(abi.Api, "cuEventQuery"),
            wait: ?@FieldType(abi.Api, "cuEventSynchronize"),
        },
    } = .none,
    pub fn releaseData(n: *Node) void {
        const a = durable;
        switch (n.data) {
            .none => {},
            .cpu => |bytes| a.free(bytes),
            .pinned => |p| {
                if (p.destroy) |drop| _ = drop(p.event.?);
                _ = p.free(p.bytes.ptr);
                memory.releaseRetainedHost(p.bytes.len);
            },
        }
    }
    pub fn destroy(n: *Node) void {
        const charge = n.charge;
        durable.destroy(n);
        budget.release(charge, 1);
    }
    pub fn release(n: *Node) void {
        n.releaseData();
        n.destroy();
    }
};
pub fn newNode() !*Node {
    try budget.acquire(@sizeOf(Node), 1);
    errdefer budget.release(@sizeOf(Node), 1);
    const n = try durable.create(Node);
    n.* = .{};
    return n;
}
