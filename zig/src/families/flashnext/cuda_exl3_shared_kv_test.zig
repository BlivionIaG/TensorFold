const std = @import("std");
const cuda = @import("cuda");
const shared = @import("cuda_exl3_shared_kv.zig");
const contract = @import("exl3_shared_kv.zig");
const wt = @import("cuda_prompt_wt.zig");

const options: shared.Options = .{ .cap = 8192, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 };

pub const Fake = struct {
    var allocations: [256]?[]u8 = @splat(null);
    var launches: usize = 0;
    var fail_launch: bool = false;
    var fail_sync: bool = false;
    var expected_generation: u64 = 0;
    var expected_key: usize = 0;
    var expected_base: usize = 0;
    var copy_calls: usize = 0;
    var fail_copy: ?usize = null;
    pub var sync_calls: usize = 0;
    pub var graph_calls: usize = 0;
    pub var graph_pending: bool = false;
    pub var fail_graph: bool = false;
    pub var fail_alloc_at: ?usize = null;
    var device_alloc_calls: usize = 0;

    fn alloc(address: *u64, length: usize) callconv(.c) c_int {
        device_alloc_calls += 1;
        if (fail_alloc_at == device_alloc_calls) return cuda.abi.error_not_ready;
        for (&allocations) |*allocation| if (allocation.* == null) {
            const memory = std.heap.page_allocator.alloc(u8, length) catch return 2;
            allocation.* = memory;
            address.* = @intFromPtr(memory.ptr);
            return 0;
        };
        return 2;
    }

    fn free(address: u64) callconv(.c) c_int {
        for (&allocations) |*allocation| if (allocation.*) |memory| if (@intFromPtr(memory.ptr) == address) {
            std.heap.page_allocator.free(memory);
            allocation.* = null;
            return 0;
        };
        return 1;
    }

    fn upload(destination: u64, source: ?*const anyopaque, length: usize) callconv(.c) c_int {
        copy_calls += 1;
        if (fail_copy == copy_calls) return cuda.abi.error_not_ready;
        const target: [*]u8 = @ptrFromInt(destination);
        const origin: [*]const u8 = @ptrCast(source.?);
        @memcpy(target[0..length], origin[0..length]);
        return 0;
    }

    fn download(destination: ?*anyopaque, source: u64, length: usize) callconv(.c) c_int {
        const target: [*]u8 = @ptrCast(destination.?);
        const origin: [*]const u8 = @ptrFromInt(source);
        @memcpy(target[0..length], origin[0..length]);
        return 0;
    }

    fn downloadAsync(destination: ?*anyopaque, source: u64, length: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return download(destination, source, length);
    }

    fn memset(address: u64, fill: c_uint, count: usize) callconv(.c) c_int {
        const target: [*]u32 = @ptrFromInt(address);
        @memset(target[0..count], fill);
        return 0;
    }

    fn memset8(address: u64, fill: u8, count: usize) callconv(.c) c_int {
        const target: [*]u8 = @ptrFromInt(address);
        @memset(target[0..count], fill);
        return 0;
    }

    fn memsetAsync(address: u64, fill: c_uint, count: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return memset(address, fill, count);
    }

    fn memset8Async(address: u64, fill: u8, count: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return memset8(address, fill, count);
    }

    fn sync() callconv(.c) c_int {
        sync_calls += 1;
        if (!fail_sync) graph_pending = false;
        return if (fail_sync) cuda.abi.error_not_ready else 0;
    }

    fn graphLaunch(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
        graph_calls += 1;
        graph_pending = true;
        return if (fail_graph) cuda.abi.error_not_ready else 0;
    }

    fn streamSync(_: cuda.abi.Stream) callconv(.c) c_int {
        return sync();
    }

    fn value(parameters: ?[*]?*anyopaque, index: usize, comptime Value: type) Value {
        const pointer: *const Value = @ptrCast(@alignCast(parameters.?[index].?));
        return pointer.*;
    }

    fn launch(function: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches += 1;
        if (fail_launch) return cuda.abi.error_not_ready;
        switch (@intFromPtr(function.?)) {
            1 => {
                std.debug.assert(value(parameters, 22, u64) == expected_generation);
                std.debug.assert(value(parameters, 11, c_int) == expected_key);
            },
            2 => std.debug.assert(value(parameters, 6, c_int) == expected_base),
            3 => std.debug.assert(value(parameters, 5, c_int) == expected_base),
            4 => {
                const inverse: [*]i32 = @ptrFromInt(value(parameters, 0, u64));
                const owners: [*]i32 = @ptrFromInt(value(parameters, 1, u64));
                const stamps: [*]u32 = @ptrFromInt(value(parameters, 2, u64));
                const generations: [*]u64 = @ptrFromInt(value(parameters, 3, u64));
                const slots: usize = @intCast(value(parameters, 4, c_int));
                const begin = value(parameters, 5, c_int);
                const end = value(parameters, 6, c_int);
                const generation = value(parameters, 7, u64);
                for (0..slots) |slot| if (owners[slot] >= begin and owners[slot] < end and generations[slot] == generation) {
                    inverse[@intCast(owners[slot])] = -1;
                    owners[slot] = -1;
                    stamps[slot] = 0;
                    generations[slot] = 0;
                };
            },
            else => unreachable,
        }
        return 0;
    }

    pub fn driver() cuda.Driver {
        launches = 0;
        fail_launch = false;
        fail_sync = false;
        copy_calls = 0;
        fail_copy = null;
        sync_calls = 0;
        graph_calls = 0;
        graph_pending = false;
        fail_graph = false;
        fail_alloc_at = null;
        device_alloc_calls = 0;
        var result: cuda.Driver = undefined;
        result.api.cuMemAlloc_v2 = alloc;
        result.api.cuMemFree_v2 = free;
        result.api.cuMemcpyHtoD_v2 = upload;
        result.api.cuMemcpyDtoH_v2 = download;
        result.api.cuMemcpyDtoHAsync_v2 = downloadAsync;
        result.api.cuMemsetD32_v2 = memset;
        result.api.cuMemsetD8_v2 = memset8;
        result.api.cuMemsetD32Async = memsetAsync;
        result.api.cuMemsetD8Async = memset8Async;
        result.api.cuCtxSynchronize = sync;
        result.api.cuStreamSynchronize = streamSync;
        result.api.cuLaunchKernel = launch;
        result.api.cuGraphLaunch = graphLaunch;
        return result;
    }

    pub fn install(arena: *shared.Arena) void {
        arena.kernels = .{ .plan = .{ .d = arena.driver, .handle = @ptrFromInt(1) }, .fetch = .{ .d = arena.driver, .handle = @ptrFromInt(2) }, .remap = .{ .d = arena.driver, .handle = @ptrFromInt(3) }, .release = .{ .d = arena.driver, .handle = @ptrFromInt(4) } };
    }
};

pub fn seed(arena: *shared.Arena, sequence: contract.Sequence, layer: usize, block: usize, slot: usize, stamp: u32, value: u8) void {
    const tables: [*]u8 = @ptrFromInt(arena.tables.ptr);
    const key = @as(usize, sequence.lane) * arena.layout.lane_words + layer * arena.layout.blocks + block;
    std.mem.writeInt(u32, tables[key * 4 ..][0..4], @intCast(slot), .little);
    std.mem.writeInt(u32, tables[arena.layout.owners + slot * 4 ..][0..4], @intCast(key), .little);
    std.mem.writeInt(u32, tables[arena.layout.stamps + slot * 4 ..][0..4], stamp, .little);
    std.mem.writeInt(u64, tables[arena.layout.owner_generations + slot * 8 ..][0..8], sequence.generation, .little);
    const old = std.mem.readInt(u32, tables[arena.layout.clock..][0..4], .little);
    std.mem.writeInt(u32, tables[arena.layout.clock..][0..4], @max(old, stamp), .little);
    const row_bytes = wt.kv_heads * wt.head_dim * 2;
    const slot_rows = arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows;
    for (0..2) |side| {
        const payload: [*]u8 = @ptrFromInt(arena.device.ptr + (side * arena.layout.side_rows + slot_rows + wt.index_ratio * slot) * row_bytes);
        @memset(payload[0 .. wt.index_ratio * row_bytes], value);
    }
}

test "exl3: shared CUDA arena fixed windows bound every upstream BF16 slot address" {
    const layout = try shared.Layout.init(.{ .cap = 200016, .lanes = 8, .window = 4096, .rows = 2048, .slots = 16384 });
    for (0..layout.options.lanes) |lane| for (0..wt.attn_n) |layer| {
        const offset = try layout.windowOffset(lane, layer);
        const base = try layout.slotBase(lane, layer);
        try std.testing.expectEqual(layout.options.lanes * wt.attn_n * layout.window_rows, offset + base);
        try std.testing.expectEqual(layout.side_rows, offset + base + wt.index_ratio * layout.options.slots);
        try std.testing.expect(base >= layout.window_rows);
    };
    try std.testing.expectEqual(layout.device_bytes + layout.table_bytes + layout.work_bytes, layout.common_bytes);
    var bad = options;
    bad.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, shared.Layout.init(bad));
    bad = options;
    bad.cap = wt.window + 1;
    try std.testing.expectError(error.SharedKvGeometry, shared.Layout.init(bad));
}

test "exl3: shared CUDA arena pager launch ABI uses rank maps tags and global relative slots" {
    var driver = Fake.driver();
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const first = try arena.register(options.cap);
    defer arena.release(first, stream) catch unreachable;
    const second = try arena.register(options.cap);
    defer arena.release(second, stream) catch unreachable;
    try std.testing.expectError(error.SharedKvCommonCap, arena.register(options.cap - 1));
    const lease = try arena.acquire(second);
    Fake.expected_generation = second.generation;
    Fake.expected_key = arena.layout.lane_words + 3 * arena.layout.blocks;
    Fake.expected_base = try arena.layout.slotBase(second.lane, 3);
    try arena.page(lease, stream, 3, 0, 1, 1, 2, 3, 4, 5);
    try std.testing.expectEqual(@as(usize, 3), Fake.launches);
    try std.testing.expectError(error.SharedKvBusy, arena.acquire(first));
    try std.testing.expectError(error.SharedKvPending, arena.cancel(lease));
    try arena.complete(lease, stream);
    const viewing = try arena.acquire(second);
    const view = try arena.view(viewing, second, 3);
    try arena.cancel(viewing);
    const row_bytes = wt.kv_heads * wt.head_dim * 2;
    try std.testing.expectEqual(arena.device.ptr + arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows * row_bytes, view.keys + view.slot_base * row_bytes);
}

test "exl3: shared CUDA owned block snapshots remap slots without copying foreign payload" {
    var driver = Fake.driver();
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const source = try arena.register(options.cap);
    const foreign = try arena.register(options.cap);
    defer arena.release(foreign, stream) catch unreachable;
    seed(arena, source, 0, 0, 0, 1, 0xa1);
    seed(arena, source, 1, 2, 1, 2, 0xa2);
    const saving = try arena.acquire(source);
    var image = try arena.saveOwnedBlocks(std.testing.allocator, saving);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(saving);
    try arena.release(source, stream);
    seed(arena, foreign, 0, 0, 0, 3, 0xb1);
    const destination = try arena.register(options.cap);
    defer arena.release(destination, stream) catch unreachable;
    try std.testing.expect(destination.generation > source.generation);
    const restoring = try arena.acquire(destination);
    try arena.restoreOwnedBlocks(restoring, &image);
    try arena.complete(restoring, stream);
    const saving_again = try arena.acquire(destination);
    var again = try arena.saveOwnedBlocks(std.testing.allocator, saving_again);
    defer again.deinit(std.testing.allocator);
    try arena.cancel(saving_again);
    try std.testing.expectEqual(@as(usize, 2), again.entries.len);
    try std.testing.expectEqualSlices(u8, image.payload, again.payload);
    try std.testing.expect(again.entries[0].source_slot != image.entries[0].source_slot);
    const foreign_saving = try arena.acquire(foreign);
    var foreign_image = try arena.saveOwnedBlocks(std.testing.allocator, foreign_saving);
    defer foreign_image.deinit(std.testing.allocator);
    try arena.cancel(foreign_saving);
    try std.testing.expectEqual(@as(usize, 1), foreign_image.entries.len);
    try std.testing.expect(std.mem.allEqual(u8, foreign_image.payload, 0xb1));
}

test "exl3: shared CUDA failed launches and completion retain allocations and lease until drain" {
    var driver = Fake.driver();
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const sequence = try arena.register(options.cap);
    const lease = try arena.acquire(sequence);
    Fake.fail_launch = true;
    try std.testing.expectError(error.NotReady, arena.page(lease, stream, 0, 0, 1, 1, 2, 3, 4, 5));
    try std.testing.expectError(error.SharedKvBusy, arena.destroy());
    try std.testing.expectError(error.SharedKvPoisoned, arena.cancel(lease));
    Fake.fail_sync = true;
    try std.testing.expectError(error.NotReady, arena.drainFailed(lease));
    try std.testing.expect(arena.active != null);
    Fake.fail_sync = false;
    Fake.fail_launch = false;
    try arena.drainFailed(lease);
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquire(sequence));
    try arena.release(sequence, stream);
    const reused = try arena.register(options.cap);
    try std.testing.expect(reused.generation > sequence.generation);
    try arena.release(reused, stream);
}

test "exl3: shared CUDA batch lease interleaves owners by layer without a host wait" {
    var driver = Fake.driver();
    var batch_options = options;
    batch_options.lanes = 3;
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, batch_options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const first = try arena.register(options.cap);
    defer arena.release(first, stream) catch unreachable;
    const second = try arena.register(options.cap);
    defer arena.release(second, stream) catch unreachable;
    const excluded = try arena.register(options.cap);
    defer arena.release(excluded, stream) catch unreachable;
    try std.testing.expectError(error.SharedKvLeaseOwners, arena.acquireMany(&.{}));
    try std.testing.expectError(error.SharedKvLeaseOwners, arena.acquireMany(&.{ first, first }));
    const lease = try arena.acquireMany(&.{ first, second });
    try std.testing.expectEqualSlices(contract.Sequence, &.{ first, second }, try arena.leaseOwners(lease));
    try std.testing.expectError(error.SharedKvLeaseOwner, arena.view(lease, excluded, 0));
    try std.testing.expectError(error.SharedKvLeaseOwner, arena.pageFor(lease, excluded, stream, 0, 0, 1, 1, 2, 3, 4, 5));
    var stale = second;
    stale.generation += 1;
    try std.testing.expectError(error.SharedKvSequence, arena.view(lease, stale, 0));
    const before = Fake.sync_calls;
    for (0..wt.attn_n) |layer| for ([_]contract.Sequence{ first, second }) |sequence| {
        Fake.expected_generation = sequence.generation;
        Fake.expected_key = @as(usize, sequence.lane) * arena.layout.lane_words + layer * arena.layout.blocks;
        Fake.expected_base = try arena.layout.slotBase(sequence.lane, layer);
        try arena.pageFor(lease, sequence, stream, layer, 0, 1, 1, 2, 3, 4, 5);
    };
    try std.testing.expectEqual(before, Fake.sync_calls);
    const alternate: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(2) };
    try std.testing.expectError(error.SharedKvStream, arena.pageFor(lease, second, alternate, 0, 0, 1, 1, 2, 3, 4, 5));
    try std.testing.expectError(error.SharedKvBusy, arena.release(second, stream));
    try std.testing.expectError(error.SharedKvSnapshotLease, arena.saveOwnedBlocks(std.testing.allocator, lease));
    try arena.completeAfterConsumers(lease);
    try std.testing.expectEqual(before, Fake.sync_calls);
    try std.testing.expectError(error.SharedKvLease, arena.completeAfterConsumers(lease));
    const next = try arena.acquire(excluded);
    try std.testing.expectError(error.SharedKvLease, arena.pageFor(lease, first, stream, 0, 0, 1, 1, 2, 3, 4, 5));
    try arena.cancel(next);
}

test "exl3: shared CUDA captured replay poisons every submitted batch owner until drain" {
    var driver = Fake.driver();
    var batch_options = options;
    batch_options.lanes = 3;
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, batch_options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const first = try arena.register(options.cap);
    const second = try arena.register(options.cap);
    const untouched = try arena.register(options.cap);
    const lease = try arena.acquireMany(&.{ first, second });
    try arena.submitted(lease);
    try arena.fail(lease);
    try std.testing.expect(arena.lanes[first.lane].failed);
    try std.testing.expect(arena.lanes[second.lane].failed);
    try std.testing.expect(!arena.lanes[untouched.lane].failed);
    try std.testing.expectError(error.SharedKvPoisoned, arena.completeAfterConsumers(lease));
    try std.testing.expectError(error.SharedKvBusy, arena.acquire(untouched));
    try arena.drainFailed(lease);
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquireMany(&.{ first, second }));
    const other = try arena.acquire(untouched);
    try arena.cancel(other);
    try arena.release(first, stream);
    try arena.release(second, stream);
    try arena.release(untouched, stream);
}

test "exl3: shared CUDA constructor allocation and upload failures refund all device ownership" {
    for ([_]usize{ 1, 2, 3 }) |failure| {
        var driver = Fake.driver();
        const before = cuda.usage(false).device;
        Fake.fail_alloc_at = failure;
        try std.testing.expectError(error.NotReady, shared.Arena.initStorage(std.testing.allocator, &driver, options));
        try std.testing.expectEqual(before, cuda.usage(false).device);
    }
    var driver = Fake.driver();
    const before = cuda.usage(false).device;
    Fake.fail_copy = 1;
    try std.testing.expectError(error.NotReady, shared.Arena.initStorage(std.testing.allocator, &driver, options));
    try std.testing.expectEqual(before, cuda.usage(false).device);
}

test "exl3: shared CUDA repeated empty snapshots advance one epoch without absolute clock amplification" {
    var driver = Fake.driver();
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const sequence = try arena.register(options.cap);
    defer arena.release(sequence, stream) catch unreachable;
    const tables: [*]u8 = @ptrFromInt(arena.tables.ptr);
    std.mem.writeInt(u32, tables[arena.layout.clock..][0..4], 1_000_000_000, .little);
    for (0..8) |iteration| {
        const saving = try arena.acquire(sequence);
        var image = try arena.saveOwnedBlocks(std.testing.allocator, saving);
        defer image.deinit(std.testing.allocator);
        try arena.cancel(saving);
        const restoring = try arena.acquire(sequence);
        try arena.restoreOwnedBlocks(restoring, &image);
        try arena.complete(restoring, stream);
        try std.testing.expectEqual(@as(u32, @intCast(1_000_000_001 + iteration)), std.mem.readInt(u32, tables[arena.layout.clock..][0..4], .little));
    }
}
