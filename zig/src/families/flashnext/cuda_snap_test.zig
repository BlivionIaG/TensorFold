const std = @import("std");
const cuda = @import("cuda");
const snap = @import("cuda_snap.zig");
const pager = @import("cuda_exl3_kvpage.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const shared_fixture = @import("cuda_exl3_shared_kv_test.zig");

const Fake = struct {
    const Copy = struct { dst: u64, src: u64, len: usize };
    var queued: [512]Copy = undefined;
    var pending: usize = 0;
    var submitted: usize = 0;
    var fail_at: ?usize = null;
    var sync_calls: usize = 0;
    var fail_sync_at: ?usize = null;
    var fail_alloc: bool = false;
    var fail_mapping: bool = false;
    var allocations: [8]?[]align(16) u8 = @splat(null);
    var freed_pending: bool = false;

    fn reset() void {
        pending = 0;
        submitted = 0;
        fail_at = null;
        sync_calls = 0;
        fail_sync_at = null;
        fail_alloc = false;
        fail_mapping = false;
        freed_pending = false;
    }

    fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        submitted += 1;
        if (fail_at == submitted) return cuda.abi.error_not_ready;
        queued[pending] = .{ .dst = dst, .src = src, .len = len };
        pending += 1;
        return 0;
    }

    fn read(dst: ?*anyopaque, src: u64, len: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
        return copy(@intFromPtr(dst), src, len, stream);
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        sync_calls += 1;
        if (fail_sync_at == sync_calls) return cuda.abi.error_not_ready;
        for (queued[0..pending]) |item| {
            const dst: [*]u8 = @ptrFromInt(item.dst);
            const src: [*]const u8 = @ptrFromInt(item.src);
            @memcpy(dst[0..item.len], src[0..item.len]);
        }
        pending = 0;
        shared_fixture.Fake.graph_pending = false;
        return 0;
    }

    fn contextSync() callconv(.c) c_int {
        return sync(null);
    }

    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) c_int {
        if (fail_alloc) return cuda.abi.error_not_ready;
        for (&allocations) |*slot| if (slot.* == null) {
            const bytes = std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return 2;
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        return 2;
    }

    fn device(out: *u64, address: ?*anyopaque, _: c_uint) callconv(.c) c_int {
        if (fail_mapping) return cuda.abi.error_not_ready;
        out.* = @intFromPtr(address);
        return 0;
    }

    fn free(address: ?*anyopaque) callconv(.c) c_int {
        freed_pending = freed_pending or pending != 0;
        for (&allocations) |*slot| if (slot.*) |bytes| if (@intFromPtr(bytes.ptr) == @intFromPtr(address)) {
            std.testing.allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }

    fn driver() cuda.Driver {
        var driver_ = shared_fixture.Fake.driver();
        driver_.api.cuMemcpyDtoDAsync_v2 = copy;
        driver_.api.cuMemcpyDtoHAsync_v2 = read;
        driver_.api.cuStreamSynchronize = sync;
        driver_.api.cuCtxSynchronize = contextSync;
        driver_.api.cuMemHostAlloc = alloc;
        driver_.api.cuMemHostGetDevicePointer_v2 = device;
        driver_.api.cuMemFreeHost = free;
        return driver_;
    }
};

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    pg: pager.Paged,
    seq: prompt.Seq,

    fn buffer(allocator: std.mem.Allocator, driver: *const cuda.Driver, len: usize, value: u8) !cuda.DeviceBuffer {
        const bytes = try allocator.alignedAlloc(u8, .@"16", len);
        @memset(bytes, value);
        return .{ .d = driver, .ptr = @intFromPtr(bytes.ptr), .len = len, .borrowed = true };
    }

    fn buffers(allocator: std.mem.Allocator, driver: *const cuda.Driver, count: usize, len: usize) ![]cuda.DeviceBuffer {
        const out = try allocator.alloc(cuda.DeviceBuffer, count);
        for (out, 0..) |*item, index| item.* = try buffer(allocator, driver, len, @intCast(index));
        return out;
    }

    fn init(driver: *const cuda.Driver) !*Fixture {
        return initCap(driver, 8192);
    }

    fn initCap(driver: *const cuda.Driver, cap: usize) !*Fixture {
        const fixture = try std.testing.allocator.create(Fixture);
        errdefer std.testing.allocator.destroy(fixture);
        fixture.arena = .init(std.testing.allocator);
        errdefer fixture.arena.deinit();
        const allocator = fixture.arena.allocator();
        const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 16 };
        const layout = try pager.SnapshotLayout.init(cap, cfg, 0);
        const host = try allocator.alignedAlloc(u8, .@"16", 24 * cap * pager.row_bytes);
        @memset(host, 0x80);
        const status = try allocator.alignedAlloc(u8, .@"16", 12 * 8 * 4);
        fixture.pg = .{ .gpa = allocator, .d = driver, .k = undefined, .cfg = cfg, .cap = cap, .wr = layout.wr, .nblocks = layout.nblocks, .host = .{ .d = driver, .bytes = host }, .host_dev = @intFromPtr(host.ptr), .status_host = .{ .d = driver, .bytes = status }, .dev = try buffer(allocator, driver, layout.device, 0xFF), .tabs = try buffer(allocator, driver, layout.tables, 0), .work = undefined };
        fixture.seq = .{
            .cap = cap,
            .fed = 0,
            .history = .{ 73, 91 },
            .paged = &fixture.pg,
            .conv = try buffers(allocator, driver, wt.linear_n, (gdn.taps - 1) * gdn.conv_dim * 2),
            .state = try buffers(allocator, driver, wt.linear_n, wt.state_n * 4),
            .kc = &.{},
            .vc = &.{},
            .ikc = try buffers(allocator, driver, wt.attn_n, cap * wt.index_dim * 2),
            .pooled = try buffers(allocator, driver, wt.attn_n, prompt.pooledBytes(cap)),
            .tail = try buffer(allocator, driver, wt.ple_tail_n * wt.wide * 2, 0x7F),
            .logits = try allocator.alloc(u8, prompt.vocabulary * 2),
        };
        @memset(fixture.seq.logits, 0x81);
        return fixture;
    }

    fn deinit(fixture: *Fixture) void {
        if (fixture.pg.cfg.shared_lanes != 0) fixture.pg.shared_arena.?.release(fixture.pg.shared_sequence, .{ .d = fixture.pg.d, .handle = null }) catch unreachable;
        fixture.arena.deinit();
        std.testing.allocator.destroy(fixture);
    }
};

fn sharedFixture(driver: *const cuda.Driver, arena: *pager.shared_device.Arena) !*Fixture {
    const fixture = try Fixture.initCap(driver, arena.layout.options.cap);
    errdefer fixture.deinit();
    const sequence = try arena.register(arena.layout.options.cap);
    fixture.pg.cfg.shared_lanes = arena.layout.options.lanes;
    fixture.pg.shared_arena = arena;
    fixture.pg.shared_sequence = sequence;
    fixture.pg.dev = arena.device;
    fixture.pg.dev.borrowed = true;
    fixture.pg.tabs = arena.tables;
    fixture.pg.tabs.borrowed = true;
    fixture.pg.work = arena.work;
    fixture.pg.work.borrowed = true;
    return fixture;
}

test "exl3: native shared snapshot version 2 isolates windows canonical data and remapped entries" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 3, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    var source: ?*Fixture = try sharedFixture(&driver, arena);
    defer if (source) |fixture| fixture.deinit();
    const foreign = try sharedFixture(&driver, arena);
    defer foreign.deinit();
    source.?.seq.fed = 20;
    for (0..wt.attn_n) |layer| for ([_]bool{ false, true }) |values| {
        const target: [*]u8 = @ptrFromInt(source.?.pg.devRows(layer, values));
        @memset(target[0 .. source.?.pg.wr * pager.row_bytes], if (values) 0xa2 else 0xa1);
        const other: [*]u8 = @ptrFromInt(foreign.pg.devRows(layer, values));
        @memset(other[0 .. foreign.pg.wr * pager.row_bytes], if (values) 0xb2 else 0xb1);
    };
    shared_fixture.seed(arena, source.?.pg.shared_sequence, 0, 0, 0, 1, 0xc1);
    shared_fixture.seed(arena, source.?.pg.shared_sequence, 1, 1, 1, 2, 0xc2);
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.?.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    try std.testing.expectEqual(@as(u32, 2), saved.version);
    try std.testing.expectEqual(@as(usize, 2), saved.owned_blocks.?.entries.len);
    try std.testing.expect(try snap.pagedBytes(64, source.?.pg.cfg, 20, false) >= saved.buf.len + @sizeOf(snap.Snap) + saved.owned_blocks.?.payload.len + saved.owned_blocks.?.entries.len * @sizeOf(pager.shared_contract.OwnedEntry));
    const original_generation = source.?.pg.shared_sequence.generation;
    source.?.deinit();
    source = null;
    shared_fixture.seed(arena, foreign.pg.shared_sequence, 2, 2, 0, 3, 0xd1);
    const destination = try sharedFixture(&driver, arena);
    defer destination.deinit();
    try std.testing.expect(destination.pg.shared_sequence.generation > original_generation);
    const pending_graph: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    try pending_graph.launchOn(stream);
    try std.testing.expect(shared_fixture.Fake.graph_pending);
    try snap.restore(stream, &destination.seq, null, saved);
    try std.testing.expect(!shared_fixture.Fake.graph_pending);
    try std.testing.expectEqual(@as(usize, 20), destination.seq.fed);
    try std.testing.expect(!destination.seq.invalidated);
    for (0..wt.attn_n) |layer| for ([_]bool{ false, true }) |values| {
        const target: [*]const u8 = @ptrFromInt(destination.pg.devRows(layer, values));
        try std.testing.expect(std.mem.allEqual(u8, target[0 .. destination.pg.wr * pager.row_bytes], if (values) 0xa2 else 0xa1));
        const other: [*]const u8 = @ptrFromInt(foreign.pg.devRows(layer, values));
        try std.testing.expect(std.mem.allEqual(u8, other[0 .. foreign.pg.wr * pager.row_bytes], if (values) 0xb2 else 0xb1));
        try std.testing.expect(std.mem.allEqual(u8, foreign.pg.hostRows(layer, values), 0x80));
        try std.testing.expect(std.mem.allEqual(u8, destination.pg.hostRows(layer, values)[0 .. 20 * pager.row_bytes], if (values) 0xa2 else 0xa1));
    };
    const foreign_lease = try arena.acquire(foreign.pg.shared_sequence);
    var foreign_blocks = try arena.saveOwnedBlocks(std.testing.allocator, foreign_lease);
    defer foreign_blocks.deinit(std.testing.allocator);
    try arena.cancel(foreign_lease);
    try std.testing.expectEqual(@as(usize, 1), foreign_blocks.entries.len);
    try std.testing.expectEqual(@as(u64, 3), foreign_blocks.entries[0].stamp);
    try std.testing.expectEqual(@as(usize, 0), foreign.seq.fed);
    try std.testing.expectEqual([2]i64{ 73, 91 }, foreign.seq.history);
    try std.testing.expect(std.mem.allEqual(u8, foreign_blocks.payload, 0xd1));
    const again = try snap.save(std.testing.allocator, &driver, stream, &destination.seq, null);
    defer snap.drop(std.testing.allocator, stream, again);
    try std.testing.expectEqualSlices(u8, saved.host.?.bytes[0..saved.buf.len], again.host.?.bytes[0..again.buf.len]);
    try std.testing.expectEqualSlices(u8, saved.owned_blocks.?.payload, again.owned_blocks.?.payload);
    for (saved.owned_blocks.?.entries, again.owned_blocks.?.entries) |original_entry, restored_entry| {
        try std.testing.expectEqual(original_entry.key, restored_entry.key);
        try std.testing.expectEqual(saved.owned_blocks.?.clock - original_entry.stamp, again.owned_blocks.?.clock - restored_entry.stamp);
    }
    try std.testing.expect(saved.owned_blocks.?.entries[0].source_slot != again.owned_blocks.?.entries[0].source_slot);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: native shared snapshot failures poison the destination after submitted work and drain its lease" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 3, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const source = try sharedFixture(&driver, arena);
    defer source.deinit();
    const destination = try sharedFixture(&driver, arena);
    defer destination.deinit();
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    const before = Fake.submitted;
    saved.version = 1;
    try std.testing.expectError(error.SnapshotVersion, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!destination.seq.invalidated);
    saved.version = 2;
    saved.owned_blocks.?.clock = @as(u64, std.math.maxInt(u32)) + 1;
    try std.testing.expectError(error.Overflow, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!destination.seq.invalidated);
    saved.owned_blocks.?.clock = 0;
    Fake.fail_at = Fake.submitted + 2;
    try std.testing.expectError(error.NotReady, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expect(destination.seq.invalidated);
    try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    try std.testing.expect(arena.active == null);
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquire(destination.pg.shared_sequence));
    const available = try arena.acquire(source.pg.shared_sequence);
    try arena.cancel(available);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: shared pager Engine sequence churn retains only the common allocation and reuses lane generations" {
    Fake.reset();
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const baseline = cuda.usage(false).device;
    var engine: prompt.Engine = undefined;
    engine.gpa = std.testing.allocator;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 256;
    engine.eos = 0;
    engine.page_config = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    engine.shared_arena = arena;
    engine.round_graph = .{};
    try std.testing.expectEqual(arena.layout.common_bytes, try engine.reserveSharedKv(64));
    try std.testing.expectEqual(baseline, cuda.usage(false).device);
    var previous_generation: u64 = 0;
    for (0..3) |_| {
        var sequence = try engine.newSeq(64);
        const pg = sequence.paged.?;
        try std.testing.expect(pg.dev.borrowed and pg.tabs.borrowed and pg.work.borrowed);
        try std.testing.expectEqual(arena.device.ptr, pg.dev.ptr);
        try std.testing.expectEqual(@as(usize, 0), pg.deviceBytes());
        try std.testing.expect(pg.shared_sequence.generation > previous_generation);
        previous_generation = pg.shared_sequence.generation;
        sequence.deinit(std.testing.allocator);
        try std.testing.expectEqual(baseline, cuda.usage(false).device);
    }
    try std.testing.expectError(error.SharedKvCommonCap, engine.newSeq(63));
    try std.testing.expectEqual(baseline, cuda.usage(false).device);
}

test "exl3: shared pager constructor mapping failures retire registrations without touching foreign slots" {
    Fake.reset();
    var driver = Fake.driver();
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const foreign = try arena.register(64);
    defer arena.release(foreign, stream) catch unreachable;
    shared_fixture.seed(arena, foreign, 0, 0, 0, 1, 0xb1);
    const baseline = cuda.usage(false);
    const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    Fake.fail_mapping = true;
    try std.testing.expectError(error.NotReady, pager.Paged.initShared(std.testing.allocator, &driver, 64, cfg, arena));
    Fake.fail_mapping = false;
    try std.testing.expectEqual(baseline.device, cuda.usage(false).device);
    try std.testing.expectEqual(baseline.host, cuda.usage(false).host);
    const available = try arena.acquire(foreign);
    var image = try arena.saveOwnedBlocks(std.testing.allocator, available);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(available);
    try std.testing.expectEqual(@as(usize, 1), image.entries.len);
    try std.testing.expect(std.mem.allEqual(u8, image.payload, 0xb1));
    const pg = try pager.Paged.initShared(std.testing.allocator, &driver, 64, cfg, arena);
    pg.destroy();
    try std.testing.expectEqual(baseline.device, cuda.usage(false).device);
    try std.testing.expectEqual(baseline.host, cuda.usage(false).host);
}

test "exl3: shared snapshot mark preflight cancels its unsubmitted lease without invalidating the source" {
    Fake.reset();
    var driver = Fake.driver();
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const source = try sharedFixture(&driver, arena);
    defer source.deinit();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var draft: @import("cuda_mtp_draft.zig").Draft = undefined;
    draft.len = 1;
    const before = Fake.submitted;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &source.seq, &draft));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!source.seq.invalidated);
    try std.testing.expect(arena.active == null);
    const lease = try arena.acquire(source.pg.shared_sequence);
    try arena.cancel(lease);
}

test "exl3: shared pager common charge admission and snapshot formats are independently checked" {
    const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 1024, .shared_lanes = 8 };
    const cap = 8192;
    const common = try pager.sharedBytesFor(cap, cfg);
    const layout = try pager.shared_device.Layout.init(.{ .cap = cap, .lanes = 8, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots });
    try std.testing.expectEqual(layout.common_bytes, common);
    try std.testing.expectEqual(@as(usize, 0), try pager.deviceBytesFor(cap, cfg));
    try std.testing.expectEqual(@as(u64, 800), try pager.admissionRoom(10000, 100, cfg));
    try std.testing.expectEqual(@as(u64, 799), try pager.admissionRoom(799, 100, cfg));
    try std.testing.expectError(error.Overflow, pager.admissionRoom(10000, std.math.maxInt(usize), cfg));
    const shared_layout = try pager.SnapshotLayout.init(cap, cfg, 64);
    var private_cfg = cfg;
    private_cfg.shared_lanes = 0;
    const private_layout = try pager.SnapshotLayout.init(cap, private_cfg, 64);
    try std.testing.expect(shared_layout.device < private_layout.device);
    try std.testing.expect(shared_layout.tables < private_layout.tables);
    try std.testing.expect(pager.configIdentity(cfg) != pager.configIdentity(private_cfg));
    try std.testing.expect(try snap.pagedBytes(cap, cfg, 64, false) > shared_layout.total + try pager.ownedSnapshotBytes(cfg));
    var bad = cfg;
    bad.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, pager.sharedBytesFor(cap, bad));
}

test "exl3: shared pager local keys bind generations and arena incarnations while TP keys remain logical" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const options: pager.shared_device.Options = .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const owner = try arena.register(64);
    var pg: pager.Paged = undefined;
    pg.cfg = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    pg.cap = 64;
    pg.wr = arena.layout.window_rows;
    pg.win0 = @splat(0);
    pg.dev = arena.device;
    pg.tabs = arena.tables;
    pg.work = arena.work;
    pg.host_dev = 1234;
    pg.shared_arena = arena;
    pg.shared_sequence = owner;
    const local = pg.graphKey();
    const logical = pg.logicalGraphKey();
    try arena.release(owner, stream);
    const replacement = try arena.register(64);
    defer arena.release(replacement, stream) catch unreachable;
    pg.shared_sequence = replacement;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
    const other = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, options);
    defer other.destroy() catch unreachable;
    shared_fixture.Fake.install(other);
    const placeholder = try other.register(64);
    defer other.release(placeholder, stream) catch unreachable;
    const peer = try other.register(64);
    defer other.release(peer, stream) catch unreachable;
    pg.shared_arena = other;
    pg.shared_sequence = peer;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
    pg.shared_sequence = placeholder;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
}

test "exl3: paged snapshot byte admission is checked and includes fixed payload and header" {
    const cfg: pager.Config = .{};
    const layout = try pager.SnapshotLayout.init(200_256, cfg, 200_000);
    try std.testing.expect(try snap.pagedBytes(200_256, cfg, 200_000, true) > layout.total + @sizeOf(snap.Snap));
    try std.testing.expectError(error.Overflow, snap.pagedBytes(std.math.maxInt(usize), cfg, 1, false));
    try std.testing.expectError(error.KvPageRows, snap.pagedBytes(3, cfg, 4, false));
    var image: pager.Snapshot = .{ .layout = layout, .win0 = @splat(196_000), .rebases = @splat(3), .written = @splat(8192) };
    try image.validate();
    image.layout.total -= 1;
    try std.testing.expectError(error.KvPageSnapshotLayout, image.validate());
}

test "exl3: paged snapshot restores distinct allocations and all state without source ownership" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var source: ?*Fixture = try Fixture.init(&driver);
    defer if (source) |fixture| fixture.deinit();
    const destination = try Fixture.init(&driver);
    defer destination.deinit();
    source.?.seq.fed = 4099;
    source.?.pg.win0 = @splat(2048);
    source.?.pg.win0[1] = 2052;
    source.?.pg.rebases = @splat(8);
    source.?.pg.written = @splat(2052);
    const controls: [*]u32 = @ptrFromInt(source.?.pg.tabs.ptr + 12 * 2 * source.?.pg.nblocks * 4);
    for (0..12) |layer| {
        controls[layer * 8] = 73;
        controls[layer * 8 + 6] = @intCast(17 + layer);
    }
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.?.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    try std.testing.expectEqual(saved.buf.len + @sizeOf(snap.Snap), try snap.pagedBytes(source.?.seq.cap, source.?.pg.cfg, saved.at, false));
    const expected = try std.testing.allocator.dupe(u8, saved.host.?.bytes[0..saved.buf.len]);
    defer std.testing.allocator.free(expected);
    source.?.deinit();
    source = null;
    const device_address = destination.pg.dev.ptr;
    for (0..2) |_| {
        try snap.restore(stream, &destination.seq, null, saved);
        try std.testing.expectEqual(device_address, destination.pg.dev.ptr);
        try std.testing.expectEqual(saved.history, destination.seq.history);
        try std.testing.expectEqual(saved.at, destination.seq.fed);
        try std.testing.expect(!destination.pg.graph_prepared and !destination.seq.invalidated);
        const again = try snap.save(std.testing.allocator, &driver, stream, &destination.seq, null);
        defer snap.drop(std.testing.allocator, stream, again);
        try std.testing.expectEqualDeep(saved.paged.?, again.paged.?);
        try std.testing.expectEqualSlices(u8, expected, again.host.?.bytes[0..again.buf.len]);
    }
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: paged snapshot submission failures drain and a partial restore stays invalid" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 4;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    for (1..125) |failure| {
        Fake.submitted = 0;
        Fake.fail_at = failure;
        fixture.seq.invalidated = false;
        const previous = fixture.seq.fed;
        try std.testing.expectError(error.NotReady, snap.restore(stream, &fixture.seq, null, saved));
        try std.testing.expect(fixture.seq.invalidated);
        try std.testing.expectEqual(previous, fixture.seq.fed);
        try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    }
    Fake.fail_at = null;
    fixture.seq.invalidated = false;
    var incompatible = saved.*;
    incompatible.version = 2;
    const submissions = Fake.submitted;
    try std.testing.expectError(error.SnapshotVersion, snap.restore(stream, &fixture.seq, null, &incompatible));
    try std.testing.expectEqual(submissions, Fake.submitted);
    try std.testing.expect(!fixture.seq.invalidated);
    fixture.pg.cfg.slots += 1;
    try std.testing.expectError(error.KvPageSnapshotLayout, snap.restore(stream, &fixture.seq, null, saved));
    try std.testing.expectEqual(submissions, Fake.submitted);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: paged snapshot preserves MTP head mark and rejects stale verification publication" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    const allocator = fixture.arena.allocator();
    var head: @import("cuda_mtp_draft.zig").Draft = undefined;
    head.cap = fixture.seq.cap;
    head.kc = try Fixture.buffer(allocator, &driver, head.cap * prompt.kv_row_bytes, 0x80);
    head.vc = try Fixture.buffer(allocator, &driver, head.cap * prompt.kv_row_bytes, 0x7F);
    head.ikc = try Fixture.buffer(allocator, &driver, head.cap * wt.index_dim * 2, 0xFC);
    head.pooled = try Fixture.buffer(allocator, &driver, prompt.pooledBytes(head.cap), 0xA5);
    head.streams = try Fixture.buffer(allocator, &driver, 2 * wt.wide * 2, 0xFF);
    head.prompt_last = head.streams.ptr + wt.wide * 2;
    head.len = 3;
    head.drafted = 0;
    head.open = false;
    head.verify_cache = .{};
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head);
    defer snap.drop(std.testing.allocator, stream, saved);
    head.drafted = 1;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head));
    head.drafted = 0;
    head.open = true;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head));
    head.open = false;
    const generation = head.verify_cache.generation;
    head.verify_cache.publish(generation, 3, 3);
    try snap.restore(stream, &fixture.seq, &head, saved);
    head.verify_cache.publish(generation, 3, 3);
    try std.testing.expect(!head.verify_cache.take(3, 3, true));
    try std.testing.expectEqual(@as(usize, 3), head.len);
    try std.testing.expect(head.drafted == 0 and head.rows == 0 and !head.open);
    try std.testing.expectEqual(head.streams.ptr, head.prompt_last);
}

test "exl3: paged restore stages data before publishing sequence metadata" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    fixture.seq.fed = 7;
    fixture.seq.history = .{ 111, 222 };
    try snap.restoreData(stream, &fixture.seq, null, saved);
    try std.testing.expectEqual(@as(usize, 7), fixture.seq.fed);
    try std.testing.expectEqual([2]i64{ 111, 222 }, fixture.seq.history);
    try std.testing.expect(fixture.seq.invalidated);
    snap.publishRestore(&fixture.seq, null, saved);
    try std.testing.expectEqual(@as(usize, 3), fixture.seq.fed);
    try std.testing.expectEqual(saved.history, fixture.seq.history);
    try std.testing.expect(!fixture.seq.invalidated);
}

test "exl3: paged save allocation, mapping and completion errors never publish or free pending copies" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, snap.save(allocator.allocator(), &driver, stream, &fixture.seq, null));
    Fake.fail_alloc = true;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    Fake.fail_alloc = false;
    Fake.fail_mapping = true;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    Fake.fail_mapping = false;
    try std.testing.expect(!fixture.seq.invalidated);
    Fake.sync_calls = 0;
    Fake.fail_sync_at = 3;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    try std.testing.expect(fixture.seq.invalidated);
    try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    try std.testing.expect(!Fake.freed_pending);
    for (Fake.allocations) |allocation| try std.testing.expect(allocation == null);
}

test "exl3: paged snapshot preflight sync failures drain and invalidate the sequence" {
    for (1..3) |failure_at| {
        Fake.reset();
        var driver = Fake.driver();
        const stream: cuda.Stream = .{ .d = &driver, .handle = null };
        const fixture = try Fixture.init(&driver);
        defer fixture.deinit();
        fixture.seq.fed = 3;
        Fake.fail_sync_at = failure_at;
        try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
        try std.testing.expect(fixture.seq.invalidated);
        try std.testing.expectEqual(@as(usize, 0), Fake.pending);
        try std.testing.expect(!Fake.freed_pending);
    }
}

test "exl3: TP2 paged restore prepares both ranks and commits only after copy acknowledgements" {
    const wire = @import("cuda_rank_wire.zig");
    const ranks = @import("cuda_rank_engine.zig");
    Fake.reset();
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 4;
    fixture.seq.rank_id = 9;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    saved.rank_id = 13;
    var engine: prompt.Engine = undefined;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.round_graph = .{};
    for ([_]bool{ false, true }) |reject| {
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer _ = std.c.close(pair[0]);
        defer _ = std.c.close(pair[1]);
        const peer: wire.Link = .{ .fd = pair[1] };
        var session: ranks.Session = .{ .gpa = std.testing.allocator, .rank = 0, .engine = &engine, .link = .{ .fd = pair[0] }, .collective = undefined };
        session.collective.comm = null;
        session.collective.failed = false;
        fixture.seq.fed = 8;
        fixture.seq.invalidated = false;
        for (1..if (reject) @as(usize, 2) else 4) |sequence| try peer.sendHeader(.{ .kind = .ack, .sequence = sequence, .object = 9, .flags = if (reject) 2 else 0 }, wire.deadline(1000));
        const before = Fake.submitted;
        if (reject) {
            try std.testing.expectError(error.RemoteRank, session.restore(&fixture.seq, null, saved));
            try std.testing.expectEqual(before, Fake.submitted);
            try std.testing.expectEqual(@as(usize, 8), fixture.seq.fed);
            try std.testing.expect(fixture.seq.invalidated and !session.available.load(.acquire));
        } else {
            try session.restore(&fixture.seq, null, saved);
            try std.testing.expectEqual(saved.at, fixture.seq.fed);
            for ([_]wire.Kind{ .restore_check, .restore, .restore_commit }, 1..) |kind, sequence| {
                const command = try peer.receiveHeader(wire.deadline(1000));
                try std.testing.expectEqual(kind, command.kind);
                try std.testing.expectEqual(sequence, command.sequence);
                try std.testing.expectEqual(@as(u64, 13), command.arg);
            }
        }
    }
}
