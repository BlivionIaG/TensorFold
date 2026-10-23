//! Lifetime placement priors: routed/metadata fingerprint; startup warmth may be discarded by prefill.
const std = @import("std");
const codec = @import("cuda_exl3_frequency.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const namespace = "flashnext-exl3-placement-v1";

pub const State = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    identity: codec.Identity,
    lock: std.Io.File,
    prior: ?codec.Counts = null,
    prefill: [codec.decoder_layers][codec.experts]u32 = @splat(@splat(0)),
    prior_loaded: bool = false,

    pub fn deinit(state: *State) void {
        state.lock.close(state.io);
        state.gpa.free(state.path);
        state.* = undefined;
    }

    pub fn save(state: *State, own: anytype) void {
        var counts = snapshot(own, @import("cuda_exl3_dplan.zig").current()) catch |err| {
            std.log.err("EXL3 placement profile rank {d}: unsafe snapshot skipped ({s})", .{ state.identity.rank, @errorName(err) });
            return;
        };
        counts.prefill = state.prefill;
        codec.write(state.gpa, state.io, state.path, state.identity, &counts) catch |err| {
            std.log.warn("EXL3 placement profile rank {d}: save failed ({s})", .{ state.identity.rank, @errorName(err) });
            return;
        };
        std.log.info("EXL3 placement profile rank {d}: saved decoder/MTP lifetime placement snapshot to {s}", .{ state.identity.rank, state.path });
    }

    pub fn restore(state: *State, own: anytype) !bool {
        return restorePools(state, own);
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, own: anytype, world: u32, rank: u32) !?State {
    const root = std.c.getenv("TF_FLASHNEXT_EXL3_PROFILE_DIR") orelse return null;
    if (std.mem.span(root).len == 0) return null;
    if (comptime @import("builtin").os.tag == .windows) {
        std.log.warn("EXL3 placement profiles require POSIX storage; persistence disabled", .{});
        return null;
    }
    const layers = own.engine.layers[0..own.engine.n_layers];
    if (!available(layers)) {
        std.log.warn("EXL3 placement profile rank {d}: unavailable routed banks; persistence disabled", .{rank});
        return null;
    }
    const before = std.Io.Clock.awake.now(io).toNanoseconds();
    const model = fingerprint(gpa, io, dir, layers, own.head) catch |err| {
        std.log.warn("EXL3 placement profile rank {d}: fingerprint unavailable ({s}); persistence disabled", .{ rank, @errorName(err) });
        return null;
    };
    std.log.info("EXL3 placement profile rank {d}: routed-bank/metadata fingerprint took {d:.3} s (excludes router/dense weights)", .{ rank, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - before)) / 1e9 });
    var state = open(gpa, io, std.mem.span(root), .{ .model = model, .world = world, .rank = rank }) catch |err| {
        std.log.warn("EXL3 placement profile rank {d}: lock/storage unavailable ({s}); persistence disabled", .{ rank, @errorName(err) });
        return null;
    };
    errdefer state.deinit();
    state.prior = codec.read(gpa, io, state.path, state.identity) catch |err| blk: {
        std.log.info("EXL3 placement profile rank {d}: cold start ({s})", .{ rank, @errorName(err) });
        break :blk null;
    };
    state.prior_loaded = true;
    if (state.prior) |*counts| {
        if (counts.mtp_present != mtpHasPool(own.head)) {
            std.log.warn("EXL3 placement profile rank {d}: cold start (MTP schema mismatch)", .{rank});
            state.prior = null;
        } else {
            decay(counts);
            state.prefill = counts.prefill;
        }
    }
    return state;
}

fn open(gpa: std.mem.Allocator, io: std.Io, root: []const u8, identity: codec.Identity) !State {
    const private = try std.fs.path.join(gpa, &.{ root, namespace });
    defer gpa.free(private);
    var directory = try std.Io.Dir.cwd().createDirPathOpen(io, private, .{
        .permissions = .fromMode(0o700),
        .open_options = .{ .iterate = true },
    });
    defer directory.close(io);
    try directory.setPermissions(io, .fromMode(0o700));
    const name = try std.fmt.allocPrint(gpa, "{s}-w{d}-r{d}.bin", .{ std.fmt.bytesToHex(&identity.model, .lower), identity.world, identity.rank });
    defer gpa.free(name);
    const path = try std.fs.path.join(gpa, &.{ private, name });
    errdefer gpa.free(path);
    const lock_path = try std.fmt.allocPrint(gpa, "{s}.lock", .{path});
    defer gpa.free(lock_path);
    const lock = try openLock(io, lock_path);
    errdefer lock.close(io);
    try lock.setPermissions(io, .fromMode(0o600));
    return .{ .gpa = gpa, .io = io, .path = path, .identity = identity, .lock = lock };
}

fn openLock(io: std.Io, path: []const u8) !std.Io.File {
    const file: std.Io.File = if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform else .{
        .handle = try std.posix.openat(std.Io.Dir.cwd().handle, path, .{
            .ACCMODE = .RDWR,
            .CREAT = true,
            .NONBLOCK = true,
            .CLOEXEC = true,
            .NOCTTY = true,
            .NOFOLLOW = true,
        }, 0o600),
        .flags = .{ .nonblocking = true },
    };
    errdefer file.close(io);
    if ((try file.stat(io)).kind != .file) return error.ProfileLockNotRegular;
    if (!try file.tryLock(io, .exclusive)) return error.WouldBlock;
    return file;
}

fn restorePools(state: *State, own: anytype) !bool {
    if (state.prior_loaded and state.prior == null) return false;
    const counts = state.prior orelse blk: {
        var disk = codec.read(state.gpa, state.io, state.path, state.identity) catch |err| {
            std.log.warn("EXL3 placement profile rank {d}: cold start ({s})", .{ state.identity.rank, @errorName(err) });
            return false;
        };
        if (disk.mtp_present != (mtpPool(own.head) != null)) return false;
        decay(&disk);
        break :blk disk;
    };
    if (!poolsAvailable(own.engine.layers[0..own.engine.n_layers])) return error.ProfileUnavailable;
    const layers = own.engine.layers[0..own.engine.n_layers];
    for (layers, 0..) |layer, index| {
        const pool = layer.exl3.?.pool.?;
        pool.counts[pool.opt.layer] = counts.decoder[index];
    }
    if (mtpPool(own.head)) |pool| pool.counts[pool.opt.layer] = counts.mtp;
    std.log.info("EXL3 placement profile rank {d}: imported decoder/MTP lifetime counts, decayed once by 2 (nonzero minimum 1)", .{state.identity.rank});
    var warmed: usize = 0;
    for (layers) |layer| warmed += try layer.exl3.?.pool.?.warmProfile();
    if (mtpPool(own.head)) |pool| warmed += try pool.warmProfile();
    std.log.info("EXL3 placement profile rank {d}: warmed {d} expert lines", .{ state.identity.rank, warmed });
    return true;
}

fn available(layers: anytype) bool {
    if (layers.len != 48) return false;
    for (layers) |layer| {
        const experts = layer.exl3 orelse return false;
        if (experts.bank == null or experts.routed.len == 0) return false;
    }
    return true;
}

test "exl3: placement history is readable before device pools attach" {
    var mock: Mock = undefined;
    mock.init();
    for (&mock.experts) |*experts| experts.pool = null;
    try std.testing.expect(available(&mock.engine.layers));
    mock.experts[17].bank = null;
    try std.testing.expect(!available(&mock.engine.layers));
}

fn poolsAvailable(layers: anytype) bool {
    if (layers.len != 48) return false;
    for (layers) |layer| {
        const experts = layer.exl3 orelse return false;
        const pool = experts.pool orelse return false;
        if (pool.opt.layer >= 48) return false;
    }
    return true;
}

fn mtpExperts(head: anytype) @TypeOf(head.?.layer.exl3) {
    const present = head orelse return null;
    return present.layer.exl3;
}

fn mtpHasPool(head: anytype) bool {
    const experts = mtpExperts(head) orelse return false;
    return experts.bank != null;
}

fn mtpPool(head: anytype) @TypeOf(head.?.layer.exl3.?.pool) {
    const present = head orelse return null;
    const experts = present.layer.exl3 orelse return null;
    return experts.pool;
}

fn decay(counts: *codec.Counts) void {
    for (&counts.decoder) |*layer| decayRow(layer);
    for (&counts.prefill) |*layer| decayRow(layer);
    decayRow(&counts.mtp);
}

fn decayRow(row: *[512]u32) void {
    for (row) |*count| {
        if (count.* != 0) count.* = @max(1, count.* / 2);
    }
}

fn snapshot(own: anytype, planner: anytype) !codec.Counts {
    const layers = own.engine.layers[0..own.engine.n_layers];
    if (!poolsAvailable(layers)) return error.ProfileUnavailable;
    try own.stream.synchronize();
    if (planner) |current| {
        if (current.indexOf(layers[0].exl3.?.pool.?) != null) {
            for (current.pools) |pool| {
                var ours = pool == mtpPool(own.head);
                for (layers) |layer| ours = ours or pool == layer.exl3.?.pool.?;
                if (!ours) return error.ProfilePlannerOwnerMismatch;
            }
            try current.toHost();
        }
    }
    var counts: codec.Counts = .{};
    for (layers, 0..) |layer, index| {
        const pool = layer.exl3.?.pool.?;
        try pool.lines.drain();
        counts.decoder[index] = pool.counts[pool.opt.layer];
    }
    if (mtpPool(own.head)) |pool| {
        try pool.lines.drain();
        counts.mtp = pool.counts[pool.opt.layer];
        counts.mtp_present = true;
    }
    return counts;
}

fn hashBytes(hash: *Sha256, bytes: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, bytes.len, .little);
    hash.update(&length);
    hash.update(bytes);
}

fn hashRouted(hash: *Sha256, routed: []const [3][]const u8) void {
    for (routed) |expert| for (expert) |projection| hashBytes(hash, projection);
}

fn fingerprint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layers: anytype, head: anytype) ![32]u8 {
    var hash: Sha256 = .init(.{});
    hashBytes(&hash, namespace);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        hashBytes(&hash, name);
        hashBytes(&hash, bytes);
    }
    hashBytes(&hash, "decoder-routed-trellis");
    for (layers) |layer| hashRouted(&hash, layer.exl3.?.routed);
    if (mtpExperts(head)) |experts| if (experts.bank != null) {
        hashBytes(&hash, "mtp-routed-trellis");
        hashRouted(&hash, experts.routed);
    };
    return hash.finalResult();
}

const MockPool = struct {
    counts: *[48][512]u32,
    opt: struct { layer: u8, lines: usize = 10 },
    src: struct { routed: []const [3][]const u8 = &.{.{ "gate", "up", "down" }} } = .{},
    lines: struct {
        drained: bool = false,
        pub fn drain(copies: *@This()) !void {
            copies.drained = true;
        }
    } = .{},
    warm_calls: usize = 0,
    fail_warm: bool = false,

    pub fn warmProfile(pool: *MockPool) !usize {
        pool.warm_calls += 1;
        if (pool.fail_warm) return error.MockCudaWarmFailure;
        return 1;
    }
};

const MockExperts = struct { pool: ?*MockPool, bank: ?u8 = 1, routed: []const [3][]const u8 = &.{.{ "gate", "up", "down" }} };
const MockLayer = struct { exl3: ?*MockExperts };
const Mock = struct {
    decoder: [48][512]u32 = @splat(@splat(0)),
    mtp: [48][512]u32 = @splat(@splat(0)),
    pools: [49]MockPool = undefined,
    experts: [49]MockExperts = undefined,
    head: struct { layer: MockLayer } = undefined,
    engine: struct { layers: [48]MockLayer, n_layers: usize = 48 } = undefined,

    fn init(mock: *Mock) void {
        mock.* = .{};
        mock.engine.n_layers = 48;
        for (&mock.pools, &mock.experts, 0..) |*pool, *experts, index| {
            pool.* = .{ .counts = if (index == 48) &mock.mtp else &mock.decoder, .opt = .{ .layer = if (index == 48) 0 else @intCast(index) } };
            experts.* = .{ .pool = pool };
            if (index < 48) mock.engine.layers[index] = .{ .exl3 = experts };
        }
        mock.head = .{ .layer = .{ .exl3 = &mock.experts[48] } };
    }

    const Stream = struct {
        syncs: usize = 0,
        fail: bool = false,
        pub fn synchronize(stream: *@This()) !void {
            if (stream.fail) return error.MockCudaSyncFailure;
            stream.syncs += 1;
        }
    };
    fn owner(mock: *Mock, drafting: bool) struct { engine: *@TypeOf(mock.engine), head: ?*@TypeOf(mock.head), stream: Stream } {
        return .{ .engine = &mock.engine, .head = if (drafting) &mock.head else null, .stream = .{} };
    }
};

const MockPlan = struct {
    pools: []const *MockPool,
    pulls: usize = 0,
    fail: bool = false,
    pub fn indexOf(plan: *MockPlan, pool: *const MockPool) ?usize {
        for (plan.pools, 0..) |held, index| if (held == pool) return index;
        return null;
    }
    pub fn toHost(plan: *MockPlan) !void {
        if (plan.fail) return error.MockCudaPullFailure;
        plan.pulls += 1;
        for (plan.pools, 0..) |pool, index| pool.counts[pool.opt.layer][7] = @intCast(100 + index);
    }
};

const test_identity: codec.Identity = .{ .model = @splat(42), .world = 2, .rank = 0 };

test "exl3: placement state lifetime lock conflicts, rank separation and private storage" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    var held = true;
    defer if (held) state.deinit();
    try std.testing.expectError(error.WouldBlock, open(allocator, io, root, test_identity));
    const empty: codec.Counts = .{};
    try codec.write(allocator, io, state.path, state.identity, &empty);
    try std.testing.expectError(error.WouldBlock, open(allocator, io, root, test_identity));
    var rank_identity = test_identity;
    rank_identity.rank = 1;
    var other_rank = try open(allocator, io, root, rank_identity);
    defer other_rank.deinit();
    try std.testing.expect(!std.mem.eql(u8, state.path, other_rank.path));
    const lock_stat = try state.lock.stat(io);
    try std.testing.expectEqual(@as(u32, 0o600), lock_stat.permissions.toMode() & 0o777);
    const directory = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(state.path).?, .{});
    defer directory.close(io);
    const directory_stat = try directory.stat(io);
    try std.testing.expectEqual(@as(u32, 0o700), directory_stat.permissions.toMode() & 0o777);
    state.deinit();
    held = false;
    var reopened = try open(allocator, io, root, test_identity);
    defer reopened.deinit();
}

test "exl3: placement state cold IO and MTP mismatch never import or warm; CUDA warm errors propagate" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    defer state.deinit();
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner(true);
    try std.testing.expect(!try restorePools(&state, &own));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = state.path, .data = "corrupt" });
    try std.testing.expect(!try restorePools(&state, &own));
    var counts: codec.Counts = .{};
    counts.decoder[0][1] = 123;
    try codec.write(allocator, io, state.path, state.identity, &counts);
    try std.testing.expect(!try restorePools(&state, &own));
    try std.testing.expectEqual(@as(u32, 0), mock.decoder[0][1]);
    try std.testing.expectEqual(@as(usize, 0), mock.pools[0].warm_calls);
    counts.mtp_present = true;
    try codec.write(allocator, io, state.path, state.identity, &counts);
    mock.pools[0].fail_warm = true;
    try std.testing.expectError(error.MockCudaWarmFailure, restorePools(&state, &own));
    try std.testing.expectEqual(@as(u32, 61), mock.decoder[0][1]);
    mock.engine.n_layers = 47;
    try std.testing.expect(!available(mock.engine.layers[0..mock.engine.n_layers]));
}

test "exl3: placement state decay and replacement snapshot do not add imported prior twice" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    defer state.deinit();
    var counts: codec.Counts = .{ .mtp_present = true };
    counts.decoder[0][0..5].* = .{ 0, 1, 2, 3, std.math.maxInt(u32) };
    counts.mtp[1] = 9;
    try codec.write(allocator, io, state.path, state.identity, &counts);
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner(true);
    try std.testing.expect(try restorePools(&state, &own));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1, 2147483647 }, mock.decoder[0][0..5]);
    try std.testing.expectEqual(@as(u32, 4), mock.mtp[0][1]);
    for (mock.pools) |pool| try std.testing.expectEqual(@as(usize, 1), pool.warm_calls);
    mock.decoder[0][1] += 8;
    mock.mtp[0][1] += 2;
    const saved = try snapshot(&own, @as(?*MockPlan, null));
    try codec.write(allocator, io, state.path, state.identity, &saved);
    const disk = try codec.read(allocator, io, state.path, state.identity);
    try std.testing.expectEqual(@as(u32, 9), disk.decoder[0][1]);
    mock.init();
    own = mock.owner(true);
    try std.testing.expect(try restorePools(&state, &own));
    try std.testing.expectEqual(@as(u32, 4), mock.decoder[0][1]);
    try std.testing.expectEqual(@as(u32, 3), mock.mtp[0][1]);
    const missing = try std.fs.path.join(allocator, &.{ root, "missing", "profile" });
    defer allocator.free(missing);
    try std.testing.expectError(error.FileNotFound, codec.write(allocator, io, missing, state.identity, &saved));
}

test "exl3: placement state pulls only owning planner before copying decoder and MTP counts" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner(true);
    var owned: [49]*MockPool = undefined;
    for (&mock.pools, &owned) |*pool, *target| target.* = pool;
    var plan: MockPlan = .{ .pools = &owned };
    const counts = try snapshot(&own, @as(?*MockPlan, &plan));
    try std.testing.expectEqual(@as(usize, 1), own.stream.syncs);
    try std.testing.expectEqual(@as(usize, 1), plan.pulls);
    try std.testing.expectEqual(@as(u32, 100), counts.decoder[0][7]);
    try std.testing.expectEqual(@as(u32, 147), counts.decoder[47][7]);
    try std.testing.expectEqual(@as(u32, 148), counts.mtp[7]);
    for (mock.pools) |pool| try std.testing.expect(pool.lines.drained);
    plan.fail = true;
    try std.testing.expectError(error.MockCudaPullFailure, snapshot(&own, @as(?*MockPlan, &plan)));
    own.stream.fail = true;
    try std.testing.expectError(error.MockCudaSyncFailure, snapshot(&own, @as(?*MockPlan, &plan)));
    own.stream.fail = false;
    var other: Mock = undefined;
    other.init();
    const foreign = [_]*MockPool{&other.pools[0]};
    var unrelated: MockPlan = .{ .pools = &foreign, .fail = true };
    _ = try snapshot(&own, @as(?*MockPlan, &unrelated));
    try std.testing.expectEqual(@as(usize, 0), unrelated.pulls);
    const mixed = [_]*MockPool{ &mock.pools[0], &other.pools[0] };
    unrelated.pools = &mixed;
    try std.testing.expectError(error.ProfilePlannerOwnerMismatch, snapshot(&own, @as(?*MockPlan, &unrelated)));
}

test "exl3: placement fingerprint reads routed and metadata bytes, ignores pool size and distinguishes MTP" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "config" });
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "index" });
    var mock: Mock = undefined;
    mock.init();
    const own = mock.owner(false);
    const original = try fingerprint(allocator, io, root, &mock.engine.layers, own.head);
    mock.pools[0].opt.lines = 23;
    try std.testing.expectEqual(original, try fingerprint(allocator, io, root, &mock.engine.layers, own.head));
    mock.experts[47].routed = &.{.{ "changed gate", "up", "down" }};
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers, own.head))));
    mock.experts[47].routed = &.{.{ "gate", "up", "down" }};
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "changed config" });
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers, own.head))));
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "config" });
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "changed index" });
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers, own.head))));
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "index" });
    const drafting = mock.owner(true);
    const with_mtp = try fingerprint(allocator, io, root, &mock.engine.layers, drafting.head);
    try std.testing.expect(!std.mem.eql(u8, &original, &with_mtp));
    mock.experts[48].routed = &.{.{ "changed mtp", "up", "down" }};
    try std.testing.expect(!std.mem.eql(u8, &with_mtp, &(try fingerprint(allocator, io, root, &mock.engine.layers, drafting.head))));
}

test "exl3: placement lock rejects nonregular storage without waiting for FIFO readers" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(path);
    try std.testing.expectError(error.ProfileLockNotRegular, openLock(io, "/dev/null"));
    const fifo = try std.fs.path.join(allocator, &.{ path, "profile.lock" });
    defer allocator.free(fifo);
    const terminated = try allocator.dupeSentinel(u8, fifo, 0);
    defer allocator.free(terminated);
    switch (@import("builtin").os.tag) {
        .linux => {
            const result = std.os.linux.mknodat(temporary.dir.handle, "profile.lock", std.posix.S.IFIFO | 0o600, 0);
            if (std.posix.errno(result) != .SUCCESS) return error.FifoCreateFailed;
        },
        .macos => {
            const Native = struct {
                extern "c" fn mkfifo(pathname: [*:0]const u8, mode: std.posix.mode_t) c_int;
            };
            if (Native.mkfifo(terminated, 0o600) != 0) return error.FifoCreateFailed;
        },
        else => return error.SkipZigTest,
    }
    try std.testing.expectError(error.ProfileLockNotRegular, openLock(io, fifo));
}
