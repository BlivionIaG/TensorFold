//! Synthetic GGUF fixtures adapted from #1835 c5947a8d2e; no trained vectors are included.

const std = @import("std");
const glp = @import("glp.zig");
const gguf = @import("glp_container.zig");

/// A metadata key a test adds beside the standard ones.
pub const Extra = struct {
    key: []const u8,
    value: union(enum) { text: []const u8, u32v: u32, f32v: f32, boolv: bool, i32v: i32, zeros: usize, texts: []const []const u8 },
};

/// What a container holds, so a test can bend one rule at a time.
pub const Shape = struct {
    architecture: ?[]const u8 = "controlvector",
    mode: ?[]const u8 = "project",
    spec_version: ?u32 = 3,
    rank: ?u32 = 1,
    hook_point: ?[]const u8 = glp.site_name,
    derived_at: ?[]const u8 = null,
    alpha: f32 = 3.5,
    /// A numeric key written as text, or the alpha as a double.
    numeric_text: ?enum { alpha, rank, version, alpha_f64 } = null,
    layers: []const u32 = &.{ 11, 12 },
    width: usize = 8,
    /// How long each direction is: a unit vector, or this multiple of one.
    scale: f32 = 1.0,
    /// Every component's value, when a test needs one that is not finite or is zero.
    value: ?f32 = null,
    digest: enum { correct, wrong, absent, number } = .correct,
    /// The base repo as "Org/Name", written as the spec's general.base_model.0 name, organization and URL.
    base_model: ?[]const u8 = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    /// The base checkpoint's commit, written as general.base_model.0.version beside the base repo.
    base_revision: ?[]const u8 = null,
    /// The first tensor's offset as the header states it, for a file whose offset is not where its bytes are.
    first_offset: ?u64 = null,
    layer_ids_key: bool = true,
    layer_ids: ?[]const u8 = null,
    /// Tensor names to write instead of "direction.<id>", so a test can name one layer twice.
    tensor_names: ?[]const []const u8 = null,
    /// The ggml type of every tensor (0 is fp32) and its dimension count.
    kind: u32 = 0,
    two_dims: bool = false,
    extra: []const Extra = &.{},
};

/// The metadata a fixture writes: the key's bytes and how many keys there are.
const Kv = struct {
    bytes: std.ArrayList(u8) = .empty,
    count: u64 = 0,
    a: std.mem.Allocator,

    fn key(self: *Kv, name: []const u8, tag: u32) !void {
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, name.len)));
        try self.bytes.appendSlice(self.a, name);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(tag));
        self.count += 1;
    }

    fn text(self: *Kv, name: []const u8, value: []const u8) !void {
        try self.key(name, 8);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, value.len)));
        try self.bytes.appendSlice(self.a, value);
    }

    fn number(self: *Kv, name: []const u8, value: u32) !void {
        try self.key(name, 4);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(value));
    }

    fn signed(self: *Kv, name: []const u8, value: i32) !void {
        try self.key(name, 5);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(value));
    }

    fn real(self: *Kv, name: []const u8, value: f32) !void {
        try self.key(name, 6);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, @bitCast(value))));
    }

    fn double(self: *Kv, name: []const u8, value: f64) !void {
        try self.key(name, 12);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, @bitCast(value))));
    }

    fn boolean(self: *Kv, name: []const u8, value: bool) !void {
        try self.key(name, 7);
        try self.bytes.append(self.a, if (value) 1 else 0);
    }

    /// An array of `n` zero bytes: items a reader has to walk past.
    fn zeros(self: *Kv, name: []const u8, n: usize) !void {
        try self.key(name, 9);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, 0)));
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, n)));
        try self.bytes.appendNTimes(self.a, 0, n);
    }

    /// An array of strings.
    fn texts(self: *Kv, name: []const u8, items: []const []const u8) !void {
        try self.key(name, 9);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, 8)));
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, items.len)));
        for (items) |s| {
            try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, s.len)));
            try self.bytes.appendSlice(self.a, s);
        }
    }
};

/// The component `i` of the direction of layer `id`: a unit vector when the shape is, distinct per layer.
pub fn component(shape: Shape, id: u32, i: usize) f32 {
    if (shape.value) |v| return v;
    // Layer id lights component (id mod width) at twice the others', so a layer shift changes the numbers read back.
    const heavy: f32 = if (i == id % shape.width) 2.0 else 1.0;
    const norm = @sqrt(@as(f32, @floatFromInt(shape.width - 1)) + 4.0);
    return shape.scale * heavy / norm;
}

/// Write a container holding one direction per listed layer.
pub fn writeFixture(shape: Shape) ![]u8 {
    const a = std.testing.allocator;
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(a);
    var offsets: std.ArrayList(u64) = .empty;
    defer offsets.deinit(a);
    for (shape.layers) |id| {
        try offsets.append(a, payload.items.len);
        for (0..shape.width) |i| try payload.appendSlice(a, &std.mem.toBytes(@as(u32, @bitCast(component(shape, id, i)))));
    }
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload.items, &sha, .{});
    const hex = std.fmt.bytesToHex(sha, .lower);

    var kv: Kv = .{ .a = a };
    defer kv.bytes.deinit(a);
    if (shape.architecture) |arch| try kv.text("general.architecture", arch);
    try kv.number("general.alignment", 32);
    if (shape.mode) |m| try kv.text("glp.mode", m);
    if (shape.spec_version) |v| if (shape.numeric_text == .version) try kv.text("glp.spec_version", "3") else try kv.number("glp.spec_version", v);
    if (shape.numeric_text == .rank) try kv.text("glp.rank", "1") else if (shape.rank) |v| try kv.number("glp.rank", v);
    switch (shape.numeric_text orelse .rank) {
        .alpha => try kv.text("glp.alpha_default", "1"),
        .alpha_f64 => try kv.double("glp.alpha_default", shape.alpha),
        else => try kv.real("glp.alpha_default", shape.alpha),
    }
    if (shape.hook_point) |h| try kv.text("glp.hook_point", h);
    if (shape.derived_at) |d| try kv.text("glp.derived_at", d);
    if (shape.base_model) |b| {
        try kv.number("general.base_model.count", 1);
        if (std.mem.lastIndexOfScalar(u8, b, '/')) |at| {
            const url = try std.fmt.allocPrint(a, "https://huggingface.co/{s}", .{b});
            defer a.free(url);
            try kv.text("general.base_model.0.name", b[at + 1 ..]);
            try kv.text("general.base_model.0.organization", b[0..at]);
            try kv.text("general.base_model.0.repo_url", url);
        } else try kv.text("general.base_model.0.name", b);
        if (shape.base_revision) |r| try kv.text("general.base_model.0.version", r);
    }
    if (shape.layer_ids_key) {
        if (shape.layer_ids) |ids| {
            try kv.text("glp.layer_ids_zero_based", ids);
        } else {
            var list: std.ArrayList(u8) = .empty;
            defer list.deinit(a);
            for (shape.layers, 0..) |id, i| try list.print(a, "{s}{d}", .{ if (i > 0) "," else "", id });
            try kv.text("glp.layer_ids_zero_based", list.items);
        }
    }
    switch (shape.digest) {
        .correct => try kv.text("glp.content_sha256", &hex),
        .wrong => try kv.text("glp.content_sha256", "0000000000000000000000000000000000000000000000000000000000000000"),
        .number => try kv.number("glp.content_sha256", 7),
        .absent => {},
    }
    for (shape.extra) |e| switch (e.value) {
        .text => |s| try kv.text(e.key, s),
        .u32v => |x| try kv.number(e.key, x),
        .f32v => |x| try kv.real(e.key, x),
        .boolv => |x| try kv.boolean(e.key, x),
        .i32v => |x| try kv.signed(e.key, x),
        .zeros => |n| try kv.zeros(e.key, n),
        .texts => |items| try kv.texts(e.key, items),
    };

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "GGUF");
    try out.appendSlice(a, &std.mem.toBytes(@as(u32, 3)));
    try out.appendSlice(a, &std.mem.toBytes(@as(u64, shape.layers.len)));
    try out.appendSlice(a, &std.mem.toBytes(kv.count));
    try out.appendSlice(a, kv.bytes.items);
    for (shape.layers, offsets.items, 0..) |id, off, i| {
        var name_buf: [32]u8 = undefined;
        const name = if (shape.tensor_names) |names| names[i] else try std.fmt.bufPrint(&name_buf, "direction.{d}", .{id});
        try out.appendSlice(a, &std.mem.toBytes(@as(u64, name.len)));
        try out.appendSlice(a, name);
        try out.appendSlice(a, &std.mem.toBytes(@as(u32, if (shape.two_dims) 2 else 1)));
        try out.appendSlice(a, &std.mem.toBytes(@as(u64, shape.width)));
        if (shape.two_dims) try out.appendSlice(a, &std.mem.toBytes(@as(u64, 1)));
        try out.appendSlice(a, &std.mem.toBytes(shape.kind));
        try out.appendSlice(a, &std.mem.toBytes(if (i == 0) shape.first_offset orelse off else off));
    }
    while (out.items.len % 32 != 0) try out.append(a, 0);
    try out.appendSlice(a, payload.items);
    return out.toOwnedSlice(a);
}

/// A vector and the arena that holds its strings and layers.
pub const Read = struct {
    arena: std.heap.ArenaAllocator,
    vector: glp.Vector,

    pub fn deinit(self: *Read) void {
        self.arena.deinit();
    }
};

pub fn readOk(bytes: []const u8) !Read {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    errdefer arena.deinit();
    var why: glp.Why = .{};
    const v = glp.read(arena.allocator(), bytes, &why) catch |err| {
        std.debug.print("refused: {s}\n", .{why.text()});
        return err;
    };
    return .{ .arena = arena, .vector = v };
}

/// Fixture bytes, read, with the bytes freed after the read's arena.
pub fn accept(shape: Shape) !struct { bytes: []u8, read: Read } {
    const bytes = try writeFixture(shape);
    errdefer std.testing.allocator.free(bytes);
    return .{ .bytes = bytes, .read = try readOk(bytes) };
}

/// The reason a file was refused, copied out so it outlives the arena; fails when the file was accepted.
pub fn reason(bytes: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: glp.Why = .{};
    if (glp.read(arena.allocator(), bytes, &why)) |_| {
        return error.NotRefused;
    } else |err| {
        if (err != error.BadGlpVector) return err;
        return std.testing.allocator.dupe(u8, why.text());
    }
}

/// Fail unless a container built from `shape` is refused with a reason holding `needle`.
pub fn expectRefused(shape: Shape, needle: []const u8) !void {
    const bytes = try writeFixture(shape);
    defer std.testing.allocator.free(bytes);
    const why = try reason(bytes);
    defer std.testing.allocator.free(why);
    if (std.mem.indexOf(u8, why, needle) == null) {
        std.debug.print("refusal \"{s}\" lacks \"{s}\"\n", .{ why, needle });
        return error.WrongReason;
    }
}

/// A unit vector of a given width, pointing along one axis.
pub fn unitDir(a: std.mem.Allocator, width: usize, axis: usize) ![]f32 {
    const dir = try a.alloc(f32, width);
    @memset(dir, 0.0);
    dir[axis] = 1.0;
    return dir;
}

/// A container and the arena that holds its metadata.
pub const Parsed = struct {
    arena: std.heap.ArenaAllocator,
    file: gguf.File,

    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }
};

/// The container a file holds, read with the generic reader.
pub fn parseFile(a: std.mem.Allocator, bytes: []const u8) !Parsed {
    var arena = std.heap.ArenaAllocator.init(a);
    errdefer arena.deinit();
    var problem: gguf.Problem = .{};
    const file = try gguf.parse(arena.allocator(), bytes, &problem);
    return .{ .arena = arena, .file = file };
}
