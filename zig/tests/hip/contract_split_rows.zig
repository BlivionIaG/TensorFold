//! The split offload's boundary rows under the contract on HIP: exact checks by value, with planted controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = hip.launches.split_rows;
const ref = hip.ops.split_rows;

fn none(_: c.Shape) usize {
    return 0;
}

/// Import and export one row each; the stamped consume, with a stale group (status 1); and two controls.
pub const Mode = enum { import, @"export", consume, timeout, short_import, stale_seq };

const rule = "every value is its input converted to the output kind (nearest even; NaN and infinity stay such), the status words are the kernel's";

/// Qwen3.5-4B's hidden 2560, a width whose last group is two thirds full, and one that needs more than one pass.
const widths = [_]usize{ 2560, 2561, 5120 };

pub fn spec(m: Mode) c.Spec {
    const name = switch (m) {
        .import => "hip tf_split_import_row",
        .@"export" => "hip tf_split_export_row",
        .consume => "hip tf_split_consume_stamped",
        .timeout => "hip tf_split_consume_stamped (a stale group: status 1)",
        .short_import => "control: tf_split_import_row converts one value short",
        .stale_seq => "control: tf_split_consume_stamped waits for the next sequence number",
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = rule },
        // one decode row, and a few rows as the nonfinite gate takes
        .served = &.{
            .{ .rows = 1, .n = 1, .k = widths[0] },
            .{ .rows = 4, .n = 1, .k = widths[1] },
            .{ .rows = 2, .n = 1, .k = widths[2] },
        },
    };
}

const seq: u32 = 0x5eed_0001;
const retries: u32 = 3;

pub const Split = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    m: Mode,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    /// the input values as fp32 bits (import, consume) or the kind's bytes (export)
    x: []u32 = &.{},
    xk: []u8 = &.{},
    src: u64 = 0,
    dst: u64 = 0,
    status: u64 = 0,
    stale_group: usize = 0,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) Split {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .m = m, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Split) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Split) g.Kernel {
        const control: ?g.Gate = if (self.m == .short_import or self.m == .stale_seq) .shapes else null;
        return .{ .spec = spec(self.m), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Split);

    fn stamped(self: *const Split) bool {
        return self.m == .consume or self.m == .timeout or self.m == .stale_seq;
    }

    fn kindInt(self: *const Split) c_int {
        return @backingInt(self.k);
    }

    fn rowBytes(self: *const Split) usize {
        return self.shape.k * if (self.m == .@"export") kit.size(self.k) else 4;
    }

    fn outRowBytes(self: *const Split) usize {
        return self.shape.k * if (self.m == .@"export") 4 else kit.size(self.k);
    }

    fn stride(self: *const Split) usize {
        return if (self.stamped()) ref.stampedBytes(self.shape.k) else self.rowBytes();
    }

    /// Random values with the tie and overflow cases at the head of each row, NaN and infinities when asked.
    pub fn prepare(self: *Split, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n != 1) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        const special = [_]f32{ 70000.0, 65504.0, 1.0 + 0x1p-8, 1.0 + 0x1p-11 };
        self.x = try a.alloc(u32, n);
        for (self.x, 0..) |*v, i| {
            const at = i % shape.k;
            const f: f32 = if (at < special.len) special[at] else @floatCast(rng.float(f64) * 16 - 8);
            v.* = @bitCast(f);
        }
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
            self.x[(i * 2711 + 9) % n] = @bitCast(v);
        };
        if (self.m == .@"export") {
            self.xk = try a.alloc(u8, n * kit.size(self.k));
            for (self.x, 0..) |v, i| kit.put(self.k, self.xk, i, @as(f32, @bitCast(v)));
            self.src = try self.rig.input(self.xk);
            self.dst = try self.rig.output(n * 4);
            return true;
        }
        self.dst = try self.rig.output(n * kit.size(self.k));
        if (!self.stamped()) {
            self.src = try self.rig.input(std.mem.sliceAsBytes(self.x));
            return true;
        }
        const groups = rows.groups(shape.k);
        const region = try a.alloc(u32, shape.rows * groups * 4);
        for (0..shape.rows) |r| {
            self.stale_group = groups / 2;
            for (0..groups) |gi| {
                const word = region[(r * groups + gi) * 4 ..][0..4];
                for (0..3) |j| {
                    const at = gi * 3 + j;
                    // a padding slot holds a value that must never reach the output
                    word[j] = if (at < shape.k) self.x[r * shape.k + at] else @bitCast(@as(f32, 1.0e30));
                }
                word[3] = if (self.m == .timeout and gi == self.stale_group) seq -% 1 else seq +% @as(u32, @intCast(r));
            }
        }
        self.src = try self.rig.input(std.mem.sliceAsBytes(region));
        self.status = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Split) !void {
        const s = self.shape;
        const st = self.rig.s.handle;
        for (0..s.rows) |r| {
            switch (self.m) {
                .import => try rows.importRow(self.l, self.src + r * self.rowBytes(), self.dst + r * self.outRowBytes(), self.kindInt(), s.k, st),
                .short_import => try rows.importRow(self.l, self.src + r * self.rowBytes(), self.dst + r * self.outRowBytes(), self.kindInt(), s.k - 1, st),
                .@"export" => try rows.exportRow(self.l, self.src + r * self.rowBytes(), self.dst + r * self.outRowBytes(), self.kindInt(), s.k, st),
                .consume, .timeout, .stale_seq => {
                    // row r's groups carry seq + r: a kernel that ignores its stamp or reads another row's fails
                    const want = if (self.m == .stale_seq) seq +% @as(u32, @intCast(r)) +% 1 else seq +% @as(u32, @intCast(r));
                    const tries: u32 = if (self.m == .stale_seq) 0 else retries;
                    try rows.consumeStamped(self.l, self.src + r * self.stride(), want, self.dst + r * self.outRowBytes(), self.kindInt(), s.k, self.status + 4 * r, tries, st);
                },
            }
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Split) usize {
        return self.shape.rows * self.outRowBytes();
    }

    pub fn reference(_: *Split, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The row's bytes by value: NaN where the input gives NaN, else the converted bits; and the status words.
    pub fn check(self: *Split, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const got = try self.rig.bytes(self.dst, self.count());
        const size: usize = if (self.m == .@"export") 4 else kit.size(self.k);
        const kind: hip.ops.Kind = @fromBackingInt(@intCast(@backingInt(self.k)));
        for (0..s.rows * s.k) |i| {
            const v: f32 = @bitCast(self.x[i]);
            // an export's input is the kind's own value, widened; the rest convert the fp32 value
            const input: f32 = if (self.m == .@"export") @floatCast(kit.get(self.k, self.xk, i)) else v;
            const want: u32 = if (self.m == .@"export") @bitCast(input) else ref.roundBits(kind, input);
            var word: u32 = 0;
            for (0..size) |b| word |= @as(u32, got[i * size + b]) << @intCast(8 * b);
            if (word == want) continue;
            if (std.math.isNan(input) and isNan(self.m == .@"export", self.k, word)) continue;
            return try std.fmt.bufPrint(&self.msg, "value {d} (row {d}) is 0x{x}, want 0x{x}", .{ i, i / s.k, word, want });
        }
        if (!self.stamped()) return null;
        const status = try self.rig.bytes(self.status, s.rows * 4);
        const want: u32 = @intFromBool(self.m == .timeout);
        for (0..s.rows) |r| {
            const word = std.mem.readInt(u32, status[r * 4 ..][0..4], .little);
            if (word != want) return try std.fmt.bufPrint(&self.msg, "row {d}'s status is {d}, want {d}", .{ r, word, want });
        }
        return null;
    }
};

fn isNan(wide: bool, k: kit.Kind, word: u32) bool {
    if (wide) return std.math.isNan(@as(f32, @bitCast(word)));
    return switch (k) {
        .f32 => std.math.isNan(@as(f32, @bitCast(word))),
        .f16 => std.math.isNan(@as(f16, @bitCast(@as(u16, @truncate(word))))),
        .bf16 => std.math.isNan(@as(f32, @bitCast(word << 16))),
    };
}
