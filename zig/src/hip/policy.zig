//! What a run may use, resolved once at open; nothing below the engine reads the environment, it reads this.

const std = @import("std");
const caps_mod = @import("caps.zig");

pub const Choice = enum { auto, on, off };
pub const Precision = enum { auto, f16, bf16, f32 };
pub const Kernels = enum { auto, shared, native, reference };
pub const Exact = enum { strict, relaxed };
/// How the kernels are launched: from Zig on the code objects, or through the embedded library's C launchers.
pub const Launch = enum { zig, library };
/// One kernel family switched to its reference, apart from `kernels=reference` that switches them all.
pub const Pick = enum { auto, reference };

pub const Mtp = struct { drafts: u8 = 3, confidence: f64 = 0.3 };
pub const Prefix = struct { slots: u32 = 8, bytes: u64 = 0 };

/// A path a rank reads locally (it never travels with the policy).
pub const Path = struct {
    buf: [160]u8 = @splat(0),

    pub fn slice(p: *const Path) ?[]const u8 {
        const n = std.mem.indexOfScalar(u8, &p.buf, 0) orelse p.buf.len;
        return if (n == 0) null else p.buf[0..n];
    }

    fn set(p: *Path, text: []const u8) error{BadValue}!void {
        if (text.len >= p.buf.len) return error.BadValue;
        p.buf = @splat(0);
        @memcpy(p.buf[0..text.len], text);
    }
};

pub const Policy = struct {
    matrix: Choice = .auto,
    activations: Precision = .auto,
    attention: Precision = .auto,
    kernels: Kernels = .auto,
    graphs: Choice = .auto,
    launch: Launch = .zig,
    /// Reference kernels one at a time: decode stream tile, prefill GEMM tile, fused decode tails, chunked recurrence.
    stream: Pick = .auto,
    gemm: Pick = .auto,
    fuse: Pick = .auto,
    gdn: Pick = .auto,
    prefill_step: u32 = 1024,
    mtp: Mtp = .{},
    prefix: Prefix = .{},
    exact: Exact = .strict,
    /// Rank whose graph capture fails on purpose, for the tests of the fallback (-1: none). Local to a rank.
    graph_fail: i32 = -1,
    /// The RCCL library to load first. Local to a rank.
    rccl_lib: Path = .{},
    /// GiB of device memory kept beside the plan, by the native backends' rule (empty: a tenth, 4 GiB at least). Local.
    reserve_gib: Path = .{},

    pub const Error = error{ UnknownKey, BadValue, StepNotAligned, ActivationsUnsupported };

    /// Whether the decode stream tile is on.
    pub fn streamOn(p: *const Policy) bool {
        return p.kernels != .reference and p.stream != .reference;
    }

    /// Whether the prefill GEMM tile (not the reference block tile) takes the products.
    pub fn gemmOn(p: *const Policy) bool {
        return p.kernels != .reference and p.gemm != .reference;
    }

    /// Whether decode.hip's merged launches are on.
    pub fn fused(p: *const Policy) bool {
        return p.kernels != .reference and p.fuse != .reference;
    }

    /// Whether the DeltaNet prefill runs chunked.
    pub fn chunked(p: *const Policy) bool {
        return p.kernels != .reference and p.gdn != .reference;
    }

    /// Whether the 64-row prefill attention tile is on.
    pub fn wideAttention(p: *const Policy) bool {
        return p.attention != .f32;
    }

    /// Whether rounds replay captured graphs on a group of `world` ranks.
    pub fn graphsOn(p: *const Policy, world: usize) bool {
        return switch (p.graphs) {
            .on => true,
            .off => false,
            .auto => world == 1,
        };
    }

    /// The defaults of a GPU: its activation type, everything else left to the kernels' own rules.
    pub fn defaults(c: caps_mod.Caps) Policy {
        return .{ .activations = if (c.act == .bf16) .bf16 else .f16 };
    }

    /// Applies `key=value,key=value` (the flags' and TF_POLICY's syntax) over this policy, all of it or none.
    pub fn apply(p: *Policy, text: []const u8) Error!void {
        var next = p.*;
        var it = std.mem.tokenizeAny(u8, text, ", ");
        while (it.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.BadValue;
            try next.set(pair[0..eq], pair[eq + 1 ..]);
        }
        if (next.prefill_step == 0 or next.prefill_step % 64 != 0) return error.StepNotAligned;
        if (next.mtp.drafts > 3) return error.BadValue;
        p.* = next;
    }

    fn set(p: *Policy, key: []const u8, value: []const u8) Error!void {
        const eql = std.mem.eql;
        if (eql(u8, key, "matrix")) p.matrix = try parseEnum(Choice, value);
        if (eql(u8, key, "activations")) p.activations = try parseEnum(Precision, value);
        if (eql(u8, key, "attention")) p.attention = try parseEnum(Precision, value);
        if (eql(u8, key, "kernels")) p.kernels = try parseEnum(Kernels, value);
        if (eql(u8, key, "graphs")) p.graphs = try parseEnum(Choice, value);
        if (eql(u8, key, "launch")) p.launch = try parseEnum(Launch, value);
        if (eql(u8, key, "stream")) p.stream = try parseEnum(Pick, value);
        if (eql(u8, key, "gemm")) p.gemm = try parseEnum(Pick, value);
        if (eql(u8, key, "fuse")) p.fuse = try parseEnum(Pick, value);
        if (eql(u8, key, "gdn")) p.gdn = try parseEnum(Pick, value);
        if (eql(u8, key, "exact")) p.exact = try parseEnum(Exact, value);
        if (eql(u8, key, "prefill_step")) p.prefill_step = std.fmt.parseInt(u32, value, 10) catch return error.BadValue;
        if (eql(u8, key, "mtp_drafts")) p.mtp.drafts = std.fmt.parseInt(u8, value, 10) catch return error.BadValue;
        if (eql(u8, key, "mtp_confidence")) p.mtp.confidence = std.fmt.parseFloat(f64, value) catch return error.BadValue;
        if (eql(u8, key, "prefix_slots")) p.prefix.slots = std.fmt.parseInt(u32, value, 10) catch return error.BadValue;
        if (eql(u8, key, "prefix_bytes")) p.prefix.bytes = std.fmt.parseInt(u64, value, 10) catch return error.BadValue;
        if (eql(u8, key, "graph_fail")) p.graph_fail = std.fmt.parseInt(i32, value, 10) catch return error.BadValue;
        if (eql(u8, key, "rccl_lib")) try p.rccl_lib.set(value);
        if (eql(u8, key, "reserve_gib")) try p.reserve_gib.set(value);
        if (!known(key)) return error.UnknownKey;
    }

    fn known(key: []const u8) bool {
        inline for (@typeInfo(Policy).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "mtp") or std.mem.eql(u8, name, "prefix")) continue;
            if (std.mem.eql(u8, key, name)) return true;
        }
        const extra = [_][]const u8{ "mtp_drafts", "mtp_confidence", "prefix_slots", "prefix_bytes" };
        for (extra) |name| if (std.mem.eql(u8, key, name)) return true;
        return false;
    }

    fn parseEnum(comptime E: type, value: []const u8) Error!E {
        return std.meta.stringToEnum(E, value) orelse error.BadValue;
    }

    /// One line for the start-up log and server info, in `apply`'s syntax; default and local fields are left out.
    pub fn format(p: Policy, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("matrix={t},activations={t},attention={t},kernels={t},graphs={t},launch={t},prefill_step={d},mtp_drafts={d},mtp_confidence={d},prefix_slots={d},prefix_bytes={d},exact={t}", .{
            p.matrix, p.activations, p.attention, p.kernels, p.graphs, p.launch, p.prefill_step, p.mtp.drafts, p.mtp.confidence, p.prefix.slots, p.prefix.bytes, p.exact,
        });
        if (p.stream != .auto) try w.print(",stream={t}", .{p.stream});
        if (p.gemm != .auto) try w.print(",gemm={t}", .{p.gemm});
        if (p.fuse != .auto) try w.print(",fuse={t}", .{p.fuse});
        if (p.gdn != .auto) try w.print(",gdn={t}", .{p.gdn});
        if (p.graph_fail >= 0) try w.print(",graph_fail={d}", .{p.graph_fail});
        if (p.rccl_lib.slice()) |path| try w.print(",rccl_lib={s}", .{path});
        if (p.reserve_gib.slice()) |gib| try w.print(",reserve_gib={s}", .{gib});
    }

    pub const word_count = 6;

    /// The policy as a fixed set of words, for rank 0 to send and every rank to use the same.
    pub fn words(p: Policy) [word_count]u32 {
        const conf: u64 = @bitCast(p.mtp.confidence);
        const pack = [_]u32{
            @backingInt(p.matrix), @backingInt(p.activations), @backingInt(p.attention), @backingInt(p.kernels), @backingInt(p.graphs), @backingInt(p.launch),
            @backingInt(p.stream), @backingInt(p.gemm),        @backingInt(p.fuse),      @backingInt(p.gdn),     @backingInt(p.exact),
        };
        var nibbles: [2]u32 = .{ 0, 0 };
        for (pack, 0..) |v, i| nibbles[i / 8] |= v << @intCast(4 * (i % 8));
        return .{ nibbles[0], nibbles[1] | @as(u32, p.mtp.drafts) << 24, p.prefill_step, @truncate(conf), @intCast(conf >> 32), p.prefix.slots };
    }

    /// `local` with every field rank 0 sent; a rank keeps its own prefix bytes, graph fault and library path.
    pub fn fromWords(w: [word_count]u32, local: Policy) Policy {
        var p = local;
        var vals: [11]u32 = undefined;
        for (&vals, 0..) |*v, i| v.* = (w[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
        p.matrix = @fromBackingInt(@intCast(vals[0]));
        p.activations = @fromBackingInt(@intCast(vals[1]));
        p.attention = @fromBackingInt(@intCast(vals[2]));
        p.kernels = @fromBackingInt(@intCast(vals[3]));
        p.graphs = @fromBackingInt(@intCast(vals[4]));
        p.launch = @fromBackingInt(@intCast(vals[5]));
        p.stream = @fromBackingInt(@intCast(vals[6]));
        p.gemm = @fromBackingInt(@intCast(vals[7]));
        p.fuse = @fromBackingInt(@intCast(vals[8]));
        p.gdn = @fromBackingInt(@intCast(vals[9]));
        p.exact = @fromBackingInt(@intCast(vals[10]));
        p.mtp = .{ .drafts = @truncate(w[1] >> 24), .confidence = @bitCast(@as(u64, w[3]) | @as(u64, w[4]) << 32) };
        p.prefill_step = w[2];
        p.prefix.slots = w[5];
        return p;
    }

    /// The environment the old switches are read from, once, at resolution.
    pub const Env = struct {
        /// Read the process's variables after `vars`.
        process: bool = false,
        vars: []const [2][]const u8 = &.{},

        pub const none: Env = .{};
        pub const current: Env = .{ .process = true };

        fn get(e: Env, name: []const u8, buf: *[96]u8) ?[]const u8 {
            for (e.vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            if (!e.process or name.len >= buf.len) return null;
            @memcpy(buf[0..name.len], name);
            buf[name.len] = 0;
            const found = std.c.getenv(buf[0..name.len :0].ptr) orelse return null;
            return std.mem.span(found);
        }
    };

    /// What resolution found besides the values: the old variables it read and TF_POLICY, for the start-up line.
    pub const Notes = struct {
        buf: [768]u8 = undefined,
        len: usize = 0,

        fn add(n: *Notes, comptime fmt: []const u8, args: anytype) void {
            const out = std.fmt.bufPrint(n.buf[n.len..], fmt, args) catch return;
            n.len += out.len;
        }

        pub fn text(n: *const Notes) []const u8 {
            return n.buf[0..n.len];
        }
    };

    /// The old variables and what each says in the policy's own words; read once, with a note that each is deprecated.
    const aliases = [_]struct { name: []const u8, rule: *const fn (value: []const u8, out: []u8) ?[]const u8 }{
        .{ .name = "TF_WMMA", .rule = wmmaRule },
        .{ .name = "TF_AFFINE_GEMV", .rule = oldRule("stream=reference") },
        .{ .name = "TF_AFFINE_GEMM", .rule = oldRule("gemm=reference") },
        .{ .name = "TF_DECODE_FUSE", .rule = oldRule("fuse=reference") },
        .{ .name = "TF_FA_WIDE", .rule = zeroRule("attention=f32") },
        .{ .name = "TF_GDN_CHUNKED", .rule = zeroRule("gdn=reference") },
        .{ .name = "TF_HIP_GRAPHS_TP", .rule = oneRule("graphs=on") },
        .{ .name = "TF_HIP_GRAPHS", .rule = zeroRule("graphs=off") },
        .{ .name = "TENSORFOLD_GRAPH", .rule = zeroRule("graphs=off") },
        .{ .name = "TF_HIP_GRAPH_FAIL", .rule = copyRule("graph_fail=") },
        .{ .name = "TF_HIP_LAUNCH", .rule = copyRule("launch=") },
        .{ .name = "TF_RCCL_LIB", .rule = copyRule("rccl_lib=") },
        .{ .name = "TENSORFOLD_MEMORY_RESERVE_GIB", .rule = copyRule("reserve_gib=") },
    };

    fn wmmaRule(value: []const u8, out: []u8) ?[]const u8 {
        _ = out;
        if (std.mem.eql(u8, value, "1")) return "matrix=on";
        if (std.mem.eql(u8, value, "0")) return "matrix=off";
        return null;
    }

    fn oldRule(comptime says: []const u8) *const fn ([]const u8, []u8) ?[]const u8 {
        return struct {
            fn rule(value: []const u8, _: []u8) ?[]const u8 {
                return if (std.mem.eql(u8, value, "old")) says else null;
            }
        }.rule;
    }

    fn zeroRule(comptime says: []const u8) *const fn ([]const u8, []u8) ?[]const u8 {
        return struct {
            fn rule(value: []const u8, _: []u8) ?[]const u8 {
                return if (value.len > 0 and value[0] == '0') says else null;
            }
        }.rule;
    }

    fn oneRule(comptime says: []const u8) *const fn ([]const u8, []u8) ?[]const u8 {
        return struct {
            fn rule(value: []const u8, _: []u8) ?[]const u8 {
                return if (std.mem.eql(u8, value, "1")) says else null;
            }
        }.rule;
    }

    fn copyRule(comptime key: []const u8) *const fn ([]const u8, []u8) ?[]const u8 {
        return struct {
            fn rule(value: []const u8, out: []u8) ?[]const u8 {
                if (value.len == 0) return null;
                return std.fmt.bufPrint(out, key ++ "{s}", .{value}) catch null;
            }
        }.rule;
    }

    /// The policy on `gpu`: defaults, then `flags` (`k=v`), old variables, then TF_POLICY; `notes` says which acted.
    pub fn resolve(gpu: caps_mod.Caps, flags: []const u8, env: Env, notes: *Notes) Error!Policy {
        var p = defaults(gpu);
        try p.apply(flags);
        var buf: [96]u8 = undefined;
        var out: [200]u8 = undefined;
        for (aliases) |a| {
            const value = env.get(a.name, &buf) orelse continue;
            const says = a.rule(value, &out) orelse continue;
            notes.add("{s}={s} is deprecated, it is {s}; ", .{ a.name, value, says });
            p.apply(says) catch {};
        }
        if (env.get("TF_POLICY", &buf)) |text| {
            notes.add("TF_POLICY={s}; ", .{text});
            try p.apply(text);
        }
        if (p.activations != .auto and p.activations != defaults(gpu).activations) return error.ActivationsUnsupported;
        return p;
    }
};

test "apply sets fields and refuses what it does not know" {
    var p: Policy = .{};
    try p.apply("matrix=off, kernels=reference,mtp_drafts=3,prefill_step=2048");
    try std.testing.expectEqual(Choice.off, p.matrix);
    try std.testing.expectEqual(Kernels.reference, p.kernels);
    try std.testing.expectEqual(@as(u8, 3), p.mtp.drafts);
    try std.testing.expectEqual(@as(u32, 2048), p.prefill_step);
    try std.testing.expectError(error.UnknownKey, p.apply("wmma=on"));
    try std.testing.expectError(error.BadValue, p.apply("matrix=maybe"));
    try std.testing.expectError(error.BadValue, p.apply("mtp_drafts=4"));
    try std.testing.expectError(error.StepNotAligned, p.apply("prefill_step=1000"));
}

test "a policy survives its words and its line" {
    var p: Policy = .{};
    try p.apply("matrix=on,attention=f32,graphs=off,launch=library,gdn=reference,mtp_confidence=0.25,prefix_slots=4,exact=relaxed,graph_fail=1");
    const back = Policy.fromWords(p.words(), .{ .graph_fail = 1 });
    try std.testing.expectEqualDeep(p, back);
    var buf: [640]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try p.format(&w);
    var again: Policy = .{};
    try again.apply(w.buffered());
    try std.testing.expectEqualDeep(p, again);
}

test "the old variables are aliases, TF_POLICY speaks last, the reference switches" {
    const rdna3 = caps_mod.Caps.of("gfx1100").?;
    var notes: Policy.Notes = .{};
    const env: Policy.Env = .{ .vars = &.{ .{ "TF_WMMA", "0" }, .{ "TF_AFFINE_GEMV", "old" }, .{ "TF_FA_WIDE", "0" }, .{ "TF_HIP_GRAPHS", "0" }, .{ "TF_HIP_GRAPHS_TP", "1" }, .{ "TF_POLICY", "matrix=on" } } };
    const p = try Policy.resolve(rdna3, "kernels=shared", env, &notes);
    try std.testing.expectEqual(Choice.on, p.matrix);
    try std.testing.expect(!p.streamOn() and p.gemmOn() and p.fused() and p.chunked() and !p.wideAttention());
    try std.testing.expect(!p.graphsOn(1) and !p.graphsOn(2));
    try std.testing.expect(std.mem.indexOf(u8, notes.text(), "TF_WMMA=0 is deprecated") != null);
    var reference: Policy = .{};
    try reference.apply("kernels=reference");
    try std.testing.expect(!reference.streamOn() and !reference.gemmOn() and !reference.fused() and !reference.chunked());
}

test "graphs follow the group unless the policy says" {
    var p: Policy = .{};
    try std.testing.expect(p.graphsOn(1) and !p.graphsOn(2));
    try p.apply("graphs=on");
    try std.testing.expect(p.graphsOn(2));
}

test "a GPU's activations are the policy's only choice" {
    var notes: Policy.Notes = .{};
    const v620 = caps_mod.Caps.of("gfx1030").?;
    try std.testing.expectEqual(Precision.f16, (try Policy.resolve(v620, "", .none, &notes)).activations);
    try std.testing.expectError(error.ActivationsUnsupported, Policy.resolve(v620, "activations=bf16", .none, &notes));
}
