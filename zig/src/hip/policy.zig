//! What a run may use, resolved once at open: the matrix cores, activation and attention precision, which kernels,
//! graphs, prefill steps, MTP and prompt reuse. Nothing below the engine reads the environment; it reads this.

const std = @import("std");

pub const Choice = enum { auto, on, off };
pub const Precision = enum { auto, f16, bf16, f32 };
pub const Kernels = enum { auto, shared, native, reference };
pub const Exact = enum { strict, relaxed };

pub const Mtp = struct { drafts: u8 = 2, confidence: f32 = 0.3 };
pub const Prefix = struct { slots: u32 = 8, bytes: u64 = 0 };

pub const Policy = struct {
    matrix: Choice = .auto,
    activations: Precision = .auto,
    attention: Precision = .auto,
    kernels: Kernels = .auto,
    graphs: Choice = .auto,
    prefill_step: u32 = 1024,
    mtp: Mtp = .{},
    prefix: Prefix = .{},
    exact: Exact = .strict,

    pub const Error = error{ UnknownKey, BadValue, StepNotAligned };

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
        if (eql(u8, key, "matrix")) p.matrix = try parseEnum(Choice, value) else if (eql(u8, key, "activations")) p.activations = try parseEnum(Precision, value) else if (eql(u8, key, "attention")) p.attention = try parseEnum(Precision, value) else if (eql(u8, key, "kernels")) p.kernels = try parseEnum(Kernels, value) else if (eql(u8, key, "graphs")) p.graphs = try parseEnum(Choice, value) else if (eql(u8, key, "exact")) p.exact = try parseEnum(Exact, value) else if (eql(u8, key, "prefill_step")) p.prefill_step = std.fmt.parseInt(u32, value, 10) catch return error.BadValue else if (eql(u8, key, "mtp_drafts")) p.mtp.drafts = std.fmt.parseInt(u8, value, 10) catch return error.BadValue else if (eql(u8, key, "mtp_confidence")) p.mtp.confidence = std.fmt.parseFloat(f32, value) catch return error.BadValue else if (eql(u8, key, "prefix_slots")) p.prefix.slots = std.fmt.parseInt(u32, value, 10) catch return error.BadValue else if (eql(u8, key, "prefix_bytes")) p.prefix.bytes = std.fmt.parseInt(u64, value, 10) catch return error.BadValue else return error.UnknownKey;
    }

    fn parseEnum(comptime E: type, value: []const u8) Error!E {
        return std.meta.stringToEnum(E, value) orelse error.BadValue;
    }

    /// One line for the start-up log and the server's info, every field in `apply`'s syntax.
    pub fn format(p: Policy, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("matrix={t},activations={t},attention={t},kernels={t},graphs={t},prefill_step={d},mtp_drafts={d},mtp_confidence={d},prefix_slots={d},prefix_bytes={d},exact={t}", .{
            p.matrix, p.activations, p.attention, p.kernels, p.graphs, p.prefill_step, p.mtp.drafts, p.mtp.confidence, p.prefix.slots, p.prefix.bytes, p.exact,
        });
    }

    /// The policy as a fixed set of words, for rank 0 to send and every rank to use the same.
    pub fn words(p: Policy) [9]u32 {
        return .{
            @backingInt(p.matrix),                                        @backingInt(p.activations), @backingInt(p.attention),
            @backingInt(p.kernels),                                       @backingInt(p.graphs),      p.prefill_step,
            @as(u32, p.mtp.drafts) | @as(u32, @backingInt(p.exact)) << 8, @bitCast(p.mtp.confidence), p.prefix.slots,
        };
    }

    pub fn fromWords(w: [9]u32, bytes: u64) Policy {
        return .{
            .matrix = @fromBackingInt(@intCast(w[0])),
            .activations = @fromBackingInt(@intCast(w[1])),
            .attention = @fromBackingInt(@intCast(w[2])),
            .kernels = @fromBackingInt(@intCast(w[3])),
            .graphs = @fromBackingInt(@intCast(w[4])),
            .prefill_step = w[5],
            .mtp = .{ .drafts = @truncate(w[6]), .confidence = @bitCast(w[7]) },
            .exact = @fromBackingInt(@intCast(w[6] >> 8)),
            .prefix = .{ .slots = w[8], .bytes = bytes },
        };
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
    try p.apply("matrix=on,attention=f32,graphs=off,mtp_confidence=0.25,prefix_slots=4,exact=relaxed");
    const back = Policy.fromWords(p.words(), p.prefix.bytes);
    try std.testing.expectEqualDeep(p, back);
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try p.format(&w);
    var again: Policy = .{};
    try again.apply(w.buffered());
    try std.testing.expectEqualDeep(p, again);
}
