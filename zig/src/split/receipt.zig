//! The probe receipt (schema tensorfold-offload-probe/1): Mac, external device and link, each number labelled.
const std = @import("std");
const strict = @import("strict.zig");

const Allocator = std.mem.Allocator;

pub const schema = "tensorfold-offload-probe/1";
const schema_family = "tensorfold-offload-probe/";

pub const Error = error{
    /// `schema` is not a tensorfold-offload-probe schema.
    UnknownSchema,
    /// A tensorfold-offload-probe schema this build does not read.
    UnsupportedVersion,
} || strict.Error;

pub const Diag = strict.Diag;

/// Where a number comes from, as in fabric/cost.zig: documented, measured, probed at startup, or assumed.
pub const Source = enum { documented, measured, probed, assumed };

/// Activation type of a device's stage boundary.
pub const Act = enum { f16, bf16 };

pub fn Labeled(comptime T: type) type {
    return struct { value: T, source: Source };
}

/// A latency distribution in nanoseconds.
pub const Latency = struct { p50: f64, p99: f64, source: Source };

pub const Mac = struct {
    name: []const u8,
    /// `recommendedMaxWorkingSetSize` of the Metal device.
    max_working_set_bytes: Labeled(u64),
    /// Weight bytes per second the Mac's decode reaches end to end.
    decode_bandwidth_bps: Labeled(f64),
};

pub const Device = struct {
    name: []const u8,
    arch: []const u8,
    /// Usable device memory after reserved regions.
    vram_bytes: Labeled(u64),
    peak_bandwidth_bps: Labeled(f64),
    /// Fraction of peak the decode kernels reach end to end.
    decode_efficiency: Labeled(f64),
    act: Act,
    /// One kernel launch that cannot overlap, in nanoseconds.
    launch_ns: Labeled(f64),
    backend: []const u8,
};

pub const Link = struct {
    h2d_bps: Labeled(f64),
    d2h_bps: Labeled(f64),
    /// External device to Metal: flag visible to the host, event wake of the next command buffer, flush.
    hop_to_host_ns: Latency,
    /// Metal to external device: host store and doorbell or poll on the device side.
    hop_to_device_ns: Latency,
    /// Metal's wake from an event signaled by the host.
    metal_wake_ns: Latency,
    /// `newBufferWithBytesNoCopy` over the host window.
    window_wrap_ns: Labeled(f64),
};

pub const Receipt = struct {
    os_build: []const u8,
    mac: Mac,
    device: Device,
    link: Link,
};

/// Pretty JSON with the schema string first. The caller frees the result.
pub fn write(gpa: Allocator, r: Receipt) Allocator.Error![]u8 {
    return std.json.Stringify.valueAlloc(gpa, .{
        .schema = schema,
        .os_build = r.os_build,
        .mac = r.mac,
        .device = r.device,
        .link = r.link,
    }, .{ .whitespace = .indent_2 });
}

/// Strict parse: exact schema, every field, no unknown keys, finite positive numbers; `deinit` frees it.
pub fn parse(gpa: Allocator, bytes: []const u8) Error!std.json.Parsed(Receipt) {
    var diag: Diag = .{};
    return parseDiag(gpa, bytes, &diag);
}

/// As `parse`; on failure `diag.path()` names the offending field.
pub fn parseDiag(gpa: Allocator, bytes: []const u8, diag: *Diag) Error!std.json.Parsed(Receipt) {
    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const root = try strict.parseDocument(a, bytes);
    var obj = try strict.object(root, "", diag);
    const sv = try strict.member(obj, "schema", "", diag);
    if (sv != .string) return strict.fail(diag, "schema", error.WrongType);
    if (!std.mem.eql(u8, sv.string, schema)) {
        if (std.mem.startsWith(u8, sv.string, schema_family)) return strict.fail(diag, "schema", error.UnsupportedVersion);
        return strict.fail(diag, "schema", error.UnknownSchema);
    }
    // the schema was checked above; the remaining members are the Receipt
    _ = obj.orderedRemove("schema");
    const r = try strict.parseValue(Receipt, a, .{ .object = obj }, "", diag);
    try check(r, diag);
    return .{ .arena = arena, .value = r };
}

fn positive(f: f64, path: []const u8, diag: *Diag) strict.Error!void {
    if (!(f > 0)) return strict.fail(diag, path, error.InvalidValue);
}

fn latency(l: Latency, path: []const u8, diag: *Diag) strict.Error!void {
    if (!(l.p50 > 0) or l.p99 < l.p50) return strict.fail(diag, path, error.InvalidValue);
}

fn check(r: Receipt, diag: *Diag) strict.Error!void {
    try positive(@floatFromInt(r.mac.max_working_set_bytes.value), "mac.max_working_set_bytes.value", diag);
    try positive(r.mac.decode_bandwidth_bps.value, "mac.decode_bandwidth_bps.value", diag);
    try positive(@floatFromInt(r.device.vram_bytes.value), "device.vram_bytes.value", diag);
    try positive(r.device.peak_bandwidth_bps.value, "device.peak_bandwidth_bps.value", diag);
    const eff = r.device.decode_efficiency.value;
    if (!(eff > 0 and eff <= 1)) return strict.fail(diag, "device.decode_efficiency.value", error.InvalidValue);
    if (r.device.launch_ns.value < 0) return strict.fail(diag, "device.launch_ns.value", error.InvalidValue);
    try positive(r.link.h2d_bps.value, "link.h2d_bps.value", diag);
    try positive(r.link.d2h_bps.value, "link.d2h_bps.value", diag);
    try latency(r.link.hop_to_host_ns, "link.hop_to_host_ns", diag);
    try latency(r.link.hop_to_device_ns, "link.hop_to_device_ns", diag);
    try latency(r.link.metal_wake_ns, "link.metal_wake_ns", diag);
    if (r.link.window_wrap_ns.value < 0) return strict.fail(diag, "link.window_wrap_ns.value", error.InvalidValue);
}

/// A receipt shaped like a Mac with a Thunderbolt-attached card, for tests: test data, not a measurement.
pub const sample: Receipt = .{
    .os_build = "25A0000",
    .mac = .{
        .name = "Apple M4 Pro",
        .max_working_set_bytes = .{ .value = 40_160_000_000, .source = .probed },
        .decode_bandwidth_bps = .{ .value = 164e9, .source = .measured },
    },
    .device = .{
        .name = "AMD Radeon RX Vega 56",
        .arch = "gfx900",
        .vram_bytes = .{ .value = 8_482_023_424, .source = .probed },
        .peak_bandwidth_bps = .{ .value = 410e9, .source = .documented },
        .decode_efficiency = .{ .value = 0.37, .source = .measured },
        .act = .f16,
        .launch_ns = .{ .value = 4400, .source = .measured },
        .backend = "hip",
    },
    .link = .{
        .h2d_bps = .{ .value = 3.19e9, .source = .measured },
        .d2h_bps = .{ .value = 3.35e9, .source = .measured },
        .hop_to_host_ns = .{ .p50 = 90_000, .p99 = 250_000, .source = .assumed },
        .hop_to_device_ns = .{ .p50 = 15_000, .p99 = 40_000, .source = .assumed },
        .metal_wake_ns = .{ .p50 = 65_000, .p99 = 120_000, .source = .measured },
        .window_wrap_ns = .{ .value = 30_000, .source = .measured },
    },
};

test "write then parse round-trips every field" {
    const gpa = std.testing.allocator;
    const text = try write(gpa, sample);
    defer gpa.free(text);
    try std.testing.expect(std.mem.startsWith(u8, text, "{\n  \"schema\": \"tensorfold-offload-probe/1\""));
    const p = try parse(gpa, text);
    defer p.deinit();
    const r = p.value;
    try std.testing.expectEqualStrings(sample.os_build, r.os_build);
    try std.testing.expectEqualStrings(sample.mac.name, r.mac.name);
    try std.testing.expectEqual(sample.mac.max_working_set_bytes, r.mac.max_working_set_bytes);
    try std.testing.expectEqual(sample.mac.decode_bandwidth_bps, r.mac.decode_bandwidth_bps);
    try std.testing.expectEqualStrings(sample.device.name, r.device.name);
    try std.testing.expectEqualStrings(sample.device.arch, r.device.arch);
    try std.testing.expectEqualStrings(sample.device.backend, r.device.backend);
    try std.testing.expectEqual(sample.device.vram_bytes, r.device.vram_bytes);
    try std.testing.expectEqual(sample.device.peak_bandwidth_bps, r.device.peak_bandwidth_bps);
    try std.testing.expectEqual(sample.device.decode_efficiency, r.device.decode_efficiency);
    try std.testing.expectEqual(Act.f16, r.device.act);
    try std.testing.expectEqual(sample.device.launch_ns, r.device.launch_ns);
    try std.testing.expectEqual(sample.link, r.link);
    // a second write of the parsed receipt is byte-identical
    const again = try write(gpa, r);
    defer gpa.free(again);
    try std.testing.expectEqualStrings(text, again);
}

fn expectParseError(gpa: Allocator, text: []const u8, err: Error, path: []const u8) !void {
    var diag: Diag = .{};
    try std.testing.expectError(err, parseDiag(gpa, text, &diag));
    try std.testing.expectEqualStrings(path, diag.path());
}

fn mutated(gpa: Allocator, from: []const u8, to: []const u8) ![]u8 {
    const text = try write(gpa, sample);
    defer gpa.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, from) != null);
    return std.mem.replaceOwned(u8, gpa, text, from, to);
}

test "planted-bad receipts get named errors" {
    const gpa = std.testing.allocator;
    const Case = struct { from: []const u8, to: []const u8, err: Error, path: []const u8 };
    const cases = [_]Case{
        .{ .from = "tensorfold-offload-probe/1", .to = "tensorfold-split/1", .err = error.UnknownSchema, .path = "schema" },
        .{ .from = "tensorfold-offload-probe/1", .to = "tensorfold-offload-probe/2", .err = error.UnsupportedVersion, .path = "schema" },
        .{ .from = "\"schema\": \"tensorfold-offload-probe/1\"", .to = "\"schema\": 1", .err = error.WrongType, .path = "schema" },
        .{ .from = "\"os_build\": \"25A0000\"", .to = "\"os_build\": 25", .err = error.WrongType, .path = "os_build" },
        .{ .from = "\"os_build\": \"25A0000\",", .to = "", .err = error.MissingField, .path = "os_build" },
        .{ .from = "\"os_build\"", .to = "\"os_buld\"", .err = error.MissingField, .path = "os_build" },
        .{ .from = "\"source\": \"probed\"", .to = "\"source\": \"guessed\"", .err = error.InvalidValue, .path = "mac.max_working_set_bytes.source" },
        .{ .from = "\"act\": \"f16\"", .to = "\"act\": \"f32\"", .err = error.InvalidValue, .path = "device.act" },
        .{ .from = "\"arch\": \"gfx900\"", .to = "\"arch\": \"gfx900\", \"extra\": 1", .err = error.UnknownField, .path = "device.extra" },
        .{ .from = "\"p99\": 250000", .to = "\"p99\": 100", .err = error.InvalidValue, .path = "link.hop_to_host_ns" },
        .{ .from = "\"value\": 0.37", .to = "\"value\": 1.5", .err = error.InvalidValue, .path = "device.decode_efficiency.value" },
        .{ .from = "\"value\": 3190000000", .to = "\"value\": \"fast\"", .err = error.WrongType, .path = "link.h2d_bps.value" },
        .{ .from = "\"value\": 3190000000", .to = "\"value\": 0", .err = error.InvalidValue, .path = "link.h2d_bps.value" },
    };
    for (cases) |c| {
        const text = try mutated(gpa, c.from, c.to);
        defer gpa.free(text);
        try expectParseError(gpa, text, c.err, c.path);
    }
    try expectParseError(gpa, "{\"schema\": \"tensorfold-offload-probe/1\"", error.InvalidJson, "");
    try expectParseError(gpa, "[]", error.WrongType, "");
    try expectParseError(gpa, "{}", error.MissingField, "schema");
    try expectParseError(gpa, "{\"schema\": \"tensorfold-offload-probe/1\"}", error.MissingField, "os_build");
}
