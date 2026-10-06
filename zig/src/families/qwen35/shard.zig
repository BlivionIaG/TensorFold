//! One safetensors file mapped read-only. core's reader refuses a header with a tensor above rank 4 (a vision patch
//! embedding has rank 5), so this one leaves such tensors, and dtypes it has no name for, out of its index.

const std = @import("std");
const Io = std.Io;
const core = @import("core");

const st = core.safetensors;
pub const Tensor = st.Tensor;

pub const Shard = struct {
    file: Io.File,
    map: Io.File.MemoryMap,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !Shard {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const names = try parse(arena.allocator(), map.memory[8..][0..header_len], len - 8 - header_len);
        return .{ .file = file, .map = map, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn close(s: *Shard, io: Io) void {
        s.map.destroy(io);
        s.file.close(io);
        s.arena.deinit();
        s.* = undefined;
    }

    pub fn get(s: *const Shard, name: []const u8) ?Tensor {
        const e = s.names.get(name) orelse return null;
        return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .bytes = s.map.memory[s.data + e.begin .. s.data + e.end] };
    }
};

/// core's `parseHeader` without its refusals of rank and dtype; sizes and ranges are still checked.
fn parse(arena: std.mem.Allocator, json: []const u8, data_len: usize) !st.Header {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    var out: st.Header = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const o = kv.value_ptr.object;
        const dtype = st.DType.parse(o.get("dtype").?.string) orelse continue;
        const shape = o.get("shape").?.array.items;
        if (shape.len > st.max_rank) continue;
        var e: st.Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
        var n: usize = dtype.size();
        for (shape, 0..) |d, i| {
            e.shape[i] = @intCast(d.integer);
            n *= e.shape[i];
        }
        const offs = o.get("data_offsets").?.array.items;
        e.begin = @intCast(offs[0].integer);
        e.end = @intCast(offs[1].integer);
        if (e.end < e.begin or e.end - e.begin != n or e.end > data_len) return error.BadSafetensors;
        try out.put(arena, kv.key_ptr.*, e);
    }
    return out;
}

test "a rank 5 tensor and an unknown dtype are left out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"a": {"dtype": "U32", "shape": [2], "data_offsets": [0, 8]},
        \\ "patch": {"dtype": "BF16", "shape": [1, 1, 1, 1, 1], "data_offsets": [8, 10]},
        \\ "f8": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [10, 12]}}
    ;
    const h = try parse(arena.allocator(), json, 12);
    try std.testing.expectEqual(@as(usize, 1), h.count());
    try std.testing.expect(h.contains("a"));
}
