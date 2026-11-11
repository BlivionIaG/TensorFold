//! The traces of expert_cache_sim.py: loading (meta.json + *.npz), keys, the leave-category-out seeds, prefill top counts, lockstep groups.
//! A trace holds its decode picks and its prefill picks as global keys (layer * 256 + expert), 40 layers x 8 picks per token.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const npy = @import("npy.zig");
const npz = @import("npz.zig");
const rng = @import("expert_cache_sim_rng.zig");

pub const row = l.NL * l.TOPK;

pub const Trace = struct {
    name: []const u8,
    cat: []const u8,
    /// decode tokens x 40 x 8 keys
    dec: []const u16,
    /// prefill tokens x 40 x 8 keys
    pre: []const u16,

    pub fn decodeTokens(self: Trace) usize {
        return self.dec.len / row;
    }

    /// The 8 keys of `layer` in token `t` of `rows` (a `dec` or `pre` array).
    pub fn picks(rows: []const u16, t: usize, layer: usize) []const u16 {
        return rows[t * row + layer * l.TOPK ..][0..l.TOPK];
    }
};

pub const Set = struct {
    /// in file name order (sorted glob), the order the original's dict iterates
    traces: []Trace,
    /// indices into `traces` ordered by prompt name (`sorted(short)`)
    by_name: []usize,

    pub fn decodeTokens(self: Set) usize {
        var n: usize = 0;
        for (self.traces) |t| n += t.decodeTokens();
        return n;
    }
};

const Meta = struct { name: []const u8, cat: []const u8 };

fn bytesLess(a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// `load(directory)` of the original.
pub fn load(arena: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Set {
    const cwd = std.Io.Dir.cwd();
    const meta_path = try std.fs.path.join(arena, &.{ dir_path, "meta.json" });
    const meta_text = try cwd.readFileAlloc(io, meta_path, arena, .limited(1 << 28));
    const meta = try std.json.parseFromSliceLeaky([]Meta, arena, meta_text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    var files: std.ArrayList([]const u8) = .empty;
    var dir = try cwd.openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .directory or entry.name.len <= 4 or entry.name[0] == '.' or !std.mem.endsWith(u8, entry.name, ".npz")) continue;
        try files.append(arena, try arena.dupe(u8, entry.name));
    }
    std.mem.sortUnstable([]const u8, files.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return bytesLess(a, b);
        }
    }.less);
    const traces = try arena.alloc(Trace, files.items.len);
    for (files.items, traces) |file, *out| {
        const name = file[0 .. file.len - 4];
        var cat: ?[]const u8 = null;
        for (meta) |m| if (std.mem.eql(u8, m.name, name)) {
            cat = m.cat;
        };
        const path = try std.fs.path.join(arena, &.{ dir_path, file });
        const bytes = try cwd.readFileAlloc(io, path, arena, .limited(1 << 32));
        defer arena.free(bytes);
        out.* = try decode(arena, name, cat orelse {
            std.debug.print("expert_cache_sim: {s} has no entry in {s}\n", .{ name, meta_path });
            return error.MissingMeta;
        }, bytes);
    }
    const by_name = try arena.alloc(usize, traces.len);
    for (by_name, 0..) |*x, i| x.* = i;
    std.mem.sortUnstable(usize, by_name, traces, struct {
        fn less(ts: []Trace, a: usize, b: usize) bool {
            return bytesLess(ts[a].name, ts[b].name);
        }
    }.less);
    return .{ .traces = traces, .by_name = by_name };
}

/// One archive: `ids` (T, 40, 8) and `kind` (T,), decode rows (kind == 1) and prefill rows (the rest) as keys.
pub fn decode(a: std.mem.Allocator, name: []const u8, cat: []const u8, bytes: []const u8) !Trace {
    var z = try npz.open(a, bytes);
    defer z.deinit(a);
    const ids_bytes = try z.array(a, "ids");
    defer a.free(ids_bytes);
    const kind_bytes = try z.array(a, "kind");
    defer a.free(kind_bytes);
    const ids = try npy.parse(ids_bytes);
    const kind = try npy.parse(kind_bytes);
    if (ids.ndim != 3 or ids.shape[1] != l.NL or ids.shape[2] != l.TOPK or kind.ndim != 1 or kind.shape[0] != ids.shape[0]) return error.BadShape;
    const t = ids.shape[0];
    var n_dec: usize = 0;
    for (0..t) |i| n_dec += @intFromBool(kind.atInt(i) == 1);
    const dec = try a.alloc(u16, n_dec * row);
    const pre = try a.alloc(u16, (t - n_dec) * row);
    var d: usize = 0;
    var p: usize = 0;
    for (0..t) |i| {
        const dst = if (kind.atInt(i) == 1) blk: {
            d += row;
            break :blk dec[d - row .. d];
        } else blk: {
            p += row;
            break :blk pre[p - row .. p];
        };
        for (dst, 0..) |*key, j| {
            const id = ids.atInt(i * row + j);
            if (id < 0 or id >= l.NE) return error.ExpertOutOfRange;
            key.* = @intCast(id + @as(i64, @intCast(j / l.TOPK * l.NE)));
        }
    }
    return .{ .name = name, .cat = cat, .dec = dec, .pre = pre };
}

pub const Seeds = struct {
    cats: []const []const u8,
    lists: []const []const u16,

    pub fn get(self: Seeds, cat: []const u8) ?[]const u16 {
        for (self.cats, self.lists) |c, list| if (std.mem.eql(u8, c, cat)) return list;
        return null;
    }
};

pub fn bincount(keys: []const u16, out: *[l.NK]u32) void {
    @memset(out, 0);
    for (keys) |k| out[k] += 1;
}

/// `build_seed`: per category, the keys with the most routed share of a prompt summed over the other categories' prompts
/// (accumulated in file order, descending, ties by key), the first `cap`.
pub fn buildSeed(a: std.mem.Allocator, short: Set, cap: usize) !Seeds {
    var cats: std.ArrayList([]const u8) = .empty;
    for (short.traces) |t| {
        var seen = false;
        for (cats.items) |c| seen = seen or std.mem.eql(u8, c, t.cat);
        if (!seen) try cats.append(a, t.cat);
    }
    std.mem.sortUnstable([]const u8, cats.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return bytesLess(x, y);
        }
    }.less);
    const lists = try a.alloc([]const u16, cats.items.len);
    const acc = try a.alloc(f64, l.NK);
    var cnt: [l.NK]u32 = undefined;
    const order = try a.alloc(u16, l.NK);
    for (cats.items, lists) |cat, *out| {
        @memset(acc, 0);
        for (short.traces) |t| {
            if (std.mem.eql(u8, t.cat, cat)) continue;
            bincount(t.dec, &cnt);
            const n: f64 = @floatFromInt(t.decodeTokens());
            for (acc, cnt) |*x, c| x.* += @as(f64, @floatFromInt(c)) / n;
        }
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sortUnstable(u16, order, acc, struct {
            fn less(s: []const f64, x: u16, y: u16) bool {
                return if (s[x] != s[y]) s[x] > s[y] else x < y;
            }
        }.less);
        out.* = try a.dupe(u16, order[0..@min(cap, l.NK)]);
    }
    return .{ .cats = cats.items, .lists = lists };
}

/// `sum(ord(ch) * (i + 1) for i, ch in enumerate(name))`.
pub fn stableSeed(name: []const u8) !u32 {
    var s: u64 = 0;
    var i: u64 = 0;
    var it = (try std.unicode.Utf8View.init(name)).iterator();
    while (it.nextCodepoint()) |cp| : (i += 1) s += @as(u64, cp) * (i + 1);
    return std.math.cast(u32, s) orelse error.SeedTooLarge;
}

/// `prefill_keys`: the prompt's most routed keys (descending count, ties by a NumPy random draw seeded from the name), at most `cap`;
/// keys never routed are not listed.
pub fn prefillKeys(a: std.mem.Allocator, pre: []const u16, cap: usize, name: []const u8) ![]u16 {
    var cnt: [l.NK]u32 = undefined;
    bincount(pre, &cnt);
    var rnd: [l.NK]f64 = undefined;
    rng.numpyRandomSample(try stableSeed(name), &rnd);
    var list: std.ArrayList(u16) = .empty;
    errdefer list.deinit(a);
    for (cnt, 0..) |c, k| if (c > 0) try list.append(a, @intCast(k));
    const Ctx = struct { cnt: *const [l.NK]u32, rnd: *const [l.NK]f64 };
    std.mem.sortUnstable(u16, list.items, Ctx{ .cnt = &cnt, .rnd = &rnd }, struct {
        fn less(c: Ctx, x: u16, y: u16) bool {
            if (c.cnt[x] != c.cnt[y]) return c.cnt[x] > c.cnt[y];
            if (c.rnd[x] != c.rnd[y]) return c.rnd[x] < c.rnd[y];
            return x < y;
        }
    }.less);
    list.shrinkRetainingCapacity(@min(list.items.len, cap));
    return list.toOwnedSlice(a);
}

/// The lockstep groups: the prompt names sorted, shuffled by `random.Random(0)`, cut in fours (indices into `set.traces`).
pub fn lockGroups(a: std.mem.Allocator, set: Set) ![]const []const usize {
    const order = try a.dupe(usize, set.by_name);
    rng.pythonShuffle(usize, 0, order);
    const n = (order.len + l.group_size - 1) / l.group_size;
    const groups = try a.alloc([]const usize, n);
    for (groups, 0..) |*g, i| g.* = order[i * l.group_size ..][0..@min(l.group_size, order.len - i * l.group_size)];
    return groups;
}
