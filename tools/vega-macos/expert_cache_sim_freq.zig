//! The frequency-aware policies of expert_cache_sim.py: TinyLFU, W-TinyLFU, aged LFU and LRU-with-frequency, over the same exact counters.
//! Same contract as expert_cache_sim_policy.zig: `step(layer, keys)` returns the call's misses; victims, ties and counters match the original.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const OList = l.OList;
const Mark = l.Mark;

/// Exact counters over the 10,240 keys, halved every W accesses; optional doorkeeper (a first sighting only sets a bit).
pub const Freq = struct {
    c: []u32,
    w: u64,
    n: u64 = 0,
    door: ?[]u32,
    epoch: u32 = 1,

    pub fn init(a: std.mem.Allocator, w: u64, door: bool) !Freq {
        const c = try a.alloc(u32, l.NK);
        @memset(c, 0);
        var stamp: ?[]u32 = null;
        if (door) {
            stamp = try a.alloc(u32, l.NK);
            @memset(stamp.?, 0);
        }
        return .{ .c = c, .w = w, .door = stamp };
    }

    pub fn touch(self: *Freq, keys: []const u16) void {
        if (self.door) |stamp| {
            for (keys) |k| {
                if (stamp[k] == self.epoch) self.c[k] += 1 else stamp[k] = self.epoch;
            }
        } else {
            for (keys) |k| self.c[k] += 1;
        }
    }

    pub fn est(self: *const Freq, k: u16) u32 {
        return self.c[k] + @intFromBool(if (self.door) |stamp| stamp[k] == self.epoch else false);
    }

    pub fn tick(self: *Freq, n: usize) void {
        self.n += n;
        if (self.n < self.w) return;
        self.n = 0;
        for (self.c) |*x| x.* >>= 1;
        if (self.door) |stamp| {
            self.epoch +%= 1;
            if (self.epoch == 0) {
                @memset(stamp, 0);
                self.epoch = 1;
            }
        }
    }
};

/// LRU victim, but a miss is admitted into a full cache only if freq(new) > freq(victim) (>= with `ge`). Frequency counts every access.
pub const TinyLfu = struct {
    cap: usize,
    f: Freq,
    ge: bool,
    od: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, w: u64, door: bool, ge: bool) !TinyLfu {
        return .{ .cap = cap, .f = try .init(a, w, door), .ge = ge, .od = try .init(a), .mark = try .init(a) };
    }

    pub fn step(self: *TinyLfu, layer: usize, keys: []const u16) u32 {
        _ = layer;
        self.f.touch(keys);
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.od.contains(k)) self.od.toEnd(k) else {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (self.od.count >= self.cap) {
                    const v = l.firstFree(&self.od, &self.mark) orelse l.noVictim();
                    const a = self.f.est(k);
                    const b = self.f.est(v);
                    if (!(if (self.ge) a >= b else a > b)) continue;
                    self.od.remove(v);
                    self.evi += 1;
                }
                self.od.append(k);
                self.adm += 1;
            }
        }
        self.f.tick(keys.len);
        return @intCast(nm);
    }
};

/// Window LRU (wfrac of the cap) in front of an LRU main; the window's overflow candidate is admitted to a full main only if it beats main's LRU.
pub const WTinyLfu = struct {
    wcap: usize,
    mcap: usize,
    f: Freq,
    win: OList,
    main: OList,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, wfrac: f64, w: u64, door: bool) !WTinyLfu {
        const wcap = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(cap)) * wfrac)));
        return .{ .wcap = wcap, .mcap = cap - wcap, .f = try .init(a, w, door), .win = try .init(a), .main = try .init(a) };
    }

    pub fn step(self: *WTinyLfu, layer: usize, keys: []const u16) u32 {
        _ = layer;
        self.f.touch(keys);
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.main.contains(k)) self.main.toEnd(k) else if (self.win.contains(k)) self.win.toEnd(k) else {
                miss[nm] = k;
                nm += 1;
            }
        }
        for (miss[0..nm]) |k| {
            self.win.append(k);
            self.adm += 1;
            if (self.win.count > self.wcap) {
                const cand = self.win.popFirst();
                if (self.main.count < self.mcap) {
                    self.main.append(cand);
                } else {
                    const v = self.main.first() orelse l.noVictim();
                    if (self.f.est(cand) > self.f.est(v)) {
                        self.main.remove(v);
                        self.main.append(cand);
                    }
                    self.evi += 1; // either the old main line or the candidate leaves the cache
                }
            }
        }
        self.f.tick(keys.len);
        return @intCast(nm);
    }
};

const Level = struct { head: u16 = none, tail: u16 = none };
const none: u16 = 0xffff;

/// Aged LFU (counts halved every W accesses), LRU tie-break; `hist` keeps the counts of non-resident keys, else a key's count restarts at eviction.
/// The residents of one count are a list in the order they reached it; the victim is the oldest unrequested line of the lowest count.
pub const Lfu = struct {
    a: std.mem.Allocator,
    cap: usize,
    w: u64,
    hist: bool,
    n: u64 = 0,
    c: []u32,
    rec: OList,
    mark: Mark,
    prev: []u16,
    next: []u16,
    levels: std.ArrayList(Level),
    min_level: usize = 0,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, w: u64, hist: bool) !Lfu {
        const c = try a.alloc(u32, l.NK);
        @memset(c, 0);
        const prev = try a.alloc(u16, l.NK);
        const next = try a.alloc(u16, l.NK);
        @memset(prev, none);
        @memset(next, none);
        return .{ .a = a, .cap = cap, .w = w, .hist = hist, .c = c, .rec = try .init(a), .mark = try .init(a), .prev = prev, .next = next, .levels = .empty };
    }

    fn add(self: *Lfu, k: u16, cnt: u32) void {
        if (self.levels.items.len <= cnt) self.levels.appendNTimes(self.a, .{}, cnt + 1 - self.levels.items.len) catch @panic("out of memory");
        const lv = &self.levels.items[cnt];
        self.prev[k] = lv.tail;
        self.next[k] = none;
        if (lv.tail == none) lv.head = k else self.next[lv.tail] = k;
        lv.tail = k;
        self.min_level = @min(self.min_level, cnt);
    }

    fn del(self: *Lfu, k: u16, cnt: u32) void {
        const lv = &self.levels.items[cnt];
        const p = self.prev[k];
        const n = self.next[k];
        if (p == none) lv.head = n else self.next[p] = n;
        if (n == none) lv.tail = p else self.prev[n] = p;
    }

    fn victim(self: *Lfu) u16 {
        const lv = self.levels.items;
        while (self.min_level < lv.len and lv[self.min_level].head == none) self.min_level += 1;
        var level = self.min_level;
        while (level < lv.len) : (level += 1) {
            var k = lv[level].head;
            while (k != none) : (k = self.next[k]) if (!self.mark.has(k)) return k;
        }
        l.noVictim();
    }

    pub fn step(self: *Lfu, layer: usize, keys: []const u16) u32 {
        _ = layer;
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.rec.contains(k)) {
                const old = self.c[k];
                self.c[k] = old + 1;
                self.del(k, old);
                self.add(k, old + 1);
                self.rec.toEnd(k);
            } else {
                self.c[k] += 1;
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (self.rec.count >= self.cap) {
                    const v = self.victim();
                    self.rec.remove(v);
                    self.del(v, self.c[v]);
                    if (!self.hist) self.c[v] = 0;
                    self.evi += 1;
                }
                self.rec.append(k);
                self.add(k, self.c[k]);
                self.adm += 1;
            }
        }
        self.n += keys.len;
        if (self.n >= self.w) {
            self.n = 0;
            // halve every count, then rebuild the levels from the residents oldest first
            var k = self.rec.first();
            while (k) |key| : (k = self.rec.after(key)) self.levels.items[self.c[key]] = .{};
            for (self.c) |*x| x.* >>= 1;
            self.min_level = std.math.maxInt(usize);
            k = self.rec.first();
            while (k) |key| : (k = self.rec.after(key)) self.add(key, self.c[key]);
        }
        return @intCast(nm);
    }
};

/// Evict, among the K oldest evictable lines, the one with the lowest (aged) frequency; ties go to the older line.
pub const LruFreq = struct {
    cap: usize,
    k: usize,
    f: Freq,
    od: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, k: usize, w: u64) !LruFreq {
        return .{ .cap = cap, .k = k, .f = try .init(a, w, false), .od = try .init(a), .mark = try .init(a) };
    }

    pub fn step(self: *LruFreq, layer: usize, keys: []const u16) u32 {
        _ = layer;
        self.f.touch(keys);
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.od.contains(k)) self.od.toEnd(k) else {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            const c = self.f.c;
            for (miss[0..nm]) |k| {
                if (self.od.count >= self.cap) {
                    var best: ?u16 = null;
                    var bc: u32 = 0;
                    var n: usize = 0;
                    var v = self.od.first();
                    while (v) |key| : (v = self.od.after(key)) {
                        if (self.mark.has(key)) continue;
                        if (best == null or c[key] < bc) {
                            best = key;
                            bc = c[key];
                        }
                        n += 1;
                        if (n >= self.k) break;
                    }
                    self.od.remove(best orelse l.noVictim());
                    self.evi += 1;
                }
                self.od.append(k);
                self.adm += 1;
            }
        }
        self.f.tick(keys.len);
        return @intCast(nm);
    }
};
