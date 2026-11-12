//! The recency policies of expert_cache_sim.py: LRU, per-layer LRU, SLRU, 2Q and ARC. `step(layer, keys)` serves one (token, layer) call and
//! returns its misses. Hits are touched first, then misses are admitted one by one; a line requested in the call is never the victim.
//! Eviction order, tie-breaks and the adm / evi counters follow the Python OrderedDict version exactly.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const OList = l.OList;
const Mark = l.Mark;

pub const Lru = struct {
    cap: usize,
    od: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize) !Lru {
        return .{ .cap = cap, .od = try .init(a), .mark = try .init(a) };
    }

    pub fn preload(self: *Lru, keys_oldest_first: []const u16) void {
        for (keys_oldest_first) |k| if (self.od.count < self.cap) self.od.put(k);
    }

    pub fn step(self: *Lru, layer: usize, keys: []const u16) u32 {
        _ = layer;
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
                    self.od.remove(v);
                    self.evi += 1;
                }
                self.od.append(k);
                self.adm += 1;
            }
        }
        return @intCast(nm);
    }
};

/// Equal split: CAP//40 lines per layer, the remainder one extra line to the first layers.
pub const PerLayer = struct {
    caps: [l.NL]usize,
    ods: [l.NL]OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize) !PerLayer {
        var self: PerLayer = .{ .caps = undefined, .ods = undefined, .mark = try .init(a) };
        for (0..l.NL) |i| {
            self.caps[i] = cap / l.NL + @intFromBool(i < cap % l.NL);
            self.ods[i] = try .init(a);
        }
        return self;
    }

    pub fn step(self: *PerLayer, layer: usize, keys: []const u16) u32 {
        const od = &self.ods[layer];
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (od.contains(k)) od.toEnd(k) else {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (od.count >= self.caps[layer]) {
                    od.remove(l.firstFree(od, &self.mark) orelse l.noVictim());
                    self.evi += 1;
                }
                od.append(k);
                self.adm += 1;
            }
        }
        return @intCast(nm);
    }
};

/// Probation + protected: a probation hit is promoted, a protected overflow is demoted to probation's newest end.
pub const Slru = struct {
    cap: usize,
    pcap: usize,
    prob: OList,
    prot: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, pfrac: f64) !Slru {
        return .{ .cap = cap, .pcap = @intFromFloat(@as(f64, @floatFromInt(cap)) * pfrac), .prob = try .init(a), .prot = try .init(a), .mark = try .init(a) };
    }

    pub fn step(self: *Slru, layer: usize, keys: []const u16) u32 {
        _ = layer;
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.prot.contains(k)) {
                self.prot.toEnd(k);
            } else if (self.prob.contains(k)) {
                self.prob.remove(k);
                self.prot.append(k);
                if (self.prot.count > self.pcap) self.prob.append(self.prot.popFirst());
            } else {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (self.prob.count + self.prot.count >= self.cap) {
                    if (l.firstFree(&self.prob, &self.mark)) |v| self.prob.remove(v) else self.prot.remove(l.firstFree(&self.prot, &self.mark) orelse l.noVictim());
                    self.evi += 1;
                }
                self.prob.append(k);
                self.adm += 1;
            }
        }
        return @intCast(nm);
    }
};

/// 2Q: FIFO A1in (kin of the cap), LRU Am, ghost list A1out of evicted A1in keys (ghost_frac of the cap); a ghost hit goes to Am.
pub const TwoQ = struct {
    cap: usize,
    kin: usize,
    gcap: usize,
    a1: OList,
    am: OList,
    out: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize, kin: f64, ghost_frac: f64) !TwoQ {
        const capf: f64 = @floatFromInt(cap);
        return .{
            .cap = cap,
            .kin = @max(1, @as(usize, @intFromFloat(capf * kin))),
            .gcap = @intFromFloat(capf * ghost_frac),
            .a1 = try .init(a),
            .am = try .init(a),
            .out = try .init(a),
            .mark = try .init(a),
        };
    }

    pub fn step(self: *TwoQ, layer: usize, keys: []const u16) u32 {
        _ = layer;
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.am.contains(k)) {
                self.am.toEnd(k);
            } else if (!self.a1.contains(k)) {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (self.a1.count + self.am.count >= self.cap) {
                    const from_a1 = if (self.a1.count > self.kin or self.am.count == 0) l.firstFree(&self.a1, &self.mark) else null;
                    if (from_a1) |v| {
                        self.a1.remove(v);
                        self.out.append(v);
                        if (self.out.count > self.gcap) _ = self.out.popFirst();
                    } else if (l.firstFree(&self.am, &self.mark)) |v| {
                        self.am.remove(v);
                    } else {
                        self.a1.remove(l.firstFree(&self.a1, &self.mark) orelse l.noVictim());
                    }
                    self.evi += 1;
                }
                if (self.out.contains(k)) {
                    self.out.remove(k);
                    self.am.append(k);
                } else {
                    self.a1.append(k);
                }
                self.adm += 1;
            }
        }
        return @intCast(nm);
    }
};

/// Adaptive Replacement Cache, one global pool; victims skip lines requested in the current call.
pub const Arc = struct {
    c: usize,
    p: usize = 0,
    t1: OList,
    t2: OList,
    b1: OList,
    b2: OList,
    mark: Mark,
    adm: u64 = 0,
    evi: u64 = 0,

    pub fn init(a: std.mem.Allocator, cap: usize) !Arc {
        return .{ .c = cap, .t1 = try .init(a), .t2 = try .init(a), .b1 = try .init(a), .b2 = try .init(a), .mark = try .init(a) };
    }

    fn demote(self: *Arc, t: *OList, b: *OList) bool {
        const v = l.firstFree(t, &self.mark) orelse return false;
        t.remove(v);
        b.append(v);
        self.evi += 1;
        return true;
    }

    fn replace(self: *Arc, in_b2: bool) void {
        if (self.t1.count + self.t2.count < self.c) return;
        const prefer1 = self.t1.count > 0 and (self.t1.count > self.p or (in_b2 and self.t1.count == self.p));
        if (prefer1) {
            if (!self.demote(&self.t1, &self.b1)) _ = self.demote(&self.t2, &self.b2);
        } else {
            if (!self.demote(&self.t2, &self.b2)) _ = self.demote(&self.t1, &self.b1);
        }
    }

    pub fn step(self: *Arc, layer: usize, keys: []const u16) u32 {
        _ = layer;
        const c = self.c;
        var miss: [l.max_keys]u16 = undefined;
        var nm: usize = 0;
        for (keys) |k| {
            if (self.t1.contains(k)) {
                self.t1.remove(k);
                self.t2.append(k);
            } else if (self.t2.contains(k)) {
                self.t2.toEnd(k);
            } else {
                miss[nm] = k;
                nm += 1;
            }
        }
        if (nm > 0) {
            self.mark.begin(keys);
            for (miss[0..nm]) |k| {
                if (self.b1.contains(k)) {
                    self.p = @min(c, self.p + @max(self.b2.count / self.b1.count, 1));
                    self.replace(false);
                    self.b1.remove(k);
                    self.t2.append(k);
                } else if (self.b2.contains(k)) {
                    self.p -|= @max(self.b1.count / self.b2.count, 1);
                    self.replace(true);
                    self.b2.remove(k);
                    self.t2.append(k);
                } else {
                    const l1 = self.t1.count + self.b1.count;
                    if (l1 >= c) {
                        if (self.t1.count < c) {
                            _ = self.b1.popFirst();
                            self.replace(false);
                        } else {
                            self.t1.remove(l.firstFree(&self.t1, &self.mark) orelse l.noVictim());
                            self.evi += 1;
                        }
                    } else {
                        const total = l1 + self.t2.count + self.b2.count;
                        if (total >= c) {
                            if (total >= 2 * c) _ = self.b2.popFirst();
                            self.replace(false);
                        }
                    }
                    self.t1.append(k);
                }
                self.adm += 1;
                // ghost lists bounded (cap hits only through protected skips)
                while (self.b1.count + self.b2.count > 2 * c) _ = (if (self.b1.count > self.b2.count) &self.b1 else &self.b2).popFirst();
            }
        }
        return @intCast(nm);
    }
};
