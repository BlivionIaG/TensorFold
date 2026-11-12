//! The two generators expert_cache_sim.py uses, bit for bit: NumPy's `RandomState(seed).random_sample` and Python's `random.Random(0).shuffle`.
//! Both are MT19937; they differ in how an integer seed is expanded (init_genrand for NumPy, init_by_array for CPython).
const std = @import("std");

pub const Mt19937 = struct {
    mt: [624]u32,
    pos: usize,

    /// NumPy's legacy seeding of an integer (`mt19937_seed`) = init_genrand.
    pub fn initGenrand(seed: u32) Mt19937 {
        var self: Mt19937 = .{ .mt = undefined, .pos = 624 };
        self.mt[0] = seed;
        for (1..624) |i| self.mt[i] = 1812433253 *% (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) +% @as(u32, @intCast(i));
        return self;
    }

    /// CPython's seeding from a list of 32-bit words (`init_by_array`); `random.seed(0)` uses the key {0}.
    pub fn initByArray(key: []const u32) Mt19937 {
        var self = initGenrand(19650218);
        var i: usize = 1;
        var j: usize = 0;
        var k: usize = @max(624, key.len);
        while (k > 0) : (k -= 1) {
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            i += 1;
            j += 1;
            if (i >= 624) {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if (j >= key.len) j = 0;
        }
        k = 623;
        while (k > 0) : (k -= 1) {
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) *% 1566083941)) -% @as(u32, @intCast(i));
            i += 1;
            if (i >= 624) {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x80000000;
        self.pos = 624;
        return self;
    }

    fn twist(self: *Mt19937) void {
        const upper: u32 = 0x80000000;
        const lower: u32 = 0x7fffffff;
        for (0..624) |kk| {
            const y = (self.mt[kk] & upper) | (self.mt[(kk + 1) % 624] & lower);
            self.mt[kk] = self.mt[(kk + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
        }
        self.pos = 0;
    }

    pub fn next32(self: *Mt19937) u32 {
        if (self.pos >= 624) self.twist();
        var y = self.mt[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// genrand_res53, what NumPy's `random_sample` and Python's `random.random()` both return.
    pub fn random(self: *Mt19937) f64 {
        const a = self.next32() >> 5;
        const b = self.next32() >> 6;
        return (@as(f64, @floatFromInt(a)) * 67108864.0 + @as(f64, @floatFromInt(b))) / 9007199254740992.0;
    }

    /// CPython `Random._randbelow(n)` for n >= 1 (getrandbits with rejection).
    pub fn randBelow(self: *Mt19937, n: u32) u32 {
        const k: u6 = @intCast(32 - @clz(n)); // n.bit_length(), 1..32
        while (true) {
            const r = self.next32() >> @intCast(32 - k);
            if (r < n) return r;
        }
    }
};

/// `np.random.RandomState(seed).random_sample(out.len)`.
pub fn numpyRandomSample(seed: u32, out: []f64) void {
    var mt: Mt19937 = .initGenrand(seed);
    for (out) |*x| x.* = mt.random();
}

/// `random.Random(seed).shuffle(items)` for a non-negative seed below 2**32 (CPython's `random_seed` key of one word).
pub fn pythonShuffle(comptime T: type, seed: u32, items: []T) void {
    var mt: Mt19937 = .initByArray(&.{seed});
    var i = items.len;
    while (i > 1) {
        i -= 1;
        const j = mt.randBelow(@intCast(i + 1));
        std.mem.swap(T, &items[i], &items[j]);
    }
}
