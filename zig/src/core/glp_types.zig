//! GLPv3 data types from #1835; format: Matt Suiche/weightless (full provenance and credits in the GLP recipe).

const std = @import("std");

/// Why a file was refused, in one sentence the server prints.
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Error = error{BadGlpVector} || std.mem.Allocator.Error;

/// What the edit does to the stream at each steered layer; both are rank one and never converted into each other.
pub const Mode = enum {
    /// x += alpha*d on every copy: llama.cpp's control vector, a bias.
    add,
    /// x -= alpha*dot(mean(x), d)*d on every copy: directional ablation (Arditi et al., NeurIPS 2024).
    project,
};

/// Where the edit lands. The spec's v3 fold site is the only one a hyper-connection model has between layers.
pub const Site = enum {
    /// Edit a block's post-FFN four-copy stream and its pre-hc-post attention writer with the same direction.
    hc_fold_pre_engram,
};

/// The spec version whose contract this reader implements: hc fold sites are valid from version 3 on.
pub const spec_version: u32 = 3;

/// One layer's direction: its zero-based layer id (`direction.N` is layer N, no offset) and the vector itself.
pub const Layer = struct {
    id: u32,
    dir: []const f32,
};

/// What the file says about itself: the keys a consumer reads before it reads the directions.
pub const Publication = struct {
    /// What the file calls itself: general.name.
    name: ?[]const u8 = null,
    /// What the file says it is: general.description.
    description: ?[]const u8 = null,
    /// Who derived it: general.author.
    author: ?[]const u8 = null,
    /// The licence the file is under: general.license.
    license: ?[]const u8 = null,
    /// The model family a consumer matches against: controlvector.model_hint.
    model_hint: ?[]const u8 = null,
    /// What the strength does, as it was measured: glp.measured_effect.
    measured_effect: ?[]const u8 = null,
    /// The day the file was made: glp.created.
    created: ?[]const u8 = null,
};

/// A vector read whole and accepted; its strings borrow the file's bytes and its layers the reader's arena.
pub const Vector = struct {
    mode: Mode,
    site: Site,
    /// glp.spec_version as the file declares it.
    spec_version: u32,
    /// glp.alpha_default: the strength the file recommends, finite and not negative.
    alpha: f32,
    hook_point: []const u8,
    derived_at: []const u8,
    /// The base repo as general.base_model.0 names it: "Org/Name", a non-hub URL as given, or empty when none.
    base_model: []const u8,
    /// The base checkpoint's commit pin, general.base_model.0.version; empty when the file names none.
    base_revision: []const u8,
    method: []const u8,
    contrast: []const u8,
    width: usize,
    /// One direction per layer, ascending by id: every id is at least 1 and no id repeats.
    layers: []const Layer,
    /// The SHA-256 over the direction tensors' bytes, in the order the file holds them.
    sha256: [32]u8,
    /// The SHA-256 of the file's exact bytes, as the reader was handed them.
    file_sha256: [32]u8,
    /// Whether the file claims a content digest (and so the reader checked it).
    digest_claimed: bool,
    transferred: bool,
    publication: Publication = .{},

    /// The direction for layer `id`, or null when the vector does not steer it.
    pub fn layer(self: Vector, id: u32) ?[]const f32 {
        var lo: usize = 0;
        var hi: usize = self.layers.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const at = self.layers[mid].id;
            if (at == id) return self.layers[mid].dir;
            if (at < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }
};

/// The control vector a serve applies to its hyper-connection stream (--glp).
pub const Glp = struct {
    /// The file's vector: one direction per layer and the checkpoint it was derived from.
    vector: Vector,
    /// The effective strength in force: --glp-alpha, or the file's own alpha_default; finite, not negative, FP32.
    alpha: f32,

    /// Whether every edit is bypassed: an alpha of exactly 0 leaves the stream's bits as an unsteered run has them.
    pub fn off(self: Glp) bool {
        return self.alpha == 0;
    }
};

/// The hook point this reader applies at, as the file names it.
pub const site_name = @tagName(Site.hc_fold_pre_engram);

/// The hub's URL prefix: a base model named by URL names the repo that follows it.
pub const hub_prefix = "https://huggingface.co/";

/// A repo id: the owner and name, whether the file names the model as an id or as the hub's URL.
pub fn repoId(name: []const u8) []const u8 {
    return if (std.mem.startsWith(u8, name, hub_prefix)) name[hub_prefix.len..] else name;
}

/// Accept explicit zero or finite, normal FP32; nonzero strengths must not silently underflow off.
pub fn strength(x: f64) ?f32 {
    if (!std.math.isFinite(x) or x < 0) return null;
    const applied: f32 = @floatCast(x);
    if (!std.math.isFinite(applied)) return null;
    if (x == 0) return 0;
    if (applied < std.math.floatMin(f32)) return null;
    return applied;
}
