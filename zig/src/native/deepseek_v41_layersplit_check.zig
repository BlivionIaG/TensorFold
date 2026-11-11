//! TF_LS_CHECK=<dir>:<k,k,..>:<ids file>: layer-split gate G3 on the Metal engine (private gate instrumentation).
//! <dir> holds CUDA's boundaries `L<k-1>.x.bf16` / `.pre.f32` for the prompt (`ids file`: its token ids, text).
const std = @import("std");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const remote = @import("deepseek_v41_remote.zig");

const vocab = 129280;
const log = std.log.scoped(.layersplit);
const tail_rows = 256;

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// The prompt through slot 0, the last `tail_rows` with every row's logits when `all` (returned, owned by `a`).
fn run(m: *ds.Model, a: std.mem.Allocator, ids: []const u32, all: bool) !?[]f32 {
    try m.reset();
    const step = m.workspace.plan.rows;
    const body = if (all) ids.len - tail_rows else ids.len;
    var at: usize = 0;
    while (at < body) {
        const end = @min(at + step, body);
        _ = try m.forward(ids[at..end], .none);
        at = end;
    }
    if (!all) return null;
    return try a.dupe(f32, try m.forward(ids[body..], .all));
}

fn compareLogits(name: []const u8, want: []const f32, got: []const f32) void {
    var agree: usize = 0;
    var worst: f64 = 0;
    var total: f64 = 0;
    for (0..tail_rows) |r| {
        const x = want[r * vocab ..][0..vocab];
        const y = got[r * vocab ..][0..vocab];
        agree += @intFromBool(argmax(x) == argmax(y));
        var num: f64 = 0;
        var den: f64 = 0;
        for (x, y) |p, q| {
            const d: f64 = @as(f64, p) - q;
            num += d * d;
            den += @as(f64, p) * p;
        }
        const e = @sqrt(num / @max(den, 1e-30));
        worst = @max(worst, e);
        total += e;
    }
    log.info("{s}: logits over the last {d} positions: top-1 agrees {d}/{d}, NRMS mean {d:.5} worst {d:.5}", .{ name, tail_rows, agree, tail_rows, total / tail_rows, worst });
}

/// Relative RMS of two bf16 row files over [r0, r1) (and, for the pre files, f32).
fn compareFiles(a: std.mem.Allocator, io: std.Io, name: []const u8, p: []const u8, q: []const u8, row: usize, f32s: bool, r0: usize, r1: usize) !void {
    const cwd = std.Io.Dir.cwd();
    const x = try cwd.readFileAlloc(io, p, a, .limited(4 << 30));
    const y = try cwd.readFileAlloc(io, q, a, .limited(4 << 30));
    defer a.free(x);
    defer a.free(y);
    const rows = @min(x.len, y.len) / row;
    var num: f64 = 0;
    var den: f64 = 0;
    var worst_row: f64 = 0;
    var mean_row: f64 = 0;
    const hi = @min(r1, rows);
    for (r0..hi) |r| {
        var rn: f64 = 0;
        var rd: f64 = 0;
        const n = row / (if (f32s) @as(usize, 4) else 2);
        for (0..n) |i| {
            const u: f64 = if (f32s) @as(f32, @bitCast(std.mem.readInt(u32, x[r * row + 4 * i ..][0..4], .little))) else bf(std.mem.readInt(u16, x[r * row + 2 * i ..][0..2], .little));
            const v: f64 = if (f32s) @as(f32, @bitCast(std.mem.readInt(u32, y[r * row + 4 * i ..][0..4], .little))) else bf(std.mem.readInt(u16, y[r * row + 2 * i ..][0..2], .little));
            rn += (u - v) * (u - v);
            rd += u * u;
        }
        num += rn;
        den += rd;
        const e = @sqrt(rn / @max(rd, 1e-30));
        worst_row = @max(worst_row, e);
        mean_row += e;
    }
    log.info("{s}: rows {d}..{d}: relative RMS {d:.5} (row mean {d:.5}, worst {d:.5})", .{ name, r0, hi, @sqrt(num / @max(den, 1e-30)), mean_row / @as(f64, @floatFromInt(@max(1, hi - r0))), worst_row });
}

pub fn runCheck(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, spec: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var it = std.mem.splitScalar(u8, spec, ':');
    const dir = it.next() orelse return error.BadSpec;
    const ks = it.next() orelse return error.BadSpec;
    const ids_path = it.next() orelse return error.BadSpec;
    const ids = try remote.parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, ids_path, a, .limited(64 << 20)));
    if (ids.len < tail_rows + 1 or ids.len > m.caches[0].plan.capacity) return error.PromptTooShortOrLong;
    log.info("{d} prompt tokens, boundaries in {s}", .{ ids.len, dir });
    const hrow: usize = 4 * 5120 * 2;
    var b0: [600]u8 = undefined;
    var b1: [600]u8 = undefined;

    m.gate.off();
    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const ref = (try run(m, a, ids, true)).?;
    log.info("Metal-only prefill + logits: {d:.2} s", .{@as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0)) / 1e9});
    // control: the same logits again (run-to-run noise of the engine itself)
    compareLogits("control (Metal-only twice)", ref, (try run(m, a, ids, true)).?);

    // the Metal-only boundary at every cut and at the decoder's first layer
    var kit = std.mem.tokenizeScalar(u8, ks, ',');
    var klist: [16]usize = undefined;
    var kn: usize = 0;
    while (kit.next()) |s| : (kn += 1) klist[kn] = try std.fmt.parseInt(usize, s, 10);
    for (klist[0..kn]) |k| {
        m.gate = .{ .stop_dir = dir, .stop_k = k };
        _ = try run(m, a, ids, false);
        const cx = try std.fmt.bufPrint(&b0, "{s}/L{d}.x.bf16", .{ dir, k - 1 });
        const mx = try std.fmt.bufPrint(&b1, "{s}/M.L{d}.x.bf16", .{ dir, k - 1 });
        const have_cuda = if (std.Io.Dir.cwd().access(io, cx, .{})) |_| true else |_| false;
        if (have_cuda) try compareFiles(a, io, try std.fmt.allocPrint(a, "cut {d}: CUDA vs Metal residual (x)", .{k}), mx, cx, hrow, false, 0, ids.len);
    }
    m.gate.off();

    for (klist[0..kn]) |k| {
        // control: enter from the engine's own boundary
        m.gate = .{ .entry_dir = dir, .entry_prefix = "M.", .k = k };
        compareLogits(try std.fmt.allocPrint(a, "cut {d}, control (Metal's own boundary entered)", .{k}), ref, (try run(m, a, ids, true)).?);
        const cuda_ok = if (std.Io.Dir.cwd().access(io, try std.fmt.bufPrint(&b0, "{s}/L{d}.x.bf16", .{ dir, k - 1 }), .{})) |_| true else |_| false;
        if (!cuda_ok) continue;
        m.gate = .{ .entry_dir = dir, .entry_prefix = "", .k = k };
        compareLogits(try std.fmt.allocPrint(a, "cut {d}, CUDA boundary entered", .{k}), ref, (try run(m, a, ids, true)).?);
        // the state at the decoder's first layer: mixed (CUDA 0..k-1, Metal k..19) against Metal-only
        m.gate = .{ .entry_dir = dir, .entry_prefix = "", .k = k, .stop_dir = dir, .stop_k = 20 };
        _ = try run(m, a, ids, false);
        const mixed = try std.fmt.bufPrint(&b0, "{s}/M.L19.x.bf16", .{dir});
        const mixed_copy = try std.fmt.allocPrint(a, "{s}/mixed{d}.L19.x.bf16", .{ dir, k });
        const data = try std.Io.Dir.cwd().readFileAlloc(io, mixed, a, .limited(4 << 30));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = mixed_copy, .data = data });
    }
    m.gate.off();
    // reference file for layer 19 again (Metal-only), then the comparisons
    m.gate = .{ .stop_dir = dir, .stop_k = 20 };
    _ = try run(m, a, ids, false);
    m.gate.off();
    for (klist[0..kn]) |k| {
        const mixed_copy = try std.fmt.allocPrint(a, "{s}/mixed{d}.L19.x.bf16", .{ dir, k });
        const only = try std.fmt.allocPrint(a, "{s}/M.L19.x.bf16", .{dir});
        try compareFiles(a, io, try std.fmt.allocPrint(a, "layer 19 output, Metal-only vs mixed (cut {d})", .{k}), only, mixed_copy, hrow, false, 0, ids.len);
    }
    const c19 = try std.fmt.allocPrint(a, "{s}/L19.x.bf16", .{dir});
    if (std.Io.Dir.cwd().access(io, c19, .{})) |_| {
        try compareFiles(a, io, "layer 19 output, Metal-only vs CUDA", try std.fmt.allocPrint(a, "{s}/M.L19.x.bf16", .{dir}), c19, hrow, false, 0, ids.len);
    } else |_| {}
}
