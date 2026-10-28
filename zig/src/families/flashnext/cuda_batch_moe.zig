//! The shared pass's MoE write-back: the slots' fp32 sum, the adds and the kept branch.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const LayerKeep = batch.LayerKeep;
const dims = wt.dims;
const slots = wt.slots;

/// After a layer's experts, when something adds to or keeps its output: the slots' fp32 sum, adds, kept branch.
pub fn moeAdds(self: anytype, layer: usize, keep: LayerKeep) !void {
    const canon = self.canon_f32;
    self.canon_f32 = false;
    const add = self.addAt(layer);
    if (add == null and keep.moe_branch == 0) return;
    const s = self.s;
    const n = self.rows;
    var a: cuda.Args = .{};
    a.add(s.y);
    a.add(s.wts);
    a.add(s.moe_f32);
    a.add(batch.cint(dims));
    a.add(batch.cint(slots));
    // the EXL3 canonical branch is already in moe_f32
    if (!canon) try cuda.launch.launch(self.k.moe_sum, .{ .grid = .{ .x = @intCast(n), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
    if (add) |m| try m.add(m.ctx, layer, s.mixed, s.picks, s.wts, s.moe_f32, n);
    if (keep.moe_branch != 0) try self.sumSlices(s.moe_f32, keep.moe_branch, true, n * dims, 1);
    self.pending_f32 = true;
}

/// The EXL3 canonical branch into moe_f32: two pick groups fma-summed in slot order and added, or one rank's [lo, hi).
pub fn groupSum(self: anytype, split: usize, lo: usize, hi: usize, both: bool) !void {
    const s = self.s;
    var a: cuda.Args = .{};
    a.add(s.y);
    a.add(s.wts);
    a.add(s.picks);
    a.add(s.moe_f32);
    for ([_]usize{ dims, slots, split, lo, hi, @intFromBool(both) }) |v| a.add(batch.cint(v));
    try cuda.launch.launch(self.k.group_sum, .{ .grid = .{ .x = @intCast(self.rows), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
    self.canon_f32 = true;
    self.pending_f32 = true;
}
