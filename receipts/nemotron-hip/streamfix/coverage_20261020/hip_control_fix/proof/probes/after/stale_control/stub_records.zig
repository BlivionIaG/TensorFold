const r = @import("records_original.zig");
pub const Record = r.Record;
pub const records = r.records;
pub const check = r.check;
pub const controlProblems = r.controlProblems;
pub const controls = blk: { var c = r.controls; _ = &c; break :blk c; };
