const r = @import("records_original.zig");
pub const Record = r.Record;
pub const records = r.records;
pub const check = r.check;
pub const controls = blk: { var c = r.controls; c[0].usage[0].vgprs = 0; break :blk c; };
