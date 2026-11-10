//! DeepSeek-V4.1 prefill/decode hand-off: CUDA state in, Metal state out, transport-independent (docs/deepseek-v41-handoff.md).
pub const geometry = @import("geometry.zig");
pub const codec = @import("codec.zig");
pub const fp4 = @import("fp4.zig");
pub const plan = @import("plan.zig");
pub const export_cuda = @import("export.zig");
pub const metal = @import("metal.zig");
pub const import_metal = @import("import.zig");
