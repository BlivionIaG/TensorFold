//! Our .hip kernels as offload bundles (one code object a gfx target) built at build time, embedded in the binary.

const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no hipcc); every image is then empty.
pub const available = options.with_kernels;

/// The gfx targets the bundles were built for, comma separated.
pub const targets: []const u8 = options.gfx;

pub const probe: []const u8 = if (available) &Blob("hsaco_probe").bytes else &.{};
pub const affine: []const u8 = if (available) &Blob("hsaco_affine").bytes else &.{};
