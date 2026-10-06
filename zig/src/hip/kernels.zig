//! Our .hip kernels as offload bundles (one code object a gfx target) built at build time, embedded in the binary.

const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no hipcc, no prebuilt bundles); every image is then empty.
pub const available = options.with_kernels;

/// The gfx targets the bundles were built for, comma separated.
pub const targets: []const u8 = options.gfx;

pub const probe: []const u8 = if (available) &Blob("hsaco_probe").bytes else &.{};

/// The ROCm kernel libraries, a GPU family each (libtf_rdna2.so for gfx103x, libtf_rdna3.so for gfx11 / gfx12).
pub const rdna2: []const u8 = if (options.with_rdna2) &Blob("lib_rdna2").bytes else &.{};
pub const rdna3: []const u8 = if (options.with_rdna3) &Blob("lib_rdna3").bytes else &.{};
