//! The GLP producer's capture kernel on this GPU, against a float64 reduction of the same chunk.
//! Source-built: no checkpoint, one small chunk, the four copies the fold keeps per token.
test {
    _ = @import("src/families/deepseek_v41/metal/glp_capture.zig");
}
