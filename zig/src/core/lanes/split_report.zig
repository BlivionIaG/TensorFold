//! What a layer-split prompt cost, for the reply's runtime block.

/// Nanoseconds: the pull's timings and the engine's own passes.
pub const SplitReport = struct {
    /// the pass entered at this layer from streamed rows (0: no split ran)
    cut: u32 = 0,
    rows: u64 = 0,
    passes: u32 = 0,
    first_row_ns: u64 = 0,
    rows_ns: u64 = 0,
    state_ns: u64 = 0,
    wait_ns: u64 = 0,
    compute_ns: u64 = 0,
    state_wait_ns: u64 = 0,
    restore_ns: u64 = 0,
    bytes: u64 = 0,
    /// the split failed and the prompt was prefilled here from 0
    fell_back: bool = false,
};
