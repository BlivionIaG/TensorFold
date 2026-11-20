// Planted faults of glp.cuh for the contract's controls (built only with the controls): each must fail its gate.
// SUM keeps the copies' sum where the mean belongs; HALF accumulates the dot in fp16; DROP_TAIL loses a row's last
// column; DROP_NONFINITE drops non-finite terms; ATOMIC adds the partials unordered; TRUNCATE rounds by truncation.

#include "glp.cuh"

using namespace dsv41_glp;

#define GLP_CONTROL(NAME, CTL)                                                                                     \
    extern "C" __global__ void __launch_bounds__(NT) dsv41_glp_stream_##NAME(const StreamArgs a) {                 \
        stream_body<CTL>(a);                                                                                       \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(NT) dsv41_glp_writer_##NAME(const WriterArgs a) {                 \
        writer_body<CTL>(a);                                                                                       \
    }

GLP_CONTROL(sum, SUM)
GLP_CONTROL(half, HALF)
GLP_CONTROL(drop_tail, DROP_TAIL)
GLP_CONTROL(drop_nonfinite, DROP_NONFINITE)
GLP_CONTROL(atomic, ATOMIC)
GLP_CONTROL(truncate, TRUNCATE)
