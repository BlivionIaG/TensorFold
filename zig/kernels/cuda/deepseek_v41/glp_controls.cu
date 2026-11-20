// Planted GLPv3 faults: wrong mean, FP16 dot, missing tail, dropped nonfinite values, unordered sums and truncation.

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
