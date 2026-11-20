// GLP steering kernels (ours, see glp.cuh): TF_DSV41's --glp fold and attention-writer edits and the steered final.
// Plain C symbols, one image only when a vector is served: dsv41_glp_stream / _writer / _final.

#include "glp.cuh"

using namespace dsv41_glp;

extern "C" {

__global__ void __launch_bounds__(NT) dsv41_glp_stream(const StreamArgs a) { stream_body<NONE>(a); }

__global__ void __launch_bounds__(NT) dsv41_glp_writer(const WriterArgs a) { writer_body<NONE>(a); }

__global__ void __launch_bounds__(NT) dsv41_glp_final(const FinalArgs a) { final_body<NONE>(a); }

}
