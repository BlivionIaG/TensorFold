// The dot2 tiles of dot2_tiles.hip again, with the expert-offload code (TF_EXPERT_PTRS: pointer-table experts, read-through admission), for the routed
// launches of an offloaded stack only. A chip without caps `expert_ptrs` builds this group empty; on gfx900 the plain tiles stay free of that code.

#if defined(TF_EXPERT_PTRS)
#include "tiles/dot2_tiles.hip"
#endif
