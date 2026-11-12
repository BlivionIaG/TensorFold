// The dot2 tiles again with the expert-offload code (TF_EXPERT_PTRS), for routed launches of an offloaded stack only.

#if defined(TF_EXPERT_PTRS)
#include "tiles/dot2_tiles.hip"
#endif
