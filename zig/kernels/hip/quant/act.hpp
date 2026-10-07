#pragma once

// ActEncoders: how a tile reads the activation rows. The identity encoders hand the rows over as the Dot's own element
// type, unchanged; an encoder that quantizes (int8 per token, fp8) turns the rows into dot operands once per launch
// and carries its scales to the epilogue.

#include "common/dot2.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

template <class T>
struct IdentityAct {
    using elem = typename T::elem;

    // Row r of the block's x (a plan's pair r reads its x row), `k` elements.
    template <class Args>
    __device__ static const elem* row(const Args& a, int r) {
        return static_cast<const elem*>(a.x) + x_row(a, r) * a.k;
    }
};

using F16Act = IdentityAct<DotF16>;
using BF16Act = IdentityAct<DotBF16>;

}  // namespace rocm
}  // namespace tf
