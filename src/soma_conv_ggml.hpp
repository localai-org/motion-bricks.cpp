#pragma once
#include "soma_aligned_ggml.hpp"

namespace motionbricks::detail {
// CPU-only diagnostic for the profiled SOMA convolution families. Operands
// are half-rounded FP32 im2col/kernel matrices in channel/spatial order.
inline ggml_tensor * soma_profiled_conv_product(ggml_context * ctx,ggml_tensor * left,
                                               ggml_tensor * right,int64_t spatial,
                                               int64_t channels,int dilation) {
    GGML_ASSERT((spatial==3 && (channels==160 || channels==512)) ||
                (spatial==1 && channels==512));
    GGML_ASSERT(left->ne[0]==spatial*channels && right->ne[0]==left->ne[0]);
    if(spatial==1)return soma_aligned_product(ctx,left,right,false,8);
    // The 160-channel first decoder Conv selects CUTLASS K=32 tiles;
    // the 512-channel root/decoder Conv families use cuDNN K=64 tiles.
    const int64_t tile=channels==160?32:64;
    const auto tiled=[&](ggml_tensor * value) {
        auto * shaped=ggml_reshape_4d(ctx,value,spatial,tile,channels/tile,value->ne[1]);
        return ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,shaped,1,0,2,3)),
                              value->ne[0],value->ne[1]);
    };
    left=tiled(left);right=tiled(right);
    if(channels==512 && dilation>=9) {
        // Twelve CUDA blocks launch; eight 64-channel slices contain data.
        // Each active slice spans all three spatial positions. Preserve FP32
        // partials and sequential additions before the caller's half store.
        ggml_tensor * output=nullptr;
        for(int part=0;part<8;++part) {
            const auto slice=[&](ggml_tensor * value) {
                return ggml_view_2d(ctx,value,192,value->ne[1],value->nb[1],
                                    static_cast<std::size_t>(part)*192*value->nb[0]);
            };
            auto * partial=soma_aligned_product(ctx,slice(left),slice(right));
            output=output?ggml_add(ctx,output,partial):partial;
        }
        return output;
    }
    return soma_aligned_product(ctx,left,right);
}
}
