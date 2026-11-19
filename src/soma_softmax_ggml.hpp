#pragma once
#include "soma_softmax.hpp"
#include <ggml.h>
namespace motionbricks::detail {
inline ggml_tensor * soma_calibrated_softmax(ggml_context * ctx,ggml_tensor * input) {
    GGML_ASSERT(input->type==GGML_TYPE_F32);
    if(!ggml_is_contiguous(input))input=ggml_cont(ctx,input);
    ggml_tensor * args[]={input};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1],input->ne[2],input->ne[3],args,1,
        [](ggml_tensor * dst,int ith,int nth,void *) {
            const auto width=static_cast<std::size_t>(dst->ne[0]);
            const auto rows=static_cast<std::size_t>(ggml_nelements(dst))/width;
            for(std::size_t row=static_cast<std::size_t>(ith);row<rows;row+=static_cast<std::size_t>(nth))
                soma_warp_softmax(static_cast<const float *>(dst->src[0]->data)+row*width,
                    static_cast<float *>(dst->data)+row*width,width);
        },GGML_N_TASKS_MAX,nullptr);
}
}
