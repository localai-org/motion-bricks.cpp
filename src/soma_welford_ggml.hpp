#pragma once
#include "soma_welford.hpp"
#include <ggml.h>

namespace motionbricks::detail {
inline ggml_tensor * soma_welford_norm(ggml_context * ctx,ggml_tensor * input,
                                       ggml_tensor * scale,ggml_tensor * bias) {
    GGML_ASSERT(input->type==GGML_TYPE_F32 && scale->type==GGML_TYPE_F32 && bias->type==GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(scale) && ggml_is_contiguous(bias));
    GGML_ASSERT(input->ne[0]==512 || input->ne[0]==1024);
    GGML_ASSERT(ggml_nelements(scale)==input->ne[0] && ggml_nelements(bias)==input->ne[0]);
    ggml_tensor * args[]={input,scale,bias};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1],input->ne[2],input->ne[3],args,3,
        [](ggml_tensor * dst,int ith,int nth,void *) {
            const auto width=static_cast<std::size_t>(dst->ne[0]);
            const auto rows=static_cast<std::size_t>(ggml_nelements(dst))/width;
            for(std::size_t row=static_cast<std::size_t>(ith);row<rows;row+=static_cast<std::size_t>(nth))
                soma_welford_normalize(static_cast<const float *>(dst->src[0]->data)+row*width,
                    static_cast<const float *>(dst->src[1]->data),static_cast<const float *>(dst->src[2]->data),
                    static_cast<float *>(dst->data)+row*width,width,1.0e-5F);
        },GGML_N_TASKS_MAX,nullptr);
}
}
