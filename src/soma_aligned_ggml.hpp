#pragma once
#include "soma_aligned_dot.hpp"
#include <ggml.h>

namespace motionbricks::detail {
// CPU-only diagnostic operation. Callers supply half-rounded FP32 operands.
inline ggml_tensor * soma_aligned_product(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b,bool split8=false,unsigned nvjet_accumulators=0,bool residue_first=true) {
    static int single_parts=1,split_parts=8,nvjet_mode=3,nvjet_output_mode=18,nvjet_split_mode=4,residue_last_mode=2;
    GGML_ASSERT(!(split8 && nvjet_accumulators));
    GGML_ASSERT(residue_first || (!split8 && !nvjet_accumulators));
    GGML_ASSERT(nvjet_accumulators==0 || nvjet_accumulators==3 || nvjet_accumulators==4 || nvjet_accumulators==8);
    const int part_count=split8?8:1;
    GGML_ASSERT(a->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32);
    GGML_ASSERT(a->ne[0]==b->ne[0] && a->ne[0]%part_count==0);
    GGML_ASSERT(a->nb[0]==sizeof(float) && b->nb[0]==sizeof(float));
    GGML_ASSERT(b->ne[2]%a->ne[2]==0 && b->ne[3]%a->ne[3]==0);
    ggml_tensor * args[]={a,b};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,a->ne[1],b->ne[1],b->ne[2],b->ne[3],args,2,
        [](ggml_tensor * dst,int ith,int nth,void * userdata) {
            const auto mode=*static_cast<const int *>(userdata);
            const auto parts=static_cast<std::size_t>(mode==2||mode==3||mode==4||mode==18?1:mode);
            const auto * weight=dst->src[0],*input=dst->src[1];
            const auto width=static_cast<std::size_t>(weight->ne[0])/parts;
            const auto round=[](float x){return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));};
            for(int64_t index=ith;index<ggml_nelements(dst);index+=nth) {
                auto remaining=index;
                const auto n=remaining%dst->ne[0];remaining/=dst->ne[0];
                const auto m=remaining%dst->ne[1];remaining/=dst->ne[1];
                const auto i2=remaining%dst->ne[2],i3=remaining/dst->ne[2];
                const auto * arow=reinterpret_cast<const float *>(static_cast<const char *>(weight->data)+
                    n*weight->nb[1]+(i2/(input->ne[2]/weight->ne[2]))*weight->nb[2]+
                    (i3/(input->ne[3]/weight->ne[3]))*weight->nb[3]);
                const auto * brow=reinterpret_cast<const float *>(static_cast<const char *>(input->data)+
                    m*input->nb[1]+i2*input->nb[2]+i3*input->nb[3]);
                float sum=0;
                if(mode==3||mode==4||mode==18) {
                    static_cast<float *>(dst->data)[index]=mode==4?soma_nvjet_split4_dot(arow,brow,width):soma_nvjet_dot(arow,brow,width,mode==3?3:8);
                    continue;
                }
                for(std::size_t part=0;part<parts;++part) {
                    const auto partial=soma_aligned_dot(arow+part*width,brow+part*width,width,mode!=2);
                    sum+=parts==1?partial:round(partial);
                }
                static_cast<float *>(dst->data)[index]=sum;
            }
        },GGML_N_TASKS_MAX,!residue_first?&residue_last_mode:nvjet_accumulators==4?&nvjet_split_mode:nvjet_accumulators==8?&nvjet_output_mode:nvjet_accumulators==3?&nvjet_mode:split8?&split_parts:&single_parts);
}
inline ggml_tensor * soma_aligned_split8(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    return soma_aligned_product(ctx,a,b,true);
}
}
