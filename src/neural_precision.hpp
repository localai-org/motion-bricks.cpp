#pragma once
#include <cstdlib>
#include <cstring>
// IEEE half rounding with FP32 accumulation. This is an explicit arithmetic
// contract, not a claim that the backend used FP16 hardware instructions.
namespace motionbricks::detail {
inline thread_local int neural_native_half_override = -1;
inline bool native_half_operands_requested() {
    if(neural_native_half_override>=0)return neural_native_half_override!=0;
    const char * value=std::getenv("MOTIONBRICKS_SOMA_NATIVE_F16");
    return value && std::strcmp(value,"1")==0;
}
inline thread_local bool neural_half_arithmetic = false;
// Scoped by public calibrated CPU requests; diagnostic runners may also set these.
inline thread_local bool neural_aligned_root_trial = false;
inline thread_local bool neural_aligned_matmul_trial = false;
inline thread_local bool neural_backend_split_root_trial = false;
inline thread_local bool neural_backend_prefix_trial = false;
inline thread_local bool neural_backend_nvjet_split4_trial = false;
inline thread_local bool neural_backend_pose_scores_trial = false;
inline thread_local bool neural_backend_decoder_conv_trial = false;
inline thread_local bool neural_backend_conditioning_trial = false;
inline thread_local bool neural_backend_softmax_trial = false;
inline thread_local bool neural_backend_root_conv_trial = false;
inline thread_local bool neural_backend_nvjet8_trial = false;
inline thread_local bool neural_backend_nvjet3_trial = false;
inline thread_local bool neural_backend_norm_trial = false;
inline thread_local bool neural_welford_trial = false;
inline thread_local bool neural_softmax_trial = false;
// Public API requests select arithmetic explicitly and restore caller state.
class soma_public_arithmetic_scope {
    bool root_=neural_aligned_root_trial, matmul_=neural_aligned_matmul_trial;
    bool split_=neural_backend_split_root_trial, norm_=neural_welford_trial, softmax_=neural_softmax_trial;
    int native_=neural_native_half_override;
    bool prefix_=neural_backend_prefix_trial, backend_norm_=neural_backend_norm_trial, nvjet3_=neural_backend_nvjet3_trial, pose_scores_=neural_backend_pose_scores_trial, decoder_conv_=neural_backend_decoder_conv_trial, conditioning_=neural_backend_conditioning_trial, backend_softmax_=neural_backend_softmax_trial, root_conv_=neural_backend_root_conv_trial, nvjet8_=neural_backend_nvjet8_trial, nvjet_split4_=neural_backend_nvjet_split4_trial;
public:
    explicit soma_public_arithmetic_scope(bool calibrated=false, bool vulkan_calibrated=false) {
        neural_aligned_root_trial=neural_aligned_matmul_trial=calibrated;
        neural_welford_trial=neural_softmax_trial=calibrated;
        neural_backend_split_root_trial=false;neural_backend_prefix_trial=false;neural_backend_norm_trial=false;neural_backend_nvjet3_trial=false;neural_backend_nvjet8_trial=false;neural_backend_root_conv_trial=false;neural_backend_softmax_trial=false;neural_backend_conditioning_trial=false;neural_backend_decoder_conv_trial=false;neural_backend_pose_scores_trial=false;neural_backend_nvjet_split4_trial=false;
        neural_native_half_override=0;
        if(vulkan_calibrated) {
            neural_backend_split_root_trial=neural_backend_prefix_trial=neural_backend_norm_trial=true;
            neural_backend_nvjet3_trial=neural_backend_nvjet8_trial=neural_backend_nvjet_split4_trial=true;
            neural_backend_root_conv_trial=neural_backend_decoder_conv_trial=true;
            neural_backend_pose_scores_trial=neural_backend_conditioning_trial=neural_backend_softmax_trial=true;
            neural_native_half_override=1;
        }
    }
    ~soma_public_arithmetic_scope() {
        neural_aligned_root_trial=root_;neural_aligned_matmul_trial=matmul_;
        neural_backend_split_root_trial=split_;neural_welford_trial=norm_;
        neural_softmax_trial=softmax_;neural_native_half_override=native_;
        neural_backend_prefix_trial=prefix_;neural_backend_norm_trial=backend_norm_;neural_backend_nvjet3_trial=nvjet3_;neural_backend_nvjet8_trial=nvjet8_;neural_backend_root_conv_trial=root_conv_;neural_backend_softmax_trial=backend_softmax_;neural_backend_conditioning_trial=conditioning_;neural_backend_decoder_conv_trial=decoder_conv_;neural_backend_pose_scores_trial=pose_scores_;neural_backend_nvjet_split4_trial=nvjet_split4_;
    }
    soma_public_arithmetic_scope(const soma_public_arithmetic_scope &)=delete;
    soma_public_arithmetic_scope & operator=(const soma_public_arithmetic_scope &)=delete;
};
class neural_precision_scope {
    bool previous_;
public:
    explicit neural_precision_scope(bool half) : previous_(neural_half_arithmetic) { neural_half_arithmetic=half; }
    ~neural_precision_scope() { neural_half_arithmetic=previous_; }
    neural_precision_scope(const neural_precision_scope &)=delete;
    neural_precision_scope & operator=(const neural_precision_scope &)=delete;
};
}
#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml.h>
#include "soma_aligned_ggml.hpp"
#include "soma_conv_ggml.hpp"
#include "soma_welford_ggml.hpp"
#include "soma_softmax_ggml.hpp"
#include <cmath>
namespace motionbricks::detail {
inline ggml_tensor * precision_norm(ggml_context * ctx,ggml_tensor * input,ggml_tensor * scale,ggml_tensor * bias) {
    if(neural_half_arithmetic && neural_backend_norm_trial) {
        // Diagnostic protocol implemented only by the experimental Vulkan shader.
        const auto width=input->ne[0], rows=ggml_nelements(input)/width;
        if(width!=512 && width!=1024)throw std::runtime_error("unsupported Vulkan Welford width");
        auto * flat=ggml_reshape_2d(ctx,ggml_cont(ctx,input),width,rows);
        auto * affine=ggml_concat(ctx,ggml_reshape_2d(ctx,ggml_cont(ctx,scale),width,1),
                                  ggml_reshape_2d(ctx,ggml_cont(ctx,bias),width,1),1);
        auto * joined=ggml_concat(ctx,flat,affine,1);
        auto * normalized=ggml_norm(ctx,joined,-1.0e-5F);
        auto * trimmed=ggml_cont(ctx,ggml_view_2d(ctx,normalized,width,rows,width*sizeof(float),0));
        return ggml_reshape(ctx,trimmed,input);
    }
    if(neural_half_arithmetic && neural_welford_trial)return soma_welford_norm(ctx,input,scale,bias);
    return ggml_add(ctx,ggml_mul(ctx,ggml_norm(ctx,input,1.0e-5F),scale),bias);
}
inline ggml_tensor * half_round(ggml_context * ctx, ggml_tensor * x) {
    if(!neural_half_arithmetic)return x;
    return ggml_cast(ctx,ggml_cast(ctx,x,GGML_TYPE_F16),GGML_TYPE_F32);
}
inline ggml_tensor * precision_add(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    return half_round(ctx,ggml_add(ctx,half_round(ctx,a),half_round(ctx,b)));
}
inline ggml_tensor * precision_sub(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    return half_round(ctx,ggml_sub(ctx,half_round(ctx,a),half_round(ctx,b)));
}
inline ggml_tensor * precision_mul(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    return half_round(ctx,ggml_mul(ctx,half_round(ctx,a),half_round(ctx,b)));
}
inline ggml_tensor * precision_product(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    if(neural_half_arithmetic && neural_aligned_matmul_trial)
        return soma_aligned_product(ctx,half_round(ctx,a),half_round(ctx,b));
    if(neural_half_arithmetic && native_half_operands_requested()) {
        auto * product=ggml_mul_mat(ctx,ggml_cast(ctx,a,GGML_TYPE_F16),ggml_cast(ctx,b,GGML_TYPE_F16));
        ggml_mul_mat_set_prec(product,GGML_PREC_F32);
        return product;
    }
    return ggml_mul_mat(ctx,half_round(ctx,a),half_round(ctx,b));
}
// Match the narrow CUDA projection's first K%16 group, then 16-wide groups.
// Pad input rows past the Vulkan vector-kernel threshold; return only real rows.
inline ggml_tensor * precision_backend_prefix_product(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    GGML_ASSERT(a->ne[0]==b->ne[0] && a->ne[0]>0);
    const auto prefix=[&](ggml_tensor * value) {
        const auto remainder=value->ne[0]%16;
        if(!remainder)return value;
        auto * first=ggml_view_4d(ctx,value,remainder,value->ne[1],value->ne[2],value->ne[3],
            value->nb[1],value->nb[2],value->nb[3],0);
        auto * rest=ggml_view_4d(ctx,value,value->ne[0]-remainder,value->ne[1],value->ne[2],value->ne[3],
            value->nb[1],value->nb[2],value->nb[3],remainder*value->nb[0]);
        return ggml_concat(ctx,ggml_pad(ctx,ggml_cont(ctx,first),static_cast<int>(16-remainder),0,0,0),ggml_cont(ctx,rest),0);
    };
    auto * lhs=prefix(a),*rhs=prefix(b);
    const auto rows=b->ne[1];
    if(rows<=8)rhs=ggml_pad(ctx,rhs,0,static_cast<int>(16-rows),0,0);
    auto * product=ggml_mul_mat(ctx,ggml_cast(ctx,lhs,GGML_TYPE_F16),ggml_cast(ctx,rhs,GGML_TYPE_F16));
    ggml_mul_mat_set_prec(product,GGML_PREC_F32);
    if(rows<=8)product=ggml_cont(ctx,ggml_view_4d(ctx,product,product->ne[0],rows,product->ne[2],product->ne[3],
        product->nb[1],product->nb[2],product->nb[3],0));
    return product;
}
// Preserve CUDA NVJET's persistent accumulation streams.
inline ggml_tensor * precision_backend_nvjet(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b,int groups=3) {
    GGML_ASSERT(groups==3 || groups==8);
    GGML_ASSERT(a->ne[0]==b->ne[0] && (a->ne[0]%128==0 || (groups==8 && a->ne[0]==64)));
    ggml_tensor * sum=nullptr;
    for(int group=0;group<std::min<int64_t>(groups,a->ne[0]/16);++group) {
        const auto gather=[&](ggml_tensor * value) {
            ggml_tensor * joined=nullptr;
            for(int64_t offset=0;offset<value->ne[0];offset+=16) {
                if(((offset%128)/16)%groups!=group)continue;
                auto * block=ggml_cont(ctx,ggml_view_4d(ctx,value,16,value->ne[1],value->ne[2],value->ne[3],
                    value->nb[1],value->nb[2],value->nb[3],offset*value->nb[0]));
                joined=joined?ggml_concat(ctx,joined,block,0):block;
            }
            return joined;
        };
        auto * partial=precision_backend_prefix_product(ctx,gather(a),gather(b));
        sum=sum?ggml_add(ctx,sum,partial):partial;
    }
    return sum;
}
// CUDA split-K uses four interleaved 128-wide tiles, each with three streams.
inline ggml_tensor * precision_backend_nvjet_split4(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    GGML_ASSERT(a->ne[0]==b->ne[0] && a->ne[0]%512==0);
    ggml_tensor * sum=nullptr;
    for(int part=0;part<4;++part) {
        const auto gather=[&](ggml_tensor * value) {
            ggml_tensor * joined=nullptr;
            for(int64_t offset=part*128;offset<value->ne[0];offset+=512) {
                auto * block=ggml_cont(ctx,ggml_view_4d(ctx,value,128,value->ne[1],value->ne[2],value->ne[3],
                    value->nb[1],value->nb[2],value->nb[3],offset*value->nb[0]));
                joined=joined?ggml_concat(ctx,joined,block,0):block;
            }
            return joined;
        };
        auto * partial=precision_backend_nvjet(ctx,gather(a),gather(b),3);
        sum=sum?ggml_add(ctx,sum,partial):partial;
    }
    return sum;
}
// Match the profiled CUDA channel/spatial traversal using Vulkan matrix products.
inline ggml_tensor * precision_backend_conv(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b,
                                            int64_t spatial,int64_t channels,int dilation) {
    GGML_ASSERT((spatial==3 && (channels==160 || channels==512)) || (spatial==1 && channels==512));
    GGML_ASSERT(a->ne[0]==spatial*channels && b->ne[0]==a->ne[0]);
    if(spatial==1)return precision_backend_nvjet(ctx,a,b,8);
    const int64_t tile=channels==160?32:64;
    const auto tiled=[&](ggml_tensor * value) {
        auto * shaped=ggml_reshape_4d(ctx,value,spatial,tile,channels/tile,value->ne[1]);
        return ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,shaped,1,0,2,3)),value->ne[0],value->ne[1]);
    };
    a=tiled(a);b=tiled(b);
    if(channels==512 && dilation>=9) {
        ggml_tensor * sum=nullptr;
        for(int part=0;part<8;++part) {
            const auto slice=[&](ggml_tensor * value) {
                return ggml_cont(ctx,ggml_view_2d(ctx,value,192,value->ne[1],value->nb[1],part*192*value->nb[0]));
            };
            auto * partial=precision_backend_prefix_product(ctx,slice(a),slice(b));
            sum=sum?ggml_add(ctx,sum,partial):partial;
        }
        return sum;
    }
    return precision_backend_prefix_product(ctx,a,b);
}
inline ggml_tensor * precision_matmul(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b) {
    return half_round(ctx,precision_product(ctx,a,b));
}
inline ggml_tensor * precision_backend_split8(ggml_context * ctx,ggml_tensor * a,ggml_tensor * b,bool force_matrix=false) {
    GGML_ASSERT(a->ne[0]==b->ne[0] && a->ne[0]%8==0);
    const auto width=a->ne[0]/8;
    ggml_tensor * sum=nullptr;
    for(int part=0;part<8;++part) {
        const auto slice=[&](ggml_tensor * value) {
            return ggml_view_4d(ctx,value,width,value->ne[1],value->ne[2],value->ne[3],
                value->nb[1],value->nb[2],value->nb[3],part*width*value->nb[0]);
        };
        auto * partial=force_matrix?half_round(ctx,precision_backend_prefix_product(ctx,slice(a),slice(b))):
            precision_matmul(ctx,slice(a),slice(b));
        sum=sum?ggml_add(ctx,sum,partial):partial;
    }
    return sum;
}
inline ggml_tensor * precision_linear(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,ggml_tensor * bias,bool fused) {
    auto * product=precision_product(ctx,weight,input);
    const std::string projection=ggml_get_name(weight);
    const bool trace_projection=projection=="_proj_local_root_value.weight" || projection=="_proj_global_root_value.weight";
    const auto stem=trace_projection?projection.substr(0,projection.size()-7):std::string{};
    if(trace_projection)ggml_set_name(product,(stem+".matmul_f32").c_str());
    if(!fused)product=half_round(ctx,product);
    if(trace_projection)ggml_set_name(product,(stem+".matmul_half").c_str());
    return half_round(ctx,ggml_add(ctx,product,half_round(ctx,bias)));
}
inline ggml_tensor * precision_convolution_bias(ggml_context * ctx,ggml_tensor * product,ggml_tensor * bias) {
    // The pinned CUDA Conv path stores a half product before adding its bias.
    // This is intentionally different from the fused Gemm arithmetic above.
    return precision_add(ctx,product,bias);
}
inline ggml_tensor * precision_blend(ggml_context * ctx,ggml_tensor * fallback,ggml_tensor * selected,ggml_tensor * mask) {
    // For binary masks and finite operands this preserves the selected value.
    // fallback + (selected-fallback)*mask can lose small selected values through
    // cancellation even in F32, before a subsequent half cast.
    fallback=half_round(ctx,fallback);selected=half_round(ctx,selected);
    fallback=ggml_repeat(ctx,fallback,selected);
    auto * inverse=ggml_scale_bias(ctx,mask,-1.0F,1.0F);
    return half_round(ctx,ggml_add(ctx,ggml_mul(ctx,selected,mask),ggml_mul(ctx,fallback,inverse)));
}
inline ggml_tensor * precision_soft_max(ggml_context * ctx,ggml_tensor * input,ggml_tensor * mask=nullptr) {
    if(neural_half_arithmetic && neural_softmax_trial)
        return soma_calibrated_softmax(ctx,mask?ggml_add(ctx,input,mask):input);
    // Negative max_bias is the explicit scoped protocol for the SOMA shader.
    return ggml_soft_max_ext(ctx,input,mask,1.0F,
        neural_half_arithmetic && neural_backend_softmax_trial?-1.0F:0.0F);
}
inline ggml_tensor * precision_attention(ggml_context * ctx,ggml_tensor * key,ggml_tensor * query,
                                        ggml_tensor * mask,float width,const char * score_name=nullptr);
inline ggml_tensor * precision_scores(ggml_context * ctx,ggml_tensor * key,ggml_tensor * query,float width) {
    if(!neural_half_arithmetic)return ggml_scale(ctx,ggml_mul_mat(ctx,key,query),1/std::sqrt(width));
    const auto round=[](float x){return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));};
    const float factor=round(std::sqrt(round(1/std::sqrt(width))));
    auto * k=half_round(ctx,ggml_scale(ctx,half_round(ctx,key),factor));
    auto * q=half_round(ctx,ggml_scale(ctx,half_round(ctx,query),factor));
    if(neural_backend_pose_scores_trial && width==64)
        return half_round(ctx,precision_backend_nvjet(ctx,k,q,8));
    if(neural_aligned_matmul_trial && width==64)
        return half_round(ctx,soma_aligned_product(ctx,k,q,false,8));
    return precision_matmul(ctx,k,q);
}
inline ggml_tensor * precision_attention(ggml_context * ctx,ggml_tensor * key,ggml_tensor * query,
                                        ggml_tensor * mask,float width,const char * score_name) {
    if(!neural_half_arithmetic)return ggml_soft_max_ext(ctx,ggml_mul_mat(ctx,key,query),mask,1/std::sqrt(width),0);
    auto * product=precision_scores(ctx,key,query,width);
    if(score_name)ggml_set_name(product,score_name);
    return precision_soft_max(ctx,product,mask);
}
}
#endif
